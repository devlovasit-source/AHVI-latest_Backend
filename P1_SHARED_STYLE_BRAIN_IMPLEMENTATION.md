# P1 — Shared Style Brain: Implementation Plan (A + B + C only)

Backend-engineer-ready. **No code in this doc is applied.** No edits, commits, deploy.
Scope: Phase A (canonical context), Phase B (inject occasion), Phase C (post-generation visual guard). **Step D (UnifiedStyleScorer in visual path) is explicitly OUT.**

Insertion points are anchored to **adjacent function calls** (grep anchors), not line numbers, since the file shifts.

Global flag (all phases gated, default ON for A/B, ON for C): `STYLE_SHARED_BRAIN` env → `os.getenv("STYLE_SHARED_BRAIN","true")`. Lets ops disable instantly without redeploy.

---

## PHASE A — `build_canonical_style_context()`

### File / function
- **New function** in `services/style_context_service.py` (next to `build_pairing_persona` @ ~:567 and `build_style_context` @ ~:295).
- Reuses existing: `_resolve_gender` (style_context_service:549), `compact_style_dna` (:496), and `build_brief` (brain/engines/style_brief.py:798).

### Signature
```
def build_canonical_style_context(
    *, query: str, user_profile: dict, intent: dict | None = None,
    router_occasion: str | None = None, weather: Any = None,
    event_context: dict | None = None, style_dna: Any = None,
) -> dict
```

### Pseudocode
```
def build_canonical_style_context(...):
    profile = user_profile or {}
    # 1. canonical occasion via the SAME engine the wardrobe path uses
    brief = build_brief(query, router_occasion=router_occasion,
                        agent_payload=intent or {}, weather=weather)   # style_brief.py:798
    canonical_occasion = brief.get("occasion") or "daily"
    # 2. gender (single resolver, prompt-override aware handled by caller)
    gender = _resolve_gender(profile)                                  # :549
    # 3. dna compacted
    dna = compact_style_dna(style_dna or profile.get("style_dna"),
                            profile.get("preferences"))                # :496
    ctx = {
        "canonical_occasion": canonical_occasion,
        "occasion_brief": brief,             # keep full brief for archetype/threshold
        "gender": gender,                    # male|female|unisex|unknown
        "style_dna": dna,
        "profile": profile,
        "weather": weather,
        "event_context": event_context or {},
    }
    logger.info("AHVI_CANONICAL_CTX occ=%s gender=%s dna=%s weather=%s",
                canonical_occasion, gender, bool(dna), bool(weather))
    return ctx
```

### New helper functions
- none beyond the function itself (all deps exist).

### Logging
- `AHVI_CANONICAL_CTX occ=… gender=… dna=… weather=…` (one per request).

### Tests (`tests/test_canonical_style_context.py`)
- `haldi` → canonical_occasion == "wedding" (matches `_ahvi_style_occasion` family) ; gender passthrough female→female.
- empty query + router_occasion="office" → canonical_occasion=="office".
- no dna → `style_dna` empty dict, no crash.
- weather passed → present in ctx.

### Risk: LOW. Pure assembly over already-used functions. No payload/Gemini change.

---

## PHASE B — inject `canonical_occasion` into `select_archetypes()` + `_asset_score()`

### File / function
- `services/style_reasoning_engine.py`, inside the visual entry (`reason()` → `_build_response`).
- Call site to build ctx: where persona is built today (anchor: the `build_pairing_persona(...)` call + the `select_archetypes(...)` call ~ the `from services.stylist_knowledge_service import select_archetypes` block).

### Insertion point
1. After resolving `user_profile`/persona and BEFORE `select_archetypes(...)`, add:
```
ctx = build_canonical_style_context(
    query=query, user_profile=_uprof, intent=intent_dict,
    router_occasion=str(context.get("occasion") or category or "") or None,
    weather=context.get("weather"), style_dna=_dna_raw,
)
occ = ctx["canonical_occasion"]          # replaces ad-hoc occasion string
gender = ctx["gender"]                    # replaces separate _resolve_asset_gender call (keep prompt override)
```
2. Replace the three occasion expressions `str(context.get("occasion") or category or "")` (the `select_archetypes(occasion=…)` arg + the `_enrich_*`/anchor `occasion=` args) with `occ`.
3. `select_archetypes(occasion=occ, gender=gender, style_dna=dna, ...)` — already supports both args (stylist_knowledge_service).

### `_asset_score` change (style_reasoning_engine.py:2546)
- Add optional kw `canonical_occasion` and use it for the occasion-match term instead of the loosely-passed occasion. Minimal: pass `occ` wherever `_asset_score(...)` is called so the occasion term is the canonical one. No signature break (default to current behavior if absent).

### New helper functions
- none (reuses Phase A + existing `select_archetypes`).

### Logging
- `AHVI_VISUAL_OCCASION_CANON raw=%s canonical=%s` at the substitution point.

### Tests (`tests/test_visual_canonical_routing.py`)
- conference → canonical occasion routes to professional archetype pool (not collapse).
- airport → travel pool.
- haldi/mehendi → festive pools (yellow→haldi, green→mehendi already in select_archetypes).
- funeral → somber_formal pool.
- assert `select_archetypes` receives canonical occasion (mock/spy).

### Risk: LOW–MED. Changes *inputs* to existing scoring; output shape unchanged. Verify no other reader depends on the raw `context["occasion"]` string downstream (grep `context.get("occasion")` in the file).

---

## PHASE C — post-generation visual guard (highest ROI)

### File / function
- `services/style_reasoning_engine.py`, in `_build_response`.
- **Insertion point (critical anchor):** immediately AFTER
  `visual_directions = _enrich_visual_directions_with_assets(...)`
  and BEFORE
  `visual_directions = _apply_editorial_polish(...)`.
```
visual_directions = _apply_style_guard(visual_directions, ctx)   # <-- NEW, here
```

### New helper: `_apply_style_guard(directions, ctx)`
```
def _apply_style_guard(directions, ctx):
    if os.getenv("STYLE_SHARED_BRAIN","true").lower() not in ("1","true","yes"):
        return directions
    from brain.engines.outfit_quality_guard import reject_board_for_occasion, normalize_occasion
    occ = normalize_occasion(ctx["canonical_occasion"])
    gender = ctx["gender"]
    out, stats = [], {"occ_reject":0,"color_drop":0,"gender_drop":0,"kept":0}
    for d in directions:
        # 1. occasion guard (whole-direction) — reuse wardrobe path's pure guard
        reject, reason = reject_board_for_occasion(d, occ)
        if reject:
            stats["occ_reject"] += 1
            logger.info("AHVI_VISUAL_GUARD_OCC_REJECT title=%r occ=%s reason=%s",
                        d.get("title"), occ, reason)
            # repair: try to strip the offending item set; if still reject, drop direction
            d = _repair_direction_for_occasion(d, occ) or None
            if d is None: continue
        # 2. color clash — drop assets/pieces clashing with the direction palette
        d = _strip_color_clashes(d, stats)
        # 3. gender — reuse existing sanitizer (already in file)
        allow_fem = gender == "female" or _prompt_allows_gendered_feminine_style(_q)
        d = _sanitize_direction_for_gender(d, target_gender=gender, allow_feminine=allow_fem)
        out.append(d); stats["kept"] += 1
    logger.info("AHVI_VISUAL_GUARD_SUMMARY occ=%s in=%d kept=%d %s",
                occ, len(directions), len(out), stats)
    return out or directions   # never return empty -> fall back to ungated
```
Notes:
- `reject_board_for_occasion` (outfit_quality_guard.py:196) + `normalize_occasion` (:121) — **reused as-is**, pure, LOW risk.
- `_sanitize_direction_for_gender` already exists in style_reasoning_engine.py — reuse.
- `or directions` fallback guarantees we never blank the screen.

### New helper: color compatibility (NO backend clash-map exists — must add one)
Reality: `color_harmony_bank.json` = feedback prose; `ahvi_color_pair_logic_v1.json` = gendered routing prose. **Neither is a machine clash-map.** `_asset_score` only *rewards* matches. So add a tiny data helper (mirror the proven Flutter `PairingEngine` map):
- **New file** `brain/engines/styling/color_compatibility.py`:
```
NEUTRALS = {"white","black","grey","gray","beige","cream","navy","tan","brown","denim","ivory","charcoal"}
HARMONY = {  # color -> set of compatible non-neutrals
  "burgundy": {"cream","beige","navy","grey","tan","white"},
  "green":    {"white","beige","tan","navy","black","cream"},
  "red":      {"black","white","navy","denim"},
  ...
}
CLASH = {  # explicit known-bad pairs (fast reject)
  frozenset({"burgundy","green"}), frozenset({"red","green"}),
  frozenset({"orange","pink"}), frozenset({"purple","brown"}), ...
}
def colors_clash(a, b) -> bool:
    a,b = a.lower(), b.lower()
    if a in NEUTRALS or b in NEUTRALS: return False
    if frozenset({a,b}) in CLASH: return True
    return b not in HARMONY.get(a, set()) and a not in HARMONY.get(b, set())
def palette_clashes(colors: list[str]) -> list[tuple[str,str]]:
    bad=[]; cs=[c.lower() for c in colors if c]
    for i,a in enumerate(cs):
        for b in cs[i+1:]:
            if colors_clash(a,b): bad.append((a,b))
    return bad
```
- `_strip_color_clashes(direction, stats)`: gather the direction's hero color + each asset's `colors`; if `colors_clash(hero, asset_color)` → drop that asset (re-pick next-best via existing `_asset_score`) and `stats["color_drop"]+=1`; log `AHVI_VISUAL_GUARD_COLOR_CLASH`.
- **Degradation (important):** if an asset has empty `colors`, clash check is **skipped** for it (log `AHVI_VISUAL_GUARD_COLOR_SKIP_NOCOLOR`). → effectiveness depends on the **209 missing colors** (P1-color track). Document this dependency.

### New helper: `_repair_direction_for_occasion(d, occ)`
- Remove the specific clashing piece flagged by `reject_board_for_occasion`'s reason (e.g. beanie in beach/airport, sandals in office); re-pick from candidate assets; return repaired direction or None if unrepairable.

### Logging (all under `AHVI_VISUAL_GUARD_*`)
- `AHVI_VISUAL_GUARD_OCC_REJECT`, `AHVI_VISUAL_GUARD_COLOR_CLASH`, `AHVI_VISUAL_GUARD_COLOR_SKIP_NOCOLOR`, `AHVI_VISUAL_GUARD_GENDER_DROP`, `AHVI_VISUAL_GUARD_SUMMARY occ in kept {stats}`.

### Frontend safety
- `_apply_style_guard` only mutates *which assets/pieces* appear inside each direction and may drop a direction; it returns the same list-of-dicts shape with the same keys. `visual_directions[]`/`cards[]`/`data.*` schema unchanged. Never returns empty (fallback).

### Risk: LOW for occasion+gender (pure reuse). LOW–MED for color (new helper + depends on color metadata). Latency: pure-python, negligible.

---

## Cross-phase test plan (`tests/test_visual_style_guard.py`) — pure, no Gemini/DB
| # | input | expected |
|---|---|---|
| 1 | direction hero=burgundy top + candidate green shirt | green dropped via `colors_clash`; `AHVI_VISUAL_GUARD_COLOR_CLASH` logged |
| 2 | beige loafers, male user, crop top candidate | crop top removed (`_sanitize_direction_for_gender`) |
| 3 | occasion=haldi | festive ethnic kept; western formal direction rejected/repaired |
| 4 | occasion=airport + beanie asset | beanie removed (`reject_board_for_occasion`) → "airport beanies die" |
| 5 | female user | only female assets survive guard |
| 6 | empty wardrobe | no wardrobe-match % (existing `has_wardrobe_signal` gate — assert unchanged) |
| 7 | "what to pair with beige loafers" | routes STYLE_PAIRING (existing) — guard still applies to any cards |
| 8 | asset with empty `colors` | clash skipped, asset kept, `..._COLOR_SKIP_NOCOLOR` logged (degradation proof) |
| 9 | guard rejects everything | returns original directions (no blank screen) |
| 10 | `STYLE_SHARED_BRAIN=false` | guard is a no-op (passthrough) |

---

## Engineer checklist (order)
1. Phase A function + tests (isolated, no wiring). 
2. Phase B wiring (swap occasion source) + tests; grep `context.get("occasion")` for other readers first.
3. Phase C: add `color_compatibility.py` + `_apply_style_guard` + `_strip_color_clashes` + `_repair_direction_for_occasion`; wire the single call after `_enrich_visual_directions_with_assets`; tests 1–10.
4. Ship behind `STYLE_SHARED_BRAIN`; canary on; watch `AHVI_VISUAL_GUARD_SUMMARY` logs.

## Hard dependency
Phase-C color clash is **blind on uncolored assets (209 missing colors)** → run the **color extraction track in parallel**; guard value scales with color coverage.

## Out of scope (do NOT implement now)
- Step D: `UnifiedStyleScorer.score_outfit` inside visual path (graph build + latency + context-shape). Later task.
- No prompt/Gemini changes. No frontend changes. No schema changes.
```
```
