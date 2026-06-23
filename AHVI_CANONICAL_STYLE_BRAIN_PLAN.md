# AHVI Canonical Style Brain — Architecture Plan (read-only audit)

Problem: occasion is classified by **5 independent, divergent resolvers**. "music festival"
becomes ethnic (kurta/bandhgala/mojari) because the visual path's own occasion→family map lumps
`festival` into `festive_general` and has **no concert/music-festival family**. No single canonical
brief governs board generation.

No code edited.

---

## The core defect: 5 occasion classifiers, no single source of truth

| # | resolver | file:loc | "festival" → |
|---|---|---|---|
| 1 | `detect_occasion_from_tokens` / `resolve_occasion_archetype` | `brain/engines/style_brief.py:267,308` (`_OCCASION_KEYWORDS` :133) | **rave** (`{rave,club,edm,festival}`) |
| 2 | `_EVENT_LEXICON` / `detect_multi_event` | `services/style_context_service.py:70` | no `festival` token → falls through |
| 3 | `normalize_occasion` | `brain/engines/outfit_quality_guard.py:121` | not mapped → passthrough (wedding cluster only) |
| 4 | `_occasion_category` | `services/style_reasoning_engine.py:3373` | own keyword buckets |
| 5 | `_OCCASION_FAMILY_RULES` / `_resolve_occasion_family` | `services/stylist_knowledge_service.py:1202,1256` | **festive_general → ethnic** ← the bug |

They disagree. The **visual board path uses #5**, which is why festival → kurta.

---

## Answers

**1. Where is occasion normalized?**
In all 5 places above. There is no one normalizer. `build_brief` (`style_brief.py:798`) is the closest
to canonical but only the wardrobe/outfit-pipeline path calls it; the visual path does not.

**2. Where does "festival" become ethnic/festive?**
`services/stylist_knowledge_service.py:1209-1210` — the `festive_general` rule list contains
`"festival"` (next to diwali/eid/ethnic). `_resolve_occasion_family` substring-matches "music
**festival**" → `festive_general` → `_FAMILY_ARCHETYPE_POOL["festive_general"]` (:1231) =
Festive Heritage / Refined Traditional / Celebration Kurta… whose `preferred_items` (:1157-1179) are
`bandhgala, silk kurta, nehru jacket, mojari`. There is **no `concert` / `music festival` / `gig`
family**, and the `festive_general` rule sits **above** `social_party` (:1217), so even a party cue
loses.

**3. Which routes bypass `build_brief`?**
- `services/style_reasoning_engine.py` `reason()` → `_build_response()` → `select_archetypes()`:
  **bypasses** `build_brief`; uses `_occasion_category` + raw `context.get("occasion")`
  (`:5341` passes `occasion=str(context.get("occasion") or category or "")`).
- `services/style_flow_service.py`: own occasion keyword logic (`:467,1349,1760`), **bypasses**.
- `routers/chat.py`: routes to the above; festive keyword list at `:1517`, **no build_brief**.
- Only the wardrobe-board / `brain/outfit_pipeline.py` path uses `build_brief` + `UnifiedStyleScorer`
  + `outfit_quality_guard`.
- Note: `services/style_context_service.build_canonical_style_context()` already exists (added for
  `STYLE_SHARED_BRAIN`) but is **flag-gated off** and only wraps occasion+gender+dna — not archetype
  allow/deny. It's the right seed, under-used.

**4. `CanonicalStyleBrief` fields**
```
canonical_occasion        # single resolved token (music_festival, wedding, office, ...)
occasion_family           # one family key (social_party | festive_general | professional | ...)
cultural_context          # western | indian_ethnic | neutral   <-- gates ethnic items
formality                 # 1..6
dress_code / vibe         # e.g. "expressive, energetic, non-ethnic"
gender                    # male|female|unisex|unknown (prompt-override aware)
weather, event_context, style_dna, profile
allowed_archetypes[]      # from family pool
forbidden_archetypes[]    # e.g. all festive_* for music_festival
allowed_item_signals[]    # graphic tee, cargo, sneakers, ...
forbidden_item_signals[]  # kurta, bandhgala, sherwani, mojari, nehru jacket (when cultural!=ethnic)
forbidden_colors / palette hints
provenance                # which resolver + tokens chose the occasion (debug)
```

**5. Reusable functions (no rewrite)**
- `build_brief` (`style_brief.py:798`) — occasion + contract (formality, forbidden_item_signals).
- `build_canonical_style_context` (`style_context_service.py`) — seed assembler; **extend** it.
- `_resolve_occasion_family` + `_FAMILY_ARCHETYPE_POOL` (`stylist_knowledge_service.py`) — family →
  archetype pool (fix the rule, then reuse).
- `select_archetypes` (`:1342`) — already takes `occasion`/`gender`; add allow/deny.
- `normalize_occasion` + `reject_board_for_occasion` (`outfit_quality_guard.py:121,196`) — guard.
- `_sanitize_direction_for_gender`, `_apply_style_guard` (`style_reasoning_engine.py`) — repair hook.
- `brain/engines/occasion_style_rules.py` — per-occasion allow/forbid rules to seed the brief.

**6. Where the brief is created**
One factory `build_canonical_brief(query, intent, profile, weather, …) -> CanonicalStyleBrief`,
created **once per request at route entry** — i.e. at the top of `style_reasoning_engine.reason()`
(and `style_flow_service` entry, and the wardrobe path), before any `_occasion_category` /
`select_archetypes` / board work. Implement as a superset of the existing
`build_canonical_style_context` (add `occasion_family` + allow/forbid lists). Single import, single call.

**7. How visual board generation consumes it**
`reason()` builds the brief, then:
- pass `brief.canonical_occasion` everywhere `_occasion_category`/raw occasion is used today.
- `select_archetypes(occasion=brief.canonical_occasion, allowed=brief.allowed_archetypes,
  forbidden=brief.forbidden_archetypes, gender=brief.gender, …)`.
- feed `brief` into the post-gen guard (`_apply_style_guard`).
The detector/Gemini prompt stays; the brief constrains **selection + filtering + repair**, not the LLM.

**8. Asset filtering rejects forbidden archetypes/items**
- `select_archetypes`: drop any archetype in `brief.forbidden_archetypes`; rank only
  `brief.allowed_archetypes`.
- `_asset_score` / `_best_style_asset`: hard-reject assets whose `archetypes ∩ forbidden_archetypes`
  or whose name/tags hit `brief.forbidden_item_signals` (kurta/bandhgala/mojari when
  `cultural_context != indian_ethnic`). Reuse the existing `avoid_for` / forbidden-signal scoring;
  make forbidden a **veto** (−∞), not a penalty.

**9. Repair when a board violates the brief**
Extend the existing `_apply_style_guard(directions, ctx)` (already runs post-enrich):
- occasion veto via `reject_board_for_occasion` (already wired).
- NEW archetype/item veto: if a direction's archetype ∈ forbidden or items hit forbidden_item_signals
  → `_repair_direction(...)` swaps to the nearest allowed archetype's assets; if unrepairable, drop.
- never return empty (existing fallback). Log `style_brief.violation` + `style_brief.repair`.

**10. Minimal patch (80/20, no full rewrite)**
1. **Fix the misclassification (1 file):** in `_OCCASION_FAMILY_RULES`
   (`stylist_knowledge_service.py:1202`) add, BEFORE `festive_general`:
   `(("music festival","concert","gig","rave","edm","club night","live show"), "social_party")`
   and **remove bare `"festival"`** from the `festive_general` list (keep diwali/eid/sangeet/etc).
   This alone fixes "music festival" → social_party (non-ethnic).
2. **One brief factory:** extend `build_canonical_style_context` → add `occasion_family`
   (`_resolve_occasion_family`), `cultural_context`, and `allowed/forbidden_archetypes`
   (from the family pool + `occasion_style_rules`).
3. **Wire visual path:** in `reason()`/`_build_response`, build the brief once and pass
   `canonical_occasion` + `allowed/forbidden_archetypes` into `select_archetypes` and `_apply_style_guard`.
   Gate behind the existing `STYLE_SHARED_BRAIN` flag for safe rollout.
4. Leave wardrobe/outfit-pipeline path as-is (already uses build_brief) — later, point it at the same
   factory to fully unify.

Result: every route resolves occasion **once**, through one brief; selection + filtering + repair all
read the same allow/forbid lists. Per-occasion patching stops.

## Risk / scope
- Step 1 is a 2-line data fix, immediate, low risk.
- Steps 2-3 reuse existing functions; flag-gated; no schema/Gemini/DB change.
- Does not rewrite the 5 resolvers — it makes the brief the **single authority the board path reads**,
  and lets the others wither.

_Read-only audit. No code edited, nothing committed or deployed._
