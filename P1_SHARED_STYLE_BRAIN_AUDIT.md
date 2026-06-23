# P1 — Shared Style Brain / Visual Path Intelligence Bridge (READ-ONLY AUDIT)

No code edited, no DB write, no commit, no deploy. All findings cite `file:line`.

## TL;DR
There are **three parallel outfit-generation systems**. Only ONE (the visual/premium path) powers `visual_inspiration`, `style_pairing`, `missing_piece`, and the advice modes — and it **imports none of the shared intelligence**. The other two paths (wardrobe-board, outfit-pipeline) use the strong engines. The premium path uses Gemini + a simple in-file `_asset_score` (color *match*, no clash/harmony, no occasion rules, no quality guard). That is why burgundy→green, odd caps, random loafers, and generic advice still appear on the best-looking screen.

---

## 1. Current flow map

| flow | intent route (chat.py) | generation handler | context source | asset selection | scoring | shared brain used |
|---|---|---|---|---|---|---|
| **visual_inspiration** | `_should_default_visual_inspiration` / cultural → `VISUAL_INSPIRATION` ([chat.py:3670](routers/chat.py)) → `style_reasoning_engine.reason()` ([chat.py:3680](routers/chat.py)) → `_style_reasoning_chat_response` | `services/style_reasoning_engine.py` `_build_response` | `context["occasion"]` = `_ahvi_style_occasion()` (chat.py) + `build_pairing_persona` + `_resolve_asset_gender` | `_style_asset_rows` → `_asset_score` (in-file) + `select_archetypes` | `_asset_score` (occasion/archetype/color-**match**/gender) | **NONE** |
| **style_pairing** | `STYLE_PAIRING` (intent_engine) → same visual path, `mode=STYLE_PAIRING` | `_build_reasoning_prompt` STYLE_PAIRING branch → `_normalize_pairing_routes` → `_pairing_routes_as_visual_directions` | `_extract_pairing_anchor(query)` + persona | same `_asset_score` | Gemini routes + `_asset_score` | **NONE** |
| **missing_piece** | within visual path | `_build_missing_piece` + `_enrich_missing_piece_with_asset` + `_dedupe_missing_piece_against_directions`; `_wardrobe_reality_explanation` uses `score_route_against_wardrobe` (stylist_knowledge_service) | LLM payload + wardrobe | keyword/archetype + asset enrich | none (heuristic) | **NONE** (partial wardrobe reality only) |
| **use_my_wardrobe** | `WARDROBE_STYLE` → `build_style_flow_response` ([chat.py:2030](routers/chat.py)) | `services/style_flow_service.py` | `build_brief()` (style_brief) | `_STYLE_DNA_TARGETS` + wardrobe | `occasion_confidence_threshold`, `reject_board_for_occasion` | **YES** — build_brief, occasion_style_rules, outfit_quality_guard, board_storyteller |
| **daily_outfit / occasion_outfit** | `brain.outfit_pipeline.get_daily_outfits` ([chat.py:1269](routers/chat.py)) | `brain/outfit_pipeline.py` | pipeline context | wardrobe_selector + graph | `UnifiedStyleScorer.score_outfit` + palette_engine + quality_guard | **YES** — full scorer + palette + guard |

**Core asymmetry:** the path with the *best UI* (visual) has the *weakest reasoning*; the paths with the *best reasoning* (wardrobe-board, outfit-pipeline) feed the older UI.

---

## 2. Bypass points (visual path = `services/style_reasoning_engine.py`)

Confirmed: `style_reasoning_engine.py` imports **none** of brief/scorer/guard/palette/storyteller/occasion_rules (grep of its import block returns empty).

| shared capability | where it lives | visual path instead does | symptom |
|---|---|---|---|
| `build_brief()` | style_brief.py:798 | ad-hoc `_ahvi_style_occasion` token + `select_archetypes` | inconsistent canonical occasion |
| `UnifiedStyleScorer.score_outfit()` | style_scorer.py:1092 | `_asset_score` (color **match** only) | no holistic outfit score / no rejection warnings |
| `occasion_style_rules` (`reject_board_for_occasion`) | outfit_quality_guard.py:196 | partial in-file `_occasion_asset_block_reason` | inappropriate items for occasion slip through |
| palette / color-harmony | brain/engines/styling/palette_engine.py + `color_harmony_bank.json`, `ahvi_color_pair_logic_v1.json` | none — `_asset_score` rewards color *match to direction palette*, never checks **clash** | **burgundy + green** survives |
| `OutfitQualityGuard` (`reject_board_for_occasion`) | outfit_quality_guard.py | none | weird cap / clashing combos |
| `BoardStoryteller` / `fallback_title_and_why` | board_storyteller.py | LLM prose + `_occasion_voice_note` | generic advice voice |
| Style DNA in scoring | style_scorer DNA weighting | DNA only feeds `select_archetypes` + prompt persona, **not** outfit scoring | personalization shallow on visual path |

---

## 3. Existing reusable functions (safe to call from the visual path)

| function | file | input | output | risk |
|---|---|---|---|---|
| `build_brief(query, *, router_occasion, agent_payload, weather)` | brain/engines/style_brief.py:798 | query + router occasion + agent payload | `{occasion, archetype, ...}` brief | **LOW** (pure; already used by wardrobe path) |
| `resolve_occasion_archetype(occasion, query)` | style_brief.py | strings | canonical occasion | LOW |
| `detect_occasion_from_tokens(query)` | style_brief.py | query | `(occasion, tokens)` | LOW |
| `reject_board_for_occasion(board, occasion)` | brain/engines/outfit_quality_guard.py:196 | board dict + occasion | `(reject: bool, reason: str)` | **LOW** (pure check; ideal post-filter) |
| `normalize_occasion(occasion)` | outfit_quality_guard.py:121 | str | canonical | LOW |
| `UnifiedStyleScorer().score_outfit(items, context, graph)` | brain/engines/style_scorer.py:1092 | items + context(style_dna…) + graph | `{score,label,reasons,breakdown,rejection_warnings}` | **MED** (needs graph build + context shape; heavier/latency) |
| `occasion_confidence_threshold(...)` | style_scorer.py | occasion | float | LOW |
| `PaletteEngine.select_palette(context)` / `build_palette_response(context)` | brain/engines/styling/palette_engine.py:51/80 | context dict | palette list/response | LOW–MED |
| `BoardStoryteller` / `fallback_title_and_why` | brain/response/board_storyteller.py:481 / 5293-usage | board/outfit/context | title + why | LOW |
| `select_archetypes(...)` | services/stylist_knowledge_service.py | anchor/occasion/dna/gender | archetype list | LOW (already used) |
| `score_route_against_wardrobe(route, wardrobe)` | stylist_knowledge_service.py | route + wardrobe | `{match_score, owned, missing, substitutions, confidence}` | LOW (already used in missing-piece) |
| color banks | brain/banks/foundational/color_harmony_bank.json, brain/banks/formulas/ahvi_color_pair_logic_v1.json | — | harmony/clash rules | LOW (data) |

---

## 4. Minimal bridge design (smallest safe patch)

**Principle:** add a shared *context* + post-LLM *guard/repair*; do **not** replace the Gemini generator or change the payload schema.

**Step A — one canonical style context** (new pure helper, e.g. `build_canonical_style_context()` in `services/style_context_service.py`):
```
{ canonical_occasion: build_brief(query, router_occasion=_ahvi_style_occasion(query), agent_payload=intent, weather).occasion,
  gender: _resolve_asset_gender(query, user_profile),
  profile, style_dna, weather, event_context }
```
Both the visual path and wardrobe path consume this one object → kills occasion drift.

**Step B — feed canonical_occasion into the visual path** where it currently uses the ad-hoc token: `select_archetypes(occasion=ctx.canonical_occasion, gender=ctx.gender, ...)` and asset scoring. No schema change.

**Step C — post-generation guard/repair (the high-ROI, low-risk core):** after `_normalize_visual_directions` / `_enrich_visual_directions_with_assets`, run pure guards per direction:
1. `reject_board_for_occasion(direction, ctx.canonical_occasion)` → if reject, drop/swap the offending asset (re-pick next-best from `_asset_score`).
2. color-clash check using `color_harmony_bank` / `ahvi_color_pair_logic_v1` against the direction palette + each asset's `colors` → drop clashing asset (fixes burgundy+green).
3. gender already handled (`_sanitize_direction_for_gender`) — keep.
This only changes *which assets* appear, not the response shape.

**Step D (optional, flagged) — holistic score gate:** behind env flag `STYLE_VISUAL_USE_UNIFIED_SCORER`, score each direction's items via `UnifiedStyleScorer.score_outfit` and reorder/drop low scores. MED risk (graph + latency) — ship A–C first.

**Frontend safety:** `visual_directions[]` / `cards[]` / `data.*` keys unchanged; bridge is selection-time only.

---

## 5. Pairing-specific plan ("what to pair with X")

Current: `STYLE_PAIRING` already builds `pairing_routes` via Gemini and *also* renders them as `visual_directions` ([style_reasoning_engine] `_pairing_routes_as_visual_directions`). It reads like boards, not advice.

Plan (advice-first):
1. **Anchor extraction**: reuse `_extract_pairing_anchor(query)` (exists) → `{name, category, color}`.
2. **Compatible palette**: from `color_harmony_bank` / `ahvi_color_pair_logic_v1` keyed by anchor color → ranked compatible colors.
3. **Do / Don't pair list**: do = compatible categories+colors; don't = clash colors + occasion-inappropriate (via `reject_board_for_occasion`).
4. **Response order**: lead the payload with a pairing **advice block** (anchor + palette + do/don't), then **optional** visual cards (existing routes) *after* the advice — gate cards behind "show looks". Keep payload keys; add an advice block (already partially present as `pairing_routes` + `what_to_avoid`).

---

## 6. Missing-piece plan (real unlock value, not keyword guess)

Current: `_build_missing_piece` (LLM/keyword) + `_wardrobe_reality_explanation` (uses `score_route_against_wardrobe`).

Plan — quantify unlock using existing `score_route_against_wardrobe`:
- **slots unlocked**: categories the missing piece completes across candidate routes.
- **occasions unlocked**: occasions that go from <threshold coverage → buildable when the piece is added (diff occasion coverage with vs without).
- **boards improved**: count routes whose `match_score` crosses a usable threshold after adding the piece.
- **wardrobe coverage delta**: `score_route_against_wardrobe` match_score before vs after.
Rank candidate missing pieces by (occasions_unlocked, boards_improved, coverage_delta) instead of LLM guess.

---

## 7. Risks

- **High blast radius**: Step D (UnifiedStyleScorer in visual path) — needs graph build + context shape; touches latency + Gemini token budget. Keep flagged/off by default.
- **Frontend contract**: any change to `visual_directions`/`cards`/`data` keys breaks the app. Bridge A–C is selection-only → LOW. Pairing advice-block reorder must remain additive.
- **Latency**: post-filter guards are pure/cheap; UnifiedStyleScorer per-direction is the latency risk (Step D only).
- **Gemini token**: A–C are post-LLM (no token change). Injecting brief into the prompt adds a small amount; keep minimal.
- **Data dependency (critical)**: clash detection needs `colors` metadata. **209 assets still lack colors** (P2). On uncolored assets, clash check is blind → bridge effectiveness is gated on the color backfill. Sequence P2 (colors) alongside.

---

## 8. Test plan (targeted, read-only fixtures)

| # | case | expected | hook |
|---|---|---|---|
| 1 | burgundy top → suggest green shirt | clash rejected; compatible palette only | color-harmony post-filter (Step C2) |
| 2 | beige loafers, male user → no crop top | crop top filtered (gender) | `_sanitize_direction_for_gender` / `_style_text_allowed_for_gender` |
| 3 | "haldi" | stays ethnic/festive archetypes + assets | `select_archetypes` family pool (festive_daytime) |
| 4 | "airport" | travel-comfort assets preferred | canonical_occasion=travel → asset scoring + occasion rules |
| 5 | female user | female assets returned | `_resolve_asset_gender` + female catalog (now 253) |
| 6 | empty wardrobe | no wardrobe-match % shown | `has_wardrobe_signal` gate in `_apply_editorial_polish` |
| 7 | "what to pair with beige loafers" | pairing-advice response first, boards optional | `is_style_pairing_request` → STYLE_PAIRING + advice block (Step 5) |

Each should be a pure-function/unit test against the guard + classifier (no Gemini, no DB), plus 1–2 integration snapshots of `_build_response` output shape.

---

## Recommended sequencing
1. **Step A+B+C** (canonical context + post-LLM guard/repair) — smallest, highest ROI, no payload/Gemini change. Fixes burgundy+green, occasion drift, inappropriate items.
2. **P2 color backfill in parallel** — clash detection is blind without colors (209 missing).
3. **Pairing advice-first (§5)** — biggest perceived-smartness win for least code.
4. **Missing-piece unlock value (§6)**.
5. **Step D (UnifiedStyleScorer gate)** — last, behind a flag, measure latency.
```
```
