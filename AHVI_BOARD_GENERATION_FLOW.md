# AHVI Board Generation Flow (audit)

Entry: `routers/chat.py` → `style_reasoning_engine.reason()` → `_build_response()` (visual path).
(Wardrobe-board path: `style_flow_service` + `brain/outfit_pipeline.py` — uses `build_brief`/UnifiedScorer.)

## Where each board element is decided (visual path)

| element | where selected | file:loc |
|---|---|---|
| **archetype** (per direction) | `select_archetypes(occasion, gender, style_dna)` → `_resolve_occasion_family` → `_FAMILY_ARCHETYPE_POOL` ranked; seeded tiebreak | `stylist_knowledge_service.py:1342, 1256, 1226` |
| direction skeleton (`title, archetype, hero_piece, items[], colors/palette`) | Gemini `_gemini_reasoning` payload → `_normalize_visual_directions` (fallbacks if Gemini empty) | `style_reasoning_engine.py:_build_response, ~6087` |
| **hero image / hero asset** | `_enrich_visual_directions_with_assets` → `_best_style_asset(direction, occasion, gender)` ranked by `_asset_score` | `style_reasoning_engine.py:3243, 2649, 2546` |
| **supporting pieces** (`items`) | from Gemini direction `items[]`, gender-sanitized; not asset-backed individually | `_sanitize_direction_for_gender` :2435 |
| **accessories / complete_the_look** | `_best_style_assets(accessory_only=True)` + `_default_complete_the_look` + occasion filter | :3318, 2945, `_filter_complete_the_look_for_occasion` |
| **footwear** | part of `items`/complete_the_look; no dedicated footwear selector in visual path | (gap) |
| **missing_piece** | `_build_missing_piece` + `_enrich_missing_piece_with_asset` | :6110 |
| title/badge/copy | `_apply_editorial_polish` (direction_name, badge, curated_for, short_note) | :5195 |
| **guards/repair** | `_apply_style_guard` (STYLE_SHARED_BRAIN): occasion→color→weather→gender + forbidden-archetype/item veto | :5106 |

## The festival failure point
1. `reason()` does **not** build a canonical brief for selection (flag-gated ctx exists but
   `select_archetypes` is called with **raw** `context.get("occasion")`, `:5341`).
2. `select_archetypes` → `_resolve_occasion_family("music festival")` → (pre-fix) `festive_general`
   → ethnic archetype pool → Gemini/fallback directions themed ethnic → `_best_style_asset` scores
   ethnic assets high (archetype +5, occasion +4) → kurta/bandhgala board.
3. Post-fix: family = `social_party` → non-ethnic pool. Guard (flag-gated) additionally vetoes any
   ethnic archetype/item that slips in.

## Structural gaps
- Selection (`select_archetypes`) and asset scoring (`_asset_score`) **never see formality/energy/
  movement** — only occasion string + archetype + color. A formal asset can win on a festival if its
  archetype/occasion tags match.
- No dedicated **footwear** slot selector in the visual path (footwear rides in items/complete_the_look)
  → "formal loafers at a festival" isn't slot-guarded.
- Gemini owns the direction skeleton; the only deterministic governance is `select_archetypes` (input)
  + `_apply_style_guard` (output). Make both consume the canonical brief.
