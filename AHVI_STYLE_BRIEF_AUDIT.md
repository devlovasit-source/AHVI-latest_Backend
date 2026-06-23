# AHVI Style Brief Audit

No function literally named `build_style_brief()`. Two brief builders exist:
- `brain/engines/style_brief.py:build_brief()` — the occasion contract (wardrobe/outfit-pipeline path).
- `services/style_context_service.py:build_canonical_style_context()` — the canonical context
  (visual path; flag-gated; now extended with family/forbidden — the de-facto canonical brief seed).

## `build_brief()` output fields (style_brief.py:867)

| field | source | required/optional/unused (by board path) |
|---|---|---|
| `occasion` | resolver precedence (router>token>agent>daily) | **required** (drives everything) |
| `sub_intent` | agent/contract | optional |
| `formality` | contract/agent ("mid" default) | optional — **NOT used by visual `_asset_score`** (unused in board scoring) |
| `movement_requirement` | contract ("medium") | **unused** in board path (no scorer reads it) |
| `polish_requirement` | contract ("mid") | **unused** in board path |
| `required_slots` / `allowed_roles` | contract | optional (wardrobe path only) |
| `forbidden_roles` | always `[]` | **unused** (never populated) |
| `preferred_item_signals` | contract | optional |
| `forbidden_item_signals` | contract + agent avoid_items | optional — used by quality_guard, **not** visual `_asset_score` |
| `board_mood` / `allowed_badges` / `allowed_titles` | contract | optional (copy) |
| `weather` | passed in (string) | optional |
| `_provenance` | debug | optional |
| `compound` / `is_compound` | `detect_compound_context` | optional |

Key gap: `formality`, `movement_requirement`, `polish_requirement` are **computed but the visual board
path ignores them**. There is **no `energy`/`movement`/`culture` field at all**. So the brief cannot
express "festival = low formality, high energy, high movement, neutral culture" — exactly the signal
needed to keep oxford/loafers off a festival board.

## `build_canonical_style_context()` fields (style_context_service.py:610)
`canonical_occasion, occasion_brief, gender, style_dna, profile, weather, event_context` + (added)
`occasion_family, cultural_context, allowed_archetypes, forbidden_archetypes, forbidden_item_signals`.

| field | status |
|---|---|
| canonical_occasion, gender | **required**, consumed by guard |
| occasion_family, cultural_context | **new, required** for gating |
| allowed/forbidden_archetypes, forbidden_item_signals | **new**, consumed by `_apply_style_guard` (flag-gated) |
| style_dna, weather, event_context | optional |
| **missing**: `formality, energy, movement, required_traits` | **NOT present** → can't score authenticity |

## Verdict
- `build_brief` carries occasion + forbidden signals but its **formality/movement/polish fields are
  dead weight in the board path**, and it has no energy/culture axis.
- `build_canonical_style_context` is the right place to become the single `CanonicalStyleBrief`; it
  already has occasion/family/cultural/allow/forbid. **Add `formality, energy, movement, required_traits`**
  and make `_asset_score` + scorer read them (today they don't). See design doc.
