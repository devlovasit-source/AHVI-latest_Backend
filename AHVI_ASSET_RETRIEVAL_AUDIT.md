# AHVI Asset Retrieval Audit

Visual path asset ranking = `_asset_score(asset, direction, occasion, target_gender)`
(`services/style_reasoning_engine.py:2546`), used by `_best_style_asset` / `_best_style_assets`.

## Actual weights in `_asset_score`

| signal | rule | weight |
|---|---|---|
| **archetype** match (direction.archetype ∈ asset.archetypes) | match / mismatch-when-asset-has-archetypes | **+5 / −2** |
| **occasion** match (occasion ∈ asset.occasions) | match / no-overlap | **+4 / −3** |
| **occasion avoid_for** (occasion ∈ asset.avoid_for) | veto-ish | **−12** |
| **style_tags** (tag ∈ archetype/occasion/direction) | each | +3 |
| **slot** (hero vs accessory in asset.allowed_slots) | in-slot / hero-only / wrong | +3 / +1 / **−5** |
| **palette** (direction color ∈ asset.colors) | each, capped 6 | +2 |
| **hero color** (extracted) match / group / clash | | **+10 / +3 / −6** |
| asset color unknown but hero colored | small penalty | −1 |
| **gender** (asset.gender == target / unisex) | | **+6 / +2** |
| private-wear / non-fashion | hard block | −8 / removed |
| hero-asset name match bonus | `_hero_asset_match_bonus` | + |
| hat/cap at coffee/date | | −3 |

## Coverage vs the requested dimensions

| dimension | weighted? | where |
|---|---|---|
| occasion | **yes** (+4/−3/−12) | `_asset_score` |
| archetype | **yes** (+5/−2) | `_asset_score` |
| color | **yes** (+10/+3/−6/+2) | `_asset_score` |
| style family | partial (via archetype + style_tags +3) | `_asset_score` |
| **formality** | **NO** | not read anywhere in visual scoring |
| **season / weather** | **NO** in scoring (only weather headwear strip in guard) | — |
| **ownership** | **NO** in visual path (wardrobe-match % is display-only, post-hoc `_wardrobe_match_pct`) | — |
| **confidence** | **NO** in asset scoring (vision confidence only gates Needs-Review at save) | — |

## Findings
- Retrieval is **occasion + archetype + color** dominated. With the (pre-fix) festive_general
  archetype, ethnic assets scored +5 (archetype) +4 (occasion) → easily top.
- **No formality / energy / movement / season / ownership / confidence weighting** in the visual
  asset scorer. So the scorer cannot prefer "movement-ready, low-formality" for a festival, nor
  penalize a high-formality oxford — it only cares whether the asset's archetype/occasion/color tags
  match the (possibly wrong) archetype.
- `avoid_for` (−12) is the only strong negative tied to occasion; it depends on asset metadata being
  populated, which is sparse.

## Recommendation
Feed the CanonicalStyleBrief into `_asset_score`: add `formality_distance` penalty
(|asset.formality − brief.formality|), `movement`/`energy` alignment, and a **hard veto** when
`asset.archetype ∈ brief.forbidden_archetypes` or item-signal ∈ `brief.forbidden_item_signals`
(today that veto only exists post-hoc in `_apply_style_guard`). Season/ownership are secondary.
