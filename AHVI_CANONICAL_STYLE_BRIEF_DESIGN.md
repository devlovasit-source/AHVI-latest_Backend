# AHVI Canonical Style Brief — Design (Phases 2-4)

Goal: one `CanonicalStyleBrief` object, built once per prompt, obeyed by every layer. No occasion
special-casing. Builds on what already ships: `build_canonical_style_context` already returns
`canonical_occasion, occasion_family, cultural_context, allowed_archetypes, forbidden_archetypes,
forbidden_item_signals`. This doc = the remaining axes + wiring + validation.

## Phase 2 — the object

```
CanonicalStyleBrief:
  occasion: str                 # canonical token (music_festival)
  occasion_family: str          # concert_social
  formality: int                # 1..5   (HAVE occasion; NEED numeric)
  energy: int                   # 1..9   (NEW)
  movement: int                 # 1..9   (NEW)
  weather: str                  # warm|cold|neutral (HAVE)
  culture: str                  # neutral|indian_ethnic  (HAVE = cultural_context)
  gender: str                   # HAVE
  allowed_archetypes: [str]     # HAVE
  forbidden_archetypes: [str]   # HAVE
  required_traits: [str]        # comfortable|expressive|movement_ready (NEW)
  forbidden_item_signals: [str] # HAVE
  style_dna, profile, provenance
```
Source the 3 new numeric axes (`formality/energy/movement` + `required_traits`) from one
`OCCASION_FAMILY_PROFILE` table keyed by `occasion_family` (single source), e.g.:
`concert_social → formality 2, energy 9, movement 9, traits[comfortable,expressive,movement_ready]`;
`professional → formality 5, energy 3, movement 3`. ~12 families, one table, no per-occasion patches.

Build site: extend `build_canonical_style_context` (already the seed). One call at `reason()` entry +
`style_flow_service` entry.

## Phase 3 — make existing engines obey (no rewrite)

| engine | change | how |
|---|---|---|
| `select_archetypes` | rank only `brief.allowed_archetypes`, drop `forbidden` | pass brief lists (today gets raw occasion) |
| `_asset_score` (visual) | add `−k·|asset.formality − brief.formality|`; reward movement/energy match; **hard veto** forbidden archetype/item | read brief instead of bare occasion string |
| `occasion_style_rules` | add `concert_social` rule + forbidden_pairings (oxford/loafers/belt/blazer/bandhgala) | one table entry |
| `outfit_quality_guard` / `_apply_style_guard` | already vetoes forbidden arch/items (shipped, flag-gated) — also check energy/formality band | extend existing guard |
| `style_scorer` | add family fit-threshold entry | one map entry |

Everything reads the SAME brief object → fixing the brief fixes all engines at once.

## Phase 4 — board validation (pre-render gate)
Extend the existing `_apply_style_guard` into a `validate_board(board, brief)`:
1. occasion match — `reject_board_for_occasion` (have).
2. forbidden archetype present → reject direction (have).
3. forbidden item signal present → strip/repair (have).
4. **NEW energy/formality check** — if board mean formality > brief.formality + 2 → low-authenticity →
   repair (swap hero to an allowed-archetype asset) or drop.
5. never empty (fallback) — have.

Examples:
- `music_festival + bandhgala` → forbidden_item_signal → reject ✓ (shipped behavior).
- `music_festival + oxford + formal loafers + belt` → formality 5 vs brief 2 → low-authenticity →
  repair to graphic tee / cargo / sneaker from `social_party` pool (NEW step 4).

## Success metric (75-prompt benchmark)
- music_festival → social_party archetypes (Gallery Night / Smart Casual Edge), NOT
  Wedding Day Ease / Sangeet Statement / Festive Heritage. **Family fix already delivers this** (live).
- Add the `concert_social` profile + energy/formality scoring to also kill oxford/loafer drift
  (not yet done — needs Phase 2 numeric axes + Phase 3 `_asset_score` term).
- Same brief improves coffee_date / conference / airport / brunch / haldi / wedding simultaneously
  because each gets a formality/energy profile instead of relying on string-match luck.

## Status / sequencing
- DONE (prod): occasion family fix; `cultural_context` + forbidden archetype/item lists; post-hoc guard veto (flag-gated).
- NEXT (this design): `OCCASION_FAMILY_PROFILE` (formality/energy/movement/traits) → into brief →
  into `_asset_score` + `occasion_style_rules` + validation step 4. That's the piece that stops
  oxford/loafer drift via *score*, not just archetype routing.
- Keep all behind `STYLE_SHARED_BRAIN` for staged rollout; the family fix stays non-gated.

_Design only — no code edited, nothing committed/deployed._
