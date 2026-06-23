# AHVI Scoring Audit — why oxford/trousers/belt/loafers for "music festival"

Two scorers exist:
- Visual board: `_asset_score` (`style_reasoning_engine.py:2546`) — archetype/occasion/color/gender only.
- Wardrobe path: `occasion_style_rules.score_item_for_occasion` (`:553`) + `style_scorer` thresholds.

## Root cause (pre-fix): nothing scored "formality/energy", everything scored "archetype/occasion match"

Chain for "music festival" before commit 8a6c1d7:
1. `select_archetypes("music festival")` → `_resolve_occasion_family` matched `"festival"` →
   `festive_general` **OR**, when no family matched cleanly, the library fell back to generic
   classic/professional archetypes (Modern Gentleman / Contemporary Classic).
2. Those archetypes' assets are **oxford shirt, tailored trousers, belt, loafers** (classic/formal).
3. `_asset_score`: archetype match **+5**, occasion tag overlap **+4**, neutral color **+10/+2** →
   oxford/trousers/loafers score top. **No formality penalty exists** (see asset audit) → a formal
   oxford is never docked for a casual occasion.
4. `occasion_style_rules` has **no `music_festival`/`concert` key** → `get_occasion_rule` returns the
   default/casual rule with empty `forbidden_pairings` → nothing rejects loafers/belt/oxford.
5. Result: formal classic board (oxford/trousers/belt/loafers) or ethnic (kurta) depending on which
   family won — both wrong for a festival.

## Why specifically oxford/trousers/belt/loafers (not just kurta)
- When the ethnic family did NOT dominate, scoring **collapsed to the library head**: classic/
  professional archetypes whose assets are formal staples (oxford/trousers/loafers/belt). Stable-sort
  + seeded tiebreak returns these when no occasion-specific signal differentiates.
- There is **no "energy/movement/expressive" axis** to reward a graphic tee / cargo / sneaker over an
  oxford. The scorer literally cannot tell a festival from an office beyond the occasion *string*,
  and the string matched nothing strong → defaulted to formal.

## Missing weights (the fix lever)
`_asset_score` has **no** terms for: formality, energy, movement, season, ownership, confidence.
`occasion_style_rules` has **no** festival/concert rule or forbidden_pairings for it.
`style_scorer` per-occasion thresholds (`:33`) have **no** festival/concert entry → default tolerance.

## Post-fix status (live)
- Family now `social_party` → archetypes Gallery Night / Smart Casual Edge / Power Casual /
  Off-Duty Tailoring → assets skew smart-casual, not oxford-formal, not ethnic.
- Guard (flag-gated) vetoes ethnic archetypes/items.
- **Still missing**: a formality/energy penalty so a stray oxford/loafer can't win on score. That
  requires the CanonicalStyleBrief carrying `formality/energy/movement` into `_asset_score`.

## Recommendation
1. Add `music_festival`/`concert` to `occasion_style_rules` with `forbidden_pairings`
   (oxford/loafers/belt/blazer/bandhgala) + a casual `min_fit`.
2. Add formality/energy distance to `_asset_score` (penalize |asset.formality − brief.formality|).
3. Hard-veto `brief.forbidden_archetypes`/`forbidden_item_signals` in scoring, not just post-hoc.
