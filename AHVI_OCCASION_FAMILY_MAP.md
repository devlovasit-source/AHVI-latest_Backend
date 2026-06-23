# AHVI Occasion Family Map (audit)

Families/occasions exist in **multiple disjoint tables** with different keys. Below = every set, then
missing/duplicate/overlapping.

## A. `stylist_knowledge_service._FAMILY_ARCHETYPE_POOL` (visual board families)
`festive_daytime, festive_evening, festive_general, christian_ceremony, somber_formal, temple_modest,
professional, evening_date, social_party, resort_summer, travel_easy, relaxed_casual`

## B. `occasion_style_rules.OCCASION_STYLE_RULES` (board scoring rules)
`beach, office, date, party, brunch, travel, workout, wedding, temple_modest, casual, coffee_date,
first_date, casual_dinner, client_dinner, beach_dinner, wedding_guest, funeral, basketball_game,
team_dinner` (+ `office_meeting, client_presentation` via aliases)

## C. `style_brief._OCCASION_KEYWORDS` clusters (text brief)
includes `rave{rave,club,edm,festival}`, office, wedding family (haldi/mehendi/sangeet→wedding), date,
travel, gym, etc.

## D. `style_context_service._EVENT_LEXICON` (multi-event)
basketball/football/soccer/cricket/tennis, gym/workout/yoga/run, office/meeting/conference/interview,
drinks, dinner/lunch/brunch, party/birthday/house_party, reception/wedding/ceremony, date, movie,
concert, travel/flight/airport

## E. `style_scorer` per-occasion fit thresholds
office/business/interview/date/wedding/festive 0.72; casual/travel/airport 0.62; beach/party/gym 0.65;
coffee_date 0.70; funeral 0.75; … (numbers, not a family taxonomy)

## F. `intent_engine.occasion_map`
date_night, office, wedding, party, travel, gym, casual, formal, event

## Cross-table comparison

| concept | A (families) | B (rules) | C (brief) | D (events) | F (intent) |
|---|---|---|---|---|---|
| office/work | professional | office, office_meeting | office | office/meeting | office |
| date | evening_date | date, first_date, coffee_date | date | date | date_night |
| wedding | festive_general | wedding, wedding_guest | wedding | wedding/reception | wedding/event |
| haldi/mehendi | festive_daytime | — (temple only) | →wedding | — | — |
| **music festival/concert** | **social_party** (post-fix) | **MISSING** | rave (festival) | concert | **MISSING** |
| travel/airport | travel_easy | travel | travel | travel/airport | travel |
| beach | resort_summer | beach, beach_dinner | (none) | — | — |
| brunch | relaxed_casual | brunch | (none) | brunch | — |
| funeral | somber_formal | funeral | (none) | — | — |
| temple | temple_modest | temple_modest | (none) | — | — |

## Findings

**Missing** (no first-class family anywhere):
- `music_festival`, `concert`, `gig`, `live_show` — only land somewhere via keyword luck (now social_party in A only).
- `coffee_date`, `conference`, `airport` present in some tables, absent in others (e.g. not in A's family list except via relaxed/professional/travel).
- `haldi`/`mehendi` only in A; B has no haldi (folds into wedding/temple).

**Duplicate** (same concept, different keys across tables):
- date: `evening_date` (A) / `date,first_date,coffee_date` (B) / `date_night` (F).
- office: `professional` (A) / `office,office_meeting` (B) / `office` (F).
- wedding: `festive_general` (A) / `wedding,wedding_guest` (B) / `event→wedding` (F).

**Overlapping / ambiguous**:
- `event`→`wedding` (occasion_style_rules ALIASES:535) — any "event" becomes a wedding.
- `festive` vs `festival` — one ethnic, one (now) social; previously collapsed.
- `party`/`social_party`/`rave`/`club` overlap between A, B, C.

## Recommendation
One enum `OCCASION_FAMILY` (single source) with explicit `music_festival/concert → concert_social`,
and a single `occasion → family` map all tables import. Add the missing concert/festival family to B
(occasion_style_rules) so the scoring rules + forbidden_pairings also cover it. See design doc.
