# AHVI Occasion Normalization Flow (audit)

Question: where do `music festival / concert / live music / gig / festival` become a canonical occasion?
Answer: in **7 independent resolvers**, none authoritative. That fragmentation is the root cause.

## The 7 occasion resolvers

| # | resolver | file:loc | festival/concert handling |
|---|---|---|---|
| 1 | `intent_engine` LLM `occasion_map` | `brain/intent_engine.py:124` | no festival/concert key → passthrough raw; `event`→`event` |
| 2 | `build_brief` → `detect_occasion_from_tokens` / `resolve_occasion_archetype` (`_OCCASION_KEYWORDS`) | `brain/engines/style_brief.py:133,267,308` | `festival`∈ **rave** cluster `{rave,club,edm,festival}` |
| 3 | `style_context_service` `_EVENT_LEXICON` / `detect_multi_event` | `services/style_context_service.py:70` | `concert`→concert/social; no `festival` token |
| 4 | `style_reasoning_engine._occasion_category` | `services/style_reasoning_engine.py:3373` | own keyword buckets (work/social/sensitive…) |
| 5 | `stylist_knowledge_service._resolve_occasion_family` (`_OCCASION_FAMILY_RULES`) | `services/stylist_knowledge_service.py:1202` | **FIXED**: `music festival/concert/gig/rave/edm/club night/live show`→`social_party`; bare `festival` removed from `festive_general` |
| 6 | `outfit_quality_guard.normalize_occasion` | `brain/engines/outfit_quality_guard.py:121` | no festival/concert → passthrough |
| 7 | `occasion_style_rules` `ALIASES` + `get_occasion_rule` | `brain/engines/occasion_style_rules.py:514,546` | no festival/concert/gig → falls to default rule |

## Pre-fix failure chain (the bug you reported)
```
"music festival"
 → intent_engine: occasion="music festival" (passthrough)
 → style_reasoning visual path calls select_archetypes(occasion="music festival")
 → _resolve_occasion_family substring-matches "festival"
 → festive_general  (was: line 1210 list contained "festival")
 → _FAMILY_ARCHETYPE_POOL["festive_general"] = Festive Heritage / Refined Traditional / Celebration Kurta …
 → preferred_items: bandhgala, silk kurta, nehru jacket, mojari
 → board = kurta/ethnic
```
Meanwhile `build_brief` (#2) said **rave**, and `outfit_quality_guard` (#6) said passthrough. Three engines, three answers. The visual board path used #5 → ethnic.

## Post-fix (live in prod, commit 8a6c1d7)
```
"music festival" → #5 social_party → Gallery Night / Smart Casual Edge / Power Casual (non-ethnic)
"diwali festival" → still festive_general (multi-word concert rule does not match)
```
But #1–#4, #6, #7 still disagree on festival/concert — only the board path (#5) is corrected. **No single canonical occasion exists.**

## Canonical-occasion gaps (no resolver knows these)
- `music_festival`, `concert`, `gig`, `live show` — no dedicated canonical token; ride on `rave`/`social_party` by keyword luck.
- `coffee_date`, `conference`, `airport` exist in some tables (occasion_style_rules / style_brief) but not others (normalize_occasion).
- Each resolver has a different vocabulary and different default (casual vs daily vs wedding-via-`event`).

## Recommendation
One `resolve_canonical_occasion(prompt, intent)` → a single token + family, consumed by all 7 call sites. See `AHVI_CANONICAL_STYLE_BRIEF_DESIGN.md`. `build_canonical_style_context` is the seed (already returns `canonical_occasion` + `occasion_family`).
