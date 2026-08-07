# AHVI Current Intelligence Delta Verification

**Verification date:** 2026-08-06
**Type:** Read-only delta against the previous audit (`AHVI_INTELLIGENCE_ORCHESTRATION_AUDIT.md`).
**Modes:** `/caveman /ponytail` — this report is normal prose; the terse/lazy mode is the chat channel.

---

## 1. Executive verdict

Two of the five findings have moved: **Finding 2 (renderer gating)** is now
**PARTIALLY FIXED** because a new frontend policy layer
(`lib/services/ahvi_response_policy.dart`, 718L) gates board rendering on a
canonical `route` taxonomy resolved from the response envelope. **Finding 5
(advice-carries-visual)** is **PARTIALLY FIXED** for the response payload
(the frontend suppresses boards when the resolved route is text-primary) and
the default visual-first upgrade is now behind the env flag
`STYLE_DEFAULT_VISUAL_INSPIRATION` (default `false`). The three remaining
findings are unchanged or worse — **Finding 3 (duplicated classifiers)** is
worse in absolute count (22 in-line classifier helpers in `routers/chat.py`,
up from 11), and **Finding 1 (response_mode)** and **Finding 4
(request_id / late-response safety)** are unchanged.

**Recommendation: CONDITIONAL GO.** The candidate pair ships a real,
observable improvement over `main` for the "advice text plus unrelated Style
board" failure. It does not ship a canonical end-to-end contract. The
smallest safe next surface is to (a) make backend responses stamp one
canonical field the frontend policy can rely on without heuristic fallbacks,
(b) add `request_id` end-to-end, and (c) mark the module-typing-bubble as
per-request-classified so the "Curating your look" loader stops showing for
information / calendar / navigation intents on the Style module.

Nothing was edited, committed, deployed, tagged, or reconfigured during
this verification.

---

## 2. Authoritative frontend / backend pair

| Repo | Branch | HEAD | 482240e / 33b2174 in ancestry | Ahead/Behind vs `origin/main` (files) |
|---|---|---|---|---|
| `devlovasit-source/ahvi-frontend` | `origin/fix/catalog-image-inplace-refresh` | `a62859189b4213cecc8c4191dd2781222c888aa9` | `482240e` YES (parent of HEAD) | large diverge — ~40 tracked files changed, incl. `lib/chat.dart`, `lib/feature/chat/services/ahvi_block_response_parser.dart`, `lib/feature/chat/widgets/blocks/visual_directions/curation_reveal.dart`, and NEW `lib/services/ahvi_response_policy.dart`, `lib/feature/chat/widgets/ahvi_processing_bubble.dart`, `lib/feature/chat/services/ahvi_processing_message.dart` |
| `devlovasit-source/ahvi-latest_backend` | `origin/fix/privacy-catalog-cutout-source` | `33b21749074376f50448e61885e0b27e005b8093` | `33b2174` IS HEAD | large diverge — `routers/chat.py`, `brain/orchestrator.py`, `services/module_chat_service.py`, `services/style_reasoning_engine.py`, `services/style_flow_service.py`, `services/stylist_knowledge_service.py`, plus NEW `services/style_asset_contract.py`, `services/style_board_shuffle_service.py`, `services/style_board_state_store.py`, `services/constrained_outfit_builder.py`, `services/home_summary_service.py`, `services/professional_safety.py`, `routers/style_boards.py`, `routers/home.py`, `routers/diet.py`, and new tests |

Frontend commit chain (last 5): `a628591 → 482240e → db7f925 → 74830f8 → a1be1ae`.
Backend commit chain (last 5): `33b2174 → ef7afea → 13a2bba → f1cd486 → 4dc60cb`.

**Both worktrees clean at start and end**; neither had uncommitted or
untracked files during verification. The prior audit's own branch
(`claude/ahvi-intelligence-audit-sopr1z`) and its docs commit (`0f9b339`
on the backend audit branch) are untouched.

---

## 3. APK / Cloud Run candidate mapping

- **Frontend source used by the currently installed APK:** **NOT PROVABLE**
  from this remote environment (APK build metadata not reachable).
- **Backend URL compiled into the current frontend:** **NOT PROVABLE** —
  resolved at build time from `lib/config/env.dart:Env.backendApiUrl`;
  the value depends on the build flavor selected by the APK build.
- **Cloud Run revision currently mapped through `cutoutfix`:** **NOT PROVABLE**
  — the user has confirmed `cutoutfix` and `stylep0-7202b5c` are Cloud Run
  traffic tags, not Git tags, and Cloud Run inspection is out of scope for
  this session.
- **Whether the cutoutfix revision was built from `33b2174`:** **NOT PROVABLE**
  for the same reason.
- **Normal production traffic split:** **NOT PROVABLE**.

The verification below is against the exact Git refs at §2. **No Cloud Run
traffic was touched, no tag was remapped, no environment variable was
changed, no APK was built.**

---

## 4. Finding 1 — Canonical response mode

**Verdict: PARTIALLY FIXED.**

The word `response_mode` still does not appear anywhere in either repo
(one hit is the diagnostic logger `lib/services/ahvi_style_diagnostics.dart:266`
reading `response['mode']` as a string, not a contract field). The FastAPI
`response_model=` uses on `routers/calendar.py:220,234,243,270` are pydantic
schema types, not the canonical contract.

**What DID land in this pair:** the frontend now has a canonical route
taxonomy in `lib/services/ahvi_response_policy.dart` (718 lines, new file):

- **Board-authorized routes** (`ahviBoardAuthorizedRoutes`, line 3):
  `visual_inspiration`, `wardrobe_style`, `style_this`, `build_outfit`.
- **Board-suppressed routes** (`ahviBoardSuppressedRoutes`, line 10):
  `style_advice`, `style_pairing`, `missing_pieces`, `supportive_conversation`,
  `medical_urgent`, `diagnosis_request`, `general_chat`, `clarification`,
  `error`.

`AhviResponsePolicy.fromResponse` (line 93) resolves the route by looking
at `response['route']` first, then falling back to `response['mode']` or
`response['intent']` only if that value is in the taxonomy. If neither
matches, the route is empty and `boardRouteAuthorized == false` — the
policy fails closed to text-primary rendering.

**But the backend never populates a top-level `route` field with these
values.** Search results:

- `routers/chat.py:4009,4029,4070,4592,5673`, `services/module_chat_service.py:50–56,209`,
  `brain/plan_pack_flow.py:872–941`, `brain/orchestrator.py:537–688`,
  `brain/daily_dependency_engine.py:134–309` all set `"route": "…"` as
  a navigation CTA target (e.g. `"/organize/calendar"`), not as a canonical
  classification.
- The only backend location that literally stamps a canonical mode into
  the response is `services/style_reasoning_engine.py:5769`
  (`"mode": "style_pairing"`).

So the frontend policy works only because `_style_reasoning_chat_response`
(`routers/chat.py:406`) stamps the internal style-mode value into the
top-level `intent` field (line 558: `"intent": mode`, where `mode` is one
of `style_advice`, `visual_inspiration`, `wardrobe_style`, `style_pairing`,
`shopping_assist`, `missing_pieces`). These strings coincidentally match
the frontend's taxonomy for six of the routes. That's a shared-constant
coincidence, not a contract:

- Backend also emits `mode` values that do NOT map: `style_reasoning`
  (`chat.py:604, 3745, 5615, 5675`), `style_intent_clarification`
  (`chat.py:364`), `greeting_bypass` / `help_identity_bypass` /
  `small_talk_bypass` (`chat.py:892, 936, 964`),
  `style_flow_service_adapter_v1` (`chat.py:2941, 2966`),
  `beta_style_clarification` (`chat.py:5185`), `visual_board`
  (`chat.py:4161`), `context_required` (`chat.py:5092`),
  `wardrobe_action` (`chat.py:5624`), `style_module_chat`
  (`chat.py:3745`), `style_flow_service_fallback_failed` (`chat.py:2825`).
  For these, the frontend policy sees route=`""` and defaults to
  text-primary. That's safe by default but silently discards legitimate
  board responses if the endpoint that emits them doesn't stamp an
  authorized route/mode/intent.
- There is no `requires_clarification`, no `confidence`, no
  `missing_information`, no `validation` block, no `request_id` on the
  envelope.

**Endpoints inspected.** `routers/chat.py`:
- `/api/text` (`text_chat`, line 4725) — no canonical route.
- `/api/module-chat` + `/api/chat/module-chat` (`module_chat`, lines 4486–4487) — no canonical route; envelope depends on which of the 4–5 return points fires (`handle_module_chat`, `_style_reasoning_chat_response`, `_module_style_response_envelope`, `_build_visual_board_envelope`, `_module_llm_response`).
- Style assistant chat: same `/api/module-chat` endpoint, same envelopes.
- Calendar module-chat: `handle_calendar_chat` (`services/module_chat_service.py:453`) returns via `_envelope` — envelope carries `intent = domain` (line 85), never a canonical route.
- Planner / Prep: `_module_plan_pack_response` (`routers/chat.py`) or `handle_module_chat(planner→calendar)` alias.

**Mutually exclusive text vs visual?** Not at the schema level. The
mutual-exclusivity is enforced *only* on the frontend by
`AhviResponsePolicy.canRenderBoards`.

---

## 5. Finding 2 — Presence-driven Flutter rendering

**Verdict: PARTIALLY FIXED.**

`lib/feature/chat/services/ahvi_block_response_parser.dart` was updated
(diff vs `origin/main`):

- Line 5: imports `ahvi_response_policy.dart`.
- Line 11: `final responsePolicy = AhviResponsePolicy.fromResponse(response);`
- Line 51: `var visualDirections = responsePolicy.canRenderBoards(response)
  ? _extractVisualDirections(response, data) : <Map<String, dynamic>>[];`
- Lines 68 & 87: `visualInspiration` and wardrobe-board conversions are all
  gated on `canRenderBoards`.
- Line 96–104: `hasVisualBoard` also gated on `canRenderBoards`.
- Lines 68–93: Style-This directions get a dedicated adapter that also
  checks `hasValidatedAnchorIn(response)` (a genuine anchor-provenance
  check — a real improvement).

**Consequence.** For any response whose resolved route is in the
board-suppressed set (`style_advice`, `style_pairing`, `missing_pieces`,
`supportive_conversation`, `medical_urgent`, `diagnosis_request`,
`general_chat`, `clarification`, `error`) OR whose resolved route cannot
be inferred, **no board blocks render**. That was the primary rendering
failure: an advice payload was carrying a `visual_directions` array and
the old parser drew it. That leak is closed.

**What is still presence-driven.** Blocks that are NOT board-related still
render on presence:

- `transition_plan`, `stylist_reasoning`, `wardrobeGap`, `image`,
  `plan`/`prep`/`checklist`, `moduleCards`, `body_proportion_advice`,
  `color_advice`, `occasion_advice`, `missing_piece_intelligence`, and the
  older `styleBoards` path when `_looksLikeModuleResponse` returns false
  and no wardrobe conversion has happened.
- `CurationReveal` (`lib/feature/chat/widgets/blocks/visual_directions/curation_reveal.dart`)
  is unchanged from the prior audit except for a text-overflow tweak
  (line 161–174 in the diff). It is still triggered per-message on any
  visual-directions block that survives the policy gate, and it still
  runs a fixed 1800 ms timer (line 79). If any board *does* pass the
  gate, the "CURATING YOUR LOOKS" loader still animates for 1.8 seconds
  regardless of when the payload actually finished computing.

**Stale-state leakage.** Not fixed. See Finding 4.

**Mutually exclusive on the frontend?** For boards, yes — the policy
suppresses them for text-primary routes. For the composite "text + visual
directions" case, yes — text always renders (from `message_text` /
`response` / `message.content`) and the visual block is suppressed. For
"text + module cards", no — the module-card block is still presence-driven
and can render alongside text.

---

## 6. Finding 3 — Duplicated classifiers

**Verdict: STILL PRESENT (absolute count is worse).**

Grep of `routers/chat.py` for classifier helpers
(`^def _is_|^def _detect_|^def _looks_like_|^def _needs_`) returns **22**
functions on the candidate branch (up from 11 in the prior audit):

```
_is_greeting                        line  628
_is_help_identity_request           line  645
_is_explicit_food_intent            line  675
_is_style_priority_query            line  680
_is_find_this_request               line  689
_is_small_talk                      line  868
_is_vague_style_prompt              line 1040
_needs_style_clarification          line 1047
_is_fast_wardrobe_count_query       line 1153
_is_use_wardrobe_action             line 1878
_is_complete_outfit_cta             line 2138
_is_alternative_look_request        line 2154
_is_generate_style_board_request    line 2172
_is_explicit_style_request          line 2979
_is_general_chat_request            line 3131   *NEW*
_detect_mode                        line 3275   *NEW*
_is_plan_pack_request               line 3752
_detect_visual_board_type           line 3786
_detect_module_summary              line 3882
_detect_quick_action_module         line 3895
_is_ask_questions_action            line 3914
_looks_like_event_create_text       line 3973
```

`_detect_mode` at line 3275 is a coarse 3-way switch (`casual` / `fashion` /
`greeting`) called only from `/api/text` at line 6036 — not a consolidating
classifier.

External decision-makers still exist and are still called from the same
paths as in the prior audit:

- `brain/intent_engine.py :: detect_intent` — LLM + `_fallback_intent`
  keyword cascade with duplicated blocks (`early_module_hits` ≡
  `module_hits`), same shape as prior audit.
- `brain/nlu/intent_router.py :: IntentRouter.classify_intent` — isolated
  singleton, still not called from the main path.
- `services/stylist_knowledge_service.py :: classify_style_mode` — still
  called from at least three places.
- `services/style_reasoning_engine.py :: reason` — 9199 lines
  (+991 lines vs the prior audit), still owns mode override.
- `brain/orchestrator.py :: AhviOrchestrator.run` — cascade still ends
  with the same occasion→outfit override at lines 1016–1020 (verbatim
  identical to prior audit's evidence).

**Frontend adds one more classification-adjacent decision:**
`_ChatScreenState._typingMessage` (`lib/chat.dart:1042–1058`) — module
keyword switch that picks the loader copy. This is not a classifier of
the message content; it is a per-module-context override that pins the
loader to a single flow copy regardless of what the user just typed.

**Total active independent decision-makers reachable from a Home chat
send:** ~26–27 (22 in-line `chat.py` + 4 external backend + 1 frontend
loader-copy picker). That is more decision surface than the prior audit,
not less.

**Canonical result authoritative end-to-end?** No. Confidence is not
preserved through the stack; each layer re-derives. The frontend
`AhviResponsePolicy` is the closest thing to a downstream authority, and
it depends on the backend happening to stamp one of six matching strings
into `mode` or `intent`.

**Occasion-word intent override** at `brain/orchestrator.py:1016–1020`
(verbatim from prior audit):

```
if (
    intent not in {"daily_outfit", "occasion_outfit", "explore_styles"}
    and occasion
):
    intent = "occasion_outfit"
```

Still present. Still overrides the classifier post-hoc.

---

## 7. Finding 4 — Request lifecycle and request ID

**Verdict: STILL PRESENT.**

Frontend grep of `lib/chat.dart` for `request_id` / `requestId` /
`requestID`: **zero hits.** Grep of `lib/services/backend_service.dart`
for the same: **zero hits.** The new `ahvi_processing_message.dart` and
`ahvi_processing_bubble.dart` files add no request identifier either.

The single-boolean loader is unchanged. `lib/chat.dart` still has:

- `bool _isTyping = false;` (line 1017)
- `_isTyping = false;` (lines 1245, 1275, 1465, 1749)
- `_isTyping = true;` (lines 1361, 1491)
- `if (_isTyping) …` (line 1946)

Six mutation sites on one boolean, no per-request scoping. The new
`_typingMessage` (line 1042) still keys off `_module` (view-level state),
not off an in-flight request. Rapid taps and topic-change cleanup are
not handled.

Backend has `http_request.state.request_id` at `routers/chat.py:6287–6294`
— a middleware-generated server-side id, not one propagated from the
client. It appears only inside `_structured_error_response` and only
inside `/api/text`. The module-chat and calendar endpoints do not surface
it. Late-response rejection, cancellation, per-request state isolation
and duplicate-submission prevention are all absent.

---

## 8. Finding 5 — Advice request carrying visual payload

**Verdict: PARTIALLY FIXED.**

Two meaningful backend changes reduce the exposure:

1. **`_should_default_visual_inspiration` is now feature-flag gated.**
   `routers/chat.py:1930–1948`:

   ```
   def _style_default_visual_inspiration_enabled() -> bool:
       return os.getenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "false")
                    .strip().lower() in {"1","true","yes","on"}

   def _should_default_visual_inspiration(query, *, intent="", …):
       if not _style_default_visual_inspiration_enabled():
           return False
       …
   ```

   Default is `false`. If the deployed env keeps the default, the
   visual-first upgrade never fires for anyone. That closes the largest
   single mechanism for advice → visual leakage.

2. **Frontend gate.** Even if visual-first fires and boards are emitted,
   `AhviResponsePolicy.canRenderBoards` on the frontend suppresses them
   for text-primary routes (§5). For an advice intent, no board block
   renders.

**What still leaks the pattern:**

- The **module-typing-bubble copy** is picked by
  `_typingMessage → _module` (frontend `chat.dart:1042`), not by the
  request's classified intent. If the user is on the Style module and
  types "give me style tips" or "what is color analysis?" or "calendar",
  the loader still says **"Curating your look"** during the wait,
  because that's the copy for
  `AhviProcessingContext.styleRecommendation` (`lib/feature/chat/services/ahvi_processing_message.dart:25–26`).
  The final rendered response can be text-only, but the wait experience
  still says "curating a look".
- The `CurationReveal` staged loader (§5) still fires for any
  visual-directions block that survives the policy gate. It does not
  observe whether the response has finished — it always animates for
  1800 ms.
- The `brain/orchestrator.py:1016–1020` occasion→outfit override is
  intact and still forces `intent = "occasion_outfit"` for /api/text
  when any occasion word is present. This does not affect /api/module-chat
  (which does not go through the orchestrator) but does affect any style
  chat that lands on /api/text via the closest-option / clarification
  branch.
- `_style_reasoning_chat_response` (`routers/chat.py:406`) still populates
  `visual_cards` (line 457) and `data.visual_directions` (line 576) for
  `mode = style_advice` when reasoning returns visual directions. The
  frontend policy suppresses their rendering, but they remain on the wire.

---

## 9. Static call traces on the candidate pair

### 9.1 "Give me style tips"

Assume: user on the Style chat, `_module = "style"`, env
`STYLE_DEFAULT_VISUAL_INSPIRATION` unset (default `false`).

1. **`lib/chat.dart:1442 _sendMessage`** — same as prior audit:
   `_isTyping = true`; no `request_id` created.
2. **`_typingMessage`** (line 1042) → `_module=='style'` →
   `AhviProcessingContext.styleRecommendation` → loader displays
   **"Curating your look"**. First observable divergence from the
   expected `text_only` flow: **`lib/chat.dart:1046–1050`.**
3. Client chooses `sendModuleChat` (line 1547 in the prior audit; still
   the same path — not the `styleViaText` branch because there is no
   `isClosestAction` / `isClarificationAnswer`).
4. `POST /api/module-chat` → `routers/chat.py:4488 module_chat`.
5. `_detect_quick_action_module("give me style tips")` → `""` (not in
   the recognised chip set).
6. `_looks_like_event_create_text("give me style tips")` → false.
7. Module in `{style, wardrobe, daily_wear}` (line 4643) AND
   `_is_explicit_style_request("give me style tips", "style")` (line
   2979) — likely true (contains `style`); style branch runs.
8. `_should_default_visual_inspiration(query, intent="style_advice",
   module_context="style")` → **`False`** because
   `_style_default_visual_inspiration_enabled()` returns False by
   default (line 1947).
9. `selected_mode = WARDROBE_STYLE` (line 4669); `_demo_style_board_payload`
   fires; `_apply_style_compliance_gate` runs; `_module_style_response_envelope`
   returns. **Second divergence:** even when the user asks for tips (advice),
   the backend goes into a wardrobe-board response, not text-only.
10. Response envelope carries `intent = wardrobe_style` (or a variant),
    `mode` from the reasoning path.
11. Frontend `AhviResponsePolicy` resolves route from `intent` →
    `wardrobe_style` (authorized) → `canRenderBoards = true` →
    the parser draws a wardrobe board block, and the outfit board card
    (`AhviOutfitBoardCard`) plus the `CurationReveal` wrapper animate.
12. **Actual visible outcome:** wardrobe board + "CURATING YOUR LOOKS"
    reveal + typing bubble said "Curating your look" during the wait.
    **Expected outcome:** text_only, no board, no visual loader.

**Verdict:** the fix at step 8 helps for the specific "visual-first
upgrade" leak; step 9 still lands on a board because the style branch is
entered for the phrase. Finding 5 is only partially fixed for this
prompt.

### 9.2 "What is color analysis?"

1. Loader: same module-based `_typingMessage` — **"Curating your look"**.
2. `sendModuleChat` → `/api/module-chat`.
3. `_detect_quick_action_module` → `""`.
4. `_looks_like_event_create_text` → false.
5. Module = `style`. Line 4643 gate:
   - `_is_explicit_style_request("what is color analysis?", "style")` —
     depends on the full body (see §6, line 2979); the function looks
     for verbs like "style this", "build outfit", "wear", etc. A
     question with a bare noun probably returns false.
   - `_needs_style_clarification("what is color analysis?", "today")` —
     depends on function body; likely true (short, vague).
   - `_ahvi_style_occasion("what is color analysis?")` returns "today"
     by default → the third disjunct is false.
6. If `_needs_style_clarification` returns true, style branch runs.
7. `_should_default_visual_inspiration` → false (default env).
8. `selected_mode = WARDROBE_STYLE` → wardrobe board payload returned.
9. Backend envelope `intent = wardrobe_style` → frontend renders a
   wardrobe board.
10. **Visible outcome:** wardrobe board (not answered as information).
    **Expected outcome:** `text_only` / `information` educational reply.

The route the classifier ought to have picked
(`style_education` / `information`) is not reachable through the
module-chat style branch — it is only reachable through
`classify_style_mode` inside `_style_reasoning_chat_response`, which is
called via the `visual_first` branch (line 4692). Since visual_first is
off by default, this reasoning path is not entered.

### 9.3 "calendar"

Assume: user on the Style chat, types the single word "calendar".

1. Loader: `_typingMessage` on `_module='style'` → **"Curating your look"**.
2. `sendModuleChat` → `/api/module-chat`.
3. `_detect_quick_action_module("calendar")` → the function normalizes to
   `"calendar"` but the recognised set is `{add event, view events,
   open calendar, open events, add reminder, plan outfit}` — bare
   "calendar" **is not in the set** → returns `""`.
4. `_looks_like_event_create_text("calendar")` → false.
5. Line 4631: `module in {"skincare", "diet", "meal", "planner",
   "calendar", "medi", "bills", "fitness"}` — `module="style"` doesn't
   match, so the calendar module handler does not run.
6. Line 4643 style branch fires with query="calendar":
   - `_is_explicit_style_request("calendar", "style")` — bare noun,
     probably false.
   - `_needs_style_clarification("calendar", "today")` — vague single
     word, likely true.
7. Style branch runs, `selected_mode = WARDROBE_STYLE`, wardrobe payload
   returned.
8. **Visible outcome:** wardrobe board plus "Curating your look" loader
   for a query that was literally the word "calendar".
    **Expected outcome:** `navigation` — open the Calendar module (or
    `text_only + list` of events).

**First divergence:** `chat.py:4643` gate assumes the module-context is
the user's intent. The word "calendar" typed inside the Style module
never gets a chance to route to calendar handling — Style module always
wins.

---

## 10. Previous audit vs current delta

### Frontend

| Prior audit finding | Current | Evidence |
|---|---|---|
| Ahvi response parser creates blocks from field presence | **CHANGED — PARTIALLY FIXED** | `AhviResponsePolicy.canRenderBoards` now gates board blocks (`lib/feature/chat/services/ahvi_block_response_parser.dart:51, 68, 87, 96`). Non-board blocks still presence-driven. |
| CurationReveal uses a fixed timer | **UNCHANGED** | `lib/feature/chat/widgets/blocks/visual_directions/curation_reveal.dart:79` still `Timer(widget.duration, …)` = 1800 ms. Diff vs main is a text-overflow tweak only. |
| Chat state uses one shared typing boolean | **UNCHANGED** | `lib/chat.dart:1017 bool _isTyping = false;` — six mutation sites. |
| No `request_id` exists | **UNCHANGED** | grep of `lib/chat.dart` / `lib/services/backend_service.dart` for `request_id\|requestId\|requestID` returns nothing. |
| Frontend keyword rules choose `/api/text` vs `/api/module-chat` | **UNCHANGED** | `lib/chat.dart:1502` `styleViaText` branch on `isClosestAction \|\| isClarificationAnswer`. |
| Previous Style/Planner context persists across unrelated requests | **UNCHANGED** | `_lastStyleContext`, `_lastPlanPackContext`, `_runningMemory`, `_clarificationResolvedByCards` still per-`_ChatScreenState` fields with no cross-module reset. |
| Two or more module-card parsers exist | **UNCHANGED** | `_moduleCardFromResponse` vs `AhviModuleCard.fromResponse` still coexist. |

### Backend

| Prior audit finding | Current | Evidence |
|---|---|---|
| `routers/chat.py` contains sequential inline intent classifiers | **UNCHANGED — worse in count** | 22 `_is_*` / `_detect_*` / `_looks_like_*` / `_needs_*` helpers (see §6). |
| Generic Style requests default to visual inspiration | **CHANGED — PARTIALLY FIXED** | `_should_default_visual_inspiration` gated by env `STYLE_DEFAULT_VISUAL_INSPIRATION`, default `false` (`routers/chat.py:1930`). |
| Occasion words override the classified intent | **UNCHANGED** | `brain/orchestrator.py:1016–1020` verbatim identical. |
| Planner is normalized to calendar on one path | **UNCHANGED** | `services/module_chat_service.py:14–34` aliases `planner\|plan\|planning → calendar` still in place. |
| Calendar creation has duplicate paths | **UNCHANGED** | `routers/chat.py` still calls `parse_plan_text_to_payload` in the module_chat handler; `services/module_chat_service.py:479` in `handle_calendar_chat` has the same call. |
| Style mode is independently classified after intent | **UNCHANGED** | `classify_style_mode` still called from at least three places; `style_reasoning_engine.reason` still overrides. |
| Reasoning / agent layers can re-decide response shape | **UNCHANGED** | `services/style_reasoning_engine.py` grew from 8208 → 9199 lines (+991). |
| Response validation does not enforce intent-to-renderer consistency | **UNCHANGED** | `brain/response_validator.py` still only text-normalizes. Enforcement moved to the frontend `AhviResponsePolicy`, so backend validation is silent about it. |

---

## 11. Focused test results

**Coverage that exists on the candidate pair (backend `tests/`):**
`test_module_chat_route.py`, `test_module_chat_board_contract.py`,
`test_api_text_board_contract.py`, `test_calendar_chat_route_reuse.py`,
`test_calendar_chat_create.py`, `test_calendar_intelligence_sync.py`,
`test_context_weather_routes.py`, `test_style_asset_contract.py`,
`test_canonical_style_board.py`, `test_canonical_style_brain.py`,
`test_style_board_state_store.py`, plus ~40 others.

**Results:** **NOT RUN** in this session.

- `pytest` is not installed in this remote environment
  (`python -m pytest --version` → `No module named pytest`;
  `pip show pytest` → `Package(s) not found`).
- Installing `pytest` (or any package) would violate the strict "do not
  update dependencies" boundary.
- Flutter SDK is likewise not installed in this environment for
  `flutter test`.

**Git status before test attempt:** clean.
**Git status after test attempt:** clean (no test ran; nothing to change).

**Coverage gap identified:** there is no test that asserts the
intent → response_mode → rendered-block invariant end-to-end. Every test
in the list covers a slice (board contract, calendar reuse, style asset
contract). The dominant failure class ("advice request produced a board
render") has no dedicated regression test on either side.

---

## 12. Smallest safe implementation surface

Ordered by leverage per line of change (ponytail ladder — stop at the
first rung that holds).

**Backend (3 files touched, no new abstractions):**

- `routers/chat.py`
  - Add `response["mode"] = <canonical>` to `_module_style_response_envelope`
    and `_style_reasoning_chat_response` so every style response stamps
    one of the frontend's known route strings. No new classifier — reuse
    the values already computed by these functions.
  - Delete the `visual_first / _should_default_visual_inspiration` branch
    entirely (line 4657–4697). The env-flag default is already `false`; if
    it is never turned on, delete the code that only fires when it is. This
    also removes the strongest advice→visual leak.
  - Gate line 4643 on `_is_explicit_style_request(...) AND NOT
    _is_general_chat_request(...)` so the Style module stops kidnapping
    bare-noun and information queries (§9.2, §9.3).
- `services/module_chat_service.py`
  - Remove the `planner → calendar` alias (line 14–34) or route it to a
    dedicated planner handler. Either is a one-line fix. Ties into the
    prior audit's Finding "prep routes to a different UI".
- `brain/orchestrator.py`
  - Delete the occasion→outfit override at line 1016–1020. That block
    survived this delta unchanged and still overrides the classifier
    post-hoc.

**Frontend (2 files touched, no new dependencies):**

- `lib/chat.dart`
  - Add a `String? _pendingRequestId;` field and a monotonically-increasing
    id (Dart's `DateTime.now().microsecondsSinceEpoch.toString()`). Set it
    in `_sendMessage` before the await, store it on the backend request
    body, and drop the response if `response['request_id'] != _pendingRequestId`.
    Replace the single `_isTyping` bool with a per-request in-flight set
    of IDs. Small diff, no new files.
  - In `_typingMessage`, if a per-request context is known (e.g. we sent
    a calendar create verbatim) use that; only fall back to `_module`
    when the request has no better hint. Kills the "Style loader on
    Calendar answer" case.
- `lib/services/ahvi_response_policy.dart`
  - Add `general_chat` and `text_only` as authorized text-primary routes
    (they are already suppressed — good) and add one line that treats a
    resolved `route == "clarification"` as `textPrimary=true` even if a
    board block is present (the file already has the plumbing; the check
    isn't wired into `canRenderBoards`).

**Backend `request_id` end-to-end** (already partly present):

- The server middleware id (`http_request.state.request_id`) is read at
  `routers/chat.py:6287`. Return it on every response envelope alongside
  the accepted-from-client id. One helper, three call sites.

No new services, no new tests infrastructure, no dependency additions.
All within existing files, and consistent with what the code already
does in adjacent branches.

---

## 13. Dependency / conflict risks

- **Env flag `STYLE_DEFAULT_VISUAL_INSPIRATION`**: if the deployed Cloud
  Run revision has this set to `true`, the visual-first upgrade is live
  and Finding 5 verdict here is invalidated. **NOT PROVABLE** from this
  environment.
- **Cloud Run tags** `cutoutfix` and `stylep0-7202b5c` are traffic tags,
  not Git refs, and were not touched. No revision remapping happened.
- **APK-configured backend URL** unknown. Even if the pair verifies
  clean, the installed APK may point at a different backend.
- **Prior audit branch** (`claude/ahvi-intelligence-audit-sopr1z`)
  contains one docs commit (`0f9b339` on the backend) that is orthogonal
  to this branch; no conflict.
- **Uncommitted files on the user's Windows worktrees**
  (`lib/wardrobe.dart`, `test/wardrobe_catalog_refresh_test.dart`,
  generated desktop registrant churn) are not in this remote environment;
  they were not staged, edited, or restored. Any implementation of the
  §12 surface must not clobber them — grep for those exact file paths in
  the merge base before landing.

---

## 14. Final recommendation

**CONDITIONAL GO.**

The candidate pair is materially safer than `origin/main` for the "advice
text plus unrelated Style board" failure. The specific mechanism that
caused it is now suppressed on the frontend for known route strings, and
the strongest backend upgrade (`_should_default_visual_inspiration`) is
off by default. That is real progress and the pair is safe to ship on
that dimension.

The conditions:

1. **Confirm `STYLE_DEFAULT_VISUAL_INSPIRATION` is unset (or `false`) in
   the Cloud Run env for the revision the APK will consume.** If it is
   `true`, the Finding 5 verdict flips back to STILL PRESENT.
2. **Land the two-file frontend `request_id` change** (see §12) before
   the APK cuts. Without it, the "typing bubble text disagrees with the
   final message" behaviour will recur on any slow / retried request.
3. **Wire the `Curating your look` typing bubble copy to a per-request
   classification signal**, not to `_module`. One helper call at the
   `_sendMessage` point; no new abstractions.
4. **Delete `brain/orchestrator.py:1016–1020`** and the
   `visual_first` branch in `routers/chat.py:4657–4697`. Both are
   post-hoc overrides that survived unchanged. Deleting them turns the
   remaining "STILL PRESENT" findings into visible improvements.

If any of (1)–(4) are not acceptable inside the ten-day window: **GO
without them** on the candidate pair still improves the visible failures
compared to `main`. But do not close the intelligence work by shipping;
the classifier consolidation (Finding 3) and `request_id` isolation
(Finding 4) remain the highest-leverage next moves.

---

## 15. Confirmation — nothing was changed

- No file was edited, staged, committed, pushed, or otherwise mutated on
  either repository during Phases 1–5 of this verification.
- The only `git` operations performed were `fetch`, `checkout` (into
  detached HEAD on the exact refs the user requested), `rev-parse`,
  `log`, `diff`, `branch`, `status`, and `merge-base --is-ancestor` —
  all read-only.
- Both worktrees are clean at the end of the verification. Head hashes:
  frontend `a62859189b4213cecc8c4191dd2781222c888aa9`, backend
  `33b21749074376f50448e61885e0b27e005b8093`.
- No Cloud Run traffic split was inspected or altered.
- The `cutoutfix` and `stylep0-7202b5c` Cloud Run tags were not touched.
- No environment variable was set or read from a running service.
- No Appwrite / Qdrant / R2 / Redis / Firebase operation was performed.
- No snapshot, golden, or generated file was regenerated.
- No dependency was installed (which is why `pytest` remained missing,
  and the focused tests in §11 are recorded as NOT RUN rather than
  smuggled behind a `pip install`).
- No APK was built.
