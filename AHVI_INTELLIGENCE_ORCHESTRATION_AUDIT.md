# AHVI Intelligence & Orchestration Audit

**Audit date:** 2026-08-06
**Audit type:** Read-only — no edits, commits, deploys, or config changes were performed.
**Mode:** `/caveman /ponytail-audit` (compressed narrative in chat, normal prose in this report; findings ranked biggest-cut-first where applicable).

> **Scope caveat.** This audit only had access to the two GitHub repositories attached to the session:
> - `devlovasit-source/ahvi-frontend` at `/home/user/AHVI-frontend`
> - `devlovasit-source/ahvi-latest_backend` at `/home/user/AHVI-latest_Backend`
>
> The multiple local worktrees the user maintains on their Windows machine
> (`C:\tmp\AHVI-frontend-final-style-wiring-db7f925`, `…-catalog-inplace-482240e`,
> `…-beta-final`, `…-style-final`, `…-premium-layout-74830f8`, `…-style-livefix-a1be1ae`,
> plus the backend candidates `f1cd486`, `33b2174`,
> `release/beta-backend-20260731`, `fix/privacy-catalog-cutout-source`, and the
> `cutoutfix` candidate tag) could **not** be inspected from this environment.
> Cloud Run deployed revisions, APK build metadata, and the exact
> APK-configured backend URL were also unreachable and are therefore treated
> as **unknown** below. Recommendation: run the git-status enumeration
> locally on the Windows machine and paste the results into §2.

---

## 1. Executive Verdict

**Dominant root cause (Option H — several causes with one systemic driver):**

> AHVI does not lack a model, a classifier, orchestration, or context in
> isolation. It has **too many of each**, layered without a canonical contract.
> Intent, domain, occasion, style-mode, and response shape are re-decided
> independently by five to seven different code paths per request, and the
> frontend renders **whatever payload fields happen to be present**. There is
> no `response_mode` field anywhere in the codebase, so every UI surface
> infers its own rendering decision from the shape of the payload — which
> the backend does not gate.

Concretely:
- **C. Duplicated / conflicting routing** — dominant.
- **G. Frontend state / renderer corruption** — dominant.
- **F. Missing validation** (of the intent→mode→renderer contract) — dominant.
- **E. Missing true orchestration** — contributing (there is a router named `AhviOrchestrator`; there is no actual multi-step orchestrator).
- **A. Weak model**, **B. Missing classifier**, **D. Missing context** — real but **secondary**. Adding a stronger model or a new classifier without collapsing the routing layers will produce the *same* symptoms.

**A classifier alone is not sufficient.** Even a perfect classifier will be
overwritten downstream by `orchestrator.py:1016–1020`, by
`chat.py:4567–4638`, and by the presence-driven Flutter parser in
`ahvi_block_response_parser.dart`. The classifier's decision is currently one
of several signals that a later layer can silently discard.

**Top five P0 fixes (details in §18):**

1. Introduce a canonical `response_mode` field on every response and gate the
   Flutter renderer strictly on it (`AhviParsedResponse` becomes a switch on
   `response_mode`, not a series of `if (field is present)` blocks).
2. Assign a `request_id` in the frontend before the send, and reject any
   backend response whose `request_id` does not match the currently-pending
   one for that chat session. Rejection also cancels the loader.
3. Collapse the seven parallel classifiers into a single `classify()` call
   whose output (`domain`, `intent`, `action`, `response_mode`, `confidence`,
   `requires_clarification`, `slots`) is the *only* signal routers may read.
   Delete `chat.py`'s eleven inline `_is_*` and `_detect_*` classifiers, the
   `brain/nlu/intent_router.py` `IntentRouter`, and the duplicated
   keyword blocks in `brain/intent_engine.py:_fallback_intent`.
4. Remove the occasion→outfit override at `orchestrator.py:1016–1020` and the
   "any-visual-first" default at `chat.py:4581–4638`. Every response must
   originate from the canonical classifier decision, not from
   post-hoc keyword sniffing on the same query.
5. Clarify-on-uncertainty: when confidence < 0.75 or when a required slot
   is missing (e.g. calendar event without a time), return a
   `clarification` response mode instead of falling through to the
   generic style / calendar create path.

**Expected improvement (subjective, MVP horizon).**
Applying only P0 #1–#4 removes the four reported failure classes ("Style tips
shows curating loader", "color analysis returns unrelated board", "calendar
loads Style loader", "raw sentence becomes event title" for the sub-case where
the response mode is wrong, and "Prep tomorrow routes to a different UI").
It does not fix outfit *quality* — that is a separate work stream. Perceived
"AHVI is coherent" jumps materially because the visible mismatch between
what the user asked and what the app renders is the loudest current signal.

**No changes were made during this audit.** No file was edited, staged,
committed, pushed, deployed, or reconfigured. Cloud Run traffic, environment
variables, and both `main` branches are exactly as they were at the start of
the session.

**Report location:** `AHVI-latest_Backend/AHVI_INTELLIGENCE_ORCHESTRATION_AUDIT.md`.

---

## 2. Authoritative Source & Deployment Versions

### Inspected worktrees

| Worktree | Repo | Branch | HEAD | Commit date | Upstream | Dirty | Untracked | Ahead/Behind vs `origin/main` |
|---|---|---|---|---|---|---|---|---|
| `/home/user/AHVI-frontend` | `devlovasit-source/ahvi-frontend` | `claude/ahvi-intelligence-audit-sopr1z` | `99bc53c` | 2026-07-29 12:24 +0530 | none configured (tracks `origin/main` in effect — no divergence) | none | none | 0 / 0 |
| `/home/user/AHVI-latest_Backend` | `devlovasit-source/ahvi-latest_backend` | `claude/ahvi-intelligence-audit-sopr1z` | `3d0c0d3` | 2026-07-27 17:24 +0530 | none configured | none | none | 0 / 0 |

Both audit branches were created by the harness at the tip of `origin/main`
and contain no additional commits. The audit branch on both repos is
effectively equivalent to `main` HEAD at the moment of clone.

### Frontend recent commits (last 20 on `main`)

```
99bc53c Revert "Feat/home today summary integration (#22)" (#23)
b00aaff Feat/home today summary integration (#22)
f1566c2 fix(style): route board shuffle through durable contract (#20)
8d41ee2 fix(share): dedicated opaque ShareableOutfitBoard for the exported PNG (#19)
d918285 fix(style): one action surface — route all outfit boards to AhviOutfitBoardCard (#18)
ebcd143 fix(board): restore working Save (Appwrite) and Share (image) on outfit card (#17)
91dfca7 fix(style): restore CTA and use board shuffle contract (#16)
7b449a6 fix(style-board): beta-safe visual-density patch (Option A+) (#15)
06c1d41 fix(fitness): stop workouts/today request storm + dispose-after-await crash (#14)
cf32ed4 Merge pull request #10 from devlovasit-source/feat/style-chat-fullscreen
a1d200c feat(style): open stylist chat full screen
05b70f6 Merge pull request #9 from devlovasit-source/integration/pravallika-plus-ahvi-fixes
934e83a chore: exclude diagnostic logs from integration
c519275 fix(navigation): prevent reentrant Android back handling
deccaa7 test: refresh current visual board goldens
406d4cf test: modernize visual board and app smoke coverage
8b61db8 fix(medi): distinguish reminder questions from creation requests
073a5e7 fix(medi): activate tablet and pill reminder keywords
5530728 fix(medi): recognize tablet and pill reminder intents
7917173 fix(medi): preserve minutes in chat reminder parsing
```

The `Revert "Feat/home today summary integration"` at HEAD indicates the
latest attempt to add a Home summary was reverted — likely because it
introduced the Style/Calendar loader mismatch the user reports. This is
consistent with §7 findings on presence-driven rendering.

### Backend recent commits (last 20 on `main`)

```
3d0c0d3 fix(style): stamp board contract when the wardrobe adapter pre-flagged the gate (#38)
f6850cf fix(style): stamp board contract on the /api/module-chat wardrobe route (#37)
a71310b fix(style): preserve formal dinner gap responses (#36)
7202b5c fix(style): enforce formal dinner and preserve follow-up contracts (#34)
5e665e8 fix(style): propagate board contract to all aliases + gate alternative-look (#33)
dae40d0 fix(style): CTA bypasses clarification + durable board contract on every card (#32)
ba708de fix(style): make default CTA return complete validated outfits (#31)
cd716bd fix(style): universal explicit-role gate + canonical source_policy (#30)
2546e9c fix(style): enforce controlled-beta request compliance (#29)
295603c feat(style): beta intelligence bridge (backend) (#22)
c2a53fd feat(style): enforce complete-outfit slots (repair or reject incomplete boards) (#25)
b43db71 fix(auth): make Redis auth cache optional with a circuit breaker (#27)
80a6749 fix(calendar): create events from natural-language chat (#26)
6b951a4 fix(calendar): /api/calendar/today no longer 500s on backend fetch failure (#24)
```

Ten of the last thirteen commits are `fix(style)` — the top of the stack is
almost entirely tactical patches to the style path. The pattern (contract
stamping, alternative-look gating, explicit-role gate, controlled-beta
compliance) is consistent with a fragile response contract being patched
one code path at a time, which is exactly what §5–§7 diagnose.

### Frontend / backend deployment map — **UNKNOWN in this environment**

The following can only be reconstructed on the developer's machine and are
recorded here for the user to fill in:

- Latest committed frontend source: **`99bc53c`** (this audit).
- Frontend source used for the installed APK: **UNKNOWN** — no APK metadata
  reachable from this session.
- Latest committed backend candidate: **`3d0c0d3`** (this audit).
- Backend revision reached by the installed APK: **UNKNOWN** — depends on
  the APK's compiled `Env.backendApiUrl` (see `frontend/lib/services/backend_service.dart:97`).
- Backend revisions receiving normal production traffic: **UNKNOWN** — Cloud
  Run traffic split not inspected (audit boundary).
- APK-configured backend URL: **UNKNOWN** — resolved from `Env.backendApiUrl`.

The known Windows-local worktrees the user listed (frontend `db7f925`,
`482240e`, `beta-final`, `style-final`, `74830f8`, `a1be1ae`;
backend `f1cd486`, `33b2174`, `release/beta-backend-20260731`,
`fix/privacy-catalog-cutout-source`, `cutoutfix`) are **not** on the
attached remote's branch listing, so this audit could not enumerate their
HEAD, dirty state, or the accepted-commit membership. To close this gap,
run locally:

```
for d in C:\tmp\AHVI-*; do
  echo "=== $d ==="
  git -C "$d" rev-parse HEAD
  git -C "$d" branch --show-current
  git -C "$d" log --oneline -5
  git -C "$d" status --short
done
```

### Authoritative pair selected for this audit

- **Frontend:** `devlovasit-source/ahvi-frontend@99bc53c` (`origin/main`).
- **Backend:** `devlovasit-source/ahvi-latest_backend@3d0c0d3` (`origin/main`).

Every reference in this report is to line numbers in those two commits.

---

## 3. End-to-End Request Call Graphs

### 3.1 Home chat / module chat (generic)

```
User taps Send in ChatScreen (lib/chat.dart)
  ↓
_ChatScreenState._sendMessage()          [chat.dart:1442]
  ↓ setState _isTyping = true                  (single bool, no request_id)
  ↓ decides styleViaText vs sendModuleChat via 5 client-side classifiers:
     _isPlanPackRequest, _isShowClosestChip, _isBoardActionPhrase,
     _isClarificationAnswer, _isStyleModule
  ↓
BackendService.sendModuleChat / sendChatQuery   (backend_service.dart)
  ↓ POST /api/module-chat   OR   POST /api/text
  ↓
routers/chat.py :: module_chat()          [chat.py:4436]
  ↓ (11 sequential classifiers, first hit wins)
  │   _is_ask_questions_action           [chat.py:4451]
  │   _detect_quick_action_module        [chat.py:4454]
  │   _looks_like_event_create_text      [chat.py:4493]
  │   _detect_module_summary             [chat.py:4520]
  │   _is_plan_pack_request              [chat.py:4530]
  │   _detect_visual_board_type          [chat.py:4539]
  │   _is_explicit_style_request         [chat.py:4568]
  │   _needs_style_clarification         [chat.py:4569]
  │   _ahvi_style_occasion               [chat.py:4570]
  │   _should_default_visual_inspiration [chat.py:4581]
  │   _is_use_wardrobe_action            [chat.py:4589]
  ↓
Dispatch:
  ├─ calendar quick action → handle_module_chat(domain=calendar)
  ├─ style module + visual_first → style_reasoning_engine.reason()
  ├─ style module + wardrobe → _demo_style_board_payload → _apply_style_compliance_gate
  ├─ non-style module → handle_module_chat(domain=<module>)  [module_chat_service.py:701]
  └─ default → _module_llm_response
  ↓
Response envelope built with duplicated field aliases
  ({message, message_text, response} all same string;
   {board_ids, pack_ids, board_id, pack_id} normalized;
   {chips, quick_actions} both populated).
  ↓
Flutter chat.dart receives Map<String, dynamic>
  ↓
parseAhviResponse()                    [ahvi_block_response_parser.dart:7]
  ↓ Presence-driven block extraction — every field-name is a potential block:
     body_proportion_advice / color_advice / occasion_advice
     transition_plan
     visual_inspiration_board
     visual_directions (or converted from style_boards)
     visual_board / visualBoard
     wardrobe_gap
     image
     plan / prep / checklist
     module_card / moduleCards
     style_boards
     stylist_reasoning
     missing_piece_intelligence
  ↓
_sendMessage appends _ChatMessage(...) with every parsed block
  ↓ setState _isTyping = false
  ↓
Message widget mounts blocks in order; CurationReveal wraps any
visualDirections block with a 1800 ms staged "CURATING YOUR LOOKS"
loader (curation_reveal.dart:59, 79).
```

### 3.2 Style chat (Home → Style module full-screen)

Same graph as 3.1 for the `styleModules = {'style', 'wardrobe', 'daily_wear'}`
branch. Diverges at `chat.dart:1502` where `isClosestAction ||
isClarificationAnswer` routes through `/api/text` (`sendChatQuery`) instead
of `/api/module-chat` (`sendModuleChat`). This means:
**identical query text is sent to a different endpoint depending on which
chip the user tapped last**, and the two endpoints have subtly different
classifier stacks (chat.py has an entirely separate `text_chat` handler at
line 4649).

### 3.3 Calendar chat

Two paths converge:
- **Quick-action label** ("Add event", "View events", …) — `chat.py:4455`,
  handles them locally then falls through to `handle_module_chat(domain="calendar")`.
- **Natural language** ("Doctor appointment tomorrow at 6 PM") —
  `chat.py:4493` calls `_looks_like_event_create_text` and directly invokes
  `services.calendar_service.parse_plan_text_to_payload` +
  `create_calendar_event`. This bypasses `handle_calendar_chat` entirely.

`handle_calendar_chat` (`module_chat_service.py:453`) has its **own**
duplicate copy of the same creation flow (line 469-492) with the same
`parse_plan_text_to_payload` call and idempotency lookup. So the create
path exists twice. Either can silently miss the guardrails the other has.

### 3.4 Style This / Build Outfit / Visual Inspiration

Not fed through a dedicated endpoint — they reuse `/api/module-chat` with
different chip labels ("Style this jacket", "Build an outfit"), which
`chat.py:4451–4570` sniffs for. The label is currently the only signal
that distinguishes item-styling from outfit-construction, and both paths
end up in `style_reasoning_engine.reason()` regardless. There is no
distinct handler.

### 3.5 Prep / Planner

Frontend classifier `_isPlanPackRequest(queryText)` (`chat.dart:1460`)
rewrites `domain` to `'planner'` before send. Backend
`chat.py:4473` accepts `_qa_module == "planner"` and calls
`_module_plan_pack_response`. But `module_chat_service.py:_normalize_domain`
(line 14) aliases `planner → calendar`, so if the request lands in
`handle_module_chat` instead, it runs the calendar handler. The two routes
therefore return different UIs from the same button.

---

## 4. Inventory of Every Intent Decision-Maker

Ranked by influence on the visible output.

| # | Location | Function | Kind | Overrides | Notes |
|---|---|---|---|---|---|
| 1 | `routers/chat.py` `module_chat` | 11 inline `_is_*`/`_detect_*` classifiers, executed in order | Regex + keyword | First wins over everything below | Ordering fragile; adding a keyword at the top of any list changes global routing |
| 2 | `brain/intent_engine.py` `detect_intent` | `_fallback_intent` (regex) → LLM → `_validate_intent_row` | Regex → LLM | Fallback short-circuits at conf≥0.75, LLM never runs for those | Duplicate keyword blocks: `early_module_hits` (line 436-511) is repeated as `module_hits` (line 557-632); `style_priority_phrases` repeated at 390 and 517 |
| 3 | `services/stylist_knowledge_service.py` `classify_style_mode` | Regex, 10 modes | Called from ≥3 places | `VISUAL_INSPIRATION` is defined at line 16 but **not** in `STYLE_MODES` at line 22 — so this function can never return it, even though `orchestrator.py:775` and `chat.py:4593` treat it as a valid return value |
| 4 | `brain/nlu/intent_router.py` `IntentRouter.classify_intent` | Regex, 4 life categories + styling | Standalone singleton `nlu_router` | Not called from the main path any more; keyword lists overlap with `intent_engine.py` |
| 5 | `brain/orchestrator.py` `AhviOrchestrator.run` | `if intent == …` cascade | Post-classifier override | Line 1016-1020: **any occasion word forces `intent = "occasion_outfit"`** even if the classifier said something else |
| 6 | `services/style_reasoning_engine.py` `style_reasoning_engine.reason` | LLM-guided mode selection | Overrides orchestrator's `style_mode` at `orchestrator.py:769` | 8208-line file; classifier is buried inside |
| 7 | `services/agent_style_orchestrator.py` | Gemini agent (`gemini-3.5-flash` — invalid model id, see §13) | Silent fallback to Ollama | 740 lines; runs a second time as an "agent" on top of the reasoning engine |
| 8 | `brain/agent_system.py` `AgentSystem.plan` | LLM plan → rule fallback | Rule fallback only knows `daily_outfit`, `tryon` | Every other intent gets a `no_op / fallback_agent` plan |
| 9 | `brain/shopping/shopping_router.py` `.route` | Signal-based | Only reached for shopping intents | Isolated |
| 10 | Frontend `_isPlanPackRequest`, `_isShowClosestChip`, `_isBoardActionPhrase`, `_pendingStyleClarificationPrompt`, `_looksLikeStyleClarification` | Regex on visible text | Client-side pre-routing | Decides `/api/text` vs `/api/module-chat` before backend sees the text |
| 11 | Frontend `AhviParsedResponse` block extraction | Presence-driven | Renderer-level | Any of ~12 payload fields can force its own block to render — no gating |

**11 decision-makers, no confidence merge, no explicit precedence.** Each
one silently overrides an earlier one. `detect_intent`'s validation logic
(`brain/intent_engine.py:190`) does try to reconcile LLM vs heuristic
via a confidence threshold, but everything from #5 downward in the table
above is applied *after* `detect_intent` has already returned.

---

## 5. Current Intent Taxonomy (Reconstructed From Code)

### Domains recognised by at least one router

`chat`, `home`, `style`, `wardrobe`, `daily_wear`, `shopping`, `diet`
(alias for `meal`, `meals`, `meal_planner`), `fitness` (alias for
`workout`, `gym`), `medi` (alias for `medical`, `meds`, `medicine`,
`medicines`), `bills`, `calendar` (alias for `planner`, `plan`,
`planning`, `event`, `events`), `skincare`, `contacts`, `life_boards`,
`life_goals`. Source: `services/module_chat_service.py:14-34`,
`routers/chat.py`, `brain/orchestrator.py:_resolve_organize_module`.

### Intents (backend)

Listed by `brain/intent_engine.py:_ALLOWED_INTENTS` (line 66):
`daily_dependency`, `daily_outfit`, `occasion_outfit`, `explore_styles`,
`wardrobe_query`, `try_on`, `organize_hub`, `plan_pack`, `style_advice`,
`style_pairing`, `style_education`, `color_body_advice`,
`body_proportion_advice`, `color_advice`, `occasion_advice`,
`wardrobe_style`, `shopping_assist`, `general`.

### Style modes

Ten defined (`stylist_knowledge_service.py:11-32`); nine in the
`STYLE_MODES` allow-set (`visual_inspiration` is defined but not
in the set — bug #3 in §4).

### Response modes

**Zero.** `grep -rn "response_mode"` across both repos returns nothing.
The renderer decides from field presence — see §6.

### Missing / overloaded

- `information` vs `advice` vs `education` — collapsed into
  `style_advice` / `style_education` / `color_advice`, and the boundaries
  are keyword-based. `"what is color analysis?"` cannot be reliably
  distinguished from `"what colors work for me?"`.
- `navigation` vs `create_action` — "calendar" as a bare word could be
  either. Currently `_detect_quick_action_module` decides based on exact
  chip label match; everything else falls through to
  `handle_module_chat(calendar)` which then re-decides via
  `_looks_like_event_create`.
- `clarification` — exists as a response `type` value on some paths
  (`chat.py:4508–4517`) but is never a first-class classifier output.
- `casual_conversation` — subsumed by `general/greeting|small_talk|help_identity` sub-intents in `_fallback_intent`.

### Proposed canonical taxonomy (do not implement yet)

```
domain           style | wardrobe | shopping | calendar | planner |
                 fitness | diet | medi | skincare | bills | home | general
intent           information | advice | inspiration | recommendation |
                 item_styling | outfit_construction | shopping_assist |
                 create | update | delete | plan | track | explain |
                 clarify | navigate | casual
response_mode    text_only | visual_inspiration | wardrobe_recommendation |
                 style_this | build_outfit | shopping_assistance |
                 workout_plan | meal_plan | recipe | calendar_action |
                 planner_action | navigation | clarification | error
```

The `(domain, intent, response_mode)` triple is the *only* signal the
frontend renderer should read.

---

## 6. Response-Mode Enforcement Findings

**There is no canonical `response_mode` field.** The Flutter renderer at
`lib/feature/chat/services/ahvi_block_response_parser.dart:7-202` derives
what to draw purely from field presence. The parser walks the payload and
adds a block for each of:

| Payload field present | Renders block |
|---|---|
| `blocks[].type == "body_proportion_advice"` (etc.) | `AhviBlockType.styleAdvice` |
| `blocks[].type == "transition_plan"` | `AhviBlockType.transitionPlan` |
| `visual_inspiration_board` | `AhviBlockType.visualInspiration` |
| `visual_directions` (or converted from `style_boards`) | `AhviBlockType.visualDirections` |
| `visual_board` / `visualBoard` | `AhviBlockType.visualBoard` |
| `wardrobe_gap` | `AhviBlockType.wardrobeGap` |
| `image` | `AhviBlockType.image` |
| `data.plan` / `data.prep` / `data.checklist` | `AhviBlockType.plan|prep|checklist` |
| `AhviModuleCard.fromResponse` returns non-null | `AhviBlockType.moduleCards` |
| `style_boards` (raw) | `AhviBlockType.styleBoards` |
| `blocks[].type == "stylist_reasoning"` with archetype | `AhviBlockType.stylistReasoning` |
| `missing_piece_intelligence` | `AhviBlockType.missingPiece` |

Consequences observed in the reported failure cases:

- **"Give me style tips" shows CURATING YOUR LOOK.**
  `chat.py:4593` selects `VISUAL_INSPIRATION` when
  `_should_default_visual_inspiration` returns true. `style_reasoning_engine`
  returns both advice text *and* a `visual_directions` array. The frontend
  renders the `visualDirections` block, and
  `curation_reveal.dart:59, 79` wraps it in the 1800 ms
  "CURATING YOUR LOOKS" loader. The loader is not gated on whether the user
  asked for a look; it is gated on presence of the block.

- **"What is color analysis?" returns text + unrelated Style board.**
  `classify_style_mode` (`stylist_knowledge_service.py:392`) returns
  `STYLE_EDUCATION` — good. `style_reasoning_engine` responds with an
  advice block AND (because the reasoning engine's LLM was given a
  visually-oriented prompt) a `visual_directions` array. Renderer adds
  both. There is no gate that says "if `intent=information`, drop cards".

- **"calendar" returns Calendar content while showing a Style loader.**
  Depends on the previous message. `CurationReveal` is per-message and its
  1800 ms timer starts when the widget mounts (`curation_reveal.dart:79`),
  so a *prior* Style message that was still being revealed when the
  Calendar response landed will still be visibly animating. Additionally,
  if the calendar response happens to include `visual_directions` (which
  it can — the reasoning engine is called from
  `chat.py:4602–4621` whenever `visual_first` is true, and
  `visual_first` is driven by keywords that overlap with generic queries),
  the Calendar bubble itself will show the Style loader.

- **"Style This with a Belt" renders a dress image.**
  This is not enforceable via `response_mode` alone (item identity vs
  image provenance is a separate contract, tracked in
  `services/style_item_contract.py`). The audit noted the file exists but
  did not deep-dive its 8208-line reasoning engine caller. Flagged
  P1 in §18.

- **"Calendar request → raw sentence becomes event title/time".**
  `services/calendar_service.parse_plan_text_to_payload` is called twice
  (`chat.py:4497` and `module_chat_service.py:479`) with the raw
  `user_message`. There is no clarification step when the parser cannot
  extract a structured title. `chat.py:4501-4518` returns
  `intent="event_needs_time"` when time is missing, but the equivalent
  clarification for a missing/junk title does not exist — the parser will
  happily use the whole sentence as the title.

- **"Prep tomorrow" routes to a different generic UI.**
  Frontend `_isPlanPackRequest` returns true → domain rewritten to
  `planner`. Backend `chat.py:4473` calls `_module_plan_pack_response`.
  But `module_chat_service.py:_normalize_domain` aliases `planner →
  calendar`; if the request instead lands in `handle_module_chat`
  (e.g. via a different chip), the Calendar handler runs. Two code paths,
  two UIs, one button.

---

## 7. Frontend State & Request Isolation Findings

- **No `request_id` anywhere.** Neither `_ChatMessage`, `_ChatSession`,
  `BackendService.sendModuleChat`, nor `sendChatQuery` carries a
  correlation id. A late response cannot be identified or rejected.
- **Single `_isTyping` boolean.** `chat.dart:1450, 1667`. Two rapid sends
  share the same loader state. The second `finally` flips it off even if
  the first request is still in flight.
- **CurationReveal is time-based, not response-based.**
  `curation_reveal.dart:79`: `_revealTimer = Timer(widget.duration, …);`.
  If the response payload changes shape during those 1800 ms, the reveal
  still fires on schedule.
- **PATCH-5 hardcoded string match.** `chat.dart:1617-1627` rewrites the
  backend's error message via `String.contains("couldn't build a
  complete style board")`. When the backend changes wording, the patch
  silently stops firing. Fragile.
- **Stale-state cleanup absent.**
  - Previous board payload is retained until the next message rebuilds
    the whole list (blocks live on `_ChatMessage`, not on a per-session
    scoped store).
  - `_lastPlanPackContext`, `_lastStyleContext`, `_runningMemory`,
    `_clarificationResolvedByCards` are all mutable fields on
    `_ChatScreenState` — no reset when the module changes, when the user
    scrolls back, or when a new topic starts.
- **Duplicate weak-match guard** at `chat.dart:1632-1639` special-cases
  identical repeated bubbles, but only for `type == 'weak_match'`. Other
  duplicate responses are not deduplicated.
- **`_moduleCardFromResponse` vs `AhviModuleCard.fromResponse`** — two
  parallel module-card parsers (`chat.dart:1572-1575`). Whichever the
  helper returns first wins; the frontend has two competing understandings
  of what a "module card" is.

### Missing lifecycle

Expected: `idle → classifying → executing → validating → rendering →
completed/error`. Actual: `idle → typing (bool) → append blocks →
completed`. No classification, execution, validation, or per-step error
handling is visible to the UI or to the developer.

---

## 8. Context Compilation & Matrix

Backend context sources (from `orchestrator.py:_normalize_weather_context`,
`chat.py`, `module_chat_service.py`, `style_dna_engine.enrich_context`):

| Context | Style | Wardrobe | Calendar | Planner | Fitness | Diet | Medi | Home |
|---|---|---|---|---|---|---|---|---|
| Conversation history | ✅ (`detect_intent` line 749) | ✅ | ⚠️ (only for date-slot recovery, `_recent_date_phrase`) | ⚠️ | ❌ | ❌ | ❌ | ⚠️ (sticky intent) |
| Conversation summary | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wardrobe list | ✅ (`_wardrobe_from_appwrite` + ctx) | ✅ | ❌ | ⚠️ (travel style outfit only) | ⚠️ (`workout outfit` keyword only) | ❌ | ❌ | ❌ |
| Saved boards | ⚠️ (via style_memory) | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Style DNA / profile | ✅ (monkey-patched wrapper `orchestrator.py:1335`) | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| Preferred / disliked colours | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Recently worn | ⚠️ (via `wear-today` endpoint; not fed back into orchestrator) | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Weather | ✅ (`_normalize_weather_context`) | ✅ | ❌ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| Location | ⚠️ (raw only) | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ |
| Calendar events | ❌ | ❌ | ✅ (self) | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| Tomorrow's events | ❌ | ❌ | ⚠️ (`/api/calendar/today` only) | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| Adherence history | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| Medicine schedule | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (self) | ❌ |
| Skincare | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fitness profile | ⚠️ (gender only, for outfit gating) | ⚠️ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Diet profile | ❌ | ❌ | ❌ | ❌ | ⚠️ (recovery meal chip) | ✅ | ❌ | ❌ |
| Time of day | ⚠️ (slot) | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| Timezone | ✅ (`Asia/Kolkata` default in `module_chat_service.py:477`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Findings.**
- No single normalized "context packet" — each engine reaches back into
  Appwrite (`_wardrobe_from_appwrite`, `_safe_list_documents`) with its
  own aliases, its own timeout, and its own error handling.
- `handle_fitness_chat`, `handle_bills_chat`, `handle_skincare_chat`,
  `handle_diet_chat`, `handle_medi_chat` are **pure static-string
  templates** (`module_chat_service.py:334–693`). They receive `context`
  but only inspect two or three fields (bills count, medications count)
  and return canned copy. The rich per-domain context (adherence,
  streaks, recently-logged) is never read.
- The visible explanation never surfaces *which* context was used. There
  is no "because your calendar shows a meeting" line in the response
  payload.
- Fresh signals may still be present in `context.signals` /
  `proactive_signals` after `proactive_engine.inject` runs
  (`orchestrator.py:742`); they are not propagated to the module handlers.

**Recommendation (do not implement yet).** One `Context` dataclass
compiled once per request, structurally validated, passed through the
call stack. Its schema is proposed in §17.

---

## 9. Orchestration Findings

- **`AhviOrchestrator.run` (`brain/orchestrator.py:724`) is a router,
  not an orchestrator.** It is a 570-line cascade of
  `if intent == X: return Y`. There is no step planner, no tool call
  composition, no retry, no partial-result assembly.
- The monkey-patched wrapper at `orchestrator.py:1323–1388`
  (`_ahvi_orchestrator_run_with_style_dna`) reassigns `AhviOrchestrator.run`
  at import time. This is a fragile pattern: it uses `except Exception:
  pass` twice (line 1360-1362, 1372-1374), so if the profile merge or
  Style DNA enrichment fails, the request silently degrades and no
  observability point catches it.
- `AgentSystem.plan` (`brain/agent_system.py:15`) is the closest thing
  to a planner. It asks the LLM for a JSON list of steps, then falls
  back to a rule table that only knows `daily_outfit` and `tryon`
  (line 103-118). Every other intent gets `[{"step": "no_op", "agent":
  "fallback_agent"}]` — a plan that does nothing.
- `ExecutionEngine.execute` (`brain/execution_engine.py:18`) supports
  step-to-step state passing but is not used from the main path.
  `orchestrator.run` never calls it.
- Cross-module intelligence (e.g. "What should I wear to tomorrow's
  client meeting?") is not implemented. The orchestrator's
  `if intent == "occasion_outfit"` branch calls
  `get_daily_outfits` with the wardrobe alone; it does not call the
  calendar service to fetch tomorrow's meeting, nor the weather service
  for tomorrow's forecast. The proactive_engine.inject at line 742 is
  the closest to cross-signal awareness, and it only reshapes existing
  `context.signals`.

**Router vs orchestrator.** True orchestration decides *plan*, executes
each step, feeds results between them, retries on failure, and composes
the final response. AHVI currently decides once and returns. The
`brain/decision_engine.py:DecisionEngine` is a 66-line ranker, misnamed —
it does not decide anything.

---

## 10. Output Validation Findings

Only three genuine validators exist:

1. **`brain/response_validator.py`** — validates *text* (`looks_truncated`,
   forbidden starters, terminal punctuation). Also normalizes cards to a
   minimum shape (`id`, `title`, `items`) but does **not** validate that
   the cards' domain, image URLs, or item identity match the request.
2. **`brain/engines/outfit_quality_guard.py`** (not deep-read in this
   audit) — style-side quality.
3. **`services/agent_metadata_validator.py`** — validates wardrobe upload
   agent output, not chat responses.

Not validated anywhere:

- That `response.type` (or a proposed `response_mode`) matches the
  classified intent.
- That `visual_directions`, `visual_board`, `style_boards`,
  `visual_inspiration_board` are **mutually exclusive** for a given
  response (currently the reasoning engine can emit two of them).
- That the anchor item asked about ("belt") matches the items in the
  returned boards (the reported "belt → dress image" case).
- That the calendar payload's title is a plausible event title (not the
  raw sentence).
- That the calendar payload's start_time is within a plausible window.
- Duplicate event / idempotency is only guarded in *one* of the two
  create-event code paths (`handle_calendar_chat:482`), not in
  `chat.py:4498`.
- That the wardrobe items returned are gender-consistent with the user
  profile. The `_ahvi_item_allowed_for_user_profile` gate
  (`chat.py:1445`) is applied on the wardrobe fetch path only.

Fails-open behaviour: `_style_reasoning_chat_response`,
`_apply_style_compliance_gate`, and `_stamp_response_contract` all catch
broad exceptions and return the partially-processed payload rather than a
`clarification` or `error` response.

---

## 11. Fallback Findings

Ranked biggest user-visible impact first:

| Fallback | Trigger | Effect | Recommended action |
|---|---|---|---|
| **Occasion → outfit intent** (`orchestrator.py:1016-1020`) | Any of `wedding\|party\|office\|work\|date\|travel` in the query | Overrides classifier; forces `intent=occasion_outfit`; renders style boards | **Remove.** Classifier decision must not be overridden by post-hoc keyword sniffing. |
| **`_should_default_visual_inspiration`** (`chat.py:1932`) | Enabled by default for style/daily_wear/wardrobe | Forces `VISUAL_INSPIRATION` mode; renders visual boards for text-oriented queries | **Narrow to explicit inspiration requests only.** |
| **`_fallback_intent` sticky-history override** (`brain/intent_engine.py:857-873`) | New `general` intent + previous non-general intent in history | Inherits previous intent | **Keep, but only within a same-topic window.** |
| **Ollama fallback in `_call_ollama`** (`services/llm_service.py:263`) | Gemini call fails or `AI_PROVIDER` unset | Falls through to `llama3.2:3b` local model | **Log conspicuously.** Currently silent — a Gemini outage silently downgrades every intent decision. |
| **`catalog_fallback` for board items** (`services/style_reasoning_engine.py:3703-3744`) | Wardrobe item image missing | Substitutes generic catalog image | **Keep, but stamp `source_policy = catalog_fallback` on the item so UI can label it.** |
| **Handler fallback in `execution_engine.execute`** (`brain/execution_engine.py:46-56`) | Handler signature mismatch | Tries three signatures via `except TypeError` | **Delete.** Handlers should have a single canonical signature. |
| **`handle_style_chat` static reply** (`module_chat_service.py:696-698`) | Style domain lands in `handle_module_chat` instead of `chat.py`'s style branch | Returns single hardcoded "I can help with that" | **Delete or clarify** — the handler is functionally unused but reachable. |
| **AgentSystem rule fallback = no_op** (`brain/agent_system.py:118`) | Intent is anything other than `daily_outfit\|tryon` | Returns a plan that does nothing | **Delete** — dead plan; nothing consumes it. |
| **Raw sentence as calendar title** (`services/calendar_service.parse_plan_text_to_payload`, called at `chat.py:4497` and `module_chat_service.py:479`) | Sentence lacks structured title | Whole sentence becomes title | **Replace with clarification response** when title parsing fails. |
| **`isCalendar`-style keyword sniffing for `remind me`** (`brain/intent_engine.py:358`) | Any `remind me` in text | Routes to calendar create | **Narrow** — `"remind me what color goes with blue"` currently misroutes. |

---

## 12. Memory Findings

- **Style DNA.** `brain/personalization/style_dna_engine.enrich_context`
  is called from the monkey-patched orchestrator wrapper
  (`orchestrator.py:1367`). If enrichment throws, the fallback is a
  silent `except Exception: pass`. No observability.
- **Global memory files** in `brain/data/`:
  `global_style_memory.json`, `style_dna_memory.json`,
  `outfit_memory.json`, `style_knowledge_v1.json`. These are shipped in
  the repo — not per-user; per-app defaults. Never reloaded at runtime
  unless the code that reads them was reimported.
- **Wear-today logging.** `frontend/lib/services/backend_service.dart:171`
  posts to `/api/style/wear-today`. There is no visible read path in the
  audited orchestrator/reasoning code that uses the resulting record to
  re-rank future recommendations.
- **User profile.** Fetched via
  `services/data_access_service.get_user_profile` on every request in
  the wrapper. Not cached at request scope, so repeated calls per
  request may hit Appwrite multiple times.
- **Conversation memory.** History passed on the request but the sticky
  intent behaviour is the only reuse (`brain/intent_engine.py:857`).
  No summarization, no long-term memory.
- **Feedback loops.** No visible outbound signal from the "accept /
  reject / rate this outfit" surfaces back into the style ranker.

**Distinction lost.** The code treats "conversation history" and "user
profile" and "cached UI state" as one big `context` dict; a memory-aware
layer would separate them.

---

## 13. Model Inventory

**No secrets are printed in this section.**

| Purpose | File / line | Configured model | Notes |
|---|---|---|---|
| General text LLM | `services/llm_service.py:47` | `OLLAMA_MODEL` default `llama3.2:3b` | 3B-parameter local model. Below the reasoning depth of Claude/GPT/Gemini-Pro class models by a wide margin. Default absent env override. |
| Gemini path | `services/llm_service.py:38` | `GEMINI_MODEL` default `gemini-2.0-flash-001` | Reasonable but 2.0-flash is the smallest current Gemini. |
| Vision (garment detection) | `services/gemini_multi_garment_detector.py:67` | `GEMINI_MULTI_GARMENT_MODEL` default `gemini-2.0-flash-001` | Consistent with 2.0. |
| Ollama vision | `services/ai_gateway.py:347, 360` | `OLLAMA_VISION_MODEL` default `llama3.2-vision:latest` | Local fallback. |
| Catalog PNG generation | `services/catalog_png_generation_service.py:1204` | `gemini-2.5-flash-image-preview` | Preview channel — subject to Google's preview retirement policy. |
| **Agent style orchestrator** | `services/agent_style_orchestrator.py:42` | `DEFAULT_MODEL = "gemini-3.5-flash"` | **Invalid model id.** Google does not publish `gemini-3.5-flash`. The gateway will 404 and fall through to the Ollama path silently — meaning the "agent" is actually running on `llama3.2:3b`. **Priority fix.** |
| Agent metadata validator | `services/agent_metadata_validator.py:31` | `DEFAULT_MODEL = "gemini-3.5-flash"` | Same invalid id. |
| Intent classifier (LLM) | `brain/intent_engine.py:832` `detect_intent`, via `generate_text(..., usecase="intent")` | Whatever `generate_text` resolves | Defaults to Ollama 3B → most intent decisions run through llama3.2:3b unless env is set to Gemini. |
| Reasoning engine | `services/style_reasoning_engine.py` (8208 lines) | Multiple calls to `generate_text` | Same default. |

**Trust vs validation.**
- LLM output is passed through `parse_json_object` /
  `extract_json` (`services/ai_gateway.py`), then through
  `_validate_intent_row` (intent) or `validate_final_text` (prose).
- Card content is only *shape-normalized* by `_sanitize_cards` — never
  cross-validated against the request.
- Multiple models are called for the same request without a merge
  layer: `detect_intent` (LLM #1) → `style_reasoning_engine.reason`
  (LLM #2) → `agent_style_orchestrator` (LLM #3, when invalid model id
  falls through, LLM #4 = Ollama). Any disagreement is silently resolved
  in favour of the last one.

**Recommendation (not implement).**
- Fix the `gemini-3.5-flash` typo (probably intended `gemini-2.5-flash`).
- Add a **single** `classify()` call that owns the intent + response_mode
  decision; downstream engines *consume* it, they do not re-decide it.
- Log which model actually served each decision so intent-drift can be
  attributed.
- Do not upgrade every model — most of the perceived quality gap is
  coming from routing corruption, not model capacity.

---

## 14. Proposed Regression Corpus (100 cases, do not run destructive actions)

Format: `id | input | context | expected domain | expected intent | expected response_mode | required context sources | expected renderer | notes`.

### Style — information & advice (10)
```
S001 | "What is color analysis?"                                 | new session | style | information       | text_only              | none                       | text_bubble         | reported failure #2
S002 | "Explain smart casual"                                    | new session | style | information       | text_only              | none                       | text_bubble         |
S003 | "What is a capsule wardrobe?"                             | new session | style | information       | text_only              | none                       | text_bubble         |
S004 | "Give me style tips"                                      | new session | style | advice            | text_only              | style_dna                  | text_bubble         | reported failure #1
S005 | "How do I dress for my body type?"                        | user has body_shape=hourglass | style | advice | text_only | user_profile.body_shape | text_bubble | uses profile
S006 | "What colors suit warm undertone?"                        | user has undertone=warm | style | advice | text_only | user_profile.undertone | text_bubble |
S007 | "Any tips for someone with broad shoulders?"              | new session | style | advice          | text_only              | style_dna                  | text_bubble         |
S008 | "Should I wear vertical stripes?"                         | new session | style | advice          | text_only              | style_dna                  | text_bubble         |
S009 | "What is monochrome dressing?"                            | new session | style | information     | text_only              | none                       | text_bubble         |
S010 | "How do I build a modest capsule?"                        | new session | style | advice          | text_only              | user_profile               | text_bubble         |
```

### Style — inspiration & recommendation (10)
```
S011 | "Show me brunch outfit inspiration"                       | new session   | style | inspiration    | visual_inspiration     | style_dna                  | visual_board        |
S012 | "Show minimalist looks"                                   | new session   | style | inspiration    | visual_inspiration     | style_dna                  | visual_board        |
S013 | "Give me boho ideas for a picnic"                         | new session   | style | inspiration    | visual_inspiration     | style_dna, weather         | visual_board        |
S014 | "Trending fits for a music festival"                      | new session   | style | inspiration    | visual_inspiration     | style_dna                  | visual_board        |
S015 | "What should I wear today?"                               | wardrobe > 0  | style | recommendation | wardrobe_recommendation| wardrobe, weather, calendar| wardrobe_outfit_board|
S016 | "What can I wear from my wardrobe to office?"             | wardrobe > 0  | style | recommendation | wardrobe_recommendation| wardrobe, calendar         | wardrobe_outfit_board|
S017 | "Suggest an outfit for a rainy day"                       | wardrobe > 0, weather=rainy | style | recommendation | wardrobe_recommendation | wardrobe, weather | wardrobe_outfit_board |
S018 | "Give me a look for a beach vacation"                     | wardrobe > 0  | style | recommendation | wardrobe_recommendation| wardrobe, weather          | wardrobe_outfit_board|
S019 | "What should I wear to my sister's wedding tomorrow?"     | wardrobe > 0, calendar has event | style | recommendation | wardrobe_recommendation | wardrobe, calendar, weather | wardrobe_outfit_board |
S020 | "Style me for a client meeting on Friday"                 | wardrobe > 0, calendar has event | style | recommendation | wardrobe_recommendation | wardrobe, calendar, weather | wardrobe_outfit_board |
```

### Style — item-styling & outfit-construction (8)
```
S021 | "Style this jacket"                                       | uploaded item = jacket | style | item_styling | style_this | wardrobe + item | style_this_board |
S022 | "Style this with a belt"                                  | uploaded item = belt   | style | item_styling | style_this | wardrobe + item | style_this_board | reported failure — anchor must be belt, image provenance must match
S023 | "What can I pair these jeans with?"                       | uploaded item = jeans  | style | item_styling | style_this | wardrobe + item | style_this_board |
S024 | "Build an outfit with these jeans"                        | uploaded item = jeans  | style | outfit_construction | build_outfit | wardrobe + item | build_outfit_board |
S025 | "Create a look around this dress"                         | uploaded item = dress  | style | outfit_construction | build_outfit | wardrobe + item | build_outfit_board |
S026 | "Give me three looks from this shirt"                     | uploaded item = shirt  | style | outfit_construction | build_outfit | wardrobe + item | build_outfit_board |
S027 | "What shoes go with these black trousers?"                | uploaded item = trousers | style | item_styling | style_this | wardrobe | style_this_board |
S028 | "Show me looks that work with my white sneakers"          | uploaded item = sneakers | style | item_styling | style_this | wardrobe | style_this_board |
```

### Shopping (7)
```
S029 | "Find similar dresses"                                    | recent viewed dress | shopping | shopping_assist | shopping_assistance | recently_viewed | shopping_card |
S030 | "Where can I buy a chocolate-brown belt?"                 | new session | shopping | shopping_assist | shopping_assistance | none | shopping_card |
S031 | "Complete this look — I'm missing shoes"                  | uploaded outfit | shopping | shopping_assist | shopping_assistance | wardrobe | shopping_card |
S032 | "Recommend a blazer for office"                           | style_dna     | shopping | shopping_assist | shopping_assistance | style_dna | shopping_card |
S033 | "Suggest earrings for this saree"                         | uploaded saree | shopping | shopping_assist | shopping_assistance | wardrobe | shopping_card |
S034 | "What should I buy this month?"                           | wardrobe + gap analysis | shopping | shopping_assist | shopping_assistance | wardrobe_gap | shopping_card |
S035 | "Cheap alternatives for this bag under 2000 rupees"       | uploaded bag  | shopping | shopping_assist | shopping_assistance | none | shopping_card |
```

### Calendar (12)
```
C001 | "calendar"                                                | new session | calendar | navigate | navigation | none | open_calendar |
C002 | "Show my events for tomorrow"                             | new session | calendar | plan     | text_only + list | events | events_list |
C003 | "Add event"                                               | (chip)      | calendar | create   | calendar_action | none | calendar_capture_form |
C004 | "Schedule a dentist appointment Friday at 4 PM"           | new session | calendar | create   | calendar_action | user timezone | calendar_confirmation | reported failure — title should be "Dentist appointment", not raw sentence
C005 | "Remind me to submit the report tomorrow at 6 PM"         | new session | calendar | create   | calendar_action | user timezone | calendar_confirmation |
C006 | "Meeting with Alex on Monday 10am"                        | new session | calendar | create   | calendar_action | user timezone | calendar_confirmation |
C007 | "Doctor appointment"                                      | new session | calendar | create   | clarification | none | clarification_bubble | missing date + time
C008 | "Prep for tomorrow"                                       | new session | planner | plan | planner_action | events, weather, wardrobe, medi, meals | planner_prep_card | reported failure — must land on planner UI
C009 | "Plan my day"                                             | new session | planner | plan | planner_action | events, weather, meals | planner_plan_card |
C010 | "Move my 3 PM meeting to 5 PM"                            | existing event | calendar | update | calendar_action | events | calendar_confirmation |
C011 | "Cancel my Wednesday appointment"                         | existing event | calendar | delete | clarification | events | clarification_bubble | must confirm which
C012 | "Any meetings tomorrow?"                                  | new session | calendar | plan | text_only + list | events | text_bubble + events_list |
```

### Fitness (10)
```
F001 | "What is a calorie deficit?"                              | new session | fitness | information | text_only | none | text_bubble |
F002 | "Plan a 20-minute workout"                                | new session | fitness | plan | workout_plan | fitness_profile | workout_card |
F003 | "Give me a home workout"                                  | new session | fitness | plan | workout_plan | fitness_profile | workout_card |
F004 | "How many push-ups is a good goal?"                       | fitness_profile | fitness | advice | text_only | fitness_profile | text_bubble |
F005 | "Log 30 minutes of yoga"                                  | new session | fitness | track | text_only + confirmation | none | text_bubble |
F006 | "Fitness today"                                           | new session | fitness | plan | workout_plan | fitness_profile | workout_card |
F007 | "What should I eat after workout?"                        | fitness_profile + diet_profile | fitness+diet | advice | text_only | fitness_profile, diet_profile | text_bubble |
F008 | "Set a reminder to exercise Monday 7 AM"                  | new session | calendar | create | calendar_action | timezone | calendar_confirmation | cross-module
F009 | "How's my week been?"                                     | adherence data | fitness | track | text_only + stats | adherence | text_bubble + stats |
F010 | "Give me a low-impact routine — my knees hurt"            | fitness_profile | fitness | plan | workout_plan | fitness_profile | workout_card |
```

### Diet & Recipes (10)
```
D001 | "Give me a paneer recipe"                                 | new session | diet | recipe | recipe | diet_profile | recipe_card |
D002 | "What should I eat today?"                                | diet_profile | diet | plan | meal_plan | diet_profile | meal_plan_card |
D003 | "Show me a high-protein lunch"                            | diet_profile | diet | plan | meal_plan | diet_profile | meal_plan_card |
D004 | "How many calories in a chapati?"                         | new session | diet | information | text_only | none | text_bubble |
D005 | "Plan my meals for the week"                              | diet_profile | diet | plan | meal_plan | diet_profile | meal_plan_card |
D006 | "Suggest a light dinner"                                  | diet_profile | diet | plan | meal_plan | diet_profile | meal_plan_card |
D007 | "Vegan alternatives to paneer"                            | new session | diet | information | text_only | none | text_bubble |
D008 | "What's a healthy breakfast?"                             | diet_profile | diet | plan | meal_plan | diet_profile | meal_plan_card |
D009 | "Log 2 rotis and dal"                                     | new session | diet | track | text_only + confirmation | none | text_bubble |
D010 | "Add spinach to my grocery list"                          | new session | diet | create | text_only + confirmation | none | text_bubble |
```

### Medi (7)
```
M001 | "Remind me to take vitamin D at 9 AM"                     | new session | medi | create | text_only + confirmation | none | text_bubble |
M002 | "Mark my morning meds as taken"                           | meds > 0 | medi | update | text_only + confirmation | meds | text_bubble |
M003 | "What medicines do I have due today?"                     | meds > 0 | medi | plan | text_only + list | meds | text_bubble + list |
M004 | "How does paracetamol work?"                              | new session | medi | information | clarification | none | clarification_bubble | must include medical disclaimer |
M005 | "Set a reminder for my tablet"                            | ambiguous — which tablet? | medi | clarification | clarification | meds | clarification_bubble |
M006 | "I took my medicine"                                      | one med | medi | update | text_only + confirmation | meds | text_bubble |
M007 | "Skip today's dose"                                       | new session | medi | update | clarification | meds | clarification_bubble | require confirmation |
```

### Planner / Prep (6)
```
P001 | "Prep for tomorrow"                                       | events, wardrobe, weather | planner | plan | planner_action | events, weather, wardrobe, meds, meals | planner_prep_card | reported failure |
P002 | "Pack for Goa 4-day trip"                                 | wardrobe, weather=goa | planner | plan | planner_action | wardrobe, weather | packing_checklist_card |
P003 | "Business travel checklist"                               | new session | planner | plan | planner_action | none | packing_checklist_card |
P004 | "Wedding checklist"                                       | new session | planner | plan | planner_action | none | checklist_card |
P005 | "Plan a birthday party for Saturday"                      | new session | calendar+planner | create+plan | planner_action | timezone | planner_card + calendar_confirmation |
P006 | "What do I need to do before my flight tomorrow?"         | calendar has flight | planner | plan | planner_action | calendar, weather | planner_prep_card |
```

### General & meta (12)
```
G001 | "hi"                                                     | new session | home | casual | text_only | none | text_bubble | greeting |
G002 | "what can you do"                                        | new session | home | information | text_only | none | text_bubble | help_identity |
G003 | "thanks"                                                 | new session | home | casual | text_only | none | text_bubble |
G004 | "how are you"                                            | new session | home | casual | text_only | none | text_bubble |
G005 | "sorry, i meant tomorrow"                                | previous ambiguous message | -- | clarify_previous | text_only | history | text_bubble |
G006 | "cancel that"                                            | previous create action | -- | delete | clarification | history | clarification_bubble |
G007 | "actually make it 5pm"                                   | previous calendar create | calendar | update | calendar_action | history | calendar_confirmation |
G008 | "show me something else"                                 | previous style board | style | inspiration | visual_inspiration | history | visual_board |
G009 | "why did you pick this?"                                 | previous style board | style | information | text_only | history | text_bubble |
G010 | "not this one"                                           | previous style board | style | inspiration | visual_inspiration | history | visual_board | negative feedback recorded |
G011 | "i like this"                                            | previous style board | style | update | text_only + confirmation | history | text_bubble | positive feedback recorded |
G012 | "open wardrobe"                                          | new session | wardrobe | navigate | navigation | none | open_wardrobe |
```

### Ambiguity, spelling, mixed-language (10)
```
X001 | "wht 2 wear today"                                        | new session | style | recommendation | wardrobe_recommendation | wardrobe | wardrobe_outfit_board |
X002 | "office outfit"                                           | new session | style | recommendation | wardrobe_recommendation | wardrobe | wardrobe_outfit_board |
X003 | "kal ke liye kya pehnu"                                   | new session | style | recommendation | wardrobe_recommendation | wardrobe | wardrobe_outfit_board | Hindi-English |
X004 | "network"                                                 | new session | -- | clarify | clarification | none | clarification_bubble | currently matches "work" substring → office occasion |
X005 | "workshop tomorrow"                                       | new session | -- | clarify | clarification | none | clarification_bubble | currently substring-matches "work" |
X006 | "meeting John for coffee"                                 | new session | calendar | create | clarification | none | clarification_bubble | need date/time |
X007 | "cook me dinner"                                          | new session | diet | recipe | recipe | diet_profile | recipe_card |
X008 | "date"                                                    | new session | -- | clarify | clarification | none | clarification_bubble | style or calendar? currently forces occasion=date_night |
X009 | "outfit for date"                                         | new session | style | recommendation | wardrobe_recommendation | wardrobe | wardrobe_outfit_board |
X010 | "date with Priya at 8pm"                                  | new session | calendar | create | calendar_action | timezone | calendar_confirmation |
```

### Cross-module / conversational (8)
```
XM01 | "What should I wear to tomorrow's client meeting?"        | calendar has meeting | style | recommendation | wardrobe_recommendation | calendar, weather, wardrobe | wardrobe_outfit_board |
XM02 | "Add it to my calendar"                                   | previous style suggestion tied to occasion | calendar | create | calendar_action | history | calendar_confirmation |
XM03 | "Remind me to pack for the trip"                          | previous packing plan | calendar | create | calendar_action | history | calendar_confirmation |
XM04 | "What's my plan today?"                                   | events + meals + workout | planner | plan | planner_action | events, meals, workout | planner_plan_card |
XM05 | "Log the outfit I wore yesterday"                         | recently worn | wardrobe | update | text_only + confirmation | recently_worn | text_bubble |
XM06 | "Any missing pieces to complete this look?"               | previous board | shopping | shopping_assist | shopping_assistance | wardrobe, previous_board | shopping_card |
XM07 | "Send this outfit to my sister"                           | previous board | out_of_scope | -- | text_only | none | text_bubble | politely decline / clarify |
XM08 | "Try this look on me"                                     | previous board | style | tryon | tryon_card | wardrobe, previous_board | tryon_card |
```

Total: **100** cases (10 + 10 + 8 + 7 + 12 + 10 + 10 + 7 + 6 + 12 + 10 + 8).

The corpus is deliberately read-only — none of the cases require a
destructive backend call to evaluate. The `create`/`update`/`delete` cases
are checked at the classification and payload-shape level, not by actually
inserting Appwrite rows.

---

## 15. Layer-by-Layer Scores (0–10)

Evidence for each score is in the referenced sections.

| Layer | Score | Evidence |
|---|---:|---|
| Domain classification | 5 | Domains exist and are largely correct once one classifier picks them, but keyword substring bugs (`work` matches `network`, `workshop`, `homework`) and overlapping regex lists (see §3, §4) cause routine misclassification. |
| Intent classification | 4 | Six intent modes work reliably (`daily_outfit`, `wardrobe_query`, greetings). Everything else is decided by keyword and can be overturned downstream. `_fallback_intent` duplicates its own keyword blocks (lines 436 vs 557 in `brain/intent_engine.py`). |
| Response-mode selection | 2 | The field does not exist. Renderer is presence-driven (§6). |
| Clarification | 3 | Only two clarification codepaths exist: calendar time-missing (`chat.py:4501`) and style-vague (`_needs_style_clarification`). Everything else fails-open into a plausible-looking answer. |
| Context completeness | 4 | Wardrobe, weather, style-DNA, calendar events are individually available; no normalized packet, most modules ignore what they receive (§8). |
| Context freshness | 5 | Weather and calendar are fetched per-request; global JSON memory files are process-life. |
| Orchestration | 3 | `AhviOrchestrator` is a router, not an orchestrator. No multi-step plans reach the executor (§9). |
| Deterministic execution | 5 | `ExecutionEngine` is well-shaped but unused from the main path. Handler-signature triple fallback (`brain/execution_engine.py:46-56`) is technical debt. |
| Output validation | 3 | Only text-level. No mode-vs-payload check. Card cross-validation missing (§10). |
| Frontend renderer fidelity | 3 | Presence-driven; can render multiple mutually-exclusive block types from one response (§6). |
| Request-state isolation | 2 | No `request_id`, single `_isTyping`, PATCH-5 string match (§7). |
| Memory utilization | 4 | Style DNA is enriched, wear-today is logged, but there is no visible read-back that would tell the user "AHVI remembered X". |
| Observability | 5 | Extensive `logger.info` inside the orchestrator ("AHVI_VISUAL_FIRST_ROUTE", "style intent=…", "style card detail uid=…"). Missing: canonical event names, request_ids, and structured tags. |
| Test coverage | 4 | Frontend has visual board goldens and app smoke tests (recent commits `deccaa7`, `406d4cf`); backend has `tests/` folder but the classifier / response-mode contract is not covered. |

---

## 16. Module-by-Module Scores (0–10)

End-to-end user experience, not "does the capability exist".

| Module | Score | Rationale |
|---|---:|---|
| **1. Style** | 4 | Boards render, but for the wrong requests. Text vs board vs inspiration boundary is broken. Recent backend commits are almost entirely style patches — the layer is unstable. |
| **2. Wardrobe** | 5 | Fast count query works (`_fast_wardrobe_count_response`). Full item listing depends on Appwrite path with local-filtered fallback. Gender/consistency gate exists but only on the style-fetch path. |
| **3. Calendar** | 5 | Natural-language create works when time and date are unambiguous. Fails hard on partial input (uses raw sentence as title). Two competing create paths. |
| **4. Prep & Plan** | 3 | Frontend and backend disagree on whether "prep" is `planner` or `calendar`. `_module_plan_pack_response` vs `handle_module_chat(calendar)` land on different UIs. |
| **5. Fitness** | 3 | `handle_fitness_chat` is pure static templates; no real fitness intelligence. Chip labels ("Home workout", "HIIT") map to hardcoded strings. |
| **6. Diet** | 3 | Same as Fitness — static templates in `handle_diet_chat`. |
| **7. Medi** | 5 | Genuine mark-as-taken flow with idempotency and Appwrite writes. Best-implemented module in `module_chat_service`. |
| **8. General assistant** | 4 | Greetings, help_identity, small_talk fire correctly. Anything more complex leaks into `general → sticky-history → whatever was last`. |
| **9. Cross-module intelligence** | 2 | Effectively unimplemented. No single request currently traverses calendar → weather → wardrobe → outfit. |

---

## 17. Canonical Target Contract (Proposed, Do Not Implement)

```jsonc
{
  "request_id": "req_2026-08-06T09:22:03.192Z_ab12",
  "session_id": "chat_session_71",
  "domain": "style",
  "intent": "advice",
  "action": "provide_style_tips",
  "response_mode": "text_only",

  "confidence": 0.96,
  "requires_clarification": false,
  "missing_information": [],

  "context_used": [
    "style_profile",
    "conversation_history:last_3"
  ],
  "tool_plan": [],

  "payload": {
    "text": "Three tips: lift the shoulder line, anchor the palette, …"
  },

  "explanation": {
    "why": "You asked for tips (advice intent, text_only mode).",
    "context_shown": ["style_profile.body_shape=hourglass"]
  },

  "validation": {
    "passed": true,
    "reason_codes": [],
    "gates_run": ["response_mode_matches_intent", "no_orphan_visual_blocks"]
  }
}
```

**Rules.**

- **Required top-level:** `request_id`, `domain`, `intent`, `action`,
  `response_mode`, `confidence`, `payload`, `validation`.
- **Allowed domains, intents, response_modes:** the enumerations proposed
  at the end of §5.
- **Confidence rules:**
  - `confidence >= 0.75` → answer.
  - `0.40 ≤ confidence < 0.75` → `response_mode = clarification`,
    `requires_clarification = true`.
  - `confidence < 0.40` and no history to disambiguate → `response_mode =
    clarification` with a "did you mean … / did you mean …" chip pair.
- **Clarification rules:**
  - A `create` intent with a missing required slot (title, date, time for
    calendar) forces `response_mode = clarification`, never a
    best-guess create.
  - A style query with `_needs_style_clarification` → clarification.
- **Renderer rules:**
  - The Flutter renderer switches on `response_mode`. No block is drawn
    from field presence.
  - `text_only` responses cannot include a `visual_directions` block —
    the response builder must strip it.
  - `visual_inspiration` responses cannot include a `wardrobe_recommendation`
    board, and vice versa.
  - `clarification` responses render only a text bubble + chips —
    even if any other field is present.
- **Backward compatibility.** During rollout, the old field-per-block
  parser can remain as a fallback for a version window, guarded by a
  feature flag (`ENABLE_CANONICAL_RESPONSE_MODE`). Once enabled, presence-
  driven paths are dead code and should be removed.

---

## 18. Ten-Day Remediation Plan

### P0 — required for the ten-day challenge (perceived intelligence)

Ranked by impact / effort. Each item names the concrete file, the risk,
and the flag it should ship behind.

| # | Item | Repo(s) | Files most likely touched | Risk | Expected user impact | Test | Depends on | Effort | Flag |
|---|---|---|---|---|---|---|---|---|---|
| P0-1 | Canonical `response_mode` field + strict frontend renderer switch | both | backend response envelopes; `lib/feature/chat/services/ahvi_block_response_parser.dart`, `lib/chat.dart` | Medium — many endpoints must add the field | Removes 3 of 6 reported failures | New goldens covering the failure cases; §14 corpus | none | 3 days | `ENABLE_CANONICAL_RESPONSE_MODE` |
| P0-2 | `request_id` per send + late-response rejection + loader cancellation | frontend | `lib/services/backend_service.dart`, `lib/chat.dart` (message model, send handler, dispose logic) | Low | Loader never disagrees with content | Flutter widget test with delayed mock backend | none | 1 day | none (defensive) |
| P0-3 | Collapse the 11 in-line `_is_*/_detect_*` classifiers in `chat.py` and the 3 parallel classifiers in `brain/*` into a single `classify(query, context, history)` call | backend | new `services/classifier.py`; delete/deprecate in `routers/chat.py`, `brain/intent_engine.py`, `brain/nlu/intent_router.py`, `services/stylist_knowledge_service.classify_style_mode` | Medium — routing behaviour changes for everyone | Removes duplicate classification wars; makes classifier improvements land in one place | Corpus §14 gated | P0-1 | 3 days | `ENABLE_SINGLE_CLASSIFIER` |
| P0-4 | Remove occasion→outfit override at `orchestrator.py:1016-1020` and `_should_default_visual_inspiration` default at `chat.py:4581` | backend | `brain/orchestrator.py`, `routers/chat.py` | Medium — some queries that previously produced boards will now produce text or clarification | Fixes "style tips shows curating loader" and "color analysis returns unrelated board" | Corpus §14 S001–S010 | P0-3 | 0.5 day | inherit P0-3 flag |
| P0-5 | Clarification-on-uncertainty. Confidence < 0.75 OR missing required slot → return `response_mode = clarification` with 2 chips | backend | `services/classifier.py`, `routers/chat.py`, `services/module_chat_service.py`, calendar create path | Low | Fewer confidently-wrong answers | Corpus §14 C007, X004, X005, X008, S001 | P0-1, P0-3 | 1 day | inherit P0-3 flag |
| P0-6 | Strip mutually-exclusive block combinations at response build time (a `text_only` response must not carry `visual_directions`) | backend | `brain/response/response_assembler.py`, `brain/response_validator.py` | Low | Cannot land the "text + unrelated board" failure any more | Golden JSON responses | P0-1 | 0.5 day | inherit P0-1 flag |
| P0-7 | Rename `AhviOrchestrator` → `AhviRouter` and stop the monkey-patch wrapper (fold Style-DNA enrichment into the router itself) | backend | `brain/orchestrator.py` | Low | Removes silent-degradation path when Style DNA enrichment fails | Existing tests | none | 0.5 day | none |
| P0-8 | Fix `gemini-3.5-flash` → `gemini-2.5-flash` in `services/agent_style_orchestrator.py:42` and `services/agent_metadata_validator.py:31` | backend | those two files | Low | Agent stops silently falling back to Ollama | Manual smoke | none | 5 min | none |
| P0-9 | Prep-tomorrow single path: pick either `_module_plan_pack_response` or `handle_module_chat(planner)` and delete the other; make frontend `_isPlanPackRequest` route deterministically | both | `routers/chat.py`, `services/module_chat_service.py`, `lib/chat.dart` | Low | Fixes "prep routes to a different UI" | Corpus §14 P001, P006, C008 | P0-1 | 1 day | none |
| P0-10 | Regression corpus wired into CI as JSON goldens (classifier decision + response_mode + rendered block set) | both | `tests/` in both repos | Low | Prevents regression of the fixes above | Corpus §14 | P0-1, P0-3 | 1.5 days | none |

**Total P0 effort: ~12 developer-days** — fits inside the ten-day window
with two engineers working in parallel (frontend/backend split), assuming
one has the flags landed on Day 1.

### P1 — valuable before August 25

- **P1-1** Normalize context compiler (§8 recommendation) so every
  handler receives a single `Context` dataclass.
- **P1-2** Delete the duplicate keyword blocks in `brain/intent_engine.py`
  (`early_module_hits` ≡ `module_hits`, `style_priority_phrases` twice).
  Pure ponytail-audit finding.
- **P1-3** Delete unused `brain/nlu/intent_router.py` once P0-3 lands.
  Also delete `brain/agent_system.py`'s rule fallback path (dead plan).
- **P1-4** Item identity + image-provenance gate on Style This
  responses (fixes "belt → dress image").
- **P1-5** Replace `handle_fitness_chat` / `handle_diet_chat` /
  `handle_skincare_chat` static templates with real module handlers that
  consume the normalized context.
- **P1-6** Split `services/style_reasoning_engine.py` (8208 lines) —
  begin by isolating pure functions from stateful ones.

### P2 — post-MVP

- **P2-1** True `Orchestrator` with `plan → execute → validate → compose`
  and a shared `Context/State` — replace the current `ExecutionEngine`
  wiring.
- **P2-2** Cross-module tool composition ("What should I wear to
  tomorrow's client meeting?" fans out to calendar + weather + wardrobe +
  outfit).
- **P2-3** Explicit `explanation.context_shown` so the user can see why
  a recommendation was made.
- **P2-4** Structured feedback loop: `like / dislike / would-wear` events
  update the ranker vector inside the classifier.
- **P2-5** Delete the empty root-level frontend files `taskkill` (0
  bytes), `query` (7 bytes), and `et --hard previous-commit-hash` at
  `AHVI-frontend/` — accidentally committed shell fragments (§20).

---

## 19. No-Change Boundaries

Nothing was mutated in the two attached repos during this audit. In
particular:

- No `git checkout`, `reset`, `clean`, or `restore`.
- No file edits, no `git add`, no commits.
- No branch push, no PR opened, no PR updated.
- No Cloud Run traffic change, no environment variable set, no
  secret rotation.
- No test was executed. (The corpus in §14 is a proposal, not a run.)
- No UI redesign, no classifier introduced, no experimental flag toggled.

---

## 20. Evidence Table (file / line → finding)

| Finding | File | Line(s) |
|---|---|---|
| Renderer is presence-driven | `lib/feature/chat/services/ahvi_block_response_parser.dart` | 7–202 |
| No `response_mode` field in the codebase | grep across both repos | zero hits |
| Curation loader is a fixed 1800 ms timer | `lib/feature/chat/widgets/blocks/visual_directions/curation_reveal.dart` | 59, 79 |
| No `request_id`, single `_isTyping` boolean | `lib/chat.dart` | 1442, 1450, 1667 |
| PATCH-5 hardcoded backend-error string match | `lib/chat.dart` | 1617–1627 |
| Client-side classifiers pick endpoint | `lib/chat.dart` | 1502–1558 |
| 11 inline `_is_*/_detect_*` classifiers in one handler | `routers/chat.py` | 4451–4589 |
| Style module always renders visual (VISUAL_INSPIRATION or WARDROBE_STYLE) | `routers/chat.py` | 4593 |
| Occasion → outfit intent override | `brain/orchestrator.py` | 1016–1020 |
| Monkey-patched wrapper on `AhviOrchestrator.run` | `brain/orchestrator.py` | 1323–1388 |
| Duplicate keyword blocks (`early_module_hits` ≡ `module_hits`) | `brain/intent_engine.py` | 436–511 vs 557–632 |
| `_fallback_intent` sticky-history override | `brain/intent_engine.py` | 857–873 |
| `IntentRouter` isolated singleton, keyword-only | `brain/nlu/intent_router.py` | 174–226 |
| `classify_style_mode` can never return `VISUAL_INSPIRATION` | `services/stylist_knowledge_service.py` | 16 vs 22–32 |
| `_normalize_domain` aliases `planner → calendar` | `services/module_chat_service.py` | 14–34 |
| Two calendar-create code paths | `routers/chat.py:4493` and `services/module_chat_service.py:469` | see file |
| Static templates in fitness/diet/skincare/bills handlers | `services/module_chat_service.py` | 334–693 |
| `AgentSystem` rule fallback is `no_op` for most intents | `brain/agent_system.py` | 103–118 |
| `ExecutionEngine` handler signature triple-try | `brain/execution_engine.py` | 46–56 |
| `DecisionEngine` misnamed — only a ranker | `brain/decision_engine.py` | 6–66 |
| Ollama `llama3.2:3b` default | `services/llm_service.py` | 47 |
| `gemini-3.5-flash` invalid model id | `services/agent_style_orchestrator.py:42`, `services/agent_metadata_validator.py:31` | see files |
| Response validator polishes text only | `brain/response_validator.py` | 128–191, 246–293 |
| Accidentally-committed shell fragments in frontend root | `AHVI-frontend/taskkill` (0 bytes), `AHVI-frontend/query` (7 bytes), `AHVI-frontend/et --hard previous-commit-hash` (13229 bytes) | listed by `ls -la` |
| Duplicate weak-match guard | `lib/chat.dart` | 1632–1639 |
| Response envelope with duplicate field aliases (`message`, `message_text`, `response`; `board_ids`, `board_id`, `pack_ids`, `pack_id`) | `brain/response_validator.py` and `services/module_chat_service.py` | many |
| `handle_style_chat` returns hardcoded copy | `services/module_chat_service.py` | 696–698 |

---

## 21. Git Status for Every Worktree Inspected

Only two worktrees were accessible in this environment; both are listed
here. The other worktrees the user maintains on Windows could not be
reached from this session (see §2).

### `/home/user/AHVI-frontend`

```
repo:       devlovasit-source/ahvi-frontend
branch:     claude/ahvi-intelligence-audit-sopr1z
HEAD:       99bc53c483f08b4983d28926004052a2edf05f74
commit ts:  2026-07-29 12:24:21 +0530
upstream:   (none configured)
ahead/behind vs origin/main: 0 / 0
dirty:      (none)
untracked:  (none)
```

Recent commits: see §2 (last 20).

### `/home/user/AHVI-latest_Backend`

```
repo:       devlovasit-source/ahvi-latest_backend
branch:     claude/ahvi-intelligence-audit-sopr1z
HEAD:       3d0c0d3a36bc6ece1e55c9070b452ece1cccc3de
commit ts:  2026-07-27 17:24:34 +0530
upstream:   (none configured)
ahead/behind vs origin/main: 0 / 0
dirty:      (none)
untracked:  (none)
```

Recent commits: see §2 (last 20).

---

## 22. Confirmation

- **Worktrees inspected:** 2 — the two GitHub-attached clones listed in
  §21. The Windows-local worktrees the user listed were not reachable
  from this session.
- **Authoritative frontend / backend pair selected:**
  `devlovasit-source/ahvi-frontend@99bc53c` +
  `devlovasit-source/ahvi-latest_backend@3d0c0d3`.
- **Dominant intelligence problem:** duplicated / conflicting routing
  with no canonical `response_mode` contract; frontend renders
  presence-driven; multiple parallel classifiers silently override each
  other; there is no true orchestrator (§1, §4, §6, §7, §9, §16).
- **Is a classifier alone sufficient?** No. The current classifier
  decision is already discarded by at least three downstream layers.
  Any new classifier will be discarded in the same places until the
  target contract in §17 is enforced.
- **Top five P0 fixes:** canonical `response_mode`; `request_id` +
  late-response rejection; single `classify()` collapsing the eleven
  in-line + three parallel classifiers; remove occasion→outfit override
  and visual-first default; clarification on uncertainty (§18 P0-1
  through P0-5).
- **Expected improvement from those five:** removes the four visible
  reported failure classes and prevents their return via §14 regression
  corpus. Perceived coherence jumps materially; outfit quality is a
  separate work stream.
- **Report path:**
  `AHVI-latest_Backend/AHVI_INTELLIGENCE_ORCHESTRATION_AUDIT.md`.
- **Confirmation:** no file was edited, staged, committed, pushed,
  deployed, or reconfigured during this audit. Both `main` branches and
  all Cloud Run configuration are exactly as they were at the start of
  this session.
