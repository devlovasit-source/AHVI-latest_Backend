# AHVI Intelligence Spine Audit — Backend Report
**Prepared by:** Jules, Senior Software Engineer
**Scope:** `~/ahvi-backend-pr22-24-25` (cloud repo)
**Status:** AUDIT ONLY (Read-Only) — No changes introduced.

---

## Phase 0 — Repository State

### 1. Git Repository Information
As executed in the live terminal session:
```bash
$ git branch --show-current
jules-3603555128623054237-05d905e5

$ git rev-parse --short HEAD
295603c

$ git status --short
# (empty - working directory clean)

$ git diff --stat
# (empty)

$ git diff --name-only
# (empty)
```

### 2. Match with origin/main
The HEAD commit matches `origin/main` at `295603c1ce8423285938899b87f3aadf71eadd49` (which incorporates PRs #22, #24, #25, #26, and #27).

### 3. Open Feature Branches or Unmerged Commits
Our remote fetch shows the following open integration/feature branches:
- `origin/integration/pr22-pr24-pr25`
- `origin/style-refactor-phase4-9`
- `origin/feature/editorial-style-boards`
- `origin/fix/style-board-variety-save`
- `origin/patch/disable-router-style-fallback`

### 4. Status of Integration-Only Backend Commit `95fd13c`
A full-depth inspection of all refs, remote branches, stash layers, and reflogs (`git log --all --reflog --grep="95fd13c"`) reveals that **commit `95fd13c` is not present** in the local or fetched remote history of the repository. It is a known missing integration branch or isolated branch commit from a parent/alternate fork that has not been pushed to this repository's origin.

---

## Phase 1 — Architecture Map

The backend request lifecycle is split into divergent, concurrent, and localized paths rather than traversing a unified canonical orchestrator pipeline. Below is the exact step-by-step mapping of how an incoming request is handled.

### 1. User Message Entry
- **Repository:** Backend
- **Exact File:** `routers/chat.py`
- **Class/Function:** `text_chat` (POST `/api/text` or POST `/api/chat/text`)
- **Input Fields:** `TextChatRequest` model (contains `messages`, `language`, `current_memory`, `user_profile`, `user_id`, `userID`, `module_context`, `wardrobe`, `style_action`, `style_state`, etc.)
- **Output Fields:** Sanitized inputs passed to down-stream routing.
- **Error Handling:** HTTP 400 if `messages` is empty or if the message text resolves to empty.

### 2. Authentication and User Binding
- **Repository:** Backend
- **Exact File:** `routers/chat.py`
- **Class/Function:** `text_chat` inline auth extraction.
- **Input Fields:** `http_request.state.user` (populated by `middleware/auth_middleware.py`)
- **Output Fields:** Bound and validated `user_id` string.
- **Error Handling:** Throws HTTP 401 if `http_request.state.user` is missing or invalid. Throws HTTP 403 if the request's supplied `user_id` does not match the authenticated session.

### 3. Intent Classification & Module Selection
The backend performs classification at multiple points using both heuristics and LLMs.
- **Repository:** Backend
- **Exact File:** `routers/chat.py` and `brain/intent_engine.py`
- **Class/Function:** `detect_intent(user_input)`
- **Input Fields:** `user_input` string.
- **Output Fields:** `Dict[str, Any]` (e.g. `{"intent": "organize_hub", "slots": {"module": "meal"}}`).
- **Additional Router Checks:**
  - `_is_explicit_style_request(user_input)` (regex/word-matching fallback)
  - `_is_general_chat_request(user_input)`
  - `_is_fast_wardrobe_count_query(user_input)`
  - `_detect_visual_board_type(user_input)`

### 4. Canonical Request-Context Creation
There is **no single canonical request/context object** shared across all modules. Instead, separate context schemas are instantiated locally for each route:
- **Style Flow Context:** Constructed in `routers/chat.py` inside `_demo_style_board_payload` or `services/style_context_service.py` (`build_style_context`).
- **Module Chat Context:** Constructed in `routers/chat.py` (`module_chat`) and passed to `services/module_chat_service.py` (`handle_module_chat`).
- **Calendar Runtime Input:** Uses the `CalendarEventInput` pydantic model in `models/calendar_models.py`.

### 5. Profile/Life Graph Context Assembly
- **Repository:** Backend
- **Exact File:** `routers/chat.py`
- **Class/Function:** `_ahvi_resolve_effective_user_profile`
- **Input Fields:** Stored profile fetched via `services/data_access_service.py` (`get_user_profile`) + request-supplied overrides.
- **Output Fields:** Merged `effective_user_profile` dictionary.

### 6. Module Execution
Depending on the classified mode, control is routed to one of several divergent execution paths:
- **A. Conversational LLM Chat:** `routers/chat.py` -> `_llm_chat_response`
- **B. Style Curation & Board Generation:** `routers/chat.py` -> `_demo_style_board_payload` -> `brain/outfit_pipeline.py` -> `build_style_flow_response`
- **C. Stylist Advice & Visual Directions:** `routers/chat.py` -> `style_reasoning_engine.reason` (`services/style_reasoning_engine.py`)
- **D. Plan & Pack Flow:** `routers/chat.py` -> `_module_plan_pack_response` -> `brain/plan_pack_flow.py` -> `build_plan_pack_response`
- **E. Module Chat Service:** `routers/chat.py` -> `module_chat` -> `services/module_chat_service.py` -> `handle_module_chat`

### 7. Candidate Generation
- **Repository:** Backend
- **Exact File:** `services/style_reasoning_engine.py` (for catalog assets) and `services/style_flow_service.py` (for wardrobe items).
- **Class/Function:** `_style_asset_rows` and `_fetch_wardrobe_for_style`.
- **Input Fields:** `user_id` and limit caps.
- **Output Fields:** Lists of matching database or catalog rows.

### 8. Rules and Safety Validation
- **Repository:** Backend
- **Exact File:** `services/beta_style_bridge.py` and `brain/engines/outfit_quality_guard.py`
- **Class/Function:** `validate_style_response` and `reject_board_for_occasion`.
- **Checks Applied:** Outfits completeness, renderable image check, occasion private-wear guard, duplicate check.

### 9. Repair/Fallback
- **Repository:** Backend
- **Exact File:** `services/beta_style_bridge.py`
- **Class/Function:** `refine_style_response` (attempts a second candidate offset of `1` if the first validation pass fails).
- **Fallback Paths:** If the orchestrator fails or times out, it drops into `_structured_error_response` with a generic timeout message or is downgraded to `_demo_style_board_payload` static fallback.

### 10. Persistence
- **Repository:** Backend
- **Exact File:** `routers/chat.py` -> `_save_plan_pack_payload` and `services/wardrobe_persistence_service.py`
- **Output Fields:** Document created in Appwrite databases (`plans`, `meds`, `med_logs`, `calendar_events`).

### 11. Final Response Generation
- **Repository:** Backend
- **Exact File:** `routers/chat.py`
- **Class/Function:** `text_chat` response formatting.
- **Output Fields:** Envelope containing `success`, `message`, `cards`, `style_boards`, `chips`, `data`, and `meta`.

---

## Phase 2 — Canonical Goal Contract

### 1. Contract Fields Presence
AHVI lacks a single unified data class or pydantic model enforcing the "Goal Contract". Instead, fields are scattered across dictionaries, custom pydantic inputs, and session memories.

| Field Name | Status | Initially Created At | Consumed By | Where Lost / Recomputed | Impact on Intent Preservation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **authenticated_user_id** | Present | `auth_middleware.py` (stored on `request.state.user`) | `routers/chat.py` | Preserved securely across all router paths. | High. Prevents cross-account data leaks. |
| **module** | Inconsistent | `intent_engine.py` / `detect_intent` | `routers/chat.py` | Recomputed in `module_chat` as `_normalize_domain`. | High. Misroutes Prepare/Prep to general Chat. |
| **intent** | Inconsistent | `intent_engine.py` / `detect_intent` | `style_reasoning_engine.py` | Lost inside fallback paths and re-classified locally. | High. Causes user intent to drift across turns. |
| **source_policy** | Partial | `beta_style_bridge.py` | `style_flow_service.py` | Re-evaluated inside local `_is_use_wardrobe_action` checks. | Medium. Silently mixes wardrobe with inspiration assets. |
| **occasion** | Inconsistent | `routers/chat.py` -> `_ahvi_style_occasion` | `style_reasoning_engine.py` | Lost and re-inferred using keyword-matching strings. | High. Forces office/date defaults on daily looks. |
| **requested_roles** | Missing | Not formally defined | None | Re-parsed locally in `beta_style_bridge.py`. | High. Garment roles like "dress" are silently dropped. |
| **timezone** | Partial | `calendar_service.py` (reads default from env) | `calendar_runtime.py` | Lost or fallback to `Asia/Kolkata` when parsing events. | High. Causes local timezone offset misalignment. |
| **Style DNA** | Present | `get_user_profile` | `style_reasoning_engine.reason` | Preserved only inside the stylist reasoning path. | Low. Ignored inside wardrobe styling fallback. |

### 2. Downstream Re-Interpretation of the Original Message
Because the modules do not pass a single immutable contract, downstream services (such as `style_reasoning_engine.py` and `module_chat_service.py`) frequently **re-parse and re-classify the original message text** from scratch. This leads to duplicate intent-classification overhead, high LLM token costs, and a high risk of localized classification mismatches (e.g. the router classifying a query as `fashion`, but the stylist engine classifying it as `general_chat`).

---

## Phase 5 — Life Graph Usage Matrix

The assembly and retrieval of contextual personalization layers are highly siloed.

| Context Source | Read By | Used in Ranking | Used in Validation | Written Back | Silent Fallback Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **User Profile** | `routers/chat.py` | No | No | No | Uses empty dict when Appwrite fails. |
| **Wardrobe Metadata** | `_fetch_wardrobe_for_style` | Yes | Yes (family caps) | No | Skips metadata enrichment when DB is cold. |
| **Style DNA** | `style_reasoning_engine.py` | Yes (aesthetic) | No | No | Ignored in wardrobe fallback. |
| **Likes/Dislikes** | `style_memory_service.py` | No | No | Yes | Ignored in real-time execution. |
| **Calendar** | `module_chat_service.py` | No | No | Yes | Empty list. |
| **Weather** | `_get_weather_cached` | No | Yes (headwear strip) | No | Falls back to default "mild" context. |

---

## Phase 8 — Validation and Repair Architecture

The backend has localized validators, but they do not form a serial, cohesive pipeline.

```
Incoming Request -> [Router Intent Parse] -> [Module Execution] -> [Candidate Selection]
                                                                        |
[Narrative Generation] <- [Final Response Envelope] <- [Repair] <- [Occasion Guard]
```

### Critical Flaw: Narrative is Generated Before Final Validation
In `style_reasoning_engine.py`, the stylist narrative (`advice` / `stylist_reasoning` / `why_it_works`) is generated **during the initial LLM reasoning pass**.

The final outfit candidates are selected and visual validators (such as `_apply_style_guard`, `_sanitize_direction_for_gender`, and `_strip_color_clashes`) run **after** the advice copy has been finalized.

This means that if the validation layer strips a clashing or weather-inappropriate garment, **the generated copy will still describe the removed garment**, creating a critical split between what the AI *says* the user should wear and what is actually rendered in the visual board.

---

## Phase 9 — Observability Gaps

There is a severe lack of structured tracing across the backend repository:
1. **No Single Request ID Propagation:** Request IDs (`request_id`) exist in `services/request_context.py` but are not universally attached to third-party API logs (Ollama, Vertex, Appwrite).
2. **Missing Decision Parameters:** Key styling parameters—such as the extracted `requested_roles`, `unresolved_constraints`, `repair_attempted`, and `fallback_reason`—are logged only in localized `logger.info` calls instead of a standardized logging payload.

---

## Phase 10 — Test Baseline (Backend)

We executed `pytest` against the current repository state:
- **Baseline Results:** 1358 passed, 13 failed (13 baseline errors on our branch as expected prior to any code edits).
- **Baselines Gaps Identified:** No existing tests verify specific multi-role outfits (e.g. S1: "Dress + shoes + bag", S2: "Outerwear + top + bottom + shoes + bag"), calendar timezone-offset duplicates, or prep routing preservation.
```,filepath: