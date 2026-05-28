# AHVI Test Plan — post-deploy / APK install

Use this checklist on the deployed Cloud Run revision + the installed APK.
Capture the observed output (text or screenshot) in the "Result" column and
send back.

Legend: **PASS** = matches expected, **FAIL** = does not, **N/A** = blocked
by environment.

---

## A. Backend health (Cloud Run)

| # | What to do | Expected | Result |
|---|---|---|---|
| A1 | `curl $SERVICE_URL/health` | HTTP 200, body `{"status":"ok"}` or similar | |
| A2 | `curl $SERVICE_URL/docs` (FastAPI) | Swagger UI loads | |
| A3 | Tail logs while sending a chat request | `ahvi.llm.token_budget`, `ahvi.agent.style_orchestration`, `ahvi.metadata.*` lines appear | |

---

## B. Agent Style Orchestrator (Phase 1)

Send these chat prompts from the APK. Each row tests one outcome of the
agent layer.

| # | Prompt | Expected behavior |
|---|---|---|
| B1 | "Can I wear shorts and slides to a client meeting?" | Firm, calm response. No emoji. No slang. AHVI says no — recommends polished alternative. Quality guard blocks boxer/slides combo. |
| B2 | "Outfit for dinner date tonight" | Cards returned. Roles: "Most effortless option", "Softer date-night option", "Stronger evening move". No "Sure!" / "Here are some ideas". |
| B3 | "Gym workout fit" | Activewear allowed. Formal items rejected. Roles: clean performance / comfort-first / outdoor-ready. |
| B4 | "Office today, raining" | Suede / canvas / open-toe rejected. Leather/closed shoes preferred. Summary mentions office composure. |
| B5 | (no wardrobe + ambiguous): "make me look good" | Asks for occasion *or* delivers a safe daily look, never returns generic chatbot fallback. |

---

## C. Metadata Validator (Phase 2)

Test by capturing/saving new wardrobe items via the APK camera/upload flow.

| # | Item to capture | Expected |
|---|---|---|
| C1 | Photo of boxer shorts | Saved as outfit. `wardrobe_style_metadata` row created with `style_role: loungewear`, `blocked_occasions` includes office/client_meeting. |
| C2 | Photo of slides | Metadata: `category: Footwear`, blocked: office/wedding. |
| C3 | Photo of blazer | Metadata: `style_role: businesswear`, allowed: office/client_meeting/interview. |
| C4 | Photo of running shorts | Metadata: `style_role: activewear`, blocked: office/wedding/formal_event. |
| C5 | Photo of low-confidence item (weird angle) | Save succeeds. Metadata has `manual_review_required: true`. |
| C6 | Re-fetch wardrobe via app | Items now carry `style_metadata` field. |

---

## D. Tone & response stabilization (Phase 3)

| # | Input | Expected output |
|---|---|---|
| D1 | "Sure! Here are some ideas 😊" (as user msg) | AHVI's reply contains no "Sure!", no "Here are some ideas", no emoji. |
| D2 | "I don't feel confident in this" | Soft, supportive, no slang ("lowkey"/"highkey"), no `!`, one next step. |
| D3 | "Can I wear shorts and slides to a client meeting?" | Firm and calm: "Shorts and slides won't land for a client meeting…", no harsh tone. |
| D4 | "Yo what's a fire fit for tonight" | Premium reply — no influencer phrases, ≤ 1 slang token. |
| D5 | Generic small talk "hi" | Lightweight reply with no "Sure!"/"Absolutely!", passes through tone polish. |

---

## E. Board storyteller (Phase 4 — premium board UX)

Trigger a styling request that returns 2–3 boards. Inspect each card.

| # | Check | Expected |
|---|---|---|
| E1 | Card header | ≤ 2 lines visible above the collage (headline + 1-line summary). |
| E2 | Bottom-left collage label | Reads role like "Safest polished option" — **not** "Refined./Relaxed.". |
| E3 | "Why this works ›" row | Collapsed by default. Tap expands. |
| E4 | Expanded panel | Shows Why this works / Personalized for you / Occasion fit / Styling tip — only rows where backend supplied copy. |
| E5 | Office-board summary | Mentions composure / sharp / ready-for-the-room register. |
| E6 | Date-board summary | Softer register — "soft" / "intentional" / "easy". |
| E7 | Workout-board summary | Practical — "move" / "performance" / "built". |
| E8 | Backend payload (DevTools / charles proxy) | Each card JSON has `story` object plus legacy `why_it_works` / `explanation` / `styling_tip` mirrors. |
| E9 | Saved board screen (`occasion.dart` list) | Preview text uses `story.summary` when present, not fallback "Custom … inspiration". |
| E10 | Old API response with no `story` field | UI still renders — falls back to `whyItWorks` cleanly. |

---

## F. Response truncation guard

| # | Test | Expected |
|---|---|---|
| F1 | Long outfit explanation request | Reply ends on complete sentence, never on a hanging word ("and"/"because"/"with"). |
| F2 | Logs during F1 | If LLM came back truncated, see `ahvi.llm.response_truncated` then either retry log or trimmed sentence. |

---

## G. Regression / fallback safety

| # | Scenario | Expected |
|---|---|---|
| G1 | `ENABLE_AGENT_STYLE_ORCHESTRATOR=0` | Style flow still works. No agent_orchestration in payload. No tone regression. |
| G2 | `ENABLE_AGENT_METADATA_VALIDATOR=0` | Wardrobe save still works. `wardrobe_style_metadata` still written via legacy enrichment. |
| G3 | Force agent call to fail (block egress to Gemini) | Backend returns boards using legacy flow. Log: `ahvi.agent.*_failed`. No 5xx. |
| G4 | Vulnerable emotion path (`emotion_state=vulnerable`) | Slang suppressed, tone soft, closer: "We can simplify it from here." |

---

## How to capture output

For each row, attach one of:

* HTTP response (curl/Postman copy)
* Screenshot from the APK
* Cloud Run log lines (5–10 around the event)
* Appwrite document JSON (for C-series metadata checks)

Then ship the filled table back.
