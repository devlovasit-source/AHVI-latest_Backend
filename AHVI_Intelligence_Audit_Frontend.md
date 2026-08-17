# AHVI Intelligence Spine Audit — Frontend Report
**Prepared by:** Jules, Senior Software Engineer
**Scope:** `C:\\tmp\\AHVI-frontend-integration` (local repo reference)
**Status:** AUDIT ONLY (Read-Only) — No changes introduced.

---

## Phase 1 — Request Lifecycle (Frontend Stages)

### 1. User Message Entry & State Bindings
- **Component:** `ChatScreen` / `ChatViewModel`
- **Input Fields:** User typing or Action Chip tap.
- **Output Fields:** Appends to the local `messages` stream and transmits a `TextChatRequest` payload to `/api/text`.

### 2. Response Parsing & Extraction
- **Component:** `ChatRepository` / `BoardExtractor`
- **Logic:** Extracts the backend response envelope.
- **Flaw:** The frontend parser looks for visual board cards across multiple different payload keys (`response`, `message_text`, `cards`, `style_boards`, `data.outfits`, `data.rendered_boards`). If keys are missing or slightly renamed (e.g. `rendered_boards` instead of `style_boards`), the frontend fails to render the card deck and defaults to plain text, prompting the user with *"I'm having trouble thinking right now."*

### 3. Rendering Layers
- **Component:** `AhviVisualBoard` / `AhviOutfitCard`
- **Logic:** Renders a carousel of styling boards, checking for the presence of `image_url` or `board_items` to present image cutouts.
- **Flaw:** Lacks slot-completion fallbacks. If the backend fails to populate images for optional accessories, the frontend silently drops the slot or fails to draw the card, resulting in half-empty layouts.

---

## Phase 6 — Calendar & Timezone Grouping Defect

Our analysis of the calendar and timezone trace reveals a severe mismatch in how events are processed and grouped.

### 1. Root Cause of Grouping Defects
The frontend groupings are fundamentally flawed due to timezone unawareness and key mismatch:
- **Date Mismatch:** The backend parses "tomorrow" at `5pm` and correctly stores it as a UTC timestamp.
- **Naive Grouping:** The frontend groups events by their `createdAt` date (which is today) instead of the actual `startAtISO` event date (which is tomorrow). This is why a "tomorrow" event is displayed under the *"Today's Plans"* header in the UI.
- **No Format Normalization:** The duplicate cards showing `5pm` and `5:00 PM` are caused by the frontend performing naive, local-string matching to de-duplicate events. Since the string formatting differs, the frontend optimistic insert layer fails to identify the duplicate and displays two separate cards.

### 2. Defect Ownership
This defect is a **combined backend & frontend defect** (Both):
- **Backend Responsibility:** The calendar persistence layer does not perform title-normalization or date-deduplication before creating a document.
- **Frontend Responsibility:** The grouping logic is timezone-naive and groups by the wrong date field (`createdAt` instead of `startTime`).
