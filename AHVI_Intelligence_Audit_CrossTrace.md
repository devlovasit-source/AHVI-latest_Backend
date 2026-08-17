# AHVI Intelligence Spine Audit — Cross-Repo Trace Report
**Prepared by:** Jules, Senior Software Engineer
**Scope:** Backend & Frontend Repositories Cross-Trace
**Status:** AUDIT ONLY (Read-Only) — No changes introduced.

---

## Phase 3 — Style End-to-End Traces

Below are the end-to-end execution traces for the three failing styling scenarios.

### Scenario S1: "Create a dinner outfit with a dress, shoes, and a bag"
- **Detected Intent:** `occasion_outfit` / `style_advice` (classified in `detect_intent`)
- **Detected Occasion:** `date night` (mapped via `_ahvi_style_occasion` from the word "dinner")
- **Extracted Requested Roles:** `dress`, `footwear`, `accessory` (mapped in `beta_style_bridge.py` under `_mentioned_roles` from the words "dress, shoes, bag")
- **Required Explicit Roles:** `dress` + `footwear` + `accessory`
- **Observed Bug:** The backend returns trousers, sneakers, and a cap.
- **Underlying Code Evidence:**
  - In `services/beta_style_bridge.py`, `_mentioned_roles` identifies "dress", "footwear", and "accessory".
  - However, in `services/style_reasoning_engine.py` (`_best_style_assets`), candidate generation filters wardrobe items using `_hero_asset_allowed`.
  - When the user's wardrobe contains only trousers and sneakers, the validation layer (`validate_style_response`) determines that the board is incomplete (since "dress" is missing), but the **fallback path silently bypasses explicit-role enforcement**. It falls back to `_demo_style_board_payload` which uses the first available items in the wardrobe (trousers and sneakers), completely disregarding the user's explicit request for a "dress" or "bag".

### Scenario S2: "Create a layered outfit with outerwear, a top, trousers, shoes, and a bag"
- **Detected Intent:** `occasion_outfit`
- **Detected Occasion:** `today` (default fallback since "layered" is not mapped)
- **Extracted Requested Roles:** `outerwear`, `top`, `bottom`, `footwear`, `accessory`
- **Observed Bug:** The returned board contains only a top, bottom, and shoes.
- **Underlying Code Evidence:**
  - The completeness validator (`validate_style_response` in `beta_style_bridge.py`) only requires `("dress" in role_set or {"top", "bottom"} <= role_set) and ("footwear" in role_set)`.
  - Outerwear and accessories (such as bags) are treated as **purely optional roles**.
  - If the database lacks valid cutout images for outerwear or bags, or if they fail the quality scorer, the engine silently discards them instead of attempting a repair or informing the user of a missing piece.

### Scenario S3: "Create a dinner styling / dinner outfit"
- **Detected Occasion:** `date night` (from the word "dinner")
- **Observed Bug:** Camouflage trousers or athletic footwear are returned for a dinner setting.
- **Underlying Code Evidence:**
  - In `brain/engines/outfit_quality_guard.py` (`reject_board_for_occasion`), the private-wear and occasion-safety guard is designed to reject inappropriate items.
  - However, because the **wardrobe fallback path (`_demo_style_board_payload`) does not run through the same strict validation and repair loop** as the stylist reasoning engine, unvalidated output reaches the final response. If a user owns camouflage trousers or athletic shoes, they are allowed to leak into a refined dinner look.

---

## Phase 4 — Alternative Board & Memory Trace

### 1. Board Loop Trace (A -> B -> C -> A)
- **Previous Board Tracking:** The previous board item IDs are carried in `TextChatRequest.exclude_style_signatures` (passed via the request body).
- **Exclusion Cumulative Check:** No, the exclusions are **not cumulative**. The backend only checks the immediate previous board signature or the directly excluded item IDs. It does not maintain a full conversation-session board history.
- **Code Evidence:** In `routers/chat.py` (`text_chat`), `exclude_style_signatures` is passed directly into `ahvi_orchestrator.run`, but is treated as a flat list of exclusions for the immediate next pass.
- **Is A -> B -> A repetition possible?** **Yes.** Since exclusions are not stashed in a durable session store and are only carried inside the request body of the immediately preceding turn, any reset of the frontend state or failure to echo back `exclude_style_signatures` allows Board A to return after Board B.

---

## Phase 7 — Prep/Prepare Routing Trace

### 1. Payload Mapping
- **Input:** `domain=prepare`, `module=prepare`, `message="I haven't met my calorie goal today. Suggest healthy meals I can prepare now."`
- **Observed Bug:** Downgraded to a generic chat refusal, with the response module changed to `chat`.

### 2. Underlying Code Evidence
- **Route Selection:** In `routers/chat.py` (`module_chat`), the router normalizes the incoming module name using `_normalize_module_name`.
- **The Defect:**
  ```python
  def _normalize_module_name(module: str) -> str:
      value = str(module or "").strip().lower().replace("-", "_")
      allowed = {
          "style",
          "wardrobe",
          "daily_wear",
          "skincare",
          "medi",
          "bills",
          "calendar",
          "meal",
          "diet",
          "fitness",
          "planner",
      }
      if value in {"plan", "planning", "reminder", "reminders"}:
          return "planner"
      return value if value in allowed else "chat"
  ```
- **The Core Flaw:** The allowed modules set **does not contain "prep" or "prepare"**.
- Since `prepare` is not in the `allowed` list, `_normalize_module_name("prepare")` falls back and returns **`chat`**.
- This is exactly why the prepare/prep module is silently downgraded to `chat`.
- Once downgraded to `chat`, it runs through `_llm_chat_response` instead of reaching the meal planner or fitness engines. The chat system prompt instructs the assistant not to handle dietary/calorie advice, triggering a hard-coded generic chat refusal.
