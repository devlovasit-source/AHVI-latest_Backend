# AHVI Backend — Closed Beta MVP

AHVI is an AI-powered personal assistant platform focused on:
- Style
- Planning
- Preparation

This repository contains the backend systems powering the AHVI closed-beta investor-ready MVP.

---

# Current MVP Scope

The current implementation focuses on stabilizing and refining the core AHVI experience across:

- AI styling orchestration
- Wardrobe intelligence
- Editorial outfit board generation
- Workout + style flows
- Plan & pack flows
- Multi-engine orchestration
- Personalized contextual recommendations

The broader AHVI ecosystem and advanced personalization layers will continue evolving iteratively beyond the current MVP milestone.

---

# Core Architecture

Main backend modules:

## Brain Layer
Located in:
- `brain/`

Contains:
- orchestration systems
- outfit pipeline
- style scoring
- planning engines
- contextual reasoning
- personalization systems
- tone systems
- workflow orchestration

---

## Routers
Located in:
- `routers/`

Handles:
- chat APIs
- wardrobe capture
- board generation
- background removal
- notifications
- styling flows
- utility endpoints

---

## Services
Located in:
- `services/`

Contains integrations for:
- LLM providers
- Appwrite
- embeddings
- vector systems
- storage
- weather
- AI gateways
- upload handling
- orchestration helpers

---

# Infrastructure

Current deployment stack:

- FastAPI
- Cloud Run
- Appwrite
- Qdrant
- Ollama
- GPU VM services
- RMBG background removal
- Cloud storage integrations

---

# Current MVP Focus

The current stabilization phase is focused on:

- outfit quality refinement
- editorial board consistency
- orchestration stability
- wardrobe flow reliability
- AI response quality
- APK/demo stability
- investor-ready walkthrough experience

---

# Repository Notes

This branch is a cleaned review branch prepared for MVP technical review.

Local cache artifacts, temporary files, and IDE-specific files have been removed for clarity.

---

# Status

Current phase:
- Closed Beta
- Investor-ready MVP stabilization
- Core architecture implemented
- Refinement and orchestration consistency ongoing


## Style-board shuffle state (durable)

Style-board shuffle revisions are stored durably in the Appwrite
`style_board_states` collection — one immutable document per
`(board_id, revision)` with a deterministic document ID
(`sha1("board_id|revision")[:36]`). Creating revision N+1 is the atomic
claim: with multiple Cloud Run instances, exactly one concurrent shuffle
from revision N succeeds; the loser receives `BOARD_REVISION_CONFLICT`.

Deployment requirements:

1. Run the idempotent migration once per environment:
   `python scripts/create_style_board_states_collection.py`
   (uses the standard `APPWRITE_ENDPOINT` / `APPWRITE_PROJECT_ID` /
   `APPWRITE_DATABASE_ID` / `APPWRITE_API_KEY` variables).
2. Optional: set `APPWRITE_COLLECTION_STYLE_BOARD_STATES` if the collection
   id differs from the default `style_board_states`.

Behavior when state storage is unavailable: board generation still returns
the board with `shuffle_available=false` plus a typed
`BOARD_STATE_UNAVAILABLE` registration error; shuffle requests fail typed
with `BOARD_STATE_UNAVAILABLE` and never advance the revision. There is no
in-memory fallback in production; boards created before this migration
return `BOARD_STATE_NOT_FOUND` (regenerate the board).
