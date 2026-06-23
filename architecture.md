# AHVI Backend Architecture Map
### *Closed Beta Investor-Ready MVP Core Systems*

AHVI is an AI-powered personal assistant platform specializing in **Style, Planning, and Preparation**. This document outlines the complete architectural mapping of the backend repository, detailing the system topology, layered software architecture, data models, AI reasoning subsystems, and infrastructure.

---

## 1. System Topology & Data Flow

Below is an ASCII map of how external clients, the API gateway, caching servers, database systems, vector search indexes, background workers, and LLM orchestrators interact.

```
                      +------------------+
                      |   Mobile Client  |
                      +--------+---------+
                               |
                               | (FastAPI REST APIs / JSON / Audio / Image upload)
                               v
+--------------------------------------------------------------------------+
|                            FASTAPI API GATEWAY                           |
|                                                                          |
|  [Auth Middleware (Appwrite JWT Validation)]  [Rate Limiter (Redis)]     |
|  [Dynamic Router Loader]                     [Thread-Safe TTLLRU Cache]  |
+---------+-------------------+---------------------+------------------+---+
          |                   |                     |                  |
          | (Sync/Async       | (Asynchronous       | (Vector          | (DB / Metadata
          |  orchestration)   |  Celery Task)       |  Search / Cosine)|  Retrieval)
          v                   v                     v                  v
  +---------------+   +---------------+     +---------------+  +---------------+
  |  Brain Layer  |   | Celery Worker |     | Qdrant Vector |  | Appwrite DB & |
  | (Orchestrator)|   | (Redis Queue) |     |   Database    |  | Auth Service  |
  +-------+-------+   +-------+-------+     +---------------+  +---------------+
          |                   |
          +---------+---------+
                    | (Model Orchestration via API / SDK / HTTP Client)
                    v
  +-----------------------------------+
  |          AI GATEWAY / LLM         |
  |                                   |
  |  - Circuit Breakers per Usecase   |
  |  - Token Budget Guardrails        |
  |  - Safe JSON extractors           |
  +-------+-------------------+-------+
          |                   |
          v                   v
+-------------------+ +-------------------+
|  Google Vertex /  | |   Ollama Engine   |
|   Gemini Flash    | | (Self-Hosted / GPU|
| (Premium Model)   | |  Ollama Fallback) |
+-------------------+ +-------------------+
```

---

## 2. Dynamic Router Loading & API Layer (`routers/`, `api/`)

FastAPI handles incoming HTTP requests. To remain lightweight and run on limited resource environments (e.g., 1-vCPU Cloud Run), the app implements an **Optional/Feature-Flagged Router Loader** in `main.py` that checks for required Python package dependencies and environment variables prior to exposing routes.

### Major API Router Modules
1. **`routers/chat.py`**: The central communication hub. It receives user prompts, triggers the translation layers (`deep-translator`), invokes NLU to categorize intents, and coordinates with the main `ahvi_orchestrator` or returns cached responses using a custom `_TTLLRUCache`.
2. **`routers/boards.py` & `services/board_service.py`**: Composes digital styling mood boards, custom "editorial boards", and organizes saved layouts.
3. **`routers/wardrobe_capture.py` & `services/upload_service.py`**: Manages uploading clothing items from physical camera rolls to MinIO S3 storage, and runs automatic categorization and tagging.
4. **`routers/garment_analyzer.py`**: Extracts fine-grained visual characteristics (silhouettes, sleeve types, necklines, fabric materials, primary/secondary colors).
5. **`routers/bg_router.py` & `routers/bg_remover.py`**: Triggers back-end background removal models utilizing an external GPU-hosted RMBG container service, with HuggingFace Inference API as a fallback.
6. **`routers/workouts.py` & `routers/skincare_adherence.py`**: Tracks fitness habits, coordinates outfit pairing for physical activities, and validates skincare regimen adherence.
7. **`routers/calendar.py` & `routers/bills.py`**: Synchronizes personal calendars and extracts schedules, deadlines, billing notifications, and financial tasks.
8. **`routers/lens_similar.py`**: Provides visual search functionality to match a user's uploaded wardrobe image with similar items in the styling catalog.
9. **`routers/ahvi_contacts.py`**: Manages personal relations, contacts, and calendar-linked event attendees.
10. **`api/ai.py`**: Direct-access route `/ai/run` implementing fine-grained API budget-tiers (`low`, `medium`, `high`) matching execution contexts with optimal cheap vs premium model endpoints.

---

## 3. The Brain Layer (`brain/`)

The core intelligence layer operates as a **hybrid decision pipeline**, combining agentic LLM reasoning with deterministic expert rules.

```
                         [User Text Input]
                                 |
                                 v
                     +-----------------------+
                     |  brain.intent_engine  | (Detects intent & extracts slots:
                     +-----------+-----------+  occasion, style, modules, time)
                                 |
                                 v
                     +-----------------------+
                     |  brain.agent_system   | (Generates execution plan)
                     +-----------+-----------+
                                 |
        +------------------------+------------------------+
        | (Dynamic LLM Plan)                              | (Deterministic Rule-based Fallback)
        v                                                 v
+-----------------------------+                 +-----------------------------+
|    Parallel Agent Tasks     |                 |  Expert Styling Rules &     |
| - context_agent             |                 |  Occasion Interpreters      |
| - style_graph_agent         |                 | - brain.engines.style_rules |
| - outfit_agent              |                 | - brain.engines.occasion_   |
| - memory_agent              |                 +-------------+---------------+
+---------------+-------------+                               |
                |                                             |
                +---------------------+-----------------------+
                                      |
                                      v
                        +---------------------------+
                        | brain.decision_engine     | (Performs weighted priority
                        | (Weighted Priority Rank)  |  ranking & persona-nudge)
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | brain.response_assembler  | (Polishes & builds final
                        +---------------------------+  multi-card JSON response)
```

### Components of the Brain Layer
* **Intent Classifier (`brain/intent_engine.py`)**: Classes categorize user prompts into core modules (`daily_dependency`, `daily_outfit`, `occasion_outfit`, `plan_pack`, etc.) and parses metadata slots.
* **Orchestrator (`brain/orchestrator.py`)**: Directs the flow of execution by collecting data, merging accessories, calling appropriate domain-specific engines, and constructing final JSON payloads.
* **Agent System (`brain/agent_system.py`)**: Leverages generative planners to divide complex queries into sub-tasks (e.g., `normalize_context` -> `build_style_graph` -> `generate_score_rank` -> `persist_and_feedback_hooks`).
* **Decision Engine (`brain/decision_engine.py`)**: Evaluates generated candidate actions and cards using deterministic weights, persona parameters (e.g., `busy_parent`, `student`), and slot context to output top suggestions.
* **Outfit Pipeline (`brain/outfit_pipeline.py`)**: Generates clothing combinations out of available wardrobe slots by calling:
  * `brain/engines/wardrobe_selector.py` (filters valid garments based on seasons, weather overlays, and occasion limits).
  * `brain/engines/style_scorer.py` & `brain/engines/outfit_quality_guard.py` (scores outfit coherence against color harmony, silhouette rules, and style blueprints).
* **Tone Engine (`brain/tone/tone_engine.py`)**: Fine-tunes textual copy before sending to clients, stabilizing vocabulary using systemic constraints mapped to user archetypes.
* **Specialized Engines (`brain/engines/`)**:
  * **Plan & Pack (`brain/engines/packing/packing_engine.py`)**: Computes packing list requirements based on trip duration, local weather, and destination activities.
  * **Fitness & Workout Pairer (`brain/engines/fitness/`)**: Evaluates workout calendars, scores training types, and pairs appropriate activewear.
  * **Meal Planner (`brain/engines/meals/`)**: Designs weekly recipes and shopping lists while rewriting ingredients dynamically for specified calorie goals.

---

## 4. Service Integration Layer (`services/`)

The infrastructure-facing tier manages downstream connections, caching, vector computations, and third-party integrations.

* **LLM Gateway (`services/ai_gateway.py` & `services/llm_service.py`)**:
  * Dual provider support for **Google Vertex GenAI (Gemini-2.0-Flash)** and local fallbacks via **Ollama**.
  * Incorporates a **Circuit Breaker** state machine on an isolated, per-usecase basis (`intent`, `styling`, `vision`, `general`) to avoid cascading failures if a model endpoint experiences latency spikes.
  * Enforces token budgets and truncation-checking with auto-retries.
* **Agent Style Orchestrator (`services/agent_style_orchestrator.py`)**:
  * Acts as a dedicated reasoning interface that executes asynchronous Gemini Agent calls, compiling detailed style parameters (avoid list, formality levels, accessory rules, required palettes) which are fed downstream to the deterministic engines.
* **Vector Search Database Interface (`services/qdrant_service.py`)**:
  * Houses client initializations for Qdrant databases.
  * Initializes four persistent index collections: `wardrobe` (text metadata), `wardrobe_images` (visual embeddings), `outfit_memory` (historical interactions), and `user_memory` (contextual preferences).
* **Text & Hybrid Embedding (`services/embedding_service.py`)**:
  * Runs text encoders using `sentence-transformers` (`all-MiniLM-L6-v2`) to translate descriptive garment metadata into high-dimensional vectors. Includes automatic async delegation to avoid blocking the single-threaded Python event loop during heavy CPU encodes.
* **Vision & Detection Models (`services/dino_service.py`)**:
  * Backed by custom visual architectures, providing item segmentation and garment classification.
* **Appwrite DB Proxy (`services/appwrite_proxy.py` & `services/appwrite_service.py`)**:
  * Manages transactional write/read patterns, JWT validation, and admin client connections to Appwrite Cloud.

---

## 5. Background Asynchronous Workers (`worker.py`)

Heavy compute procedures (such as media compression, voice transcription, visual analysis, and long-form document processing) are offloaded to background workers using a **Celery-on-Redis** cluster architecture.

* **Broker/Backend**: Redis (`redis://localhost:6379/0`) manages queue storage and tracks result metadata.
* **Task Retrying (`_retry_or_fail`)**: Built-in exponential backoff routines retry failed jobs up to a limit before flagging errors in Appwrite's client database via the `job_tracker`.
* **Sentry Integration**: Initialized dynamically on workers to track system faults and performance metrics separately from primary HTTP threads.

---

## 6. Testing, Quality Assurance, & Scripts (`tests/`, `scripts/`)

AHVI maintains a high-integrity verification suite covering dynamic components:
* **`tests/test_agent_style_orchestrator.py`**: Ensures model-generated style payloads are coerced to solid formats, resolving missing confidence markers and handling malformed data gracefully.
* **`tests/test_adherence_engine.py`**: Validates rules concerning skincare tracking and habit adherence scoring.
* **`tests/test_ahvi_tone_stabilization.py`**: Asserts constraints over text-generators, verifying system prompt formatting and tone consistencies.
* **`scripts/` & `tools/`**: Ingestion scripts that parse fashion data from DOCX outlines and backfill metadata, indexes, and Appwrite databases.

---

## Summary of Architectural Strengths

1. **Failure Containment**: Built-in circuit breakers and structural validation loops ensure that even if downstream LLM/Agent APIs fail or output corrupted structures, the backend fails over to deterministic expert rules without disrupting client sessions.
2. **Dynamic Adaptation**: Optional/conditional loading of routers, models, and features allows the identical codebase to scale down seamlessly to 1-vCPU Cloud Run instances or scale up to multi-GPU containers in production.
3. **Hybrid Precision**: Merges advanced generative planning (LLM) with hard mathematical scoring boundaries (deterministic color harmony, seasonal weather checks, and priority metrics) to maintain professional-grade recommendations.
