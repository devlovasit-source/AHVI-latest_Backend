# GPU Dependency Audit — ahvi-gpu (`35.200.172.152`)

Read-only. Traces every caller of RMBG / `bg_service.py` / `remove-bg` / Ollama and what breaks if
the GPU VM is terminated.

## ahvi-gpu hosts TWO services (prod, serving revision `00445-4c4`)
| port | service | env var |
|---|---|---|
| `:8010/remove-bg` | RMBG-2.0 background removal | `RMBG_SERVICE_URL=http://35.200.172.152:8010/remove-bg` |
| `:11434/api` | Ollama text + vision | `OLLAMA_URL` / `OLLAMA_VISION_URL` / `OLLAMA_BASE_URL` / `OLLAMA_HOST` = `http://35.200.172.152:11434/api` |

Prod flags: `AI_PROVIDER=vertex` (gemini-2.5-flash), `ENABLE_BG_REMOVER=true`, `ENABLE_VISION=true`,
`ENABLE_GEMINI_MULTI_GARMENT_PREVIEW=true`, `HF_TOKEN=<set>`.

---

## 1. RMBG background removal (`bg_service.remove_bg_bytes`)

**Fail behavior (key):** `bg_service.py:120-157` — when `RMBG_SERVICE_URL` is set, any RMBG error/non-200
→ `return image_bytes` (the ORIGINAL, un-cut image). The HuggingFace fallback (`HF_TOKEN`, lines 159+)
is **only reached when `RMBG_SERVICE_URL` is empty** — so in prod HF is configured but **bypassed**.
`remove_bg_bytes` never raises → all callers fail-open.

| caller | prod active? | Gemini fallback? | on ahvi-gpu death |
|---|---|---|---|
| `routers/wardrobe_capture.py:791` (`_full_image_fallback_item`) | yes | none | returns original crop; item saved WITH background |
| `routers/wardrobe_capture.py:2014` (save flow) | yes | none | same — `maskedUrl`=raw, no cutout |
| save-selected catalog hook (`_maybe_generate_catalog_image`) | flag-off (dormant) | none | masked=raw → catalog gets non-cutout → validation rejects → `catalog_failed` (save still ok) |
| `services/hybrid_detection_service.py:225` (per-crop) | yes | none | crops keep background; detection degraded |
| `brain/engines/style_board_renderer.py:11` (`remove_bg_external_sync`) | yes | none | board composites with backgrounds (uglier), no crash |
| `routers/bg_router.py:34` `POST /remove-bg` | yes (`ENABLE_BG_REMOVER=true`) | none | endpoint returns original bytes (no-op removal) |
| `routers/bg_remover.py:26` `remove_bg_external_sync` | mounted | none | same no-op |
| `main.py:837` `POST /api/background/remove-bg`, `/api/remove-bg` | yes | none | returns original bytes |

**Verdict: DEGRADED, not blocking.** No Gemini equivalent for background removal. Everything fails open
to the un-cut image. No 5xx, no save failures.

## 2. Ollama TEXT (`llm_service` / `ai_gateway.generate_text` + `chat_completion`)

Routing: `llm_service.py:357` — if `_gemini_enabled()` (AI_PROVIDER ∈ vertex/gemini/google) → Vertex
first; Ollama (`OLLAMA_URL`) is only hit when Gemini returns empty (`:385+`).

| caller | prod active? | Gemini fallback? | on ahvi-gpu death |
|---|---|---|---|
| `ai_gateway.generate_text/chat_completion` → `routers/chat.py`, `services/style_flow_service.py`, `brain/response/response_assembler.py`, `routers/ops.py` | yes | **Vertex is PRIMARY** | ~no impact — text served by Vertex; Ollama was only the empty-result fallback |

**Verdict: SAFE.** Text path is Vertex-first; GPU Ollama is a rarely-hit fallback. (If Vertex ALSO
fails, the Ollama fallback is already unreachable and the circuit breaker returns a friendly string.)

## 3. Ollama VISION (`ai_gateway.ollama_vision_json`)

| caller | prod active? | Gemini fallback? | on ahvi-gpu death |
|---|---|---|---|
| `routers/wardrobe_capture.py:729` (`_vision_extract_attributes`, gated `ENABLE_VISION=true`) | yes | **yes** — `except → heuristic`, and `ENABLE_GEMINI_MULTI_GARMENT_PREVIEW=true` makes `gemini_multi_garment_detector` (Vertex) the primary multi-garment path | vision enrichment fails → falls back to Gemini-multi + heuristic labels (degraded classification) |
| `routers/vision.py:213` `POST /analyze-image` (gated `ENABLE_VISION=true`) | mounted | **none** (Ollama-only) | endpoint 5xx / errors — no fallback |

**Verdict: DEGRADED** for wardrobe (Gemini-multi + heuristic cover it). `routers/vision.py /analyze-image`
is the one **hard-dependent** path (no fallback) — but it appears legacy (wardrobe capture uses
`gemini_multi`, not this endpoint).

---

## Summary

### Critical blockers (break, no fallback)
- **`routers/vision.py POST /analyze-image`** — Ollama-vision only, no Gemini fallback → errors if GPU
  down. Mitigated only by it being legacy/unused by the main wardrobe flow. **Confirm no client calls it**;
  if unused, downgrade to "safe-to-remove".
- Otherwise **NONE** — no save/checkout/chat path hard-fails.

### Degraded paths (work, worse quality)
- Wardrobe capture + save-selected: items saved WITH background (`maskedUrl`=raw). No crash.
- Catalog normalization (when enabled): non-cutout input → mostly `catalog_failed`, save proceeds.
- Style board renderer: composites over original backgrounds.
- `/remove-bg`, `/api/background/remove-bg`, `/api/remove-bg`: return original (no-op).
- Vision classification: Gemini-multi + heuristic instead of Ollama vision.

### Unused / legacy
- **Ollama TEXT fallback** — never primary in prod (Vertex), effectively dead weight.
- **HuggingFace bg fallback** — `HF_TOKEN` set but code bypasses it whenever `RMBG_SERVICE_URL` is set;
  a latent, currently-unreachable fallback.
- **`routers/vision.py` Ollama `/analyze-image`** — superseded by `gemini_multi`.

### Safe-to-remove / safe-to-cutover (no code change here — recommendations)
- To make background removal GPU-independent: **unset `RMBG_SERVICE_URL`** → `bg_service` automatically
  uses the HuggingFace RMBG path (`HF_TOKEN` already set). One env change, no code edit. (Or add a
  Gemini/Vertex segmentation fallback — larger work.)
- Ollama text/vision URLs can be dropped once it's confirmed Vertex + gemini_multi cover all calls;
  keep the circuit breaker.
- `routers/vision.py` + `routers/bg_remover.py` are candidates for removal if no client hits them.

### Bottom line
Terminating ahvi-gpu **does not hard-break** the app (everything fails open to original images / Vertex),
**except** the legacy `routers/vision.py /analyze-image`. Real cost is **quality**: no background removal
(items + boards + catalog show backgrounds) until `RMBG_SERVICE_URL` is unset to fail over to HuggingFace,
or a managed RMBG replacement is wired.

_Read-only audit. No code changed._
