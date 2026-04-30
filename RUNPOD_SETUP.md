# RunPod Setup (rembg 2.x + Ollama Text/Vision)

## 1) Install Python dependencies
Use the RunPod profile requirements so rembg and model toolchain are present.

```bash
pip install -r requirements.runpod.txt
```

## 2) Install and start Ollama
On the pod:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Keep `ollama serve` running (separate tmux/screen/session).

For your 3-port setup, run two Ollama daemons:
- `11434` for chat model
- `11435` for vision model

## 3) Pull required models
Text + vision models:

```bash
ollama pull phi3:latest
ollama pull tinyllama
ollama pull llama3.2-vision:latest
```

Optional fallback if you want a second vision model:

```bash
ollama pull llava:7b
```

## 4) Configure environment
Copy `.env.runpod.example` to `.env` and update endpoint values:

- `RUNPOD_BG_SINGLE_URL`
- `RUNPOD_BG_BATCH_URL`

If Ollama runs in the same pod, keep:

- `OLLAMA_URL=http://127.0.0.1:11434/api`
- `OLLAMA_VISION_URL=http://127.0.0.1:11435/api`

Recommended anti-hallucination runtime:

- `OLLAMA_NUM_CTX=4096` (use `8192` if GPU allows)
- `OLLAMA_TEMPERATURE=0.2`
- `OLLAMA_PIN_TEXT_MODEL=true`
- Keep `OLLAMA_MODEL_FALLBACKS` empty for chat stability

## 5) Run backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 6) Smoke checks
1. Text generation path (via app flow):
   - `POST /api/chat/text` with a basic message should return a non-`none` assistant response.
2. Vision path:
   - `POST /api/vision/analyze-image` with `image_base64` should return `meta.vision_model_used`.
3. Background removal path:
   - `POST /api/background/remove-bg` should return `bg_removed: true` when RunPod endpoint is reachable.
4. Runtime metrics:
   - `GET /api/ops/metrics` should return LLM queue/in-flight stats and GPU memory data.

## 7) Deployment split (recommended)
- Chat pod:
  - `ENABLE_VISION=false`
  - `ENABLE_BG_REMOVER=false`
  - `ENABLE_LOCAL_REMBG_FALLBACK=false`
- Vision/rembg pod:
  - `ENABLE_VISION=true`
  - `ENABLE_BG_REMOVER=true`
  - `ENABLE_LOCAL_REMBG_FALLBACK=true` only if RunPod BG endpoint is unavailable.

## 8) Exact 3-port process plan (single RunPod machine)
1. Start backend/rembg API on `8000`:
   - `uvicorn main:app --host 0.0.0.0 --port 8000`
2. Start Ollama chat on `11434`:
   - `OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=/workspace/ollama-chat ollama serve`
   - `OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=/workspace/ollama-chat ollama pull phi3:latest`
3. Start Ollama vision on `11435`:
   - `OLLAMA_HOST=0.0.0.0:11435 OLLAMA_MODELS=/workspace/ollama-vision ollama serve`
   - `OLLAMA_HOST=0.0.0.0:11435 OLLAMA_MODELS=/workspace/ollama-vision ollama pull llama3.2-vision:latest`

Use tmux/screen so all three services stay up.

## Notes
- `services/llm_service.py` is now Ollama-enabled for text routes.
- `services/ai_gateway.py` already handles Ollama vision with model fallbacks.
- `routers/bg_remover.py` now supports env-based RunPod URLs and a sync compatibility helper used by legacy routes.
