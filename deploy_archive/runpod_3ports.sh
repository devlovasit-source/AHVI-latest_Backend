#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./runpod_3ports.sh backend
#   ./runpod_3ports.sh ollama-chat
#   ./runpod_3ports.sh ollama-vision

mode="${1:-}"

case "$mode" in
  backend)
    exec uvicorn main:app --host 0.0.0.0 --port 8000
    ;;
  ollama-chat)
    export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama-chat}"
    mkdir -p "$OLLAMA_MODELS"
    exec ollama serve
    ;;
  ollama-vision)
    export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11435}"
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama-vision}"
    mkdir -p "$OLLAMA_MODELS"
    exec ollama serve
    ;;
  *)
    echo "Unknown mode: $mode"
    echo "Expected one of: backend | ollama-chat | ollama-vision"
    exit 1
    ;;
esac
