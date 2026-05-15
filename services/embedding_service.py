import asyncio
import logging
import os
from typing import List, Optional

logger = logging.getLogger("ahvi.embedding")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as _import_err:  # pragma: no cover - optional dep
    SentenceTransformer = None
    _IMPORT_ERR = _import_err
else:
    _IMPORT_ERR = None


# =========================
# 🔥 SINGLETON MODEL
# =========================
_model = None
_load_failed = False

_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)


def embeddings_enabled() -> bool:
    """Quick check for callers — true only if the model is importable AND
    has not previously failed to load. Use this instead of try/except around
    every encode call.
    """
    return SentenceTransformer is not None and not _load_failed


def _get_model():
    global _model, _load_failed

    if SentenceTransformer is None:
        raise RuntimeError(
            f"sentence-transformers is not installed: {_IMPORT_ERR}"
        )

    if _load_failed:
        raise RuntimeError("embedding model previously failed to load")

    if _model is None:
        logger.info("ahvi.embedding.loading model=%s", _MODEL_NAME)
        try:
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("ahvi.embedding.ready model=%s", _MODEL_NAME)
        except Exception as exc:
            _load_failed = True
            logger.exception("ahvi.embedding.load_failed model=%s err=%s", _MODEL_NAME, exc)
            raise

    return _model


def get_model():
    """Public accessor (used across app)."""
    return _get_model()


# =========================
# TEXT BUILDING
# =========================
def _build_text(data: dict) -> str:
    category = str(data.get("category") or "")
    sub_category = str(data.get("sub_category") or "")
    color = str(data.get("color_code") or "")
    pattern = str(data.get("pattern") or "")
    style = str(data.get("style") or "")

    occasions_raw = data.get("occasions", [])
    if isinstance(occasions_raw, list):
        occasions = " ".join(str(x) for x in occasions_raw if x)
    else:
        occasions = str(occasions_raw or "")

    return " ".join([category, sub_category, color, pattern, style, occasions]).strip()


# =========================
# 🔥 CORE ENCODERS
# =========================
def encode_text(text: str) -> List[float]:
    """Synchronous encode. NOTE: SentenceTransformer.encode is CPU-bound
    and can take 0.5-3s per call on first warm-up. Inside async handlers
    use `encode_text_async()` to avoid blocking the event loop.
    """
    if not text or not embeddings_enabled():
        return []
    try:
        model = _get_model()
        vector = model.encode(text)
        return vector.tolist()
    except Exception as exc:
        logger.warning("ahvi.embedding.encode_text_failed err=%s", exc)
        return []


async def encode_text_async(text: str) -> List[float]:
    """Async wrapper that runs the CPU-bound encode in a worker thread.

    This is the fix for the 62-second style flow latency seen in
    production: previously _semantic_retrieval() called encode() directly
    inside an async route, freezing the FastAPI event loop.
    """
    if not text or not embeddings_enabled():
        return []
    try:
        return await asyncio.to_thread(encode_text, text)
    except Exception as exc:
        logger.warning("ahvi.embedding.encode_text_async_failed err=%s", exc)
        return []


def encode_metadata(data: dict) -> List[float]:
    if not embeddings_enabled():
        return []
    try:
        text = _build_text(data)
        return encode_text(text)
    except Exception as exc:
        logger.warning("ahvi.embedding.encode_metadata_failed err=%s", exc)
        return []


async def encode_metadata_async(data: dict) -> List[float]:
    if not embeddings_enabled():
        return []
    try:
        text = _build_text(data)
        return await encode_text_async(text)
    except Exception as exc:
        logger.warning("ahvi.embedding.encode_metadata_async_failed err=%s", exc)
        return []


# =========================
# OPTIONAL SERVICE WRAPPER
# =========================
class EmbeddingService:
    def encode_text(self, text: str) -> List[float]:
        return encode_text(text)

    async def encode_text_async(self, text: str) -> List[float]:
        return await encode_text_async(text)

    def encode_metadata(self, data: dict) -> List[float]:
        return encode_metadata(data)

    async def encode_metadata_async(self, data: dict) -> List[float]:
        return await encode_metadata_async(data)

    @property
    def enabled(self) -> bool:
        return embeddings_enabled()


# 🔥 singleton instance
embedding_service = EmbeddingService()
