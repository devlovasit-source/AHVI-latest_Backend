import base64
import io
import os
import asyncio
from typing import Any, List

import httpx
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = None
    CLIPProcessor = None

# 🔥 TEXT MODEL (now merged here)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# =========================
# CONFIG
# =========================
_device = torch.device("cuda" if (torch and torch.cuda.is_available()) else "cpu") if torch else "cpu"

_IMAGE_MODEL = None
_IMAGE_PROCESSOR = None
_TEXT_MODEL = None

_IMAGE_MODEL_NAME = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")
_TEXT_MODEL_NAME = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

_URL_CACHE: dict[str, list] = {}
_URL_CACHE_MAX = 512


# =========================
# MODEL LOADERS
# =========================
def _load_image_model():
    global _IMAGE_MODEL, _IMAGE_PROCESSOR

    if _IMAGE_MODEL is not None:
        return _IMAGE_MODEL, _IMAGE_PROCESSOR

    _IMAGE_PROCESSOR = CLIPProcessor.from_pretrained(_IMAGE_MODEL_NAME)
    _IMAGE_MODEL = CLIPModel.from_pretrained(_IMAGE_MODEL_NAME)

    _IMAGE_MODEL.to(_device)
    _IMAGE_MODEL.eval()

    return _IMAGE_MODEL, _IMAGE_PROCESSOR


def _load_text_model():
    global _TEXT_MODEL

    if _TEXT_MODEL is not None:
        return _TEXT_MODEL

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")

    _TEXT_MODEL = SentenceTransformer(_TEXT_MODEL_NAME)
    return _TEXT_MODEL


# =========================
# NORMALIZATION
# =========================
def _normalize(vec: List[float]) -> List[float]:
    if not vec:
        return vec

    if torch:
        t = torch.tensor(vec)
        return (t / t.norm()).tolist()

    return vec


# =========================
# IMAGE EMBEDDING
# =========================
def _encode_image_bytes(image_bytes: bytes) -> List[float]:
    if not image_bytes:
        return []

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        model, processor = _load_image_model()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)

        return features[0].cpu().tolist()

    except Exception as e:
        print("[Image Embedding Error]", str(e))
        return []


async def encode_image_bytes(image_bytes: bytes) -> List[float]:
    return await asyncio.to_thread(_encode_image_bytes, image_bytes)


# =========================
# TEXT EMBEDDING (INLINE)
# =========================
def _build_text(metadata: dict) -> str:
    return f"""
    A {metadata.get("color_code", "")} {metadata.get("pattern", "")} {metadata.get("sub_category", "")}.
    Category: {metadata.get("category", "")}.
    Suitable for: {", ".join(metadata.get("occasions", []))}.
    """.strip()


def _encode_text(metadata: dict) -> List[float]:
    try:
        model = _load_text_model()
        text = _build_text(metadata)

        if not text:
            return []

        emb = model.encode(text)
        return _normalize(emb)

    except Exception as e:
        print("[Text Embedding Error]", str(e))
        return []


async def encode_text(metadata: dict) -> List[float]:
    return await asyncio.to_thread(_encode_text, metadata)


# =========================
# 🔥 HYBRID EMBEDDING
# =========================
def _combine(image_vec, text_vec, alpha=0.7):
    if not image_vec:
        return text_vec
    if not text_vec:
        return image_vec

    # ensure same length
    min_len = min(len(image_vec), len(text_vec))

    return [
        alpha * image_vec[i] + (1 - alpha) * text_vec[i]
        for i in range(min_len)
    ]


async def encode_hybrid(
    *,
    image_bytes: bytes,
    metadata: dict,
    alpha: float = 0.7
) -> List[float]:
    """
    🔥 MAIN FUNCTION (USE THIS EVERYWHERE)
    """

    image_vec, text_vec = await asyncio.gather(
        encode_image_bytes(image_bytes),
        encode_text(metadata)
    )

    return _combine(image_vec, text_vec, alpha)


# =========================
# URL SUPPORT (ASYNC)
# =========================
async def encode_image_url(url: str, timeout: float = 6.0) -> List[float]:
    if not url:
        return []

    if url in _URL_CACHE:
        return _URL_CACHE[url]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            vec = await encode_image_bytes(resp.content)

            if vec:
                _URL_CACHE[url] = vec

            if len(_URL_CACHE) > _URL_CACHE_MAX:
                _URL_CACHE.pop(next(iter(_URL_CACHE)), None)

            return vec

    except Exception:
        return []
