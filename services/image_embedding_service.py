import base64
import io
import os
from typing import Any, List

import requests
from PIL import Image

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    torch = None
    CLIPModel = None
    CLIPProcessor = None


# =========================
# 🔥 SINGLETON MODEL
# =========================
_model = None
_processor = None

_DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch is not None else "cpu"
)

_MODEL_NAME = os.getenv(
    "IMAGE_EMBEDDING_MODEL_NAME",
    "openai/clip-vit-base-patch32"
)


def _get_model():
    global _model, _processor

    if torch is None or CLIPModel is None:
        raise RuntimeError("transformers/torch not installed")

    if _model is None:
        print(f"[image-embedding] loading model: {_MODEL_NAME}")
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _model = CLIPModel.from_pretrained(_MODEL_NAME)

        _model.to(_DEVICE)
        _model.eval()

    return _model, _processor


# =========================
# 🔥 CORE ENCODER
# =========================
def encode_image_bytes(image_bytes: bytes) -> List[float]:
    try:
        if not image_bytes:
            return []

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        model, processor = _get_model()

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)

        return features[0].cpu().tolist()

    except Exception as e:
        print("[image-embedding] encode error:", e)
        return []


# =========================
# BASE64
# =========================
def encode_image_base64(value: Any) -> List[float]:
    try:
        text = str(value or "").strip()
        if not text:
            return []

        if "," in text:
            text = text.split(",", 1)[1]

        image_bytes = base64.b64decode(text, validate=True)
        return encode_image_bytes(image_bytes)

    except Exception:
        return []


# =========================
# URL
# =========================
def encode_image_url(url: Any, timeout: float = 8.0) -> List[float]:
    try:
        if not url:
            return []

        response = requests.get(str(url).strip(), timeout=timeout)
        response.raise_for_status()

        return encode_image_bytes(response.content)

    except Exception:
        return []
