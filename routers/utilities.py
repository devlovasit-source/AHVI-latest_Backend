import logging
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.auth_helpers import bind_request_user
from services.r2_storage import R2StorageError
from services.qdrant_service import qdrant_service
from services import upload_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["utilities"])


class AnthropicMessage(BaseModel):
    role: str
    content: Any


class AnthropicRequest(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 380
    system: Optional[str] = None
    messages: List[AnthropicMessage]


class AvatarUploadRequest(BaseModel):
    user_id: str = ""
    image_base64: str = Field(..., min_length=10)


class WardrobeUploadRequest(BaseModel):
    file_id: str
    raw_image_base64: str = Field(..., min_length=10)
    masked_image_base64: str = Field(..., min_length=10)
    user_id: str = ""


def _safe_file_id(file_id: str) -> str:
    """Storage keys are built from file_id — reject separators / traversal."""
    value = str(file_id or "").strip()
    if (
        not value
        or len(value) > 128
        or ".." in value
        or any(ch in value for ch in ("/", "\\", "\0"))
    ):
        raise HTTPException(status_code=400, detail="Invalid file_id")
    return value


@router.post("/api/anthropic")
def anthropic_messages(request: AnthropicRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500, detail="ANTHROPIC_API_KEY is not configured"
        )

    payload: Dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [m.model_dump() for m in request.messages],
    }
    if request.system:
        payload["system"] = request.system

    try:
        res = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic request failed: {exc}")

    if res.status_code >= 400:
        logger.warning(
            "anthropic_upstream_error status=%d body_len=%d",
            res.status_code,
            len(res.text or ""),
        )
        raise HTTPException(
            status_code=502 if res.status_code >= 500 else 400,
            detail="Upstream provider error",
        )
    return res.json()


@router.get("/api/weather")
def weather(latitude: float, longitude: float):
    # Open-Meteo is keyless and suitable for quick weather snapshots.
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current=temperature_2m,weather_code,wind_speed_10m"
    )
    try:
        res = requests.get(url, timeout=20)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather request failed: {exc}")

    if res.status_code >= 400:
        logger.warning(
            "weather_upstream_error status=%d body_len=%d",
            res.status_code,
            len(res.text or ""),
        )
        raise HTTPException(
            status_code=502 if res.status_code >= 500 else 400,
            detail="Upstream weather error",
        )
    return res.json()


@router.post("/api/uploads/avatar")
def upload_avatar(request: AvatarUploadRequest, http_request: Request = None):
    # Avatar keys are derived from user_id — bind to the authenticated user so
    # one account can never write into another account's avatar namespace.
    user_id = bind_request_user(http_request, request.user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        avatar_url = upload_service.upload_avatar(
            user_id=user_id,
            image_base64=request.image_base64,
        )
        return {"avatar_url": avatar_url}
    except R2StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid avatar payload: {exc}")


@router.post("/api/uploads/wardrobe")
def upload_wardrobe_images(request: WardrobeUploadRequest, http_request: Request = None):
    # Wardrobe storage keys are derived from file_id. Namespace the key by
    # the authenticated user so one account can never overwrite another
    # account's objects, and reject separator/traversal file_ids outright.
    user_id = bind_request_user(http_request, request.user_id)
    file_id = _safe_file_id(request.file_id)
    if user_id and not file_id.startswith(f"{user_id}-"):
        file_id = f"{user_id}-{file_id}"
    try:
        result = upload_service.upload_wardrobe_images(
            file_id=file_id,
            raw_image_base64=request.raw_image_base64,
            masked_image_base64=request.masked_image_base64,
        )
        return result
    except R2StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid wardrobe upload payload: {exc}"
        )


@router.get("/api/qdrant/status")
def qdrant_status():
    return qdrant_service.status()


@router.get("/api/architecture/status")
def architecture_status():
    return {
        "experience_layer": {
            "frontend": "external_flutter_app",
            "boards": {"style_boards": True, "life_boards": True},
        },
        "core_intelligence_layer": {
            "context_engine": True,
            "style_graph_engine": True,
            "style_dna_engine": True,
            "agent_system": True,
            "execution_engine": True,
        },
        "api_ai_processing": {
            "fastapi_backend": True,
            "ai_gateway": True,
            "vision_pipeline": True,
            "async_processing": True,
        },
        "data_storage": {
            "appwrite": True,
            "qdrant": qdrant_service.status().get("enabled", False),
            "cloudflare_r2": True,
        },
        "response_flow": "boards_updated -> ui_rendered -> user_feedback",
    }
