import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain.personalization.style_dna_engine import style_dna_engine
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.style_flow_service import build_style_flow_response

router = APIRouter()
logger = logging.getLogger("ahvi.stylist")


class ItemContextRequest(BaseModel):
    main_category: str
    sub_category: str
    color_hex: str


@router.post("/item-suggestions")
def get_item_suggestions(request: ItemContextRequest):
    system_instruction = (
        "You are Ahvi's Fashion Knowledge Engine. The user just uploaded a new garment. "
        "Return JSON with: name, tags (4), pairing_rules (2). Output ONLY JSON."
    )
    user_prompt = (
        f"Item: {request.sub_category}\n"
        f"Category: {request.main_category}\n"
        f"Color Hex: {request.color_hex}"
    )
    try:
        messages = [{"role": "user", "content": user_prompt}]
        return ai_gateway.chat_json_object(
            messages,
            system_instruction=system_instruction,
            model="llama3.1",
        )
    except Exception as exc:
        print(f"[item-suggestions] error={str(exc)}")
        return {
            "name": request.sub_category.title(),
            "tags": ["versatile", "casual"],
            "pairing_rules": [
                "Pair with neutral basics.",
                "Layer depending on weather.",
            ],
        }


class OutfitPipelineRequest(BaseModel):
    user_id: str
    query: str = "What should I wear today?"
    wardrobe: Any = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    include_base64: bool = True
    upload_style_boards_to_r2: bool = False


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@router.post("/pipeline")
def run_outfit_pipeline(request: OutfitPipelineRequest):
    appwrite = AppwriteProxy()
    context = dict(request.context or {})
    context["query"] = request.query
    context["user_id"] = request.user_id

    wardrobe = request.wardrobe
    if wardrobe is None:
        try:
            wardrobe = appwrite.list_documents("outfits", user_id=request.user_id)
        except Exception:
            wardrobe = []

    style_dna = style_dna_engine.build(
        {
            "user_id": request.user_id,
            "user_profile": request.user_profile or {},
            "history": context.get("history", []),
            "wardrobe": wardrobe,
        }
    )
    context["style_dna"] = style_dna

    try:
        response = build_style_flow_response(
            user_id=request.user_id,
            query=request.query,
            wardrobe=wardrobe,
            user_profile=request.user_profile or {},
            context=context,
            include_base64=bool(request.include_base64),
            upload_to_r2=bool(request.upload_style_boards_to_r2),
            cache_bypass=True,
        )
        response["meta"] = {
            **_dict(response.get("meta")),
            "query": request.query,
            "analysis_source": "style_flow_service",
        }
        return response
    except Exception as exc:
        logger.exception(
            "stylist.pipeline failed user_id=%s error=%s", request.user_id, str(exc)
        )
        return {
            "success": False,
            "board": "style",
            "type": "cards",
            "message": "Pipeline temporarily unavailable. Please try again.",
            "cards": [],
            "board_ids": "",
            "data": {
                "outfits": [],
                "visual_intelligence": {},
                "pipeline": {},
                "rendered_boards": [],
                "board_item_ids": [],
            },
            "meta": {
                "count": 0,
                "query": request.query,
                "analysis_source": "outfit_pipeline",
                "error": "outfit_pipeline_failed",
            },
        }
