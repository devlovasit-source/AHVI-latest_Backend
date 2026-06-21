import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain.personalization.style_dna_engine import style_dna_engine
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.style_flow_service import build_style_flow_response, item_role

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


# --------------------------------------------------------------------------
# Item-detail CTAs: Style This (3 directions) + Build Outfit (1 outfit).
# Reuses the existing style-flow pipeline, anchored on a wardrobe item, with a
# dress-pairing guard so dresses never lead with men's/formal leather shoes.
# --------------------------------------------------------------------------

_FRIENDLY_FAIL = (
    "AHVI could not build a complete outfit yet. Try adding shoes or accessories."
)

_STYLE_DIRECTIONS = (
    ("Casual Brunch", "a relaxed casual brunch"),
    ("Date Night", "a date night"),
    ("Vacation Day", "an easy vacation day"),
)

# Footwear that should NOT lead a dress look (men's / formal leather).
_BAD_DRESS_FOOTWEAR_TOKENS = (
    "loafer", "oxford", "derby", "brogue", "monk", "wingtip",
    "dress shoe", "formal shoe", "leather shoe", "leather shoes",
    "men's", "mens", "combat boot",
)
_DRESS_FOOTWEAR_SUGGESTIONS = [
    {"label": "White sneakers", "reason": "Fresh, casual contrast that lets the dress lead.", "cta": "Find this"},
    {"label": "Nude or black sandals", "reason": "Clean feminine line for a dress.", "cta": "Find this"},
    {"label": "Ballet flats", "reason": "Soft daytime option that keeps the look easy.", "cta": "Find this"},
]


class ItemStyleRequest(BaseModel):
    user_id: str
    mode: str = "build_outfit"  # "build_outfit" | "style_this"
    occasion: Optional[str] = None
    wardrobe_only: bool = False
    anchor_item: Dict[str, Any] = Field(default_factory=dict)
    wardrobe: Any = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


def _txt(value: Any) -> str:
    return str(value or "").strip()


def _item_id_of(item: Dict[str, Any]) -> str:
    return _txt(item.get("item_id") or item.get("id") or item.get("$id"))


def _items_of(card: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(card, dict):
        return out
    for key in ("items", "accessories"):
        value = card.get(key)
        if isinstance(value, list):
            out.extend([dict(x) for x in value if isinstance(x, dict)])
    for key in ("dress", "top", "bottom", "footwear", "shoes", "outerwear"):
        value = card.get(key)
        if isinstance(value, dict):
            out.append(dict(value))
    return out


def _first_card(resp: Any) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return {}
    cards = resp.get("cards")
    if isinstance(cards, list) and cards and isinstance(cards[0], dict):
        return cards[0]
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    outfits = data.get("outfits")
    if isinstance(outfits, list) and outfits and isinstance(outfits[0], dict):
        return outfits[0]
    return {}


def _resp_missing(resp: Any, card: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in (card, (resp.get("data") if isinstance(resp, dict) else {}), resp):
        if isinstance(src, dict):
            value = src.get("missing_items")
            if isinstance(value, list):
                out.extend([dict(x) for x in value if isinstance(x, dict)])
    # de-dup by label
    seen, deduped = set(), []
    for m in out:
        key = _txt(m.get("label")).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def _anchor_is_dress(anchor: Dict[str, Any]) -> bool:
    blob = " ".join(
        _txt(anchor.get(k)) for k in ("category", "sub_category", "subcategory", "name", "label")
    ).lower()
    return any(t in blob for t in ("dress", "frock", "gown"))


def _is_bad_dress_footwear(item: Dict[str, Any]) -> bool:
    if item_role(item) != "footwear":
        return False
    blob = " ".join(
        _txt(item.get(k)) for k in ("name", "label", "category", "sub_category", "subcategory", "type")
    ).lower()
    return any(t in blob for t in _BAD_DRESS_FOOTWEAR_TOKENS)


def _apply_dress_pairing(
    items: List[Dict[str, Any]], missing: List[Dict[str, Any]], anchor: Dict[str, Any]
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Drop bad dress footwear; if that leaves no footwear, suggest a good
    missing piece instead of a bad owned pairing."""
    if not _anchor_is_dress(anchor):
        return items, missing
    kept = [i for i in items if not _is_bad_dress_footwear(i)]
    out_missing = list(missing)
    if len(kept) != len(items) and not any(item_role(i) == "footwear" for i in kept):
        existing = {_txt(m.get("label")).lower() for m in out_missing}
        for suggestion in _DRESS_FOOTWEAR_SUGGESTIONS:
            if suggestion["label"].lower() not in existing:
                out_missing.append(dict(suggestion))
                break
    return kept, out_missing


def _inject_anchor(items: List[Dict[str, Any]], anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not anchor:
        return items
    aid = _item_id_of(anchor)
    if aid and any(_item_id_of(i) == aid for i in items):
        return items
    return [anchor, *items]


def _resolve_wardrobe(request: ItemStyleRequest) -> List[Dict[str, Any]]:
    if isinstance(request.wardrobe, list):
        return [i for i in request.wardrobe if isinstance(i, dict)]
    try:
        rows = AppwriteProxy().list_documents("outfits", user_id=request.user_id) or []
        return [i for i in rows if isinstance(i, dict)]
    except Exception:
        return []


def _resolve_anchor(
    request: ItemStyleRequest, item_id: str, wardrobe: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if isinstance(request.anchor_item, dict) and request.anchor_item:
        anchor = dict(request.anchor_item)
        anchor.setdefault("item_id", item_id)
        return anchor
    for item in wardrobe:
        if _item_id_of(item) == _txt(item_id):
            return dict(item)
    return {"item_id": _txt(item_id)}


def _anchor_desc(anchor: Dict[str, Any]) -> str:
    parts = [
        _txt(anchor.get(k))
        for k in ("color_name", "color", "sub_category", "subcategory", "category", "name")
    ]
    seen, words = set(), []
    for p in parts:
        low = p.lower()
        if p and low not in seen:
            seen.add(low)
            words.append(p)
    return (" ".join(words) or "wardrobe item")[:120]


def _build_one(request: ItemStyleRequest, query: str, wardrobe: List[Dict[str, Any]]) -> Dict[str, Any]:
    context = dict(request.context or {})
    context["query"] = query
    context["user_id"] = request.user_id
    context["anchor_item_id"] = _item_id_of(request.anchor_item) or None
    if request.occasion:
        context["occasion"] = request.occasion
    return build_style_flow_response(
        user_id=request.user_id,
        query=query,
        wardrobe=wardrobe,
        user_profile=request.user_profile or {},
        context=context,
        include_base64=False,
        upload_to_r2=False,
        cache_bypass=True,
    )


def _map_look(resp: Any, anchor: Dict[str, Any], title: str, note_key: str) -> Dict[str, Any]:
    card = _first_card(resp)
    items = _inject_anchor(_items_of(card), anchor)
    missing = _resp_missing(resp, card)
    items, missing = _apply_dress_pairing(items, missing, anchor)
    note = _txt(
        card.get("styling_note")
        or card.get("reason")
        or card.get("editorial")
        or card.get("description")
        if isinstance(card, dict)
        else ""
    ) or (_txt(resp.get("message")) if isinstance(resp, dict) else "")
    return {
        "title": title or _txt(card.get("title")) or "Your Look",
        "items": items,
        "missing_items": missing,
        note_key: note or "Styled around your selected piece.",
    }


def _outfit_title(occasion: Optional[str]) -> str:
    occ = _txt(occasion).lower()
    return {
        "office": "Office-Ready Look",
        "date": "Date Night Look",
        "casual": "Casual Day Look",
    }.get(occ, "Casual Day Look")


def _style_fallback(mode: str, anchor: Dict[str, Any]) -> Dict[str, Any]:
    missing = (
        list(_DRESS_FOOTWEAR_SUGGESTIONS)
        if _anchor_is_dress(anchor)
        else [{"label": "Versatile shoes", "reason": "Completes the look.", "cta": "Find this"}]
    )
    anchor_items = [anchor] if anchor else []
    if mode == "style_this":
        return {
            "success": False,
            "mode": mode,
            "anchor_item": anchor,
            "message": _FRIENDLY_FAIL,
            "style_directions": [
                {"title": t, "items": anchor_items, "missing_items": missing, "styling_note": _FRIENDLY_FAIL}
                for t, _ in _STYLE_DIRECTIONS
            ],
        }
    return {
        "success": False,
        "mode": mode,
        "anchor_item": anchor,
        "message": _FRIENDLY_FAIL,
        "outfit": {"title": "Your Look", "items": anchor_items, "missing_items": missing, "reason": _FRIENDLY_FAIL},
    }


@router.post("/items/{item_id}/style")
def style_wardrobe_item(item_id: str, request: ItemStyleRequest) -> Dict[str, Any]:
    """Power the item-detail CTAs.

    mode=style_this -> 3 editorial styling directions.
    mode=build_outfit -> 1 practical outfit anchored on the item.
    Never raises: returns a friendly fallback so the UI never dead-ends.
    """
    mode = _txt(request.mode).lower()
    if mode not in {"build_outfit", "style_this"}:
        mode = "build_outfit"
    wardrobe = _resolve_wardrobe(request)
    anchor = _resolve_anchor(request, item_id, wardrobe)
    desc = _anchor_desc(anchor)

    try:
        if mode == "style_this":
            def _one(direction: "tuple[str, str]") -> Dict[str, Any]:
                title, vibe = direction
                query = (
                    f"Style my {desc} for {vibe}. Prefer pieces I already own and "
                    f"suggest what's missing."
                )
                return _map_look(_build_one(request, query, wardrobe), anchor, title, "styling_note")

            with ThreadPoolExecutor(max_workers=len(_STYLE_DIRECTIONS)) as pool:
                directions = list(pool.map(_one, _STYLE_DIRECTIONS))
            logger.info(
                "stylist.item_style mode=style_this item_id=%s directions=%d", item_id, len(directions)
            )
            return {"success": True, "mode": mode, "anchor_item": anchor, "style_directions": directions}

        occ = request.occasion or "a casual everyday"
        query = (
            f"Build one complete, practical outfit using my {desc} as the anchor piece "
            f"for {occ}. Prefer items I already own; only suggest missing pieces if needed."
        )
        outfit = _map_look(_build_one(request, query, wardrobe), anchor, _outfit_title(request.occasion), "reason")
        logger.info(
            "stylist.item_style mode=build_outfit item_id=%s items=%d missing=%d",
            item_id, len(outfit.get("items") or []), len(outfit.get("missing_items") or []),
        )
        return {"success": True, "mode": mode, "anchor_item": anchor, "outfit": outfit}
    except Exception as exc:  # noqa: BLE001 - CTA must never dead-end the UI
        logger.exception("stylist.item_style failed item_id=%s mode=%s err=%s", item_id, mode, str(exc))
        return _style_fallback(mode, anchor)
