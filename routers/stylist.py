import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from brain.engines.outfit_quality_guard import is_complete_board
from brain.personalization.style_dna_engine import style_dna_engine
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.auth_helpers import enforce_owner
from services.style_flow_service import build_style_flow_response, item_role
from services.location_weather_context import resolve_location_weather_context
from services.style_board_image_readiness import is_board_renderable, project_board_image_fields
from services.style_board_reasoning import FALLBACK_STYLING_NOTE, build_styling_note
from services.style_board_shuffle_service import _default_position, register_board
from services.style_item_contract import (
    canonical_accessory_type,
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
)
from services.style_this_anchor import canonical_style_this_anchor
from services.stylist_knowledge_service import resolve_style_archetypes
from services.wardrobe_sanitizer import sanitize_fashion_wardrobe_items

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
    resolved = resolve_location_weather_context(
        user_id=request.user_id,
        request_data={**request.model_dump(exclude_none=True), **context},
        profile=request.user_profile,
    )
    context["location_context"] = resolved["location"]
    context["weather"] = resolved["weather"]
    context["weather_context"] = resolved["weather"]

    wardrobe = request.wardrobe
    if wardrobe is None:
        try:
            wardrobe = appwrite.list_documents("outfits", user_id=request.user_id)
        except Exception:
            wardrobe = []
    wardrobe = sanitize_fashion_wardrobe_items(
        wardrobe, source="stylist.pipeline"
    )

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
            user_profile=resolved["profile"],
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
        response["context_usage"] = resolved["context_usage"]
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
            "context_usage": resolved["context_usage"],
        }


# --------------------------------------------------------------------------
# Item-detail CTAs: Style This (3 directions) + Build Outfit (1 outfit).
# Reuses the existing style-flow pipeline, anchored on a wardrobe item, with a
# dress-pairing guard so dresses never lead with men's/formal leather shoes.
# --------------------------------------------------------------------------

_FRIENDLY_FAIL = (
    "AHVI could not build a complete outfit yet. Try adding shoes or accessories."
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
    scenario: Optional[str] = None
    anchor_item_id: Optional[str] = None
    occasion: Optional[str] = None
    wardrobe_only: bool = False
    anchor_item: Dict[str, Any] = Field(default_factory=dict)
    wardrobe: Any = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    weather: Any = None
    weather_context: Any = None
    location: Any = None
    coordinates: Any = None
    latitude: float | None = None
    longitude: float | None = None


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
        return sanitize_fashion_wardrobe_items(
            request.wardrobe, source="stylist.request"
        )
    try:
        rows = AppwriteProxy().list_documents("outfits", user_id=request.user_id) or []
        return sanitize_fashion_wardrobe_items(
            rows, source="stylist.appwrite"
        )
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


def _resolve_style_this_anchor(
    request: ItemStyleRequest,
    item_id: str,
    wardrobe: List[Dict[str, Any]],
    *,
    allow_legacy_metadata: bool = False,
) -> Dict[str, Any]:
    """Resolve Style This identity from the authoritative item ID only."""
    requested_id = _txt(request.anchor_item_id or item_id)
    if requested_id != _txt(item_id):
        return {}
    authoritative = next(
        (dict(item) for item in wardrobe if _item_id_of(item) == requested_id),
        None,
    )
    if authoritative is None:
        return {}
    # This row came from the authenticated user's authoritative wardrobe
    # collection. Legacy rows may omit provenance, so derive that missing
    # field here without accepting client-supplied source metadata.
    if not _txt(authoritative.get("source")):
        authoritative["source"] = "wardrobe"
    return canonical_style_this_anchor(
        authoritative,
        expected_item_id=requested_id,
        allow_missing_image=allow_legacy_metadata,
    ) or {}


def _list_all_documents(
    proxy: AppwriteProxy, resource: str, *, user_id: str | None = None
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        page = proxy.list_documents(
            resource,
            user_id=user_id,
            limit=100,
            offset=offset,
            return_meta=True,
        ) or []
        if isinstance(page, dict):
            documents = page.get("documents") or []
            meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
            has_more = bool(meta.get("has_more"))
            next_offset = meta.get("next_offset")
        else:
            documents = page
            has_more = len(documents) >= 100
            next_offset = None
        clean = [dict(item) for item in documents if isinstance(item, dict)]
        rows.extend(clean)
        if not has_more:
            return rows
        offset = int(next_offset) if next_offset is not None else offset + len(documents)


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


def _style_fallback(
    mode: str,
    anchor: Dict[str, Any],
    strategies: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    missing = (
        list(_DRESS_FOOTWEAR_SUGGESTIONS)
        if _anchor_is_dress(anchor)
        else [{"label": "Versatile shoes", "reason": "Completes the look.", "cta": "Find this"}]
    )
    anchor_items = [anchor] if anchor else []
    if mode == "style_this":
        fallback_strategies = strategies or [
            {"direction_title": f"Style Edit {index + 1}"} for index in range(3)
        ]
        return {
            "success": False,
            "mode": mode,
            "anchor_item": anchor,
            "message": _FRIENDLY_FAIL,
            "style_directions": [
                {
                    "title": str(strategy.get("direction_title") or "Style Edit"),
                    "items": anchor_items,
                    "missing_items": missing,
                    "styling_note": _FRIENDLY_FAIL,
                }
                for strategy in fallback_strategies[:3]
            ],
        }
    return {
        "success": False,
        "mode": mode,
        "anchor_item": anchor,
        "message": _FRIENDLY_FAIL,
        "outfit": {"title": "Your Look", "items": anchor_items, "missing_items": missing, "reason": _FRIENDLY_FAIL},
    }


def _style_this_failure(
    anchor: Dict[str, Any], code: str, message: str
) -> Dict[str, Any]:
    logger.info(
        "AHVI_STYLE_THIS_TYPED_FAILURE code=%s anchor_item_id=%s",
        code,
        canonical_item_id(anchor) if isinstance(anchor, dict) else "",
    )
    return {
        "success": False,
        "mode": "style_this",
        "anchor_item": anchor,
        "style_directions": [],
        "error": {"code": code, "message": message},
        "message": message,
    }


def _legacy_build_outfit_cta_failure() -> Dict[str, Any]:
    """Stop obsolete item CTA calls without touching conversational styling."""
    message = "Try-On is coming soon."
    return {
        "success": False,
        "mode": "build_outfit",
        "intent": "try_on_coming_soon",
        "action": "try_on_coming_soon",
        "response_mode": "text_only",
        "message": message,
        "message_text": message,
        "style_directions": [],
        "error": {
            "code": "TRY_ON_COMING_SOON",
            "message": message,
        },
    }


# --------------------------------------------------------------------------
# Lite wardrobe pairing — fast, deterministic CTA path.
# The full style_flow pipeline takes ~90s for an interactive button; this
# builds a look from owned items with simple rules in ~1-2s (wardrobe load
# only). No agent/qdrant/board rendering.
# --------------------------------------------------------------------------
_LITE_FOOTWEAR = ("shoe", "sneaker", "loafer", "boot", "sandal", "heel", "flat", "footwear", "mule", "pump", "oxford", "espadrille")
_LITE_BOTTOM = ("jean", "trouser", "pant", "chino", "short", "skirt", "legging", "bottom")
_LITE_TOP = ("top", "shirt", "tee", "tshirt", "t-shirt", "polo", "kurta", "blouse", "jacket", "blazer", "coat", "hoodie", "sweater", "knit", "cardigan", "overshirt")
_LITE_DRESS = ("dress", "gown", "saree", "sari", "lehenga", "jumpsuit", "one-piece", "one piece", "frock", "kaftan", "anarkali")
_LITE_ACCESSORY = ("watch", "belt", "sunglass", "bag", "clutch", "handbag", "purse", "necklace", "bracelet", "ring", "earring", "scarf", "hat", "cap", "jewel")
# Non-fashion junk that got mis-saved into wardrobes — never pair these.
_LITE_NON_FASHION = (
    "charger", "cable", "adapter", "bottle", "phone", "remote", "mouse",
    "keyboard", "laptop", "earbud", "headphone", "airpod", "power bank",
    "powerbank", "plug", "wire", "battery", "speaker", "camera", "mug",
    "cup", "pen", "book", "box",
)
# Activity / swim / sport gear — real apparel but NOT stylable into everyday
# looks (e.g. a swim cap should never land in a brunch or date outfit).
_LITE_SPORT_SWIM = (
    "swim", "goggle", "wetsuit", "snorkel", "flipper", "cleat",
    "shin guard", "helmet", "ski ", "snowboard", "life jacket",
    "life vest", "boxing glove", "swimsuit", "swimwear",
)

_LITE_MISSING = {
    "footwear": {"label": "Clean sneakers or sandals", "reason": "Completes the look.", "cta": "Find this"},
    "top": {"label": "A simple top", "reason": "Pairs with this piece.", "cta": "Find this"},
    "bottom": {"label": "Tailored trousers or jeans", "reason": "Anchors the outfit.", "cta": "Find this"},
    "accessory": {"label": "A small bag or jewelry", "reason": "Finishes the look.", "cta": "Find this"},
    "dress": {"label": "A standout dress", "reason": "Gives the accessory something to sit on.", "cta": "Find this"},
}

def _lite_role(item: Dict[str, Any]) -> str:
    blob = " ".join(
        _txt(item.get(k))
        for k in ("role", "category", "sub_category", "subcategory", "type", "name", "label")
    ).lower()
    if any(t in blob for t in _LITE_NON_FASHION):
        return "unknown"
    if any(t in blob for t in _LITE_SPORT_SWIM):
        return "unknown"
    if any(t in blob for t in _LITE_DRESS):
        return "dress"
    if any(t in blob for t in _LITE_FOOTWEAR):
        return "footwear"
    if any(t in blob for t in _LITE_BOTTOM):
        return "bottom"
    if any(t in blob for t in _LITE_ACCESSORY):
        return "accessory"
    if any(t in blob for t in _LITE_TOP):
        return "top"
    # Unrecognized item — do NOT assume it's an accessory (that pulled in
    # chargers/bottles). Excluded from pairing.
    return "unknown"


def _lite_image(item: Dict[str, Any]) -> str:
    # Raw/display fallback only - deliberately does NOT fall back to
    # masked_url/cutout_url/etc. If it did, an item whose only board-safe
    # field is e.g. masked_url (no distinct raw upload photo) would end up
    # with image_url == masked_url, which is_board_renderable() correctly
    # treats as fabricated provenance (masked_url aliasing the raw photo) -
    # rejecting a genuinely renderable item because this function manufactured
    # the appearance of aliasing. normalized_url is the one safe exception:
    # it's an unconditional, never-alias-checked catalog field by contract
    # (see services.style_board_image_readiness).
    return _txt(
        item.get("normalized_url") or item.get("normalizedUrl")
        or item.get("image_url") or item.get("imageUrl")
    )


def _lite_item(item: Dict[str, Any], role: str) -> Dict[str, Any]:
    return {
        "item_id": _item_id_of(item),
        "name": _txt(item.get("name") or item.get("label")) or "Item",
        "category": _txt(item.get("category")),
        "image_url": _lite_image(item),
        "role": role,
        "owned": True,
        # Carry the winning board-safe field (+ required status) under its
        # own canonical name so a downstream is_board_renderable() check on
        # THIS item still finds it - collapsing to image_url-only here is
        # exactly how a renderable item became an empty hanger.
        **project_board_image_fields(item),
    }


def _lite_group(wardrobe: List[Dict[str, Any]], exclude_id: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {"dress": [], "top": [], "bottom": [], "footwear": [], "accessory": []}
    for item in wardrobe:
        if not isinstance(item, dict) or _item_id_of(item) == exclude_id:
            continue
        if not is_board_renderable(item):
            continue  # no genuine board-safe image - never a support-piece candidate
        groups.setdefault(_lite_role(item), []).append(item)
    return groups


def _lite_pick(
    groups: Dict[str, List[Dict[str, Any]]],
    role: str,
    is_dress: bool,
    prefer=(),
    variant: int = 0,
    strategy: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    cands = list(groups.get(role) or [])
    if is_dress and role == "footwear":
        cands = [c for c in cands if not _is_bad_dress_footwear(c)]
    if not cands:
        return None
    if strategy:
        avoid = [str(value).strip().lower() for value in strategy.get("avoid") or [] if str(value).strip()]
        safe = [
            candidate for candidate in cands
            if not any(token in _lite_item_blob(candidate) for token in avoid)
        ]
        cands = safe
        if not cands:
            return None

        palette = [str(value).strip().lower() for value in strategy.get("palette") or [] if str(value).strip()]
        target_formality = _lite_formality_value(strategy.get("formality"))

        def _strategy_score(candidate: Dict[str, Any]) -> float:
            score = 0.0
            color = _txt(
                candidate.get("color")
                or candidate.get("colour")
                or candidate.get("color_name")
                or candidate.get("color_code")
            ).lower()
            if palette and color and any(color == p or color in p or p in color for p in palette):
                score += 3.0
            meta = candidate.get("style_metadata") if isinstance(candidate.get("style_metadata"), dict) else {}
            item_formality = _lite_formality_value(candidate.get("formality") or meta.get("formality"))
            if target_formality is not None and item_formality is not None:
                score += max(0.0, 2.0 - abs(target_formality - item_formality) * 0.4)
            return score

        ranked = [(candidate, _strategy_score(candidate)) for candidate in cands]
        best_score = max(score for _, score in ranked)
        best = [candidate for candidate, score in ranked if score == best_score]
        return best[variant % len(best)]
    if prefer:
        def _score(c: Dict[str, Any]) -> int:
            blob = " ".join(_txt(c.get(k)) for k in ("name", "label", "sub_category", "subcategory")).lower()
            return 0 if any(p in blob for p in prefer) else 1
        cands.sort(key=_score)
        return cands[0]
    # No preference (tops/bottoms/accessories): rotate by variant so different
    # directions show different owned pieces when more than one exists.
    return cands[variant % len(cands)]


def _lite_item_blob(item: Dict[str, Any]) -> str:
    return " ".join(
        _txt(item.get(key))
        for key in (
            "name", "label", "category", "sub_category", "subcategory",
            "style", "pattern", "material", "tags", "style_tags",
        )
    ).lower()


def _lite_formality_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    key = _txt(value).lower().replace("-", "_").replace(" ", "_")
    return {
        "homewear": 1.0,
        "athletic": 2.0,
        "casual": 3.0,
        "smart_casual": 5.0,
        "business_casual": 6.0,
        "formal": 8.0,
        "business_formal": 8.0,
    }.get(key)


def _lite_missing(role: str, is_dress: bool) -> Dict[str, Any]:
    if role == "footwear" and is_dress:
        return dict(_DRESS_FOOTWEAR_SUGGESTIONS[0])
    return dict(_LITE_MISSING.get(role, _LITE_MISSING["accessory"]))


def _lite_accessory_needed_slots(
    groups: Dict[str, List[Dict[str, Any]]],
    variant: int,
    strategy: Optional[Dict[str, Any]],
) -> List[str]:
    """Accessory anchors prefer a one-piece hero garment (dress) + footwear.
    Most wardrobes are separates-only though, so before committing to that
    shape, try the same pick this direction would actually use for "dress"
    (respecting avoid/palette filtering, not just bool(groups["dress"])) -
    if nothing survives that filtering, fall back to top+bottom+footwear so
    the board doesn't collapse to an incomplete accessory+footwear pairing."""
    dress_pick = _lite_pick(groups, "dress", True, variant=variant, strategy=strategy)
    if dress_pick:
        return ["dress", "footwear"]
    return ["top", "bottom", "footwear"]


def _lite_needed_slots(anchor_role: str) -> List[str]:
    if anchor_role == "dress":
        return ["footwear", "accessory"]
    if anchor_role == "top":
        return ["bottom", "footwear", "accessory"]
    if anchor_role == "bottom":
        return ["top", "footwear", "accessory"]
    if anchor_role == "footwear":
        return ["top", "bottom", "accessory"]
    if anchor_role == "outerwear":
        # A jacket/blazer/coat/cardigan/overshirt/shacket anchor is a layer,
        # not a top-slot substitute - it needs the same separates a top
        # anchor would (never the accessory-anchor dress fallback below).
        return ["top", "bottom", "footwear", "accessory"]
    return ["dress", "footwear"]  # accessory anchor fallback default (see _lite_accessory_needed_slots)


def _lite_weather_note(weather: Dict[str, Any]) -> str:
    if weather.get("status") != "available":
        return ""
    condition = _txt(weather.get("condition") or weather.get("weather_type")).lower()
    temp_level = _txt(weather.get("temp_level") or weather.get("temperature_band")).lower()
    humidity = weather.get("humidity")
    if any(token in condition for token in ("rain", "storm", "drizzle", "snow")):
        return "Rain-safe adjustment: add a water-resistant layer and closed footwear."
    if temp_level in {"hot", "very_hot", "extreme_heat", "warm"} or (
        isinstance(humidity, (int, float)) and humidity >= 70
    ):
        return "Heat adjustment: favor breathable fabrics and minimal layering."
    if temp_level in {"cold", "very_cold"}:
        return "Cold-weather adjustment: add a warm outer layer."
    signals = weather.get("signals") if isinstance(weather.get("signals"), dict) else {}
    if signals.get("avoid_loose_flow"):
        return "Wind adjustment: favor secure layers over loose, trailing pieces."
    return ""


def _lite_build_outfit(
    anchor: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
    occasion: Optional[str],
    *,
    title: Optional[str] = None,
    prefer=(),
    note: str = "",
    variant: int = 0,
    weather: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    identity_role: Optional[str] = None,
) -> Dict[str, Any]:
    is_dress = _anchor_is_dress(anchor) or _lite_role(anchor) == "dress"
    anchor_role = identity_role or ("dress" if is_dress else _lite_role(anchor))
    groups = _lite_group(wardrobe, _item_id_of(anchor))
    items = [_lite_item(anchor, identity_role or "hero")]
    missing: List[Dict[str, Any]] = []
    needed_slots = (
        _lite_accessory_needed_slots(groups, variant, strategy)
        if anchor_role == "accessory"
        else _lite_needed_slots(anchor_role)
    )
    for slot in needed_slots:
        pick = _lite_pick(
            groups, slot, is_dress,
            prefer=prefer if slot == "footwear" else (),
            variant=variant,
            strategy=strategy,
        )
        if pick:
            items.append(_lite_item(pick, "accent" if slot == "accessory" else "support"))
            groups[slot] = [g for g in groups[slot] if _item_id_of(g) != _item_id_of(pick)]
        else:
            missing.append(_lite_missing(slot, is_dress))
    if strategy:
        reason = build_styling_note(anchor, items, strategy)
    else:
        reason = note or FALLBACK_STYLING_NOTE
    return {
        "title": title or _outfit_title(occasion),
        "items": items,
        "missing_items": missing,
        "reason": " ".join(
            part for part in (
                reason,
                _lite_weather_note(weather or {}),
            ) if part
        ),
    }


def _lite_directions(
    anchor: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
    weather: Optional[Dict[str, Any]] = None,
    strategies: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    selected = strategies or resolve_style_archetypes(
        {"occasion": "daily"}, anchor_item=anchor, direction_count=3
    )
    directions = []
    for idx, strategy in enumerate(selected[:3]):
        title = _txt(strategy.get("direction_title")) or f"Style Edit {idx + 1}"
        look = _lite_build_outfit(
            anchor, wardrobe, None, title=title,
            variant=idx, weather=weather, strategy=strategy,
            identity_role=canonical_item_role(anchor),
        )
        directions.append({
            "title": title,
            "items": look["items"],
            "missing_items": look["missing_items"],
            "styling_note": look["reason"],
            "archetype_id": strategy.get("archetype_id"),
            "formality": strategy.get("formality"),
            "palette": list(strategy.get("palette") or []),
            "avoid": list(strategy.get("avoid") or []),
            "reasoning_intent": strategy.get("reasoning_intent"),
            "style_strategy": dict(strategy),
        })
    return directions


def _style_this_registration_error(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def _register_style_this_direction(
    direction: Dict[str, Any],
    *,
    anchor: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
    user_id: str,
    occasion: Optional[str],
) -> Dict[str, Any]:
    board_id = str(uuid4())
    canonical_anchor = canonical_style_this_anchor(
        anchor,
        expected_item_id=canonical_item_id(anchor),
        allow_missing_image=True,
    ) or dict(anchor)
    anchor = canonical_anchor
    anchor_id = canonical_item_id(anchor)
    wardrobe_by_id = {
        canonical_item_id(item): dict(item)
        for item in wardrobe
        if canonical_item_id(item)
    }
    canonical_items: List[Dict[str, Any]] = []
    error: Optional[Dict[str, str]] = None

    if not anchor_id:
        error = _style_this_registration_error(
            "INVALID_ANCHOR_ITEM",
            "The selected item does not have a stable item ID.",
        )

    seen_ids = set()
    for item in direction.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_id = canonical_item_id(item)
        source_item = wardrobe_by_id.get(item_id, {})
        current = {**item, **source_item}
        if item_id == anchor_id:
            current = {**current, **anchor}
        if isinstance(item.get("position"), dict):
            current["position"] = dict(item["position"])
        current_id = canonical_item_id(current)
        if not current_id:
            error = error or _style_this_registration_error(
                "INVALID_ITEM_ID",
                "A board item does not have a stable item ID.",
            )
            continue
        if current_id in seen_ids:
            error = error or _style_this_registration_error(
                "DUPLICATE_ITEM_ID",
                "A board cannot contain the same item more than once.",
            )
            continue
        seen_ids.add(current_id)

        board_role = str(current.get("board_role") or current.get("role") or "").strip()
        role = canonical_item_role(current)
        source = canonical_item_source(current)
        if source == "unknown" and current_id in wardrobe_by_id:
            source = "wardrobe"
        if role == "unknown":
            error = error or _style_this_registration_error(
                "INVALID_SLOT",
                f"Board item {current_id} does not have a canonical slot.",
            )
        if source not in {"wardrobe", "style_asset"}:
            error = error or _style_this_registration_error(
                "UNKNOWN_ITEM_SOURCE",
                f"Board item {current_id} does not have a trusted source.",
            )

        current["item_id"] = current_id
        current["id"] = current_id
        current["role"] = role
        current["slot"] = role
        current["source"] = source
        current["locked"] = current_id == anchor_id
        if current_id == anchor_id:
            current["anchor_item_id"] = anchor_id
            current["anchor"] = True
            current["safe_image_url"] = anchor.get("safe_image_url")
            current["source_kind"] = anchor.get("source_kind")
            current["expected_transparent"] = anchor.get("expected_transparent")
        if board_role:
            current["board_role"] = board_role
        # NOT canonical_image_url() here: it falls back to masked_url, which
        # for an item with no distinct raw photo makes image_url == masked_url
        # - is_board_renderable() then rejects it as fabricated provenance.
        # _lite_image() uses the same restricted (normalized_url / raw
        # image_url only) fallback _lite_item already applied, so this can't
        # re-introduce that collision. canonical_image_url() itself is left
        # unchanged - it's a general "display image" helper used elsewhere
        # for non-board comparisons, not specific to board serialization.
        image_url = _lite_image(current)
        if image_url:
            current["image_url"] = image_url
        # Re-project the winning board-safe field(s) explicitly: the
        # wardrobe_by_id merge above already carries them through in the
        # common case, but this makes the guarantee direct rather than
        # incidental, so registration can never hand back an item that was
        # renderable before this function and isn't after it.
        current.update(project_board_image_fields(current))
        if role == "accessory" and not current.get("accessory_type"):
            current["accessory_type"] = canonical_accessory_type(current)
        if not isinstance(current.get("position"), dict):
            current["position"] = _default_position(current_id, role)
        canonical_items.append(current)

    anchor_matches = [item for item in canonical_items if item["item_id"] == anchor_id]
    if len(anchor_matches) != 1:
        error = _style_this_registration_error(
            "INVALID_ANCHOR_ITEM",
            "The selected anchor is missing from this Style This direction.",
        )
    elif anchor_matches[0].get("role") != anchor.get("role"):
        error = _style_this_registration_error(
            "ANCHOR_ROLE_MISMATCH",
            "The selected anchor role changed while building this Style This direction.",
        )
    if not is_complete_board(canonical_items):
        error = error or _style_this_registration_error(
            "BOARD_REGISTRATION_INVALID",
            "A complete Style This board could not be built from the available pieces.",
        )

    sources = {item.get("source") for item in canonical_items}
    source_policy = "wardrobe" if sources == {"wardrobe"} else ""
    if sources and sources != {"wardrobe"}:
        error = error or _style_this_registration_error(
            "INSUFFICIENT_WARDROBE",
            "Style This can only be completed from wardrobe pieces.",
        )

    registration: Dict[str, Any]
    if error or not source_policy:
        registration = {
            "ok": False,
            "error": error
            or _style_this_registration_error(
                "BOARD_REGISTRATION_INVALID",
                "The board source policy could not be determined.",
            ),
        }
    else:
        registration = register_board(
            board_id=board_id,
            revision=1,
            scenario="style_this",
            anchor_item_id=anchor_id,
            source_policy=source_policy,
            allow_wardrobe_fallback=False,
            occasion=occasion,
            style_direction=str(direction.get("title") or ""),
            style_strategy=(
                direction.get("style_strategy")
                if isinstance(direction.get("style_strategy"), dict)
                else None
            ),
            items=canonical_items,
            styling_note=_txt(direction.get("styling_note")) or None,
            user_id=user_id,
        )

    direction["board_id"] = board_id
    direction["revision"] = 1
    direction["scenario"] = "style_this"
    direction["interaction_mode"] = "style_this"
    direction["source_policy"] = source_policy or None
    direction["anchor_item_id"] = anchor_id or None
    direction["originating_item_id"] = anchor_id or None
    direction["board_items"] = canonical_items
    direction["items"] = canonical_items
    direction["shuffle_available"] = bool(registration.get("ok"))
    direction["can_shuffle"] = bool(registration.get("ok"))
    if not registration.get("ok"):
        direction["shuffle_state_error"] = registration.get("error")

    user_fingerprint = hashlib.sha256(str(user_id or "").encode("utf-8")).hexdigest()[:12]
    logger.info(
        "AHVI_STYLE_THIS_BOARD_REGISTERED user_id=%s anchor_item_id=%s board_id=%s "
        "revision=1 source_policy=%s item_count=%s locked_count=%s shuffle_available=%s",
        user_fingerprint,
        anchor_id,
        board_id,
        source_policy or "unknown",
        len(canonical_items),
        sum(1 for item in canonical_items if item.get("locked")),
        str(bool(registration.get("ok"))).lower(),
    )
    return direction


@router.post("/items/{item_id}/style")
def style_wardrobe_item(
    item_id: str, request: ItemStyleRequest, http_request: Request = None
) -> Dict[str, Any]:
    """Power the item-detail CTAs (fast lite-pairing path).

    mode=style_this -> 3 editorial styling directions.
    mode=build_outfit -> 1 practical outfit anchored on the item.
    Never raises: returns a friendly fallback so the UI never dead-ends.
    """
    if http_request is not None:
        user_id = enforce_owner(http_request, request.user_id)
        requested_anchor_id = _txt(request.anchor_item_id or item_id)
        if requested_anchor_id != _txt(item_id):
            raise HTTPException(
                status_code=400,
                detail="anchor_item_id must match the selected item path.",
            )
        proxy = AppwriteProxy()
        try:
            wardrobe = _list_all_documents(proxy, "outfits", user_id=user_id)
        except Exception:
            wardrobe = []
        anchor = next(
            (dict(item) for item in wardrobe if _item_id_of(item) == _txt(item_id)),
            None,
        )
        if anchor is None:
            raise HTTPException(status_code=404, detail="Wardrobe item not found")
        anchor.setdefault("source", "wardrobe")
        request = request.model_copy(
            update={
                "user_id": user_id,
                "wardrobe": wardrobe,
                "anchor_item": anchor,
            }
        )

    mode = _txt(request.mode).lower()
    scenario = _txt(request.scenario).lower()
    if http_request is not None and (
        mode == "build_outfit" or scenario == "build_outfit"
    ):
        return _legacy_build_outfit_cta_failure()
    if mode not in {"build_outfit", "style_this"}:
        mode = "build_outfit"
    wardrobe = _resolve_wardrobe(request)
    if mode == "style_this":
        wardrobe = [
            item
            for item in wardrobe
            if canonical_item_source(item) in {"", "unknown", "wardrobe"}
        ]
    anchor = (
        _resolve_style_this_anchor(
            request,
            item_id,
            wardrobe,
            allow_legacy_metadata=http_request is None,
        )
        if mode == "style_this"
        else _resolve_anchor(request, item_id, wardrobe)
    )
    if mode == "style_this" and not anchor:
        return _style_this_failure(
            {},
            "STYLE_THIS_ANCHOR_UNAVAILABLE",
            "The selected wardrobe item could not be verified. Refresh your wardrobe and try again.",
        )
    # Explicit Style This anchor boundary: the user chose THIS item, so an
    # unrenderable image must never be silently swapped for another garment
    # or shown as-is. Fail with a typed error instead - the item is still
    # fully usable everywhere else (wardrobe view, Try-On, etc.), just not as
    # a Style This anchor until it has a real processed image.
    # Gated on http_request (the real production path) so direct test/legacy
    # callers keep the same allow_missing_image leniency
    # _resolve_style_this_anchor already grants them.
    #
    # Checked against the RAW authoritative wardrobe row, not the transformed
    # `anchor` dict: canonical_style_this_anchor() deliberately sets
    # anchor["image_url"] = safe_image_url (whatever field won), which would
    # make image_url equal normalized_url/masked_url on an already-safe item
    # and false-positive as a "raw alias" under this same check.
    if mode == "style_this" and http_request is not None:
        _anchor_row = next(
            (item for item in wardrobe if _item_id_of(item) == _txt(anchor.get("item_id"))),
            anchor,
        )
        if not is_board_renderable(_anchor_row):
            return _style_this_failure(
                anchor,
                "ANCHOR_IMAGE_NOT_BOARD_READY",
                "This item's photo needs to be reprocessed before Style This can use it.",
            )
    resolved = resolve_location_weather_context(
        user_id=request.user_id,
        request_data={**request.model_dump(exclude_none=True), **request.context},
        profile=request.user_profile,
    )
    strategies: List[Dict[str, Any]] = []

    try:
        if mode == "style_this":
            profile_dna = (
                request.user_profile.get("style_dna")
                if isinstance(request.user_profile.get("style_dna"), dict)
                else {}
            )
            resolver_context = {
                **dict(request.context or {}),
                "occasion": request.occasion or request.context.get("occasion") or "daily",
                "style_dna": request.context.get("style_dna") or profile_dna,
                "gender": request.user_profile.get("style_gender") or request.user_profile.get("gender") or "unknown",
            }
            strategies = resolve_style_archetypes(
                resolver_context, anchor_item=anchor, direction_count=3
            )
            directions = _lite_directions(
                anchor, wardrobe, resolved["weather"], strategies=strategies
            )
            directions = [
                direction
                for direction in directions
                if is_complete_board(direction.get("items"))
            ]
            if not directions:
                return _style_this_failure(
                    anchor,
                    "INSUFFICIENT_WARDROBE",
                    "The available wardrobe cannot form a complete Style This look.",
                )
            directions = [
                _register_style_this_direction(
                    direction,
                    anchor=anchor,
                    wardrobe=wardrobe,
                    user_id=request.user_id,
                    occasion=request.occasion,
                )
                for direction in directions
            ]
            registration_error = next(
                (
                    direction.get("shuffle_state_error")
                    for direction in directions
                    if (direction.get("shuffle_state_error") or {}).get("code")
                    in {"INSUFFICIENT_WARDROBE", "SOURCE_POLICY_VIOLATION"}
                ),
                None,
            )
            if registration_error:
                return _style_this_failure(
                    anchor,
                    registration_error["code"],
                    registration_error["message"],
                )
            logger.info(
                "stylist.item_style mode=style_this item_id=%s directions=%d wardrobe=%d",
                item_id, len(directions), len(wardrobe),
            )
            return {
                "success": True,
                "mode": mode,
                "anchor_item": anchor,
                "anchor_item_id": canonical_item_id(anchor),
                "style_directions": directions,
                "context_usage": resolved["context_usage"],
            }

        outfit = _lite_build_outfit(
            anchor,
            wardrobe,
            request.occasion,
            weather=resolved["weather"],
        )
        logger.info(
            "stylist.item_style mode=build_outfit item_id=%s items=%d missing=%d wardrobe=%d",
            item_id, len(outfit["items"]), len(outfit["missing_items"]), len(wardrobe),
        )
        return {"success": True, "mode": mode, "anchor_item": anchor, "outfit": outfit, "context_usage": resolved["context_usage"]}
    except Exception as exc:  # noqa: BLE001 - CTA must never dead-end the UI
        logger.exception("stylist.item_style failed item_id=%s mode=%s err=%s", item_id, mode, str(exc))
        if mode == "style_this":
            fallback = _style_fallback(mode, anchor, strategies=strategies)
            fallback["error"] = {
                "code": "STYLE_THIS_GENERATION_FAILED",
                "message": "AHVI could not build a verified Style This board. Please try again.",
            }
            return fallback
        fallback = _style_fallback(mode, anchor, strategies=strategies)
        fallback["context_usage"] = resolved["context_usage"]
        return fallback
