import logging
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from brain.personalization.style_dna_engine import style_dna_engine
from services.auth_helpers import bind_request_user
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.style_flow_service import (
    build_style_flow_response,
    item_role,
    _is_professional_safe,
    _is_office_occasion,
)
from services.style_reasoning_engine import _occasion_display_name
from services.professional_safety import is_professional_occasion
from services.constrained_outfit_builder import (
    ConstrainedOutfitBuilder,
    ConstrainedOutfitError,
    replaceable_slots_for_fixed_items,
)
from services.style_board_shuffle_service import _default_position, register_board
from services.style_execution_policy import (
    run_board_registration,
    server_style_execution,
)
from services.style_item_contract import (
    FixedItemLostError,
    assert_fixed_items_preserved,
    canonical_accessory_type,
    canonical_image_url,
    canonical_item_id,
    canonical_item_role,
)
from services.style_context_service import build_canonical_style_context

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
    user_id: str = ""
    query: str = "What should I wear today?"
    wardrobe: Any = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    include_base64: bool = True
    upload_style_boards_to_r2: bool = False


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@router.post("/pipeline")
@server_style_execution
def run_outfit_pipeline(request: OutfitPipelineRequest, http_request: Request = None):
    # Bind to the authenticated user; a mismatched body user_id is a 403.
    user_id = bind_request_user(http_request, request.user_id)
    appwrite = AppwriteProxy()
    request_context = dict(request.context or {})

    wardrobe = request.wardrobe
    if wardrobe is None:
        try:
            wardrobe = appwrite.list_documents("outfits", user_id=user_id)
        except Exception:
            wardrobe = []

    context = build_canonical_style_context(
        query=request.query,
        user_id=user_id,
        user_profile=request.user_profile or {},
        intent=_dict(request_context.get("agent_orchestration")),
        router_occasion=request_context.get("canonical_occasion") or request_context.get("occasion"),
        weather=request_context.get("weather_context") or request_context.get("weather"),
        event_context=_dict(request_context.get("event_context")),
        style_dna=request_context.get("style_dna"),
        style_mode=request_context.get("style_mode") or request_context.get("mode") or "wardrobe_style",
        style_action=request_context.get("style_action") or request_context.get("action") or "",
        wardrobe_items=wardrobe,
        last_style_context=request_context.get("last_style_context"),
        request_context=request_context,
    )

    try:
        response = build_style_flow_response(
            user_id=user_id,
            query=request.query,
            wardrobe=wardrobe,
            user_profile=context.get("user_profile") or {},
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
            "stylist.pipeline failed user_id=%s error=%s", user_id, str(exc)
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
    user_id: str = ""
    mode: str = "build_outfit"  # "build_outfit" | "style_this"
    occasion: Optional[str] = None
    wardrobe_only: bool = False
    anchor_item: Dict[str, Any] = Field(default_factory=dict)
    wardrobe: Any = None
    style_assets: Any = None
    allow_wardrobe_fallback: bool = False
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


def _resolve_wardrobe(
    request: ItemStyleRequest, user_id: Optional[str] = None
) -> "tuple[List[Dict[str, Any]], bool]":
    # user_id is the bound (authenticated) id from the route; direct callers
    # without an auth context fall back to the request's own user_id.
    uid = user_id if user_id is not None else request.user_id
    if isinstance(request.wardrobe, list):
        return [dict(i) for i in request.wardrobe if isinstance(i, dict)], False
    try:
        rows = AppwriteProxy().list_documents("outfits", user_id=uid) or []
        return [
            {**i, "source": i.get("source") or "wardrobe"}
            for i in rows if isinstance(i, dict)
        ], True
    except Exception:
        return [], True


def _resolve_style_assets(request: ItemStyleRequest) -> "tuple[List[Dict[str, Any]], bool]":
    if isinstance(request.style_assets, list):
        return [dict(i) for i in request.style_assets if isinstance(i, dict)], False
    try:
        rows = AppwriteProxy().list_documents("style_assets") or []
        return [
            {**i, "source": i.get("source") or "style_asset"}
            for i in rows if isinstance(i, dict)
        ], True
    except Exception:
        return [], True


def _resolve_anchor(
    request: ItemStyleRequest,
    item_id: str,
    wardrobe: List[Dict[str, Any]],
    wardrobe_trusted: bool = False,
) -> Optional[Dict[str, Any]]:
    """Resolve the fixed anchor for the direct item CTA.

    Trusted wardrobe (server-resolved from the authenticated user's Appwrite
    collection): the SERVER record is the authority for identity, ownership
    and source - the client anchor_item may only supply display/context
    fields it does not override.  A requested item_id absent from the
    authenticated wardrobe returns None (INVALID_ANCHOR_ITEM), even when the
    caller claims source="wardrobe".

    Untrusted wardrobe (caller-inlined, tests/offline): the caller owns the
    payload, so the anchor passes through unchanged - missing/unknown source
    stays unknown and is rejected downstream by the constrained builder.
    """
    client_anchor = (
        dict(request.anchor_item)
        if isinstance(request.anchor_item, dict) and request.anchor_item
        else {}
    )
    wanted_id = _txt(item_id) or canonical_item_id(client_anchor)
    server_record = next(
        (dict(i) for i in wardrobe if canonical_item_id(i) == wanted_id), None
    )

    if not wardrobe_trusted:
        if client_anchor:
            return client_anchor
        return server_record if server_record else {"item_id": _txt(item_id)}

    if server_record is None:
        return None  # not in the authenticated wardrobe -> INVALID_ANCHOR_ITEM
    # Client fields fill display gaps only; the server record wins on every
    # overlapping key, then trusted provenance is stamped at this boundary.
    merged = {**client_anchor, **server_record}
    merged["source"] = "wardrobe"
    merged["owned"] = True
    return merged


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
    request_context = dict(request.context or {})
    request_context["anchor_item_id"] = _item_id_of(request.anchor_item) or None
    context = build_canonical_style_context(
        query=query,
        user_id=request.user_id,
        user_profile=request.user_profile or {},
        router_occasion=request.occasion,
        weather=request_context.get("weather_context") or request_context.get("weather"),
        event_context=_dict(request_context.get("event_context")),
        style_dna=request_context.get("style_dna"),
        style_mode="wardrobe_style",
        style_action=request.mode,
        wardrobe_items=wardrobe,
        request_context=request_context,
    )
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
    fixed = {
        "office": "Office-Ready Look",
        "date": "Date Night Look",
        "casual": "Casual Day Look",
    }
    if occ in fixed:
        return fixed[occ]
    if occ:
        # "client_meeting" -> "Client Meeting Look", never the raw slug.
        return f"{_occasion_display_name(occ)} Look"
    return "Casual Day Look"


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

_LITE_STYLE_DIRECTIONS = [
    ("Casual Brunch", ("sneaker", "flat", "loafer", "mule"), "Relaxed and easy — let the piece breathe."),
    ("Date Night", ("heel", "sandal", "boot", "pump"), "A touch sharper for the evening."),
    ("Vacation Day", ("sandal", "flat", "sneaker", "espadrille"), "Light, breezy, low-effort."),
]

# Occasions on which anchor-item safety is enforced (same set as
# _PROFESSIONAL_OCCASIONS in style_flow_service.py).
_LITE_PROFESSIONAL_OCCASIONS = frozenset(
    {
        "office", "client_meeting", "client meeting", "corporate_office",
        "corporate office", "interview", "business_formal", "business formal",
        "business_casual", "business casual", "presentation", "startup_office",
    }
)


def _is_lite_professional(occasion: Optional[str]) -> bool:
    return is_professional_occasion(occasion)


def _anchor_safe_for_occasion(anchor: Dict[str, Any], occasion: Optional[str]) -> tuple:
    """Return (is_safe: bool, reason: str).

    Calls the shared _is_professional_safe checker from style_flow_service.
    Always returns (True, "") for non-professional occasions so the lite path
    is unaffected for casual / date / party etc.
    """
    if not _is_lite_professional(occasion):
        return True, ""
    return _is_professional_safe(anchor, occasion or "")


def _filter_wardrobe_for_occasion(
    wardrobe: List[Dict[str, Any]], occasion: Optional[str]
) -> List[Dict[str, Any]]:
    """Drop unsafe items from wardrobe before pairing for professional occasions."""
    if not _is_lite_professional(occasion):
        return wardrobe
    safe = []
    for item in wardrobe:
        ok, _ = _is_professional_safe(item, occasion or "")
        if ok:
            safe.append(item)
    return safe


def _lite_role(item: Dict[str, Any]) -> str:
    # DEPRECATED from live paths: the live CTA flow now uses
    # services.style_item_contract.canonical_item_role (which has the
    # "dress shirt"->top / "shorts"-token / blazer->outerwear fixes).  This
    # function remains only for the _lite_* fallback path below.
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
    return _txt(
        item.get("normalized_url") or item.get("normalizedUrl")
        or item.get("masked_url") or item.get("maskedUrl")
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
    }


def _lite_group(wardrobe: List[Dict[str, Any]], exclude_id: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {"dress": [], "top": [], "bottom": [], "footwear": [], "accessory": []}
    for item in wardrobe:
        if not isinstance(item, dict) or _item_id_of(item) == exclude_id:
            continue
        groups.setdefault(_lite_role(item), []).append(item)
    return groups


def _lite_pick(
    groups: Dict[str, List[Dict[str, Any]]], role: str, is_dress: bool, prefer=(), variant: int = 0
) -> Optional[Dict[str, Any]]:
    cands = list(groups.get(role) or [])
    if is_dress and role == "footwear":
        cands = [c for c in cands if not _is_bad_dress_footwear(c)]
    if not cands:
        return None
    if prefer:
        def _score(c: Dict[str, Any]) -> int:
            blob = " ".join(_txt(c.get(k)) for k in ("name", "label", "sub_category", "subcategory")).lower()
            return 0 if any(p in blob for p in prefer) else 1
        cands.sort(key=_score)
        return cands[0]
    # No preference (tops/bottoms/accessories): rotate by variant so different
    # directions show different owned pieces when more than one exists.
    return cands[variant % len(cands)]


def _lite_missing(role: str, is_dress: bool) -> Dict[str, Any]:
    if role == "footwear" and is_dress:
        return dict(_DRESS_FOOTWEAR_SUGGESTIONS[0])
    return dict(_LITE_MISSING.get(role, _LITE_MISSING["accessory"]))


def _lite_needed_slots(anchor_role: str) -> List[str]:
    if anchor_role == "dress":
        return ["footwear", "accessory"]
    if anchor_role == "top":
        return ["bottom", "footwear", "accessory"]
    if anchor_role == "bottom":
        return ["top", "footwear", "accessory"]
    if anchor_role == "footwear":
        return ["top", "bottom", "accessory"]
    return ["dress", "footwear"]  # accessory anchor -> hero garment + shoes


def _lite_build_outfit(
    anchor: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
    occasion: Optional[str],
    *,
    title: Optional[str] = None,
    prefer=(),
    note: str = "",
    variant: int = 0,
) -> Dict[str, Any]:
    # For professional occasions, filter supporting items so only safe pieces
    # are considered as pairing candidates.  The anchor safety check is handled
    # by the caller (style_wardrobe_item) before _lite_build_outfit is invoked.
    safe_wardrobe = _filter_wardrobe_for_occasion(wardrobe, occasion)
    is_dress = _anchor_is_dress(anchor) or _lite_role(anchor) == "dress"
    anchor_role = "dress" if is_dress else _lite_role(anchor)
    groups = _lite_group(safe_wardrobe, _item_id_of(anchor))
    items = [_lite_item(anchor, "hero")]
    missing: List[Dict[str, Any]] = []
    for slot in _lite_needed_slots(anchor_role):
        pick = _lite_pick(
            groups, slot, is_dress,
            prefer=prefer if slot == "footwear" else (),
            variant=variant,
        )
        if pick:
            items.append(_lite_item(pick, "accent" if slot == "accessory" else "support"))
            groups[slot] = [g for g in groups[slot] if _item_id_of(g) != _item_id_of(pick)]
        else:
            missing.append(_lite_missing(slot, is_dress))
    return {
        "title": title or _outfit_title(occasion),
        "items": items,
        "missing_items": missing,
        "reason": note or "Built from pieces you already own, anchored on this item.",
    }


def _lite_directions(anchor: Dict[str, Any], wardrobe: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    directions = []
    for idx, (title, prefer, note) in enumerate(_LITE_STYLE_DIRECTIONS):
        look = _lite_build_outfit(
            anchor, wardrobe, None, title=title, prefer=prefer, note=note, variant=idx
        )
        directions.append({
            "title": title,
            "items": look["items"],
            "missing_items": look["missing_items"],
            "styling_note": note,
        })
    return directions


# --------------------------------------------------------------------------
# Constrained-builder path (live). Legacy lite helpers above are retained for
# compatibility with direct imports, but are not reachable from this endpoint.
# --------------------------------------------------------------------------
_builder = ConstrainedOutfitBuilder()


def _builder_item_to_response(item: Dict[str, Any]) -> Dict[str, Any]:
    slot = canonical_item_role(item)
    image_url = str(item.get("image_url") or "").strip() or canonical_image_url(item)
    response = {
        "item_id": canonical_item_id(item),
        "id": item.get("id") or canonical_item_id(item),
        "name": item.get("name") or "Item",
        "category": item.get("category", ""),
        "sub_category": item.get("sub_category", ""),
        "image_url": image_url,
        "role": slot,
        "slot": slot,
        "board_role": "hero" if item.get("locked") else (
            "accent" if slot == "accessory" else "support"
        ),
        "source": item.get("source", "unknown"),
        "owned": bool(item.get("owned")),
        "locked": bool(item.get("locked")),
    }
    if slot == "accessory":
        response["accessory_type"] = item.get("accessory_type") or canonical_accessory_type(item)
    for key in (
        "masked_url", "board_image_url", "normalized_url", "position",
        "x", "y", "width", "height", "scale", "z", "rotation",
    ):
        value = item.get(key)
        if value not in (None, ""):
            response[key] = value
    return response


def _raise_builder_failure(result: Dict[str, Any]) -> None:
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    raise ConstrainedOutfitError(
        str(error.get("code") or "CONSTRAINED_BUILDER_FAILED"),
        str(error.get("message") or "AHVI could not safely complete this look."),
        error,
    )


def _builder_outfit(
    anchor: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
    style_assets: List[Dict[str, Any]],
    occasion: Optional[str],
    *,
    mode: str = "build_outfit",
    title: Optional[str] = None,
    prefer=(),
    note: str = "",
    variant: int = 0,
    allow_wardrobe_fallback: bool = False,
    exclude_item_ids: "tuple[str, ...]" = (),
    user_id: str = "",
) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    fixed = [dict(anchor)] if anchor else []
    replaceable_slots = replaceable_slots_for_fixed_items(fixed)
    if mode == "style_this":
        allowed_sources = ["style_asset"]
        candidates = list(style_assets)
        if allow_wardrobe_fallback:
            allowed_sources.append("wardrobe")
            candidates.extend(wardrobe)
        policy = {"allowed_completion_sources": allowed_sources}
        board_policy = "mixed" if allow_wardrobe_fallback else "style_asset"
        source_meta = {
            "completion_source": board_policy,
            "source_policy": board_policy,
            "allow_wardrobe_fallback": allow_wardrobe_fallback,
            "wardrobe_fallback_enabled": allow_wardrobe_fallback,
        }
    else:
        policy = {"allowed_completion_sources": ["wardrobe"]}
        candidates = list(wardrobe)
        board_policy = "wardrobe"
        source_meta = {
            "completion_source": "wardrobe",
            "source_policy": "wardrobe",
            "allow_wardrobe_fallback": False,
            "wardrobe_fallback_enabled": False,
        }
    result = _builder.generate(
        scenario=mode,
        fixed_items=fixed,
        replaceable_slots=replaceable_slots,
        exclude_item_ids=list(exclude_item_ids),
        source_policy=policy,
        context={
            "wardrobe": candidates,
            "occasion": occasion,
            "variant": variant,
            "prefer_footwear": tuple(prefer),
        },
    )
    if not result.get("success"):
        _raise_builder_failure(result)
    is_dress = _anchor_is_dress(anchor) or canonical_item_role(anchor) == "dress"
    serialized_items = [_builder_item_to_response(i) for i in result.get("items", [])]
    for item in serialized_items:
        if not isinstance(item.get("position"), dict):
            item["position"] = _default_position(item["item_id"], item["slot"])
    try:
        assert_fixed_items_preserved(fixed, serialized_items, stage="stylist_response")
    except FixedItemLostError as exc:
        raise ConstrainedOutfitError(
            "FIXED_ITEM_LOST",
            "The selected item could not be preserved exactly in the look.",
            {"fixed_item_ids": exc.missing_ids, "mismatched_fields": exc.mismatched_fields},
        ) from exc
    outfit = {
        "board_id": str(uuid4()),
        "revision": 1,
        "scenario": mode,
        "source_policy": board_policy,
        "allow_wardrobe_fallback": allow_wardrobe_fallback if mode == "style_this" else False,
        "title": title or _outfit_title(occasion),
        "items": serialized_items,
        "missing_items": [
            _lite_missing(slot, is_dress) for slot in result.get("missing_items", [])
        ],
        "reason": note or "Built from pieces you already own, anchored on this item.",
    }
    # Durably persist the board contract so a later shuffle can resolve
    # "inherit" from stored state instead of guessing from locked-item
    # sources. If state storage is down the board is still returned, but
    # shuffle is explicitly unavailable (typed error, never a silent
    # in-memory fallback).
    registration = run_board_registration(
        register_board,
        board_id=outfit["board_id"],
        revision=1,
        scenario=mode,
        source_policy=board_policy,
        allow_wardrobe_fallback=outfit["allow_wardrobe_fallback"],
        occasion=occasion,
        style_direction=title,
        items=serialized_items,
        user_id=user_id,
    )
    outfit["shuffle_available"] = bool(registration.get("ok"))
    if not registration.get("ok"):
        outfit["shuffle_state_error"] = registration.get("error")
    return outfit, source_meta


def _builder_directions(
    anchor: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
    style_assets: List[Dict[str, Any]],
    *,
    allow_wardrobe_fallback: bool = False,
    user_id: str = "",
) -> "tuple[List[Dict[str, Any]], Dict[str, Any]]":
    directions = []
    source_meta: Dict[str, Any] = {}
    for idx, (title, prefer, note) in enumerate(_LITE_STYLE_DIRECTIONS):
        look, source_meta = _builder_outfit(
            anchor, wardrobe, style_assets, None,
            mode="style_this", title=title, prefer=prefer, note=note, variant=idx,
            allow_wardrobe_fallback=allow_wardrobe_fallback,
            user_id=user_id,
        )
        directions.append({**look, "styling_note": note})
    return directions, source_meta


def _anchor_in_items(anchor: Dict[str, Any], items: List[Dict[str, Any]]) -> bool:
    aid = canonical_item_id(anchor)
    if not aid:
        return False
    return any(canonical_item_id(i) == aid for i in items)


def _typed_failure(
    mode: str,
    anchor: Dict[str, Any],
    code: str,
    message: str,
    response_meta: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    error = {"code": code, "message": message}
    error.update(details or {})
    response: Dict[str, Any] = {
        "success": False,
        "mode": mode,
        "anchor_item": anchor,
        "error": error,
        "message": message,
        **(response_meta or {}),
    }
    if mode == "style_this":
        response["style_directions"] = []
    else:
        response["outfit"] = {"title": _outfit_title((response_meta or {}).get("occasion")), "items": [], "missing_items": []}
    return response


@router.post("/items/{item_id}/style")
@server_style_execution
def style_wardrobe_item(
    item_id: str, request: ItemStyleRequest, http_request: Request = None
) -> Dict[str, Any]:
    """Power direct item actions using hard constrained-generation rules."""
    # Bind to the authenticated user; a mismatched body user_id is a 403 and
    # all server-side wardrobe reads use the authenticated id only.
    user_id = bind_request_user(http_request, request.user_id)
    mode = _txt(request.mode).lower()
    if mode not in {"build_outfit", "style_this"}:
        mode = "build_outfit"
    wardrobe, wardrobe_trusted = _resolve_wardrobe(request, user_id)
    style_assets, style_assets_trusted = _resolve_style_assets(request)
    anchor = _resolve_anchor(request, item_id, wardrobe, wardrobe_trusted)
    if anchor is None:
        # Trusted wardrobe resolved, but the requested item is not in the
        # authenticated user's collection - caller-claimed source is never
        # trusted without this server verification.
        client_anchor = request.anchor_item if isinstance(request.anchor_item, dict) else {}
        return _typed_failure(
            mode, dict(client_anchor), "INVALID_ANCHOR_ITEM",
            "This item is not in your wardrobe, so it cannot anchor a look.",
        )

    anchor_id = canonical_item_id(anchor)
    if not anchor_id:
        return _typed_failure(
            mode, anchor, "INVALID_ITEM_ID",
            "The selected item does not have a stable item ID.",
        )

    # --- Professional anchor safety check ---
    occasion = _txt(request.occasion) or None
    anchor_safe, anchor_block_reason = _anchor_safe_for_occasion(anchor, occasion)
    is_professional = _is_lite_professional(occasion)

    occasion_display = _occasion_display_name(occasion) if occasion else ""

    # Response metadata fields (contract addition)
    response_meta: Dict[str, Any] = {
        "occasion": occasion,
        "occasion_label": occasion_display or None,
        "formality": "professional" if is_professional else "casual",
        "source": "style_asset" if mode == "style_this" else "wardrobe",
        "wardrobe_source_trusted": wardrobe_trusted,
        "style_asset_source_trusted": style_assets_trusted,
        "validation_passed": anchor_safe or not is_professional,
        "anchor_blocked": not anchor_safe and is_professional,
        "anchor_included": anchor_safe or not is_professional,
    }
    if not anchor_safe and is_professional:
        response_meta["anchor_block_reason"] = anchor_block_reason

    if not anchor_safe and is_professional:
        if mode == "build_outfit":
            # Safe-alternative mode: the professional guard rejected the
            # anchor, so generation is explicitly UN-anchored — the unsafe
            # item is excluded, the original anchor is returned only as
            # reference metadata, and the fixed-item guarantee is not
            # claimed (there is no fixed item).
            try:
                alt_outfit, source_meta = _builder_outfit(
                    {}, wardrobe, style_assets, occasion,
                    mode="build_outfit",
                    note=(
                        "Built a professional-safe look from your wardrobe; "
                        "the selected item was not suitable for this occasion."
                    ),
                    exclude_item_ids=(anchor_id,),
                    user_id=user_id,
                )
            except ConstrainedOutfitError as exc:
                # No safe complete alternative — typed failure that carries
                # why the anchor was rejected.
                return _typed_failure(
                    mode, anchor, "ANCHOR_INCOMPATIBLE_WITH_OCCASION",
                    "This selected item is not suitable for the requested occasion.",
                    response_meta,
                    {"anchor_block_reason": anchor_block_reason, "alternative_error": exc.code},
                )
            response_meta.update(source_meta)
            # The returned alternative itself passed professional validation.
            response_meta["validation_passed"] = True
            response_meta["anchor_included"] = False
            alt_outfit["safe_alternative"] = True
            logger.info(
                "stylist.item_style mode=build_outfit item_id=%s anchor_blocked=True "
                "safe_alternative items=%d missing=%d reason=%s",
                item_id, len(alt_outfit.get("items") or []),
                len(alt_outfit.get("missing_items") or []), anchor_block_reason,
            )
            return {
                "success": True,
                "mode": mode,
                "anchor_item": anchor,  # reference metadata only, not in items
                "outfit": alt_outfit,
                **response_meta,
            }
        return _typed_failure(
            mode, anchor, "ANCHOR_INCOMPATIBLE_WITH_OCCASION",
            "This selected item is not suitable for the requested occasion.",
            response_meta,
            {"anchor_block_reason": anchor_block_reason},
        )

    try:
        if mode == "style_this":
            directions, source_meta = _builder_directions(
                anchor,
                wardrobe,
                style_assets,
                allow_wardrobe_fallback=bool(request.allow_wardrobe_fallback),
                user_id=user_id,
            )
            response_meta.update(source_meta)
            if not all(_anchor_in_items(anchor, d.get("items") or []) for d in directions):
                raise ConstrainedOutfitError(
                    "FIXED_ITEM_LOST",
                    "The selected item could not be preserved in the look.",
                    {"fixed_item_ids": [anchor_id]},
                )
            logger.info(
                "stylist.item_style mode=style_this item_id=%s directions=%d wardrobe=%d anchor_blocked=%s",
                item_id, len(directions), len(wardrobe), response_meta["anchor_blocked"],
            )
            return {
                "success": True,
                "mode": mode,
                "anchor_item": anchor,
                "style_directions": directions,
                **response_meta,
            }

        outfit, source_meta = _builder_outfit(
            anchor, wardrobe, style_assets, occasion, mode="build_outfit",
            user_id=user_id,
        )
        response_meta.update(source_meta)
        if not _anchor_in_items(anchor, outfit.get("items") or []):
            raise ConstrainedOutfitError(
                "FIXED_ITEM_LOST",
                "The selected item could not be preserved in the outfit.",
                {"fixed_item_ids": [anchor_id]},
            )
        logger.info(
            "stylist.item_style mode=build_outfit item_id=%s items=%d missing=%d wardrobe=%d anchor_blocked=%s",
            item_id, len(outfit.get("items") or []), len(outfit.get("missing_items") or []),
            len(wardrobe), response_meta["anchor_blocked"],
        )
        return {
            "success": True,
            "mode": mode,
            "anchor_item": anchor,
            "outfit": outfit,
            **response_meta,
        }
    except ConstrainedOutfitError as exc:
        return _typed_failure(mode, anchor, exc.code, exc.message, response_meta, exc.details)
    except Exception as exc:  # noqa: BLE001
        logger.exception("stylist.item_style constrained builder failed item_id=%s mode=%s err=%s", item_id, mode, str(exc))
        return _typed_failure(
            mode, anchor, "CONSTRAINED_BUILDER_FAILED",
            "AHVI could not safely complete this look.", response_meta,
        )
