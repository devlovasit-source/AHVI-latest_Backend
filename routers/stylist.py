import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain.personalization.style_dna_engine import style_dna_engine
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.style_flow_service import (
    build_style_flow_response,
    item_role,
    _is_professional_safe,
    _is_office_occasion,
)

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
    if not occasion:
        return False
    key = str(occasion).strip().lower().replace(" ", "_")
    return key in {o.strip().lower().replace(" ", "_") for o in _LITE_PROFESSIONAL_OCCASIONS}


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


@router.post("/items/{item_id}/style")
def style_wardrobe_item(item_id: str, request: ItemStyleRequest) -> Dict[str, Any]:
    """Power the item-detail CTAs (fast lite-pairing path).

    mode=style_this -> 3 editorial styling directions.
    mode=build_outfit -> 1 practical outfit anchored on the item.
    Never raises: returns a friendly fallback so the UI never dead-ends.

    Professional occasion guardrail (new):
    - If occasion is professional (client_meeting, office, interview, etc.) and
      the anchor item is unsafe for that occasion (camo, shiny gold shirt, etc.),
      the anchor is NOT forced into the outfit.  Instead we return a fallback
      outfit from safe wardrobe items and signal anchor_blocked=True so the UI
      can surface a helpful message.
    """
    mode = _txt(request.mode).lower()
    if mode not in {"build_outfit", "style_this"}:
        mode = "build_outfit"
    wardrobe = _resolve_wardrobe(request)
    anchor = _resolve_anchor(request, item_id, wardrobe)

    # --- Professional anchor safety check ---
    occasion = _txt(request.occasion) or None
    anchor_safe, anchor_block_reason = _anchor_safe_for_occasion(anchor, occasion)
    is_professional = _is_lite_professional(occasion)

    # Response metadata fields (contract addition)
    response_meta: Dict[str, Any] = {
        "occasion": occasion,
        "formality": "professional" if is_professional else "casual",
        "source": "wardrobe",
        "validation_passed": anchor_safe or not is_professional,
        "anchor_blocked": not anchor_safe and is_professional,
    }
    if not anchor_safe and is_professional:
        response_meta["anchor_block_reason"] = anchor_block_reason

    try:
        # style_this mode: always built from wardrobe (occasion-aware filtering
        # happens inside _lite_build_outfit via safe_wardrobe), anchor included
        # only if it passes professional check.
        if mode == "style_this":
            if anchor_safe or not is_professional:
                directions = _lite_directions(anchor, wardrobe)
            else:
                # Build directions from safe wardrobe, without anchoring on the
                # blocked item.  Use the first safe owned item if one exists.
                safe_wardrobe = _filter_wardrobe_for_occasion(wardrobe, occasion)
                safe_alt = next(
                    (i for i in safe_wardrobe if _item_id_of(i) != _item_id_of(anchor)),
                    None,
                )
                if safe_alt:
                    directions = _lite_directions(safe_alt, safe_wardrobe)
                else:
                    directions = [
                        {
                            "title": t,
                            "items": [],
                            "missing_items": [{"label": "Office-ready pieces", "reason": "Add professional wardrobe items to get styling suggestions.", "cta": "Find this"}],
                            "styling_note": f"This item works better for party or evening occasions. I don't see enough {occasion} pieces in your wardrobe yet.",
                        }
                        for t, _, _ in _LITE_STYLE_DIRECTIONS
                    ]
                anchor_name = _txt(anchor.get("name") or anchor.get("label")) or "This item"
                for d in directions:
                    d["_anchor_note"] = f"{anchor_name} works better for party/evening occasions. Here's a {occasion} look from your wardrobe instead."
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

        # build_outfit mode
        if anchor_safe or not is_professional:
            outfit = _lite_build_outfit(anchor, wardrobe, occasion)
        else:
            # Anchor is unsafe for the professional occasion — do not force it.
            # Build an outfit from safe wardrobe items instead.
            safe_wardrobe = _filter_wardrobe_for_occasion(wardrobe, occasion)
            safe_alt = next(
                (i for i in safe_wardrobe if _item_id_of(i) != _item_id_of(anchor)),
                None,
            )
            if safe_alt:
                outfit = _lite_build_outfit(safe_alt, safe_wardrobe, occasion)
                anchor_name = _txt(anchor.get("name") or anchor.get("label")) or "This item"
                outfit["_anchor_note"] = (
                    f"{anchor_name} works better for party/evening occasions. "
                    f"Here's a {occasion} look from your wardrobe instead."
                )
            else:
                outfit = {
                    "title": _outfit_title(occasion),
                    "items": [],
                    "missing_items": [
                        {
                            "label": "Office-ready pieces",
                            "reason": "I don't see enough client-meeting pieces in your wardrobe yet.",
                            "cta": "Find this",
                        }
                    ],
                    "reason": f"I don't see enough {occasion} pieces in your wardrobe yet.",
                }
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
    except Exception as exc:  # noqa: BLE001 - CTA must never dead-end the UI
        logger.exception("stylist.item_style failed item_id=%s mode=%s err=%s", item_id, mode, str(exc))
        return _style_fallback(mode, anchor)
