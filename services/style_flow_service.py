import base64
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

try:
    from services.r2_storage import R2Storage, R2StorageError
except Exception:  # pragma: no cover - optional deploy dependency
    R2Storage = None
    R2StorageError = Exception

logger = logging.getLogger("ahvi.style_flow")


STYLE_ACTION_CHIPS = ["More looks", "Next best options", "Try different shoes"]


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: Any) -> set[str]:
    import re

    return set(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def item_image(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _safe_text(
        item.get("masked_url")
        or item.get("maskedUrl")
        or item.get("image_url")
        or item.get("imageUrl")
        or item.get("raw_url")
        or item.get("rawUrl")
        or item.get("url")
        or item.get("image")
    )


def item_key(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _safe_text(
        item.get("$id")
        or item.get("id")
        or item.get("item_id")
        or item.get("itemId")
        or item.get("image_id")
        or item.get("name")
        or item.get("label")
    ).lower()


def item_role(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "unknown"
    blob = " ".join(
        _safe_text(item.get(k))
        for k in (
            "role",
            "slot",
            "type",
            "category",
            "sub_category",
            "subcategory",
            "main_category",
            "name",
            "label",
            "description",
        )
    )
    tokens = _tokens(blob)

    if tokens.intersection(
        {
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "loafer",
            "loafers",
            "boot",
            "boots",
            "sandal",
            "sandals",
            "footwear",
            "slipper",
            "slippers",
            "slider",
            "sliders",
        }
    ):
        return "footwear"
    if tokens.intersection(
        {
            "watch",
            "belt",
            "sunglass",
            "sunglasses",
            "eyewear",
            "bag",
            "bracelet",
            "ring",
            "necklace",
            "earring",
            "jewelry",
            "jewellery",
            "accessory",
            "accessories",
            "cap",
            "hat",
            "scarf",
        }
    ):
        return "accessory"
    if tokens.intersection(
        {
            "top",
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "polo",
            "jacket",
            "blazer",
            "sweater",
            "hoodie",
            "kurta",
            "blouse",
        }
    ):
        return "top"
    if tokens.intersection(
        {
            "bottom",
            "bottoms",
            "pant",
            "pants",
            "trouser",
            "trousers",
            "jean",
            "jeans",
            "chino",
            "chinos",
            "shorts",
            "skirt",
        }
    ):
        return "bottom"
    if tokens.intersection({"dress", "dresses", "saree", "sari", "lehenga", "gown"}):
        return "dress"
    return "unknown"


def normalize_item(item: Dict[str, Any], role: Optional[str] = None) -> Dict[str, Any]:
    row = dict(item or {})
    resolved = role or item_role(row)
    if resolved in {"top", "bottom", "dress", "footwear", "accessory"}:
        row["role"] = resolved
        row["slot"] = resolved
    if resolved == "top":
        row.setdefault("category", "Tops")
    elif resolved == "bottom":
        row.setdefault("category", "Bottoms")
    elif resolved == "dress":
        row.setdefault("category", "Dresses")
    elif resolved == "footwear":
        row.setdefault("category", "Footwear")
    elif resolved == "accessory":
        row.setdefault("category", "Accessories")

    image = item_image(row)
    if image:
        row.setdefault("image_url", image)
        row.setdefault("imageUrl", image)
        row.setdefault("masked_url", image)
        row.setdefault("maskedUrl", image)
    return row


def card_signature(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    parts = []
    for item in _card_items(card, include_slots=True):
        key = item_key(item)
        if key:
            parts.append(key)
    if parts:
        return "|".join(sorted(set(parts)))
    return _safe_text(card.get("id") or card.get("title") or card.get("name")).lower()


def core_card_signature(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    by_role: Dict[str, str] = {}
    for item in _card_items(card, include_slots=True):
        role = item_role(item)
        if role not in {"top", "bottom", "dress", "footwear"}:
            continue
        key = item_key(item)
        if key and role not in by_role:
            by_role[role] = key
    if "dress" in by_role:
        parts = [by_role.get("dress", ""), by_role.get("footwear", "")]
    else:
        parts = [
            by_role.get("top", ""),
            by_role.get("bottom", ""),
            by_role.get("footwear", ""),
        ]
    clean = [p for p in parts if p]
    return "|".join(clean)


def _card_items(card: Dict[str, Any], *, include_slots: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in ("items", "accessories"):
        value = card.get(key)
        if isinstance(value, list):
            out.extend([dict(x) for x in value if isinstance(x, dict)])
    if include_slots:
        for key in ("top", "bottom", "dress", "shoes", "footwear", "outerwear"):
            value = card.get(key)
            if isinstance(value, dict):
                out.append(dict(value))
    return out


def card_is_complete(card: Dict[str, Any]) -> bool:
    roles = {item_role(item) for item in _card_items(card, include_slots=True)}
    if "dress" in roles and "footwear" in roles:
        return True
    return {"top", "bottom", "footwear"}.issubset(roles)


def _accessory_type(item: Dict[str, Any]) -> str:
    blob = " ".join(_safe_text(item.get(k)) for k in ("name", "label", "category", "type", "sub_category"))
    tokens = _tokens(blob)
    for typ, words in (
        ("watch", {"watch", "watches"}),
        ("eyewear", {"sunglass", "sunglasses", "eyewear", "glasses"}),
        ("bag", {"bag", "bags", "purse", "tote", "clutch"}),
        ("belt", {"belt", "belts"}),
        ("headwear", {"cap", "hat"}),
        ("scarf", {"scarf", "scarves"}),
        ("jewelry", {"ring", "necklace", "bracelet", "earring", "jewelry", "jewellery"}),
    ):
        if tokens.intersection(words):
            return typ
    return "accessory"


def _canonicalize_card(card: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    raw_items = _card_items(card, include_slots=True)
    if not raw_items:
        return None

    chosen: Dict[str, Dict[str, Any]] = {}
    accessories: List[Dict[str, Any]] = []
    seen_item_keys: set[str] = set()
    seen_accessory_types: set[str] = set()

    for item in raw_items:
        role = item_role(item)
        if role not in {"top", "bottom", "dress", "footwear", "accessory"}:
            continue
        if not item_image(item):
            continue
        key = item_key(item)
        if key and key in seen_item_keys:
            continue
        if role == "accessory":
            typ = _accessory_type(item)
            if typ in seen_accessory_types:
                continue
            accessories.append(normalize_item(item, "accessory"))
            seen_accessory_types.add(typ)
            if key:
                seen_item_keys.add(key)
            continue
        if role not in chosen:
            chosen[role] = normalize_item(item, role)
            if key:
                seen_item_keys.add(key)

    if "dress" in chosen:
        chosen.pop("top", None)
        chosen.pop("bottom", None)

    final_items: List[Dict[str, Any]] = []
    for role in ("dress", "top", "bottom", "footwear"):
        if role in chosen:
            final_items.append(chosen[role])
    final_items.extend(accessories[:4])

    fixed = dict(card)
    fixed["items"] = final_items
    fixed["accessories"] = accessories[:4]
    fixed.setdefault("id", f"style_card_{index + 1}")

    if not card_is_complete(fixed):
        return None
    return fixed


def _occasion_flags(query: str) -> Dict[str, bool]:
    q = str(query or "").lower()
    return {
        "office": any(k in q for k in ("office", "work", "meeting", "client")),
        "date": any(k in q for k in ("date", "dinner", "night")),
        "party": any(k in q for k in ("party", "club", "after-hours", "night out")),
    }


def _card_blob(card: Dict[str, Any]) -> str:
    return " ".join(
        [
            _safe_text(card.get("title")),
            _safe_text(card.get("occasion")),
            _safe_text(card.get("vibe")),
            *[
                " ".join(
                    _safe_text(item.get(k))
                    for k in ("name", "label", "category", "sub_category", "subcategory", "style", "pattern", "color")
                )
                for item in card.get("items", [])
                if isinstance(item, dict)
            ],
        ]
    ).lower()


def _quality_score(card: Dict[str, Any], query: str) -> float:
    text = _card_blob(card)
    flags = _occasion_flags(query)
    score = float(card.get("score") or 0.0) / 100.0
    roles = {item_role(item) for item in card.get("items", []) if isinstance(item, dict)}
    score += len(roles.intersection({"top", "bottom", "dress", "footwear"}))
    score += min(2, len([x for x in card.get("accessories", []) if isinstance(x, dict)])) * 0.35

    if flags["office"]:
        if any(k in text for k in ("button-down", "button down", "shirt", "trouser", "black pants", "off white", "loafer")):
            score += 3.0
        if any(k in text for k in ("watch", "belt", "sneaker")):
            score += 1.0
        if any(k in text for k in ("tropical", "hawaiian", "vacation", "beach", "party", "loud")):
            score -= 5.0
        if any(k in text for k in ("shorts", "slipper", "slides")):
            score -= 2.0
    if flags["date"] and any(k in text for k in ("watch", "black", "off white", "loafer")):
        score += 1.5
    if flags["party"] and any(k in text for k in ("print", "pattern", "statement", "black")):
        score += 1.0
    return score


def _title_for(card: Dict[str, Any], query: str, index: int) -> str:
    existing = _safe_text(card.get("look_name") or card.get("title") or card.get("name"))
    lower = existing.lower()
    generic = (
        not existing
        or lower in {
            "styled look",
            "ahvi style board",
            "hero look",
            "easy win",
            "signature combo",
            "polished daily",
            "today's edit",
        }
        or lower.startswith("look ")
    )
    if not generic:
        return existing

    flags = _occasion_flags(query)
    titles = (
        ["Boardroom Casual", "Sharp Daily", "Clean Friday", "Polished Neutral", "Smart Ease", "Workday Edit"]
        if flags["office"]
        else ["Date Night Edit", "Evening Ease", "Polished Dinner", "Soft Statement", "After-Dark Smart", "Clean Romance"]
        if flags["date"]
        else ["After-Hours Edit", "Statement Ease", "Night-Out Sharp", "Clean Contrast", "Smart Presence", "Polished Edge"]
        if flags["party"]
        else ["Polished Neutral", "Sharp Daily", "Smart Ease", "Clean Edit", "Refined Casual", "Signature Fit"]
    )
    return titles[index % len(titles)]


def finalize_style_cards(
    cards: Any,
    *,
    query: str,
    exclude_signatures: Any = None,
    requested_count: Optional[int] = None,
    default_limit: int = 6,
) -> List[Dict[str, Any]]:
    excluded = {
        _safe_text(x).lower()
        for x in (exclude_signatures or [])
        if _safe_text(x)
    }
    limit = max(1, min(6, requested_count or default_limit))

    canonical: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        fixed = _canonicalize_card(card, idx)
        if not fixed:
            continue
        sig = card_signature(fixed)
        core_sig = core_card_signature(fixed) or sig
        if not sig or core_sig in seen or sig in excluded or core_sig in excluded:
            continue
        fixed["_style_signature"] = sig
        fixed["_style_core_signature"] = core_sig
        fixed["_style_quality_score"] = _quality_score(fixed, query)
        seen.add(core_sig)
        canonical.append(fixed)

    canonical.sort(key=lambda c: float(c.get("_style_quality_score") or 0.0), reverse=True)
    for idx, card in enumerate(canonical):
        title = _title_for(card, query, idx)
        card["title"] = title
        card["name"] = title
    return canonical[:limit]


def board_item_ids(cards: Any) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        for item in card.get("items", []):
            if not isinstance(item, dict):
                continue
            ident = _safe_text(item.get("$id") or item.get("id") or item.get("item_id") or item.get("itemId") or item.get("image_id"))
            if ident and ident not in seen:
                seen.add(ident)
                ids.append(ident)
    return ids


def visual_intelligence_from_outfit(outfit: Dict[str, Any]) -> Dict[str, Any]:
    parts = [
        _dict(outfit.get("top")),
        _dict(outfit.get("bottom")),
        _dict(outfit.get("dress")),
        _dict(outfit.get("shoes")),
    ] + [x for x in (outfit.get("accessories") or []) if isinstance(x, dict)]
    colors = [_safe_text(p.get("color")).lower() for p in parts if _safe_text(p.get("color"))]
    patterns = [_safe_text(p.get("pattern")).lower() for p in parts if _safe_text(p.get("pattern"))]
    styles = [_safe_text(p.get("style")).lower() for p in parts if _safe_text(p.get("style"))]
    return {
        "dominant_palette": sorted(set(colors))[:4],
        "pattern_mix": sorted(set(patterns))[:4],
        "style_signals": sorted(set(styles))[:4],
        "composition_score": float(outfit.get("score") or 0.0),
        "story": _safe_text(_dict(outfit.get("story")).get("subtitle") or outfit.get("explanation")),
    }


def render_style_boards(
    cards: List[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    user_id: str,
    include_base64: bool,
    upload_to_r2: bool = False,
) -> List[Dict[str, Any]]:
    if not cards:
        return []
    if not include_base64 and not upload_to_r2:
        return []

    storage = None
    if upload_to_r2 and R2Storage is not None:
        try:
            storage = R2Storage()
        except Exception:
            storage = None

    rendered: List[Dict[str, Any]] = []
    style_dna = _dict(context.get("style_dna"))
    try:
        from brain.engines.style_board_engine import style_board_engine
        from brain.engines.style_board_renderer import style_board_renderer
    except Exception as exc:
        logger.warning("style board renderer unavailable: %s", exc)
        return []

    for idx, card in enumerate(cards):
        items = card.get("items") if isinstance(card.get("items"), list) else []
        if not items:
            continue
        board = {}
        image_bytes = b""
        try:
            board = style_board_engine.build_board({"items": items, "score": card.get("score")}, context)
            image_bytes = style_board_renderer.render(board)
        except Exception as exc:
            logger.warning("style board render failed user=%s idx=%s error=%s", user_id, idx, exc)

        image_base64 = base64.b64encode(image_bytes).decode("ascii") if include_base64 and image_bytes else None
        image_url = None
        upload_error = None
        if storage and image_bytes:
            try:
                uploaded = storage.upload_style_board_image(
                    user_id=str(user_id or "user"),
                    image_bytes=image_bytes,
                    extension="png",
                )
                image_url = uploaded.get("image_url")
            except (R2StorageError, Exception) as exc:
                upload_error = str(exc)

        rendered.append(
            {
                "board_id": str(uuid.uuid4()),
                "type": "style",
                "label": card.get("title") or "AHVI Style Board",
                "score": card.get("score"),
                "aesthetic": board.get("aesthetic"),
                "vibe": board.get("vibe"),
                "items": items,
                "image_base64": image_base64,
                "image_url": image_url,
                "upload_error": upload_error,
                "style_signature": card.get("_style_signature") or card_signature(card),
                "board_payload": {
                    "items": items,
                    "aesthetic": board.get("aesthetic"),
                    "vibe": board.get("vibe"),
                    "score": card.get("score"),
                    "style_dna": style_dna,
                    "card_id": card.get("id") or f"style_card_{idx + 1}",
                },
            }
        )
    return rendered


def _style_signature_hash(signatures: List[str]) -> str:
    return hashlib.sha1("|".join(signatures).encode("utf-8")).hexdigest() if signatures else ""


def finalize_style_response_payload(
    result: Dict[str, Any],
    *,
    user_id: str,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    include_base64: bool = False,
    upload_to_r2: bool = False,
    style_action: str = "",
    exclude_style_signatures: Any = None,
    requested_board_count: Optional[int] = None,
    cache_bypass: bool = True,
) -> Dict[str, Any]:
    ctx = dict(context or {})
    raw_cards = result.get("cards") if isinstance(result.get("cards"), list) else []
    raw_outfits = result.get("outfits") if isinstance(result.get("outfits"), list) else []
    candidates = list(raw_outfits or []) + list(raw_cards or [])

    cards = finalize_style_cards(
        candidates,
        query=query,
        exclude_signatures=exclude_style_signatures,
        requested_count=requested_board_count if style_action in {"more_options", "more_looks", "next_best"} else None,
    )
    ids = board_item_ids(cards)
    rendered = render_style_boards(
        cards,
        ctx,
        user_id=user_id,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
    )
    signatures = [card.get("_style_signature") or card_signature(card) for card in cards]
    core_signatures = [card.get("_style_core_signature") or core_card_signature(card) for card in cards]
    style_signature = _style_signature_hash([s for s in signatures if s])
    primary_board_id = ids[0] if ids else ""

    logger.info(
        "style_flow.final_response user=%s cards=%d core_signatures=%s signatures=%s style_action=%s cache_bypass=%s",
        user_id,
        len(cards),
        core_signatures,
        signatures,
        style_action or "",
        bool(cache_bypass),
    )

    data = {
        "outfits": cards,
        "visual_intelligence": visual_intelligence_from_outfit(raw_outfits[0]) if raw_outfits and isinstance(raw_outfits[0], dict) else {},
        "pipeline": _dict(result.get("pipeline")),
        "rendered_boards": rendered or cards,
        "board_item_ids": ids,
    }
    return {
        "cards": cards,
        "style_boards": cards,
        "board_ids": primary_board_id,
        "data": data,
        "meta": {
            "style_action": style_action or None,
            "style_signature": style_signature or None,
            "core_style_signatures": core_signatures,
            "board_count": len(cards),
            "has_more_style_options": bool(cards),
            "cache_bypass": bool(cache_bypass),
            "style_cache_bypass": bool(cache_bypass),
        },
    }


def build_style_flow_response(
    *,
    user_id: str,
    query: str,
    wardrobe: Any,
    user_profile: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    include_base64: bool = False,
    upload_to_r2: bool = False,
    style_action: str = "",
    exclude_style_signatures: Any = None,
    requested_board_count: Optional[int] = None,
    cache_bypass: bool = True,
) -> Dict[str, Any]:
    from brain.outfit_pipeline import get_daily_outfits

    ctx = dict(context or {})
    ctx.setdefault("query", query)
    ctx.setdefault("user_id", user_id)
    if user_profile is not None:
        ctx.setdefault("user_profile", user_profile)

    result = get_daily_outfits(
        {
            "user_id": user_id,
            "wardrobe": wardrobe,
            "context": ctx,
        }
    )
    if not isinstance(result, dict):
        result = {}

    finalized = finalize_style_response_payload(
        result,
        user_id=user_id,
        query=query,
        context=ctx,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
        style_action=style_action,
        exclude_style_signatures=exclude_style_signatures,
        requested_board_count=requested_board_count,
        cache_bypass=cache_bypass,
    )
    cards = finalized["cards"]
    return {
        "success": bool(cards),
        "message": result.get("context")
        or (
            "I pulled together wardrobe-based looks that match your request, occasion, and style profile."
            if cards
            else "I couldn't build a reliable style board from your wardrobe yet."
        ),
        "board": "style",
        "type": "cards" if cards else "missing_outfit_cards",
        "cards": cards,
        "style_boards": finalized["style_boards"],
        "chips": STYLE_ACTION_CHIPS if cards else [],
        "board_ids": finalized["board_ids"],
        "data": finalized["data"],
        "meta": {
            **_dict(result.get("meta")),
            **finalized["meta"],
            "analysis_source": "style_flow_service",
        },
    }
