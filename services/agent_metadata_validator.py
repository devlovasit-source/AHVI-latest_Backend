"""AHVI Metadata Validator Agent — enrich wardrobe items with structured
style metadata from the Gemini-backed Metadata Validator Agent.

The validator is fully gated by ENABLE_AGENT_METADATA_VALIDATOR. If the
flag is off or the agent call fails, callers get a safe default payload
(or an existing fallback) so the wardrobe save flow never breaks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.agent.metadata_validator")


# ---------------------------------------------------------------------------
# Env knobs
# ---------------------------------------------------------------------------

ENV_ENABLE = "ENABLE_AGENT_METADATA_VALIDATOR"
ENV_MODEL = "AGENT_METADATA_VALIDATOR_MODEL"
ENV_TIMEOUT = "AGENT_METADATA_VALIDATOR_TIMEOUT_SECONDS"
ENV_LOW_CONFIDENCE = "AGENT_METADATA_LOW_CONFIDENCE_THRESHOLD"
ENV_ENDPOINT = "AGENT_METADATA_VALIDATOR_ENDPOINT"
ENV_API_KEY = "AGENT_METADATA_VALIDATOR_API_KEY"

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_TIMEOUT = 12.0
DEFAULT_LOW_CONFIDENCE = 0.55

# Appwrite string attributes have a default upper bound. Keep us well under.
APPWRITE_STRING_SOFT_LIMIT = 16000
_SCORE_KEYS = (
    "professionalism_score",
    "client_meeting_score",
    "boardroom_score",
    "capsule_score",
    "versatility_score",
    "date_night_score",
)


def is_enabled() -> bool:
    return str(os.getenv(ENV_ENABLE, "")).strip().lower() in {"1", "true", "yes", "on"}


def _model_name() -> str:
    return os.getenv(ENV_MODEL, DEFAULT_MODEL) or DEFAULT_MODEL


def _timeout_seconds() -> float:
    try:
        return float(os.getenv(ENV_TIMEOUT, str(DEFAULT_TIMEOUT)))
    except Exception:
        return DEFAULT_TIMEOUT


def _low_confidence_threshold() -> float:
    try:
        return float(os.getenv(ENV_LOW_CONFIDENCE, str(DEFAULT_LOW_CONFIDENCE)))
    except Exception:
        return DEFAULT_LOW_CONFIDENCE


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_KEYS = (
    "category",
    "subcategory",
    "formality",
    "style_role",
    "allowed_occasions",
    "blocked_occasions",
    "layering_role",
    "silhouette_type",
    "compatible_footwear",
    "incompatible_footwear",
    "compatible_accessories",
    "incompatible_accessories",
    "suitable_seasons",
    "climate_appropriateness",
    "material_characteristics",
    "visual_noise",
    "pattern_intensity",
    "statement_level",
    "professionalism_score",
    "client_meeting_score",
    "boardroom_score",
    "capsule_score",
    "versatility_score",
    "date_night_score",
    "risk_flags",
    "styling_notes",
    "confidence",
)


def _coerce_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _coerce_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _coerce_confidence(value: Any) -> float:
    return _coerce_score(value, default=0.0)


def _coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        c = float(value)
    except Exception:
        return default
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


def _coerce_enum(value: Any, allowed: set[str], default: str) -> str:
    text = _coerce_str(value, default).strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in allowed else default


def _tokens_from(*values: Any) -> set[str]:
    import re

    return set(
        re.sub(r"[^a-z0-9]+", " ", " ".join(str(v or "") for v in values).lower())
        .strip()
        .split()
    )


def _blob_from(base: Dict[str, Any], raw: Dict[str, Any]) -> str:
    parts: List[str] = []
    for source in (base, raw):
        for key in (
            "name",
            "title",
            "label",
            "category",
            "subcategory",
            "sub_category",
            "color",
            "pattern",
            "material",
            "fabric",
            "description",
        ):
            parts.append(str(source.get(key) or ""))
        for key in ("vision_labels", "labels", "occasions"):
            value = source.get(key)
            if isinstance(value, list):
                parts.extend(str(x) for x in value)
            else:
                parts.append(str(value or ""))
    return " ".join(parts).lower()


def _add_unique(values: List[str], additions: List[str]) -> List[str]:
    out = list(values or [])
    seen = {str(x).strip().lower() for x in out}
    for value in additions:
        text = str(value or "").strip()
        if text and text.lower() not in seen:
            out.append(text)
            seen.add(text.lower())
    return out


def _max_score(meta: Dict[str, Any], key: str, value: float) -> None:
    meta[key] = max(_coerce_score(meta.get(key), 0.0), float(value))


def _all_scores_zero(meta: Dict[str, Any]) -> bool:
    return all(_coerce_score(meta.get(key), 0.0) <= 0.0 for key in _SCORE_KEYS)


def _set_score_defaults(
    meta: Dict[str, Any],
    *,
    scores: Dict[str, float],
    visual_noise: Optional[str] = None,
    statement_level: Optional[str] = None,
    pattern_intensity: Optional[str] = None,
    risk_flags: Optional[List[str]] = None,
    confidence: float = 0.75,
) -> None:
    for key in _SCORE_KEYS:
        meta[key] = _coerce_score(scores.get(key), 0.0)
    if visual_noise:
        meta["visual_noise"] = visual_noise
    if statement_level:
        meta["statement_level"] = statement_level
    if pattern_intensity:
        meta["pattern_intensity"] = pattern_intensity
    if risk_flags:
        meta["risk_flags"] = _add_unique(meta.get("risk_flags", []), risk_flags)
    meta["confidence"] = max(_coerce_confidence(meta.get("confidence")), confidence)
    meta["manual_review_required"] = False


def _apply_metadata_score_defaults(
    meta: Dict[str, Any], *, base: Dict[str, Any], raw: Dict[str, Any], force: bool = False
) -> Dict[str, Any]:
    """Fill deterministic score vectors only when the agent produced none."""
    if not force and not _all_scores_zero(meta):
        return meta

    blob = _blob_from(base, raw)
    tokens = _tokens_from(blob)

    def has_any(words: List[str]) -> bool:
        return any(word in tokens or word in blob for word in words)

    applied = ""
    if has_any(["formal trouser", "formal trousers", "trouser", "trousers", "chino", "chinos", "slacks", "tailored pants"]):
        formal_trousers = has_any(["formal trouser", "formal trousers", "tailored pants", "slacks"])
        meta.update(
            {
                "category": "bottom",
                "subcategory": "formal_trousers" if formal_trousers else ("chinos" if has_any(["chino", "chinos", "khaki"]) else "trousers"),
                "formality": "business_casual" if formal_trousers else "smart_casual",
                "style_role": "businesswear",
            }
        )
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.85 if formal_trousers else 0.80,
                "client_meeting_score": 0.80 if formal_trousers else 0.75,
                "boardroom_score": 0.75 if formal_trousers else 0.65,
                "capsule_score": 0.80,
                "versatility_score": 0.85,
                "date_night_score": 0.55,
            },
            visual_noise="low",
            statement_level="core",
            pattern_intensity="none",
        )
        applied = "trousers"
    elif has_any(["button down", "button up", "buttondown", "buttonup", "button-up", "button-down", "dress shirt", "oxford", "formal shirt"]):
        meta.update({"category": "top", "formality": "business_casual", "style_role": "businesswear"})
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.75,
                "client_meeting_score": 0.72,
                "boardroom_score": 0.60,
                "capsule_score": 0.75,
                "versatility_score": 0.82,
                "date_night_score": 0.60,
            },
        )
        applied = "button_down"
    elif has_any(["polo", "polo shirt"]):
        meta.update({"category": "top", "formality": "smart_casual", "style_role": "casualwear"})
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.55,
                "client_meeting_score": 0.50,
                "boardroom_score": 0.30,
                "capsule_score": 0.70,
                "versatility_score": 0.80,
                "date_night_score": 0.55,
            },
            confidence=0.70,
        )
        applied = "polo"
    elif has_any(["graphic tee", "graphic t shirt", "graphic t-shirt", "graphic shirt", "loud graphic"]):
        meta.update({"category": "top", "formality": "casual", "style_role": "casualwear"})
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.10,
                "client_meeting_score": 0.05,
                "boardroom_score": 0.00,
                "capsule_score": 0.25,
                "versatility_score": 0.35,
                "date_night_score": 0.20,
            },
            visual_noise="high",
            statement_level="statement",
            pattern_intensity="loud",
            risk_flags=["too_casual_for_professional", "high_visual_noise"],
            confidence=0.70,
        )
        applied = "graphic_tee"
    elif has_any(["blazer", "sport coat", "suit jacket"]):
        meta.update({"category": "outerwear", "subcategory": "blazer", "formality": "business_casual", "style_role": "businesswear"})
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.95,
                "client_meeting_score": 0.90,
                "boardroom_score": 0.90,
                "capsule_score": 0.75,
                "versatility_score": 0.70,
                "date_night_score": 0.65,
            },
        )
        applied = "blazer"
    elif has_any(["loafer", "loafers", "oxford", "oxfords", "derby", "derbies"]):
        meta.update({"category": "footwear", "formality": "business_casual", "style_role": "businesswear"})
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.85,
                "client_meeting_score": 0.80,
                "boardroom_score": 0.75,
                "capsule_score": 0.70,
                "versatility_score": 0.70,
                "date_night_score": 0.65,
            },
        )
        applied = "loafers_oxfords"
    elif has_any(["sneaker", "sneakers", "trainer", "trainers"]):
        meta.update({"category": "footwear", "formality": "casual", "style_role": "casualwear"})
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.30,
                "client_meeting_score": 0.25,
                "boardroom_score": 0.10,
                "capsule_score": 0.60,
                "versatility_score": 0.80,
                "date_night_score": 0.45,
            },
            confidence=0.70,
        )
        applied = "sneakers"
    elif has_any(["cap", "hat", "baseball cap"]):
        meta.update({"category": "accessories", "formality": "casual", "style_role": "casualwear"})
        _set_score_defaults(
            meta,
            scores={
                "professionalism_score": 0.10,
                "client_meeting_score": 0.05,
                "boardroom_score": 0.00,
                "capsule_score": 0.35,
                "versatility_score": 0.45,
                "date_night_score": 0.20,
            },
            confidence=0.70,
        )
        applied = "cap_hat"

    if applied:
        logger.info(
            "ahvi.metadata.score_defaults_applied item=%s category=%s subcategory=%s professionalism_score=%.2f client_meeting_score=%.2f boardroom_score=%.2f capsule_score=%.2f versatility_score=%.2f date_night_score=%.2f",
            _coerce_str(base.get("name") or raw.get("name") or base.get("title") or raw.get("title") or applied),
            meta.get("category"),
            meta.get("subcategory"),
            _coerce_score(meta.get("professionalism_score")),
            _coerce_score(meta.get("client_meeting_score")),
            _coerce_score(meta.get("boardroom_score")),
            _coerce_score(meta.get("capsule_score")),
            _coerce_score(meta.get("versatility_score")),
            _coerce_score(meta.get("date_night_score")),
        )
    return meta


def _apply_category_defaults(meta: Dict[str, Any], *, base: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    blob = _blob_from(base, raw)
    tokens = _tokens_from(blob)

    def has_any(words: List[str]) -> bool:
        return any(word in tokens or word in blob for word in words)

    professional_garment = False
    patterned = has_any(["stripe", "striped", "check", "checked", "plaid", "print", "printed", "graphic", "paisley", "embroidered"])
    loud = has_any(["loud", "graphic", "neon", "bold print", "large print", "statement", "logo"])
    specific_signal = any(
        _coerce_str(source.get(key)).strip().lower() not in {"", "unknown", "none", "null"}
        for source in (base, raw)
        for key in ("name", "title", "label", "subcategory", "sub_category", "description")
    )

    if has_any(["boxer", "brief", "underwear", "undergarment", "innerwear", "trunks", "loungewear", "pajama", "pyjama"]):
        meta.update(
            {
                "category": "innerwear",
                "subcategory": "private_wear",
                "formality": "homewear",
                "style_role": "loungewear",
                "visual_noise": meta.get("visual_noise") or "low",
                "pattern_intensity": meta.get("pattern_intensity") or "none",
                "statement_level": meta.get("statement_level") or "core",
            }
        )
        meta["blocked_occasions"] = _add_unique(
            meta.get("blocked_occasions", []),
            ["office", "client_meeting", "boardroom", "formal_event", "public_outing"],
        )
        meta["allowed_occasions"] = _add_unique(meta.get("allowed_occasions", []), ["home", "private", "lounge"])
        return meta

    if has_any(["running shorts", "gym shorts", "training shorts", "track", "activewear", "workout", "gym", "running"]):
        meta.update(
            {
                "style_role": "activewear",
                "formality": "athletic",
                "visual_noise": meta.get("visual_noise") or "low",
                "pattern_intensity": meta.get("pattern_intensity") or "none",
                "statement_level": meta.get("statement_level") or "core",
            }
        )
        meta["blocked_occasions"] = _add_unique(
            meta.get("blocked_occasions", []),
            ["office", "client_meeting", "boardroom", "wedding", "formal_event"],
        )
        return meta

    if has_any(["formal trousers", "tailored pants", "tailored trouser", "slacks"]):
        professional_garment = True
        meta.update({"category": "bottom", "subcategory": "formal_trousers", "formality": "business_casual", "style_role": "businesswear"})
        _max_score(meta, "professionalism_score", 0.85)
        _max_score(meta, "client_meeting_score", 0.80)
        _max_score(meta, "boardroom_score", 0.75)
        _max_score(meta, "capsule_score", 0.72)
        _max_score(meta, "versatility_score", 0.72)
    elif has_any(["trouser", "trousers", "chino", "chinos", "slack", "slacks"]) and not has_any(["jogger", "sweatpant", "track pant", "lounge", "pajama", "pyjama", "shorts", "distressed", "beach"]):
        professional_garment = True
        meta.update(
            {
                "category": "bottom",
                "subcategory": "chinos" if has_any(["chino", "chinos", "khaki"]) else "trousers",
                "formality": "business_casual" if has_any(["formal", "tailored"]) else "smart_casual",
                "style_role": "businesswear",
            }
        )
        _max_score(meta, "professionalism_score", 0.75)
        _max_score(meta, "client_meeting_score", 0.70)
        _max_score(meta, "boardroom_score", 0.60)
        _max_score(meta, "capsule_score", 0.70)
        _max_score(meta, "versatility_score", 0.70)
    elif has_any(["button down", "button-down", "dress shirt", "oxford", "formal shirt"]):
        professional_garment = True
        meta.update({"category": "top", "formality": "business_casual", "style_role": "businesswear"})
        _max_score(meta, "professionalism_score", 0.75)
        _max_score(meta, "client_meeting_score", 0.70)
        _max_score(meta, "capsule_score", 0.65)
        _max_score(meta, "versatility_score", 0.65)
    elif has_any(["blazer", "sport coat", "suit jacket"]):
        professional_garment = True
        meta.update({"category": "outerwear", "subcategory": "blazer", "formality": "business_casual", "style_role": "businesswear"})
        _max_score(meta, "professionalism_score", 0.90)
        _max_score(meta, "client_meeting_score", 0.85)
        _max_score(meta, "boardroom_score", 0.85)
        _max_score(meta, "capsule_score", 0.75)
    elif has_any(["loafer", "loafers", "oxford", "oxfords", "derby", "derbies"]):
        professional_garment = True
        meta.update({"category": "footwear", "formality": "business_casual", "style_role": "businesswear"})
        _max_score(meta, "professionalism_score", 0.80)
        _max_score(meta, "client_meeting_score", 0.75)
    elif has_any(["slides", "slipper", "slippers", "flip flop", "flip-flop", "flipflop"]):
        meta.update({"category": "footwear", "formality": "casual", "style_role": "resortwear"})
        meta["blocked_occasions"] = _add_unique(
            meta.get("blocked_occasions", []),
            ["office", "client_meeting", "boardroom", "formal_event"],
        )

    if professional_garment:
        meta["allowed_occasions"] = _add_unique(
            meta.get("allowed_occasions", []),
            ["office", "client_meeting", "business_lunch", "travel", "dinner"],
        )
        meta["blocked_occasions"] = [
            occasion
            for occasion in meta.get("blocked_occasions", [])
            if str(occasion).strip().lower()
            not in {
                "office",
                "work",
                "client_meeting",
                "boardroom",
                "executive_meeting",
                "business_lunch",
                "dinner",
            }
        ]
        meta["blocked_occasions"] = _add_unique(
            meta.get("blocked_occasions", []),
            ["workout", "gym", "beach", "sleepwear"],
        )
        meta["risk_flags"] = [
            flag
            for flag in meta.get("risk_flags", [])
            if flag not in {"too_casual_for_professional"}
        ]

    if loud:
        meta["visual_noise"] = "high"
        meta["pattern_intensity"] = "loud"
        meta["statement_level"] = "statement" if meta.get("statement_level") != "risky" else "risky"
        meta["risk_flags"] = _add_unique(meta.get("risk_flags", []), ["high_visual_noise"])
        meta["client_meeting_score"] = min(_coerce_score(meta.get("client_meeting_score"), 0.0), 0.45)
    else:
        meta["visual_noise"] = meta.get("visual_noise") or ("medium" if patterned else "low")
        meta["pattern_intensity"] = meta.get("pattern_intensity") or ("subtle" if patterned else "none")
        meta["statement_level"] = meta.get("statement_level") or "core"

    if not professional_garment and (specific_signal or loud):
        if _coerce_score(meta.get("client_meeting_score"), 0.0) < 0.50:
            meta["risk_flags"] = _add_unique(meta.get("risk_flags", []), ["too_casual_for_professional"])
            meta["blocked_occasions"] = _add_unique(meta.get("blocked_occasions", []), ["client_meeting", "boardroom", "executive_meeting"])
        if _coerce_score(meta.get("boardroom_score"), 0.0) < 0.50:
            meta["risk_flags"] = _add_unique(meta.get("risk_flags", []), ["too_casual_for_professional"])
            meta["blocked_occasions"] = _add_unique(meta.get("blocked_occasions", []), ["boardroom", "executive_meeting"])

    return meta


def validate_metadata_payload(
    payload: Any, *, base_item: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Normalize agent payload to documented schema. Never raises.

    `base_item` (the original outfit row) supplies safe defaults for
    `category` / `subcategory` when the agent omits them.
    """
    raw = payload if isinstance(payload, dict) else {}
    base = base_item if isinstance(base_item, dict) else {}
    raw_scores = {key: _coerce_score(raw.get(key), 0.0) for key in _SCORE_KEYS}
    raw_all_scores_zero = all(value <= 0.0 for value in raw_scores.values())

    default_category = _coerce_str(base.get("category"), "unknown")
    default_subcategory = _coerce_str(
        base.get("subcategory") or base.get("sub_category"), "unknown"
    )

    validated: Dict[str, Any] = {
        "category": _coerce_str(raw.get("category"), default_category),
        "subcategory": _coerce_str(raw.get("subcategory"), default_subcategory),
        "formality": _coerce_str(raw.get("formality"), "casual"),
        "style_role": _coerce_str(raw.get("style_role"), "casualwear"),
        "allowed_occasions": _coerce_list(raw.get("allowed_occasions")),
        "blocked_occasions": _coerce_list(raw.get("blocked_occasions")),
        "layering_role": _coerce_str(raw.get("layering_role"), "standalone"),
        "silhouette_type": _coerce_str(raw.get("silhouette_type"), "unknown"),
        "compatible_footwear": _coerce_list(raw.get("compatible_footwear")),
        "incompatible_footwear": _coerce_list(raw.get("incompatible_footwear")),
        "compatible_accessories": _coerce_list(raw.get("compatible_accessories")),
        "incompatible_accessories": _coerce_list(raw.get("incompatible_accessories")),
        "suitable_seasons": _coerce_list(raw.get("suitable_seasons")),
        "climate_appropriateness": _coerce_list(raw.get("climate_appropriateness")),
        "material_characteristics": _coerce_list(raw.get("material_characteristics")),
        "visual_noise": _coerce_enum(raw.get("visual_noise"), {"low", "medium", "high"}, "low"),
        "pattern_intensity": _coerce_enum(raw.get("pattern_intensity"), {"none", "subtle", "moderate", "loud"}, "none"),
        "statement_level": _coerce_enum(raw.get("statement_level"), {"core", "accent", "statement", "risky"}, "core"),
        "professionalism_score": _coerce_score(raw.get("professionalism_score"), 0.0),
        "client_meeting_score": _coerce_score(raw.get("client_meeting_score"), 0.0),
        "boardroom_score": _coerce_score(raw.get("boardroom_score"), 0.0),
        "capsule_score": _coerce_score(raw.get("capsule_score"), 0.0),
        "versatility_score": _coerce_score(raw.get("versatility_score"), 0.0),
        "date_night_score": _coerce_score(raw.get("date_night_score"), 0.0),
        "risk_flags": _coerce_list(raw.get("risk_flags")),
        "styling_notes": _coerce_list(raw.get("styling_notes")),
        "confidence": _coerce_confidence(raw.get("confidence")),
    }
    validated = _apply_category_defaults(validated, base=base, raw=raw)
    validated = _apply_metadata_score_defaults(
        validated,
        base=base,
        raw=raw,
        force=raw_all_scores_zero,
    )
    if validated["confidence"] < _low_confidence_threshold():
        validated["manual_review_required"] = True
    elif validated.get("manual_review_required") is False:
        validated.pop("manual_review_required", None)
    return validated


def default_metadata(base_item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return validate_metadata_payload(None, base_item=base_item)


# ---------------------------------------------------------------------------
# Agent transport
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are the AHVI Metadata Validator. Given a garment record and any "
    "vision/AI hints, return a STRICT JSON object with exactly these keys: "
    + ", ".join(_SCHEMA_KEYS)
    + ". Return ONLY the JSON object, no prose."
)


def _build_prompt(
    *,
    item: Dict[str, Any],
    vision_result: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> str:
    safe_item = {
        "id": item.get("$id") or item.get("id") or item.get("item_id"),
        "name": item.get("name") or item.get("title") or item.get("label"),
        "category": item.get("category"),
        "sub_category": item.get("sub_category") or item.get("subcategory"),
        "color": item.get("color") or item.get("color_code"),
        "pattern": item.get("pattern"),
        "material": item.get("material") or item.get("fabric"),
        "occasions": item.get("occasions") or [],
        "image_url": item.get("image_url") or item.get("masked_url"),
    }
    payload = {
        "item": safe_item,
        "vision": dict(vision_result or {}),
        "context": dict(context or {}),
    }
    return json.dumps(payload, ensure_ascii=False)


async def _call_agent(prompt: str, *, model: str, timeout: float) -> Optional[Dict[str, Any]]:
    endpoint = os.getenv(ENV_ENDPOINT, "").strip()
    api_key = os.getenv(ENV_API_KEY, "").strip()

    try:
        from services import ai_gateway  # type: ignore

        call = getattr(ai_gateway, "call_agent", None) or getattr(
            ai_gateway, "call_gemini", None
        )
        if callable(call):
            result = call(
                model=model,
                system=_SYSTEM_PROMPT,
                prompt=prompt,
                timeout=timeout,
                response_format="json",
            )
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=timeout)
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except Exception:
                    return None
    except Exception:
        logger.debug("ahvi.metadata.gateway_unavailable", exc_info=True)

    if not endpoint:
        return None

    # Vertex AI Reasoning Engine resource id — call the Vertex SDK rather
    # than treating the string as an HTTP URL.
    try:
        from services._reasoning_engine_client import (
            call_reasoning_engine,
            looks_like_resource_id,
        )

        if looks_like_resource_id(endpoint):
            engine_result = await call_reasoning_engine(
                endpoint,
                system=_SYSTEM_PROMPT,
                prompt=prompt,
                timeout=timeout,
            )
            if isinstance(engine_result, dict):
                return engine_result
            return None
    except Exception:
        logger.debug("ahvi.metadata.reasoning_engine_path_failed", exc_info=True)

    try:
        import httpx  # type: ignore
    except Exception:
        return None

    if not endpoint.startswith(("http://", "https://")):
        logger.warning(
            "ahvi.metadata.endpoint_not_http endpoint=%s — skipping HTTP fallback",
            endpoint[:80],
        )
        return None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "input": prompt,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("ahvi.metadata.transport_failed error=%s", str(exc)[:200])
        return None

    if isinstance(data, dict):
        if isinstance(data.get("output"), dict):
            content = data["output"].get("content") or data["output"].get("text")
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except Exception:
                    return None
        if isinstance(data.get("content"), str):
            try:
                return json.loads(data["content"])
            except Exception:
                return None
        return data
    return None


# ---------------------------------------------------------------------------
# Public API — async + sync
# ---------------------------------------------------------------------------

async def validate_wardrobe_metadata(
    item: Dict[str, Any],
    user_id: Optional[str] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the Metadata Validator agent and return a validated payload.

    Always returns a schema-conformant dict — never raises, never blocks the
    save flow.
    """
    logger.info(
        "ahvi.metadata.validation_started user_id=%s item_id=%s category=%s",
        user_id,
        (item or {}).get("$id") or (item or {}).get("id"),
        (item or {}).get("category"),
    )
    if not is_enabled():
        return default_metadata(item)

    model = _model_name()
    timeout = _timeout_seconds()
    prompt = _build_prompt(item=item, vision_result=vision_result, context=context)

    try:
        raw = await asyncio.wait_for(
            _call_agent(prompt, model=model, timeout=timeout), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(
            "ahvi.metadata.validation_failed reason=timeout user_id=%s", user_id
        )
        return default_metadata(item)
    except Exception as exc:
        logger.warning(
            "ahvi.metadata.validation_failed reason=exception user_id=%s err=%s",
            user_id,
            str(exc)[:200],
        )
        return default_metadata(item)

    if not raw:
        logger.warning(
            "ahvi.metadata.validation_failed reason=empty_response user_id=%s",
            user_id,
        )
        return default_metadata(item)

    validated = validate_metadata_payload(raw, base_item=item)
    confidence = float(validated.get("confidence") or 0.0)

    if confidence < _low_confidence_threshold():
        logger.warning(
            "ahvi.metadata.low_confidence user_id=%s item_id=%s category=%s "
            "subcategory=%s formality=%s style_role=%s confidence=%.2f "
            "blocked_occasions=%s",
            user_id,
            (item or {}).get("$id"),
            validated.get("category"),
            validated.get("subcategory"),
            validated.get("formality"),
            validated.get("style_role"),
            confidence,
            validated.get("blocked_occasions"),
        )
    else:
        logger.info(
            "ahvi.metadata.validation_success user_id=%s item_id=%s category=%s "
            "subcategory=%s formality=%s style_role=%s client_meeting_score=%.2f "
            "boardroom_score=%.2f capsule_score=%.2f versatility_score=%.2f "
            "visual_noise=%s risk_flags=%s confidence=%.2f blocked_occasions=%s",
            user_id,
            (item or {}).get("$id"),
            validated.get("category"),
            validated.get("subcategory"),
            validated.get("formality"),
            validated.get("style_role"),
            float(validated.get("client_meeting_score") or 0.0),
            float(validated.get("boardroom_score") or 0.0),
            float(validated.get("capsule_score") or 0.0),
            float(validated.get("versatility_score") or 0.0),
            validated.get("visual_noise"),
            validated.get("risk_flags"),
            confidence,
            validated.get("blocked_occasions"),
        )
    return validated


def validate_wardrobe_metadata_sync(
    item: Dict[str, Any],
    user_id: Optional[str] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sync wrapper for sync callers (e.g. wardrobe persistence service)."""
    if not is_enabled():
        return default_metadata(item)
    coro = validate_wardrobe_metadata(
        item=item, user_id=user_id, vision_result=vision_result, context=context
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(
                    timeout=_timeout_seconds() + 2
                )
        return loop.run_until_complete(coro)
    except RuntimeError:
        try:
            return asyncio.run(coro)
        except Exception as exc:
            logger.warning("ahvi.metadata.sync_wrapper_failed err=%s", str(exc)[:200])
            return default_metadata(item)
    except Exception as exc:
        logger.warning("ahvi.metadata.sync_wrapper_failed err=%s", str(exc)[:200])
        return default_metadata(item)


# ---------------------------------------------------------------------------
# Appwrite persistence helper
# ---------------------------------------------------------------------------

def _safe_doc_id(item_id: str) -> str:
    import re

    raw = str(item_id or "").strip()
    if not raw:
        return ""
    return re.sub(r"[^A-Za-z0-9_]+", "", raw)[:36] or raw[:36]


def _json_compact(metadata: Dict[str, Any]) -> str:
    text = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
    if len(text) <= APPWRITE_STRING_SOFT_LIMIT:
        return text
    # Trim long list fields first.
    trimmed = dict(metadata)
    for key in (
        "material_characteristics",
        "climate_appropriateness",
        "suitable_seasons",
        "compatible_accessories",
        "incompatible_accessories",
        "compatible_footwear",
        "incompatible_footwear",
        "allowed_occasions",
        "blocked_occasions",
        "risk_flags",
        "styling_notes",
    ):
        if isinstance(trimmed.get(key), list) and len(trimmed[key]) > 8:
            trimmed[key] = trimmed[key][:8]
    text = json.dumps(trimmed, separators=(",", ":"), ensure_ascii=False)
    return text[:APPWRITE_STRING_SOFT_LIMIT]


def upsert_wardrobe_style_metadata_sync(
    user_id: str, item_id: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Sync upsert into the `wardrobe_style_metadata` collection.

    Returns `{"status": "created"|"updated"|"failed", "doc_id": <id>}`.
    Never raises — wardrobe save flow must not be blocked.
    """
    from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError

    doc_id = _safe_doc_id(item_id)
    if not doc_id or not user_id:
        return {"status": "failed", "doc_id": doc_id, "reason": "missing ids"}

    payload = {
        "item_id": str(item_id),
        "userId": str(user_id),
        "style_metadata": _json_compact(metadata),
    }
    proxy = AppwriteProxy()
    try:
        proxy.update_document("wardrobe_style_metadata", doc_id, payload)
        status = "updated"
    except AppwriteProxyError as exc:
        if "404" not in str(exc):
            logger.warning(
                "ahvi.metadata.save_failed item=%s user=%s err=%s",
                item_id,
                user_id,
                str(exc)[:200],
            )
            return {"status": "failed", "doc_id": doc_id, "reason": str(exc)[:200]}
        try:
            proxy.create_document(
                "wardrobe_style_metadata", payload, document_id=doc_id
            )
            status = "created"
        except Exception as exc2:
            logger.warning(
                "ahvi.metadata.save_failed item=%s user=%s err=%s",
                item_id,
                user_id,
                str(exc2)[:200],
            )
            return {"status": "failed", "doc_id": doc_id, "reason": str(exc2)[:200]}
    except Exception as exc:
        logger.warning(
            "ahvi.metadata.save_failed item=%s user=%s err=%s",
            item_id,
            user_id,
            str(exc)[:200],
        )
        return {"status": "failed", "doc_id": doc_id, "reason": str(exc)[:200]}

    logger.info(
        "ahvi.metadata.saved item=%s user=%s status=%s confidence=%.2f",
        item_id,
        user_id,
        status,
        float(metadata.get("confidence") or 0.0),
    )
    return {"status": status, "doc_id": doc_id}


async def upsert_wardrobe_style_metadata(
    user_id: str, item_id: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Async wrapper exposed per the integration spec."""
    return await asyncio.to_thread(
        upsert_wardrobe_style_metadata_sync, user_id, item_id, metadata
    )


# ---------------------------------------------------------------------------
# Wardrobe enrichment helpers
# ---------------------------------------------------------------------------

def _parse_style_metadata_string(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def merge_style_metadata_into_wardrobe_items(
    wardrobe_items: List[Dict[str, Any]],
    metadata_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach parsed `style_metadata` to each wardrobe item.

    Items missing metadata are returned untouched. Existing fields are
    never removed or renamed.
    """
    if not isinstance(wardrobe_items, list):
        return wardrobe_items

    by_item: Dict[str, Dict[str, Any]] = {}
    for doc in metadata_docs or []:
        if not isinstance(doc, dict):
            continue
        item_id = str(doc.get("item_id") or doc.get("itemId") or "").strip()
        if not item_id:
            continue
        parsed = _parse_style_metadata_string(doc.get("style_metadata"))
        if parsed:
            by_item[item_id] = parsed

    if not by_item:
        return wardrobe_items

    enriched: List[Dict[str, Any]] = []
    for item in wardrobe_items:
        if not isinstance(item, dict):
            enriched.append(item)
            continue
        item_id = str(
            item.get("$id") or item.get("id") or item.get("item_id") or ""
        ).strip()
        meta = by_item.get(item_id)
        if meta:
            new_item = dict(item)
            new_item["style_metadata"] = meta
            enriched.append(new_item)
        else:
            enriched.append(item)
    return enriched


def fetch_style_metadata_docs_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Best-effort fetch of style metadata docs for a user. Never raises."""
    if not user_id:
        return []
    try:
        from services.appwrite_proxy import AppwriteProxy

        docs = AppwriteProxy().list_documents(
            "wardrobe_style_metadata", user_id=user_id, limit=200
        )
        if isinstance(docs, dict):
            return list(docs.get("documents") or docs.get("items") or [])
        if isinstance(docs, list):
            return docs
    except Exception as exc:
        logger.debug("ahvi.metadata.fetch_failed user=%s err=%s", user_id, str(exc)[:200])
    return []


__all__ = [
    "validate_wardrobe_metadata",
    "validate_wardrobe_metadata_sync",
    "validate_metadata_payload",
    "default_metadata",
    "upsert_wardrobe_style_metadata",
    "upsert_wardrobe_style_metadata_sync",
    "merge_style_metadata_into_wardrobe_items",
    "fetch_style_metadata_docs_for_user",
    "is_enabled",
]
