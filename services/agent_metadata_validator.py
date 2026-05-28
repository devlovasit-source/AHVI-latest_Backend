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
    try:
        c = float(value)
    except Exception:
        return 0.0
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


def validate_metadata_payload(
    payload: Any, *, base_item: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Normalize agent payload to documented schema. Never raises.

    `base_item` (the original outfit row) supplies safe defaults for
    `category` / `subcategory` when the agent omits them.
    """
    raw = payload if isinstance(payload, dict) else {}
    base = base_item if isinstance(base_item, dict) else {}

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
        "confidence": _coerce_confidence(raw.get("confidence")),
    }
    if validated["confidence"] < _low_confidence_threshold():
        validated["manual_review_required"] = True
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

    try:
        import httpx  # type: ignore
    except Exception:
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
