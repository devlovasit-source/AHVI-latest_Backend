"""Bounded semantic intent resolution for the Style module.

This is deliberately a seam, not a second router. Deterministic pre-classifier
results remain authoritative for obvious requests. The model is consulted only
for unresolved Style language, and its structured result is validated before it
can reach module-chat response handling.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from services.ai_gateway import extract_json
from services.ai_gateway import generate_text as _generate_text
from services.response_contract import ALLOWED_RESPONSE_MODES
from services.style_conversation_context import ACTIVITY_TYPES, normalize_activity

logger = logging.getLogger("ahvi.semantic_intent")

_SEMANTIC_DOMAINS = {"style", "calendar", "planner"}
_SEMANTIC_INTENTS = {
    "greeting",
    "small_talk",
    "help_identity",
    "supportive_conversation",
    "advice",
    "color_advice",
    "information",
    "inspiration",
    "recommendation",
    "navigate",
    "clarification",
    "modify_current_look",
    "generate_alternative",
    "explain_current_look",
}
_SEMANTIC_ACTIONS = {
    "respond_greeting",
    "respond_small_talk",
    "respond_identity",
    "respond_supportively",
    "provide_style_advice",
    "provide_color_advice",
    "explain_style_concept",
    "provide_visual_inspiration",
    "recommend_wardrobe",
    "open_calendar",
    "request_clarification",
    "modify_current_look",
    "generate_alternative",
    "explain_current_look",
}
_SEMANTIC_RESPONSE_MODES = {
    "text_only",
    "visual_inspiration",
    "wardrobe_recommendation",
    "calendar_navigation",
    "clarification",
}
_CONTEXT_KEYS = {
    "occasion",
    "activity",
    "activity_type",
    "venue",
    "date_context",
    "daypart",
    "date",
    "date_text",
    "time_period",
    "style_context",
    "last_style_context",
    "current_memory",
    "garment_references",
}
_RESOLVED_CONTEXT_KEYS = {
    "occasion",
    "activity",
    "activity_type",
    "venue",
    "date_context",
    "daypart",
    "date",
    "date_text",
    "time_period",
    "positive_constraints",
    "negative_constraints",
}
_REFERENT_KEYS = {"kind", "ordinal", "text", "resolved_to", "confidence", "type", "label", "temporal"}
_REFERENT_TEMPORAL_KEYS = {"relative_date", "daypart"}
_CONSTRAINT_KEYS = {"required", "avoid"}
_OPERATION_TYPES = {"modify", "generate_alternative", "explain_current_look"}
_OPERATION_KEYS = {
    "type",
    "replace_roles",
    "preserve_roles",
    "remove_roles",
    "constraints",
    "style_adjustments",
    "alternative_scope",
    "explanation_target",
}
_OPERATION_ROLES = {
    "top", "bottom", "dress", "outerwear", "footwear", "accessory", "bag"
}
_CONSTRAINT_DIMENSIONS = {
    "color", "material", "garment_trait", "footwear_type", "fit", "style", "occasion"
}
_CONSTRAINT_OPERATORS = {"avoid", "require"}
_ADJUSTMENT_KEYS = {"formality", "polish", "energy", "fit", "palette"}
_ADJUSTMENT_VALUES = {
    "lower", "raise", "neutral", "casual", "formal", "polished", "sporty",
    "relaxed", "playful", "colorful", "monochrome", "tailored",
}
_TOP_LEVEL_KEYS = {
    "domain",
    "intent",
    "action",
    "response_mode",
    "confidence",
    "requires_clarification",
    "resolved_context",
    "constraints",
    "referent",
    "reason_codes",
    "missing_information",
    "operation",
}
_CONTEXTUAL_LANGUAGE = (
    "something like",
    "same but",
    "for this",
    "something for",
    "make it",
    "what about",
    "another one",
    "another look",
    "not the ",
    "this one",
    "that one",
    "later",
    "change",
    "replace",
    "swap",
    "switch",
    "keep",
    "hold onto",
    "on my feet",
    "outer layer",
    "lower half",
    "avoid",
    "without",
    "more casual",
    "more relaxed",
    "more formal",
    "sportier",
    "alternative",
    "completely different",
    "why this",
    "why these",
)
_FORBIDDEN_TOKENS = re.compile(
    r"\b(?:execute|tool|command|deploy|delete|write|api[_ -]?key|"
    r"anchor[_ -]?item[_ -]?id|locked[_ -]?item[_ -]?ids|item[_ -]?id)\b",
    re.IGNORECASE,
)
_MIN_CONFIDENCE = 0.65
_MAX_LIST_ITEMS = 6
_MAX_STRING_LENGTH = 160


def _bounded_string(value: Any, *, max_length: int = _MAX_STRING_LENGTH) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > max_length or _FORBIDDEN_TOKENS.search(text):
        return None
    return text


def _bounded_string_list(value: Any) -> Optional[List[str]]:
    if not isinstance(value, list) or len(value) > _MAX_LIST_ITEMS:
        return None
    result: List[str] = []
    for item in value:
        text = _bounded_string(item)
        if text is None:
            return None
        result.append(text)
    return result


def _normalize_context(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    sources = [value]
    for nested_key in ("conversation_context", "resolved_context", "style_conversation_context"):
        nested = value.get(nested_key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    result: Dict[str, Any] = {}
    for key in _CONTEXT_KEYS:
        item = next((source.get(key) for source in sources if source.get(key) is not None), None)
        safe_item = _safe_context_value(item)
        if safe_item is not None:
            result[key] = safe_item
    return result


def _safe_context_value(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and len(text) <= 500 and not _FORBIDDEN_TOKENS.search(text):
            return text
        return None
    if depth >= 2:
        return None
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in list(value.items())[:12]:
            key_text = str(key).strip()
            if not key_text or _FORBIDDEN_TOKENS.search(key_text):
                return None
            safe_item = _safe_context_value(item, depth + 1)
            if safe_item is not None:
                result[key_text] = safe_item
        return result
    if isinstance(value, list) and len(value) <= _MAX_LIST_ITEMS:
        result = []
        for item in value:
            safe_item = _safe_context_value(item, depth + 1)
            if safe_item is None:
                return None
            result.append(safe_item)
        return result
    return None


def _contextual_request(message: str) -> bool:
    normalized = " ".join(str(message or "").lower().split())
    return any(phrase in normalized for phrase in _CONTEXTUAL_LANGUAGE)


def _safe_history(history: Iterable[Mapping[str, Any]] | None) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for item in list(history or [])[-6:]:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role not in {"user", "assistant"}:
            role = "user"
        content = _bounded_string(item.get("content") or item.get("text"), max_length=800)
        if content:
            result.append({"role": role, "content": content})
    return result


def _safe_board_context(value: Any) -> Dict[str, Any]:
    """Expose bounded board facts without exposing authoritative item IDs."""
    if not isinstance(value, Mapping):
        return {}
    items = value.get("items") if isinstance(value.get("items"), list) else []
    safe_items = []
    for index, item in enumerate(items[:8], start=1):
        if not isinstance(item, Mapping):
            continue
        role = _bounded_string(item.get("role"), max_length=32)
        name = _bounded_string(item.get("name"), max_length=80)
        if not role:
            continue
        safe_items.append({
            "ordinal": index,
            "role": role.lower(),
            "name": name or role.lower(),
            "protected": bool(item.get("protected")),
        })
    return {
        "has_current_board": bool(value.get("has_current_board")),
        "interaction_mode": _bounded_string(value.get("interaction_mode"), max_length=32) or "",
        "scenario": _bounded_string(value.get("scenario"), max_length=32) or "",
        "items": safe_items,
    }


def _prompt(
    *,
    message: str,
    history: Iterable[Mapping[str, Any]] | None,
    module_hint: str,
    conversation_context: Mapping[str, Any] | None,
    board_context: Mapping[str, Any] | None,
) -> str:
    payload = {
        "current_message": str(message or "").strip()[:800],
        "recent_history": _safe_history(history),
        "module_hint": module_hint,
        "known_context": _normalize_context(conversation_context),
        "current_board": _safe_board_context(board_context),
    }
    schema = {
        "domain": "style|calendar|planner",
        "intent": "greeting|small_talk|help_identity|supportive_conversation|advice|color_advice|information|inspiration|recommendation|navigate|clarification|modify_current_look|generate_alternative|explain_current_look",
        "action": "respond_greeting|respond_small_talk|respond_identity|respond_supportively|provide_style_advice|provide_color_advice|explain_style_concept|provide_visual_inspiration|recommend_wardrobe|open_calendar|request_clarification|modify_current_look|generate_alternative|explain_current_look",
        "response_mode": "text_only|visual_inspiration|wardrobe_recommendation|calendar_navigation|clarification",
        "confidence": 0.0,
        "requires_clarification": False,
        "resolved_context": {
            "date_context": None,
            "occasion": None,
            "activity": None,
            "activity_type": None,
            "venue": None,
            "daypart": None,
            "positive_constraints": [],
            "negative_constraints": [],
        },
        "constraints": {"required": [], "avoid": []},
        "referent": "null or {type: activity|occasion|context|garment, text, label, resolved_to, confidence}",
        "reason_codes": [],
        "missing_information": [],
        "operation": {
            "type": "modify|generate_alternative|explain_current_look",
            "replace_roles": [],
            "preserve_roles": [],
            "remove_roles": [],
            "constraints": [],
            "style_adjustments": {},
            "alternative_scope": "default|broad|null",
            "explanation_target": None,
        },
    }
    return (
        "You are AHVI's semantic intent resolver. Return only JSON matching the "
        "schema. Interpret meaning, referents, occasion, activity, date context, "
        "and positive or negative Style constraints. Resolve textual garment "
        "referents such as 'this shirt' without inventing wardrobe IDs. For a "
        "current-board request, "
        "return one bounded operation object: type modify, generate_alternative, "
        "or explain_current_look. Use roles and semantic constraints only; never "
        "return item IDs, lock IDs, anchor IDs, or selected IDs. Do not answer the user, "
        "generate an outfit, execute actions, invent wardrobe ownership or "
        "Calendar facts, change item or lock IDs, or return tools/commands. "
        "When context is insufficient, request clarification rather than "
        "assuming.\n\nSchema:\n"
        + json.dumps(schema, separators=(",", ":"))
        + "\n\nInput:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def validate_semantic_decision(raw: Any) -> Optional[Dict[str, Any]]:
    """Validate and normalize model output; return None on any unsafe shape."""
    if not isinstance(raw, Mapping):
        return None
    if set(raw) - _TOP_LEVEL_KEYS:
        return None

    domain = str(raw.get("domain") or "").strip().lower()
    intent = str(raw.get("intent") or "").strip().lower()
    action = str(raw.get("action") or "").strip().lower()
    mode = str(raw.get("response_mode") or "").strip().lower()
    if domain not in _SEMANTIC_DOMAINS:
        return None
    if intent not in _SEMANTIC_INTENTS:
        return None
    if action not in _SEMANTIC_ACTIONS:
        return None
    if mode not in ALLOWED_RESPONSE_MODES or mode not in _SEMANTIC_RESPONSE_MODES:
        return None

    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None

    requires_clarification = raw.get("requires_clarification")
    if not isinstance(requires_clarification, bool):
        return None

    resolved_context = raw.get("resolved_context") or {}
    if not isinstance(resolved_context, Mapping):
        return None
    if set(resolved_context) - _RESOLVED_CONTEXT_KEYS:
        return None
    normalized_context: Dict[str, Any] = {}
    for key, value in resolved_context.items():
        if value is None:
            continue
        if key in {"positive_constraints", "negative_constraints"}:
            values = _bounded_string_list(value)
            if values is None:
                return None
            normalized_context[key] = values
        elif key == "activity_type":
            _activity, activity_type = normalize_activity(
                resolved_context.get("activity"), value
            )
            if not activity_type or activity_type not in ACTIVITY_TYPES:
                return None
            normalized_context[key] = activity_type
        else:
            text = _bounded_string(value)
            if text is None:
                return None
            normalized_context[key] = text

    constraints = raw.get("constraints") or {}
    if not isinstance(constraints, Mapping) or set(constraints) - _CONSTRAINT_KEYS:
        return None
    normalized_constraints: Dict[str, List[str]] = {}
    for key in _CONSTRAINT_KEYS:
        values = _bounded_string_list(constraints.get(key, []))
        if values is None:
            return None
        normalized_constraints[key] = values

    operation_raw = raw.get("operation")
    normalized_operation = None
    if intent in {"modify_current_look", "generate_alternative", "explain_current_look"}:
        if not isinstance(operation_raw, Mapping) or set(operation_raw) - _OPERATION_KEYS:
            return None
        operation_type = str(operation_raw.get("type") or "").strip().lower()
        expected_type = {
            "modify_current_look": "modify",
            "generate_alternative": "generate_alternative",
            "explain_current_look": "explain_current_look",
        }[intent]
        if operation_type != expected_type or operation_type not in _OPERATION_TYPES:
            return None

        def _roles(key: str) -> Optional[List[str]]:
            values = operation_raw.get(key, [])
            if not isinstance(values, list) or len(values) > _MAX_LIST_ITEMS:
                return None
            result = []
            for value in values:
                role = _bounded_string(value, max_length=32)
                if role is None or role.lower() not in _OPERATION_ROLES:
                    return None
                result.append(role.lower())
            return list(dict.fromkeys(result))

        replace_roles = _roles("replace_roles")
        preserve_roles = _roles("preserve_roles")
        remove_roles = _roles("remove_roles")
        if replace_roles is None or preserve_roles is None or remove_roles is None:
            return None

        raw_operation_constraints = operation_raw.get("constraints", [])
        if not isinstance(raw_operation_constraints, list) or len(raw_operation_constraints) > _MAX_LIST_ITEMS:
            return None
        operation_constraints = []
        for constraint in raw_operation_constraints:
            if not isinstance(constraint, Mapping) or set(constraint) - {"dimension", "operator", "value"}:
                return None
            dimension = _bounded_string(constraint.get("dimension"), max_length=32)
            operator = _bounded_string(constraint.get("operator"), max_length=16)
            value = _bounded_string(constraint.get("value"), max_length=80)
            if (
                dimension is None or dimension.lower() not in _CONSTRAINT_DIMENSIONS
                or operator is None or operator.lower() not in _CONSTRAINT_OPERATORS
                or value is None
            ):
                return None
            operation_constraints.append({
                "dimension": dimension.lower(),
                "operator": operator.lower(),
                "value": value.lower(),
            })

        raw_adjustments = operation_raw.get("style_adjustments") or {}
        if not isinstance(raw_adjustments, Mapping) or set(raw_adjustments) - _ADJUSTMENT_KEYS:
            return None
        style_adjustments = {}
        for key, value in raw_adjustments.items():
            normalized_key = str(key).strip().lower()
            normalized_value = _bounded_string(value, max_length=32)
            if normalized_value is None or normalized_value.lower() not in _ADJUSTMENT_VALUES:
                return None
            style_adjustments[normalized_key] = normalized_value.lower()

        alternative_scope = operation_raw.get("alternative_scope")
        if alternative_scope is not None:
            alternative_scope = _bounded_string(alternative_scope, max_length=16)
            if alternative_scope is None or alternative_scope.lower() not in {"default", "broad"}:
                return None
            alternative_scope = alternative_scope.lower()
        explanation_target = operation_raw.get("explanation_target")
        if explanation_target is not None:
            explanation_target = _bounded_string(explanation_target, max_length=48)
            if explanation_target is None:
                return None

        normalized_operation = {
            "type": operation_type,
            "replace_roles": replace_roles,
            "preserve_roles": preserve_roles,
            "remove_roles": remove_roles,
            "constraints": operation_constraints,
            "style_adjustments": style_adjustments,
            "alternative_scope": alternative_scope,
            "explanation_target": explanation_target,
        }
    elif operation_raw is not None:
        return None

    referent = raw.get("referent")
    normalized_referent = None
    if referent is not None:
        if not isinstance(referent, Mapping) or set(referent) - _REFERENT_KEYS:
            return None
        normalized_referent = {}
        for key, value in referent.items():
            if key == "ordinal":
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
                    return None
                normalized_referent[key] = value
            elif key == "confidence":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    return None
                normalized_referent[key] = float(value)
            elif key == "temporal":
                if not isinstance(value, Mapping) or set(value) - _REFERENT_TEMPORAL_KEYS:
                    return None
                temporal = {}
                for temporal_key, temporal_value in value.items():
                    text = _bounded_string(temporal_value)
                    if text is None:
                        return None
                    temporal[temporal_key] = text
                normalized_referent[key] = temporal
            elif key == "type":
                text = _bounded_string(value)
                if text is None or text.lower() not in {"activity", "occasion", "context", "garment"}:
                    return None
                normalized_referent[key] = text.lower()
            else:
                text = _bounded_string(value)
                if text is None:
                    return None
                normalized_referent[key] = text

    reason_codes = _bounded_string_list(raw.get("reason_codes", []))
    missing_information = _bounded_string_list(raw.get("missing_information", []))
    if reason_codes is None or missing_information is None:
        return None

    if confidence < _MIN_CONFIDENCE:
        requires_clarification = True
    if requires_clarification:
        mode = "clarification"
        intent = "clarification"
        action = "request_clarification"
        if not missing_information:
            missing_information = ["more_context"]

    if mode == "calendar_navigation" and (domain != "calendar" or action != "open_calendar"):
        return None
    if mode == "visual_inspiration" and (
        domain != "style" or action != "provide_visual_inspiration"
    ):
        return None
    if mode == "wardrobe_recommendation" and (
        domain != "style"
        or action not in {"recommend_wardrobe", "modify_current_look", "generate_alternative"}
    ):
        return None
    if mode == "text_only" and domain != "style":
        return None

    if intent in {"modify_current_look", "generate_alternative"} and mode != "wardrobe_recommendation":
        return None
    if intent == "explain_current_look" and mode != "text_only":
        return None

    return {
        "domain": domain,
        "intent": intent,
        "action": action,
        "response_mode": mode,
        "confidence": confidence,
        "requires_clarification": requires_clarification,
        "resolved_context": normalized_context,
        "constraints": normalized_constraints,
        "referent": normalized_referent,
        "reason_codes": reason_codes,
        "missing_information": missing_information,
        "operation": normalized_operation,
    }


def _clarification_decision(reason: str = "context_required") -> Dict[str, Any]:
    return {
        "domain": "style",
        "intent": "clarification",
        "action": "request_clarification",
        "response_mode": "clarification",
        "confidence": 0.0,
        "requires_clarification": True,
        "resolved_context": {},
        "constraints": {"required": [], "avoid": []},
        "referent": None,
        "reason_codes": [reason],
        "missing_information": ["more_context"],
    }


def resolve_semantic_intent(
    *,
    current_message: str,
    recent_history: Iterable[Mapping[str, Any]] | None = None,
    module_hint: str = "style",
    conversation_context: Mapping[str, Any] | None = None,
    board_context: Mapping[str, Any] | None = None,
    deterministic: Optional[Mapping[str, Any]] = None,
    request_id: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Return a validated semantic decision or None for existing routing."""
    if deterministic is not None:
        result = dict(deterministic)
        result["decision_source"] = "deterministic_fast_path"
        result.setdefault("confidence", 1.0)
        result.setdefault("requires_clarification", False)
        return result

    module = str(module_hint or "").strip().lower()
    # Defense-in-depth only (see routers/chat.py's early_intent-based board
    # gate for the primary safeguard): the LLM-backed contextual classifier
    # below can now also veto a board for wardrobe-surface follow-ups, not
    # just style-surface ones.
    if module not in {"style", "daily_wear", "wardrobe"}:
        return None

    contextual = _contextual_request(current_message)
    has_existing_context = bool(_safe_history(recent_history)) or bool(
        _normalize_context(conversation_context)
    )
    # Keep ordinary one-turn Style prompts on the established deterministic /
    # service path. The model is for referents and carried context, not a
    # replacement for every existing Style heuristic.
    if not contextual and not has_existing_context:
        return None
    raw: Any = None
    try:
        raw_text = _generate_text(
            _prompt(
                message=current_message,
                history=recent_history,
                module_hint=module,
                conversation_context=conversation_context,
                board_context=board_context,
            ),
            options={"temperature": 0.0, "max_output_tokens": 500},
            signals={"context_mode": "semantic_intent"},
            timeout_seconds=20,
            usecase="intent",
            request_id=request_id,
        )
        raw = extract_json(raw_text)
    except Exception as exc:
        logger.info(
            "semantic_intent_failed source=llm_semantic_resolver error_type=%s",
            type(exc).__name__,
        )

    decision = validate_semantic_decision(raw)
    if decision is not None:
        decision["decision_source"] = "llm_semantic_resolver"
        logger.info(
            "semantic_intent_decision source=llm_semantic_resolver domain=%s intent=%s response_mode=%s confidence=%.2f clarification=%s",
            decision["domain"],
            decision["intent"],
            decision["response_mode"],
            decision["confidence"],
            decision["requires_clarification"],
        )
        return decision

    if contextual:
        decision = _clarification_decision("invalid_or_missing_semantics")
        decision["decision_source"] = "llm_semantic_resolver"
        return decision
    return None
