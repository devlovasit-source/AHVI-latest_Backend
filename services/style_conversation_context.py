"""Bounded semantic context for a single Style conversation.

This module owns only conversational facts. It is intentionally separate from
board state, wardrobe persistence, and action execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping


MAX_CONTEXT_STRING = 120
MAX_CONTEXT_LIST = 6
MAX_HISTORY_ITEMS = 8

ACTIVITY_TYPES = {
    "court_sport",
    "running",
    "training",
    "outdoor_active",
    "studio_movement",
    "field_sport",
}

_REFERENT_TYPES = {"activity", "occasion", "context", "garment"}

_GARMENT_NOUNS = (
    "t-shirt", "shirt", "tee", "blouse", "top", "sweater", "hoodie",
    "cardigan", "blazer", "jacket", "coat", "dress", "trousers", "pants",
    "jeans", "skirt", "shorts", "sneakers", "shoes", "loafers", "boots",
    "heels", "sandals", "flats", "saree", "kurta", "jumpsuit", "tie",
    "belt", "bag",
)
_GARMENT_MODIFIERS = (
    "blue", "black", "white", "red", "green", "navy", "cream", "beige",
    "grey", "gray", "brown", "tan", "pink", "purple", "yellow", "orange",
    "denim", "linen", "silk", "leather", "button-down", "button down",
    "oversized", "tailored", "casual", "formal", "striped", "plain", "light",
    "dark",
)
_GARMENT_NOUN_PATTERN = "|".join(re.escape(value) for value in _GARMENT_NOUNS)
_GARMENT_PATTERN = re.compile(
    rf"\b(?P<phrase>(?:(?:{'|'.join(re.escape(value) for value in _GARMENT_MODIFIERS)})\s+)*"
    rf"(?:{_GARMENT_NOUN_PATTERN}))\b",
    re.IGNORECASE,
)
_GARMENT_TARGET_PATTERN = re.compile(
    rf"\b(?:with|for|on|from|about)\s+(?:the|this|that|my|your)\s+"
    rf"(?P<noun>{_GARMENT_NOUN_PATTERN})\b",
    re.IGNORECASE,
)

_ACTIVITY_ALIASES = {
    "badminton": ("badminton", "court_sport"),
    "tennis": ("tennis", "court_sport"),
    "squash": ("squash", "court_sport"),
    "pickleball": ("pickleball", "court_sport"),
    "running": ("running", "running"),
    "jogging": ("running", "running"),
    "gym": ("gym", "training"),
    "strength training": ("strength training", "training"),
    "hiit": ("hiit", "training"),
    "hiking": ("hiking", "outdoor_active"),
    "trekking": ("trekking", "outdoor_active"),
    "yoga": ("yoga", "studio_movement"),
    "pilates": ("pilates", "studio_movement"),
    "football": ("football", "field_sport"),
    "basketball": ("basketball", "field_sport"),
    "cricket": ("cricket", "field_sport"),
}

_OCCASION_ALIASES = (
    ("client meeting", "client_meeting"),
    ("casual outing", "casual_outing"),
    ("formal event", "formal_event"),
    ("office", "office"),
    ("interview", "interview"),
    ("work", "work"),
    ("wedding", "wedding"),
    ("party", "party"),
    ("brunch", "brunch"),
    ("dinner", "dinner"),
    ("travel", "travel"),
    ("date", "date"),
    ("casual", "casual"),
    ("formal", "formal_event"),
)

_DATE_PATTERNS = (
    ("this_morning", r"\bthis\s+morning\b"),
    ("this_afternoon", r"\bthis\s+afternoon\b"),
    ("this_evening", r"\bthis\s+evening\b"),
    ("this_weekend", r"\bthis\s+weekend\b"),
    ("tonight", r"\btonight\b"),
    ("tomorrow", r"\btomorrow\b"),
    ("today", r"\btoday\b"),
    ("saturday", r"\bsaturday\b"),
    ("sunday", r"\bsunday\b"),
    ("monday", r"\bmonday\b"),
    ("tuesday", r"\btuesday\b"),
    ("wednesday", r"\bwednesday\b"),
    ("thursday", r"\bthursday\b"),
    ("friday", r"\bfriday\b"),
)

_DAYPART_PATTERNS = (
    ("morning", r"\b(?:this\s+)?morning\b"),
    ("afternoon", r"\b(?:this\s+)?afternoon\b"),
    ("evening", r"\b(?:this\s+)?evening\b"),
    ("night", r"\b(?:tonight|night)\b"),
)


def _clean(value: Any, limit: int = MAX_CONTEXT_STRING) -> str:
    text = str(value or "").strip()
    return text[:limit] if text else ""


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def normalize_activity(activity: Any, activity_type: Any = None) -> tuple[str | None, str | None]:
    """Return the bounded activity label and family, or safe nulls."""
    raw_activity = _clean(activity).lower().replace("_", " ")
    raw_type = _clean(activity_type).lower().replace("-", "_").replace(" ", "_")
    if raw_activity:
        for alias, (label, family) in _ACTIVITY_ALIASES.items():
            if raw_activity == alias or alias in raw_activity:
                return label, family
    if raw_type in ACTIVITY_TYPES:
        return (raw_activity or None), raw_type
    return (raw_activity or None), None


def _activity_from_text(text: str) -> tuple[str | None, str | None]:
    candidates = sorted(_ACTIVITY_ALIASES, key=len, reverse=True)
    for alias in candidates:
        if _contains(text, alias):
            return normalize_activity(*_ACTIVITY_ALIASES[alias])
    return None, None


def _occasion_from_text(text: str) -> str | None:
    for alias, value in _OCCASION_ALIASES:
        if alias == "work" and re.search(r"\bwork\s+with\b", text):
            continue
        if alias == "casual" and "more casual" in text:
            continue
        if alias == "formal" and "more formal" in text:
            continue
        if _contains(text, alias):
            return value
    return None


def _date_from_text(text: str) -> str | None:
    for value, pattern in _DATE_PATTERNS:
        if re.search(pattern, text):
            return value
    # Keep explicit month/day text bounded without trying to parse a calendar.
    explicit = re.search(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b",
        text,
    )
    return explicit.group(0) if explicit else None


def _is_explicit_context_correction(message: str) -> bool:
    text = " ".join(str(message or "").lower().split())
    return bool(re.search(r"\b(?:actually|instead|change|make that)\b", text))


def _daypart_from_text(text: str) -> str | None:
    for value, pattern in _DAYPART_PATTERNS:
        if re.search(pattern, text):
            return value
    return None


def _extract_garment_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for match in _GARMENT_PATTERN.finditer(text):
        phrase = " ".join(match.group("phrase").lower().replace("-", " ").split())
        if phrase and phrase not in mentions:
            mentions.append(phrase)
    return mentions[-MAX_CONTEXT_LIST:]


def _garment_noun(value: str) -> str | None:
    match = re.search(rf"\b({_GARMENT_NOUN_PATTERN})\b", value, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _constraints_from_text(text: str) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    for phrase in ("more casual", "more formal", "sportier"):
        if phrase in text:
            positive.append(phrase)
    for match in re.finditer(r"\b(?:no|avoid)\s+([a-z][a-z -]{1,40})", text):
        value = re.split(r"\b(?:for|with|and|but)\b|[,.;!?]", match.group(1), maxsplit=1)[0]
        value = _clean(value, 50).strip()
        if value:
            negative.append(value)
    return positive[:MAX_CONTEXT_LIST], negative[:MAX_CONTEXT_LIST]


@dataclass
class StyleConversationContext:
    date_context: str | None = None
    daypart: str | None = None
    occasion: str | None = None
    activity: str | None = None
    activity_type: str | None = None
    venue: str | None = None
    style_constraints: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    garment_references: list[str] = field(default_factory=list)
    referent: dict[str, Any] | None = None
    previous_intent: str | None = None
    previous_response_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_context": self.date_context,
            "daypart": self.daypart,
            "occasion": self.occasion,
            "activity": self.activity,
            "activity_type": self.activity_type,
            "venue": self.venue,
            "style_constraints": list(self.style_constraints[:MAX_CONTEXT_LIST]),
            "negative_constraints": list(self.negative_constraints[:MAX_CONTEXT_LIST]),
            "garment_references": list(self.garment_references[:MAX_CONTEXT_LIST]),
            "referent": dict(self.referent) if isinstance(self.referent, dict) else None,
            "previous_intent": self.previous_intent,
            "previous_response_mode": self.previous_response_mode,
        }

    def overlay(self, other: "StyleConversationContext", *, fill_only: bool = False) -> "StyleConversationContext":
        result = StyleConversationContext(**self.to_dict())
        scalar_fields = (
            "date_context", "daypart", "occasion", "activity", "activity_type", "venue",
            "referent", "previous_intent", "previous_response_mode",
        )
        for field_name in scalar_fields:
            incoming = getattr(other, field_name)
            if incoming is not None and (not fill_only or getattr(result, field_name) is None):
                setattr(result, field_name, incoming)
        if other.style_constraints:
            result.style_constraints = _merge_lists(
                result.style_constraints, other.style_constraints, replace=not fill_only
            )
        if other.negative_constraints:
            result.negative_constraints = _merge_lists(
                result.negative_constraints, other.negative_constraints, replace=not fill_only
            )
        if other.garment_references:
            result.garment_references = _merge_lists(
                result.garment_references, other.garment_references, replace=False
            )
        return result

    @classmethod
    def from_mapping(cls, value: Any) -> "StyleConversationContext":
        if not isinstance(value, Mapping):
            return cls()
        date = value.get("date_context") or value.get("date") or value.get("date_text") or value.get("time_period")
        activity, activity_type = normalize_activity(value.get("activity"), value.get("activity_type"))
        raw_type = _clean(value.get("activity_type")).lower().replace("-", "_").replace(" ", "_")
        if raw_type not in ACTIVITY_TYPES:
            raw_type = activity_type
        referent = value.get("referent")
        safe_referent = None
        if isinstance(referent, Mapping):
            text = _clean(referent.get("text"), 40)
            resolved_to = _clean(referent.get("resolved_to"), 100)
            confidence = referent.get("confidence")
            label = _clean(referent.get("label"), 100)
            referent_type = _clean(referent.get("type"), 24).lower()
            temporal = referent.get("temporal")
            safe_referent = {}
            if text:
                safe_referent["text"] = text
            if resolved_to:
                safe_referent["resolved_to"] = resolved_to
            if label:
                safe_referent["label"] = label
            if referent_type in _REFERENT_TYPES:
                safe_referent["type"] = referent_type
            if isinstance(temporal, Mapping):
                safe_temporal = {
                    key: _clean(temporal.get(key), 60)
                    for key in ("relative_date", "daypart")
                    if _clean(temporal.get(key), 60)
                }
                if safe_temporal:
                    safe_referent["temporal"] = safe_temporal
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                safe_referent["confidence"] = max(0.0, min(1.0, float(confidence)))
            if "kind" in referent:
                safe_referent["kind"] = _clean(referent.get("kind"), 24)
            if isinstance(referent.get("ordinal"), int) and not isinstance(referent.get("ordinal"), bool):
                safe_referent["ordinal"] = referent["ordinal"]
            if not safe_referent:
                safe_referent = None
        style_constraints = value.get("style_constraints") or value.get("positive_constraints") or value.get("required") or []
        negative_constraints = value.get("negative_constraints") or value.get("avoid") or []
        garment_references = (
            value.get("garment_references")
            or value.get("garments")
            or value.get("garment_referents")
            or []
        )
        return cls(
            date_context=_clean(date) or None,
            daypart=_clean(value.get("daypart")) or None,
            occasion=_clean(value.get("occasion")) or None,
            activity=activity,
            activity_type=raw_type,
            venue=_clean(value.get("venue")) or None,
            style_constraints=_bounded_list(style_constraints),
            negative_constraints=_bounded_list(negative_constraints),
            garment_references=_bounded_list(garment_references),
            referent=safe_referent,
            previous_intent=_clean(value.get("previous_intent")) or None,
            previous_response_mode=_clean(value.get("previous_response_mode")) or None,
        )


def _bounded_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(_clean(item, 60) for item in value if _clean(item, 60)))[:MAX_CONTEXT_LIST]


def _merge_lists(left: list[str], right: list[str], *, replace: bool) -> list[str]:
    values = right if replace else [*left, *right]
    return list(dict.fromkeys(values))[:MAX_CONTEXT_LIST]


def extract_current_turn_context(message: str) -> StyleConversationContext:
    text = " ".join(str(message or "").lower().split())
    activity, activity_type = _activity_from_text(text)
    positive, negative = _constraints_from_text(text)
    return StyleConversationContext(
        date_context=_date_from_text(text),
        daypart=_daypart_from_text(text),
        occasion=_occasion_from_text(text),
        activity=activity,
        activity_type=activity_type,
        style_constraints=positive,
        negative_constraints=negative,
        garment_references=_extract_garment_mentions(text),
    )


def extract_history_context(history: Iterable[Mapping[str, Any]] | None) -> StyleConversationContext:
    result = StyleConversationContext()
    rows = list(history or [])[-MAX_HISTORY_ITEMS:]
    for item in rows:
        if not isinstance(item, Mapping) or str(item.get("role") or "user").lower() != "user":
            continue
        result = result.overlay(extract_current_turn_context(item.get("content") or item.get("text") or ""))
    return result


def _structured_carried_context(value: Mapping[str, Any] | None) -> StyleConversationContext:
    if not isinstance(value, Mapping):
        return StyleConversationContext()
    candidates = [value]
    for key in ("conversation_context", "resolved_context", "style_conversation_context"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    result = StyleConversationContext()
    for candidate in candidates:
        result = result.overlay(StyleConversationContext.from_mapping(candidate))
    return result


def _supported_semantic_context(
    semantic: Mapping[str, Any] | None,
    *,
    message: str,
    history: Iterable[Mapping[str, Any]] | None,
    carried: StyleConversationContext,
) -> StyleConversationContext:
    """Discard model facts that have no evidence in the supplied conversation."""
    if not isinstance(semantic, Mapping):
        return StyleConversationContext()
    evidence = " ".join(
        [str(message or "").lower()]
        + [
            str(item.get("content") or item.get("text") or "").lower()
            for item in list(history or [])[-MAX_HISTORY_ITEMS:]
            if isinstance(item, Mapping) and str(item.get("role") or "user").lower() == "user"
        ]
    )
    candidate = StyleConversationContext.from_mapping(semantic)
    allowed = StyleConversationContext()
    for field_name in ("date_context", "daypart", "occasion", "activity", "venue"):
        value = getattr(candidate, field_name)
        if field_name == "occasion" and value in {"casual", "formal_event"} and (
            f"more {value.split('_', 1)[0]}" in evidence
        ):
            continue
        if value and (getattr(carried, field_name) is not None or _contains(evidence, str(value).replace("_", " "))):
            setattr(allowed, field_name, value)
    if candidate.activity and candidate.activity_type and allowed.activity:
        allowed.activity_type = candidate.activity_type
    allowed.style_constraints = [
        item for item in candidate.style_constraints if item.lower() in evidence
    ]
    allowed.negative_constraints = [
        item for item in candidate.negative_constraints if item.lower() in evidence
    ]
    return allowed


def _resolve_referent(
    message: str,
    context: StyleConversationContext,
    semantic_referent: Any = None,
) -> dict[str, Any] | None:
    lowered = str(message or "").lower()
    garment_mentions = context.garment_references
    current_garment_mentions = _extract_garment_mentions(lowered)
    prior_garment_mentions = [
        mention
        for mention in reversed(garment_mentions)
        if mention not in current_garment_mentions
    ]
    target_match = _GARMENT_TARGET_PATTERN.search(lowered)
    target_noun = target_match.group("noun").lower() if target_match else None
    has_garment_pronoun = bool(
        re.search(r"\b(?:it|this|that)\b", lowered)
    )
    if garment_mentions and (target_noun or has_garment_pronoun or _extract_garment_mentions(lowered)):
        selected = None
        if target_noun:
            selected = next(
                (
                    mention
                    for mention in prior_garment_mentions
                    if _garment_noun(mention) == target_noun
                ),
                None,
            )
        if selected is None and has_garment_pronoun:
            selected = prior_garment_mentions[0] if prior_garment_mentions else None
        selected = selected or garment_mentions[-1]
        return {
            "text": selected,
            "type": "garment",
            "label": selected,
            "resolved_to": selected,
            "confidence": 0.97,
        }

    if isinstance(semantic_referent, Mapping):
        text = _clean(semantic_referent.get("text"), 40)
        if text:
            semantic_context = StyleConversationContext.from_mapping({"referent": semantic_referent}).referent
            if semantic_context:
                semantic_context = _generic_referent(context, text, semantic_context)
                return semantic_context
    token = next((value for value in ("this", "that", "it") if re.search(rf"\b{value}\b", lowered)), "")
    if not token:
        return context.referent
    return _generic_referent(context, token)


def _generic_referent(
    context: StyleConversationContext,
    token: str,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if existing and existing.get("type") == "garment":
        label = _clean(existing.get("label") or existing.get("text"), 100)
        if label:
            return {
                "text": token,
                "type": "garment",
                "label": label,
                "resolved_to": label,
                "confidence": float(existing.get("confidence") or 0.97),
            }
    subject = context.activity or context.occasion
    if not subject:
        if existing and existing.get("resolved_to"):
            return dict(existing)
        return {"text": token, "resolved_to": "unresolved", "confidence": 0.0}
    existing_label = _clean(existing.get("label"), 100) if existing else ""
    label = existing_label or str(subject).replace("_", " ").strip()
    temporal: dict[str, str] = {}
    if context.date_context:
        temporal["relative_date"] = context.date_context.replace("_", " ")
    if context.daypart:
        temporal["daypart"] = context.daypart
    resolved_to = " ".join(
        part for part in (label, temporal.get("relative_date"), temporal.get("daypart")) if part
    )
    referent = {
        "text": token,
        "type": "activity" if context.activity else "occasion",
        "label": label,
        "temporal": temporal,
        "resolved_to": resolved_to,
        "confidence": 0.97,
    }
    if existing:
        if isinstance(existing.get("confidence"), (int, float)):
            referent["confidence"] = existing["confidence"]
        if existing.get("type") in _REFERENT_TYPES:
            referent["type"] = existing["type"]
    return referent


def resolve_style_conversation_context(
    *,
    current_message: str,
    recent_history: Iterable[Mapping[str, Any]] | None = None,
    carried_context: Mapping[str, Any] | None = None,
    semantic_context: Mapping[str, Any] | None = None,
    semantic_referent: Any = None,
) -> tuple[StyleConversationContext, dict[str, Any]]:
    """Merge facts using current turn > carried state > semantic history."""
    current = extract_current_turn_context(current_message)
    history_context = extract_history_context(recent_history)
    carried = _structured_carried_context(carried_context)
    resolved = history_context.overlay(carried)
    resolved = resolved.overlay(
        _supported_semantic_context(
            semantic_context,
            message=current_message,
            history=recent_history,
            carried=resolved,
        ),
        fill_only=True,
    )
    resolved = resolved.overlay(current)
    if _is_explicit_context_correction(current_message):
        # An explicit replacement must not retain the superseded context
        # dimension from an earlier turn.
        if current.occasion and not current.activity:
            resolved.activity = None
            resolved.activity_type = None
        elif current.activity and not current.occasion:
            resolved.occasion = None
    resolved.referent = _resolve_referent(current_message, resolved, semantic_referent)

    missing = []
    request_text = str(current_message or "").lower()
    needs_style_context = bool(
        re.search(r"\b(?:need something|show(?: me)? .*inspiration|show inspiration|outfit|what should i wear|dress(?:ing)?)\b", request_text)
    )
    if needs_style_context and not resolved.activity and not resolved.occasion:
        missing = ["occasion_or_activity"]
    context_used: list[str] = []
    for field_name in ("date_context", "daypart", "occasion", "activity", "activity_type", "venue"):
        if getattr(current, field_name):
            context_used.append(f"current_turn.{field_name}")
        elif getattr(carried, field_name) or getattr(history_context, field_name):
            context_used.append(f"conversation.{field_name}")
    if resolved.referent and resolved.referent.get("resolved_to") not in {None, "unresolved"}:
        context_used.append("conversation.referent")
    return resolved, {
        "current_turn_context": current.to_dict(),
        "carried_context": carried.to_dict(),
        "resolved_context": resolved.to_dict(),
        "context_used": list(dict.fromkeys(context_used)),
        "requires_clarification": bool(missing),
        "missing_information": missing,
    }


def activity_compatibility_issues(direction: Mapping[str, Any], activity_type: str | None) -> list[str]:
    if activity_type != "court_sport" or not isinstance(direction, Mapping):
        return []
    text = " ".join(
        str(direction.get(key) or "")
        for key in ("title", "archetype", "hero_piece", "description", "strategy", "items", "pieces")
    ).lower()
    banned = (
        "formal blazer", "oxford", "dress shirt", "chinos", "loafer", "dress shoes",
        "scarf", "office tailoring", "formal tailoring", "formal archetype",
        "refined weekend", "contemporary classic",
    )
    return [term for term in banned if term in text]
