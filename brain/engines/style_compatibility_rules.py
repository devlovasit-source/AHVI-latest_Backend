"""Thin adapter over the versioned P0 negative-compatibility knowledge pack
(garment_to_garment, garment_to_footwear, occasion_incompatibilities,
dress_code_violations, occasion_based_exceptions).

Pure knowledge evaluator: given outfit items + occasion/query context, returns
structured CompatibilityViolation entries. Never builds outfits, never
queries Qdrant/Gemini/APIs, never owns source policy or board registration.
brain.engines.outfit_quality_guard remains the single final quality
authority; this module only supplies additional pairwise/contextual
evidence for it to fold in.

Deferred (not evaluated here — narrower P0 scope):
  garment_to_accessories, footwear_to_accessories, color/fabric/silhouette
  rules, body-balance rules. occasion_incompatibilities and
  dress_code_violations carry rules for those categories too (mixed in the
  same JSON files) - only rows with category in {"garment", "footwear"} are
  evaluated; everything else is skipped, not partially enforced.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.style_item_contract import canonical_item_id, canonical_item_role

logger = logging.getLogger("ahvi.style_compatibility_rules")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "style_compatibility",
)

_P0_FAMILY_FILES = {
    "garment_to_garment": "garment_to_garment.json",
    "garment_to_footwear": "garment_to_footwear.json",
    "occasion_incompatibilities": "occasion_incompatibilities.json",
    "dress_code_violations": "dress_code_violations.json",
    "occasion_based_exceptions": "occasion_based_exceptions.json",
}

SEVERITY_HARD = "HARD"
SEVERITY_STRONG = "STRONG"
SEVERITY_SOFT = "SOFT"
_BUCKET_ORDER = {SEVERITY_HARD: 3, SEVERITY_STRONG: 2, SEVERITY_SOFT: 1}

# Severity 5 only promotes to HARD when the RAW occasion passed in (not a
# loosely-aliased umbrella like "office") is itself one of these strict
# tokens. AHVI's coarse everyday occasions (office/wedding/date_night/...)
# never auto-promote through this path - "Do NOT treat Drive severity=5 as
# automatically HARD" (P0 spec). The realistic HARD trigger for ordinary
# chat input is the explicit-dress-code path (_detect_explicit_dress_code),
# which reads free text rather than requiring a caller to pass one of these
# literal tokens.
_STRICT_RAW_OCCASIONS = {
    "black_tie", "white_tie", "black_tie_optional", "formal_business",
    "business_formal", "wedding_guest", "red_carpet",
}

_ONLY_GARMENT_FOOTWEAR_CATEGORIES = {"garment", "footwear"}
_CATEGORY_ROLE_MAP = {
    "garment": frozenset({"top", "bottom", "dress", "outerwear"}),
    "footwear": frozenset({"footwear"}),
}

# AHVI's own normalize_occasion() vocabulary -> knowledge-pack occasion
# vocabulary. Multiple aliases per AHVI occasion; any one matching a rule's
# occasion key counts.
_OCCASION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "office": ("formal_business", "business_formal", "business_casual", "interview"),
    "client_dinner": ("formal_business", "business_formal", "dinner", "cocktail"),
    "wedding": ("wedding_guest", "traditional_festive", "festive"),
    "date_night": ("date_night", "dinner", "cocktail"),
    "cocktail": ("cocktail",),
    "party": ("party", "festive"),
    "beach": ("beach", "resort", "vacation"),
    "temple_modest": ("religious_ceremony",),
    "travel": ("travel", "vacation"),
    "brunch": ("brunch", "casual", "smart_casual"),
    "casual": ("casual", "weekend"),
    "casual_dinner": ("dinner", "casual", "smart_casual"),
    "workout": ("athleisure",),
    "rave": ("streetwear", "concert", "festival", "party"),
    "daily": ("casual", "weekend"),
}

_DRESS_CODE_TRIGGERS: Dict[str, Tuple[str, ...]] = {
    "black_tie": ("black tie",),
    "black_tie_optional": ("black tie optional",),
    "white_tie": ("white tie",),
    "cocktail": ("cocktail attire", "cocktail dress code"),
    "formal_business": ("business formal", "formal business"),
    "business_professional": ("business professional",),
    "business_casual": ("business casual",),
    "smart_casual": ("smart casual",),
    "festive": ("festive dress code",),
    "traditional_festive": ("traditional festive",),
    "beach_formal": ("beach formal",),
    "resort_casual": ("resort casual",),
}

_BOLD_INTENT_TERMS = {
    "bold", "streetwear", "sporty", "sneakerhead", "statement",
    "avant garde", "avant-garde", "experimental", "editorial", "fashion forward",
}


@dataclass
class CompatibilityViolation:
    rule_id: str
    family: str
    severity: str  # HARD | STRONG | SOFT
    reason: str
    offending_item_ids: List[str] = field(default_factory=list)
    offending_roles: List[str] = field(default_factory=list)
    repairable: bool = True
    source: str = "style_compatibility_rules"
    exception_applied: str = ""


# ---------------------------------------------------------------------------
# Knowledge loading - cached, fail-safe (never raise into a caller)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: Optional[Dict[str, Any]] = None
_load_failed_families: List[str] = []


def _load_family(filename: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(_DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("root is not an object")
        if not isinstance(data.get("rules"), list) and not isinstance(data.get("exceptions"), list):
            raise ValueError("missing rules/exceptions array")
        return data
    except Exception:
        logger.warning(
            "STYLE_COMPAT_EVALUATION_FAILED stage=load file=%s", filename, exc_info=True
        )
        return None


def _load_all() -> Dict[str, Any]:
    global _cache, _load_failed_families
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        packs: Dict[str, Any] = {}
        failed: List[str] = []
        for family, filename in _P0_FAMILY_FILES.items():
            data = _load_family(filename)
            if data is not None:
                packs[family] = data
            else:
                failed.append(family)
        _cache = packs
        _load_failed_families = failed
        if failed:
            logger.warning("STYLE_COMPAT_EVALUATION_FAILED stage=load_summary failed_families=%s", failed)
        return packs


def knowledge_pack_available() -> bool:
    return bool(_load_all())


# ---------------------------------------------------------------------------
# Item / occasion matching helpers
# ---------------------------------------------------------------------------


def _item_blob(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    parts = [
        item.get("name"), item.get("label"), item.get("category"),
        item.get("sub_category"), item.get("subcategory"), item.get("type"),
        item.get("garment_type"), item.get("style"), item.get("material"),
        item.get("fabric"), item.get("color"),
    ]
    return " ".join(str(p or "").strip().lower() for p in parts if p)


def _item_tokens(blob: str) -> set:
    return set(re.sub(r"[^a-z0-9]+", " ", blob).split())


def _rule_term_matches(term: Any, item_tokens: set, item_blob: str) -> bool:
    term_norm = str(term or "").strip().lower().replace("-", "_")
    if not term_norm:
        return False
    words = [w for w in term_norm.split("_") if w]
    if not words:
        return False
    if len(words) == 1:
        return words[0] in item_tokens
    phrase = " ".join(words)
    return phrase in item_blob or set(words).issubset(item_tokens)


def _item_role(item: Dict[str, Any]) -> str:
    explicit = str(item.get("role") or item.get("slot") or "").strip().lower()
    return explicit or canonical_item_role(item)


def _find_matching_item(
    items: List[Dict[str, Any]], term: Any, allowed_roles: Optional[frozenset] = None
) -> Optional[Dict[str, Any]]:
    if not term:
        return None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if allowed_roles is not None and _item_role(item) not in allowed_roles:
            continue
        if _rule_term_matches(term, _item_tokens(_item_blob(item)), _item_blob(item)):
            return item
    return None


def _normalize_occasion_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return " ".join(sorted(text.split()))


def _raw_occasion_is_strict(occasion: str) -> bool:
    occ = str(occasion or "").strip().lower().replace("-", "_")
    return occ in _STRICT_RAW_OCCASIONS


def _candidate_occasion_keys(occasion: str) -> List[str]:
    occ = str(occasion or "").strip().lower().replace("-", "_")
    if not occ:
        return []
    keys = [occ, *_OCCASION_ALIASES.get(occ, ())]
    return [_normalize_occasion_key(k) for k in keys if k]


def _lookup_occasion_severity(
    occasions: Dict[str, Any], candidate_keys: List[str]
) -> Tuple[Optional[int], str]:
    """Return (severity, matched_rule_occasion_normalized) or (None, "")."""
    if not candidate_keys or not isinstance(occasions, dict):
        return None, ""
    candidates = set(candidate_keys)
    for key, value in occasions.items():
        norm = _normalize_occasion_key(key)
        if norm in candidates:
            try:
                return int(value), norm
            except Exception:
                return None, ""
    return None, ""


def _severity_bucket(severity: int, raw_occasion: str) -> str:
    if severity >= 5 and _raw_occasion_is_strict(raw_occasion):
        return SEVERITY_HARD
    if severity >= 3:
        return SEVERITY_STRONG
    if severity >= 1:
        return SEVERITY_SOFT
    return ""


def _dress_code_severity_bucket(severity: int) -> str:
    # Reached only when the user explicitly stated this dress code, so
    # severity 5 here IS the hard, non-negotiable case (matches the pack's
    # own HV_001/OVR_003 hard_violation_logic semantics).
    if severity >= 5:
        return SEVERITY_HARD
    if severity >= 3:
        return SEVERITY_STRONG
    if severity >= 1:
        return SEVERITY_SOFT
    return ""


def _detect_explicit_dress_code(occasion: str, query: str) -> str:
    # Only the user's own free text counts as "explicitly stated" - `occasion`
    # is an internal normalized bucket (e.g. AHVI's own "business_casual"
    # classification) and must not be mistaken for the user having typed a
    # dress code, or every board generated for that occasion bucket would
    # silently activate hard dress-code enforcement it never asked for.
    text = f" {str(query or '').lower()} ".replace("_", " ").replace("-", " ")
    for code, phrases in _DRESS_CODE_TRIGGERS.items():
        if any(phrase in text for phrase in phrases):
            return code
    return ""


def _iid(item: Dict[str, Any]) -> str:
    return canonical_item_id(item)


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------


def _eval_pair_rule(
    rule: Dict[str, Any],
    items: List[Dict[str, Any]],
    key_a: str,
    key_b: str,
    family: str,
    candidate_keys: List[str],
    raw_occasion: str,
    allowed_roles_a: Optional[frozenset],
    allowed_roles_b: Optional[frozenset],
) -> Optional[CompatibilityViolation]:
    term_a, term_b = rule.get(key_a), rule.get(key_b)
    if not term_a or not term_b:
        return None
    severity, matched_key = _lookup_occasion_severity(rule.get("occasions") or {}, candidate_keys)
    if severity is None or severity <= 0:
        return None
    match_a = _find_matching_item(items, term_a, allowed_roles_a)
    match_b = _find_matching_item(items, term_b, allowed_roles_b)
    if not match_a or not match_b or _iid(match_a) == _iid(match_b):
        return None
    bucket = _severity_bucket(severity, raw_occasion)
    if not bucket:
        return None
    return CompatibilityViolation(
        rule_id=str(rule.get("rule_id") or ""),
        family=family,
        severity=bucket,
        reason=str(rule.get("reason") or ""),
        offending_item_ids=[_iid(match_a), _iid(match_b)],
        offending_roles=[_item_role(match_a), _item_role(match_b)],
        repairable=True,
    )


def _eval_occasion_incompatibility_rule(
    rule: Dict[str, Any], items: List[Dict[str, Any]], candidate_keys: List[str], raw_occasion: str
) -> Optional[CompatibilityViolation]:
    category = str(rule.get("category") or "").strip().lower()
    if category not in _ONLY_GARMENT_FOOTWEAR_CATEGORIES:
        return None
    rule_occ_norm = _normalize_occasion_key(rule.get("occasion"))
    if not rule_occ_norm or rule_occ_norm not in candidate_keys:
        return None
    try:
        severity = int(rule.get("severity") or 0)
    except Exception:
        return None
    if severity <= 0:
        return None
    match = _find_matching_item(items, rule.get("item"), _CATEGORY_ROLE_MAP.get(category))
    if not match:
        return None
    bucket = _severity_bucket(severity, raw_occasion)
    if not bucket:
        return None
    return CompatibilityViolation(
        rule_id=str(rule.get("rule_id") or ""),
        family="occasion_incompatibilities",
        severity=bucket,
        reason=str(rule.get("reason") or ""),
        offending_item_ids=[_iid(match)],
        offending_roles=[_item_role(match)],
        repairable=True,
    )


def _eval_dress_code_rule(
    rule: Dict[str, Any], items: List[Dict[str, Any]], explicit_dress_code: str
) -> Optional[CompatibilityViolation]:
    if str(rule.get("dress_code") or "").strip().lower() != explicit_dress_code:
        return None
    category = str(rule.get("category") or "").strip().lower()
    if category not in _ONLY_GARMENT_FOOTWEAR_CATEGORIES:
        return None
    try:
        severity = int(rule.get("severity") or 0)
    except Exception:
        return None
    if severity <= 0:
        return None
    match = _find_matching_item(items, rule.get("item"), _CATEGORY_ROLE_MAP.get(category))
    if not match:
        return None
    bucket = _dress_code_severity_bucket(severity)
    if not bucket:
        return None
    return CompatibilityViolation(
        rule_id=str(rule.get("rule_id") or ""),
        family="dress_code_violations",
        severity=bucket,
        reason=str(rule.get("reason") or ""),
        offending_item_ids=[_iid(match)],
        offending_roles=[_item_role(match)],
        repairable=True,
    )


def _dedupe_same_conflict(violations: List[CompatibilityViolation]) -> List[CompatibilityViolation]:
    """Same offending item-id set flagged by more than one rule -> keep only
    the single strongest violation (anti-double-counting, LOOP 5)."""
    best: Dict[Tuple[str, ...], CompatibilityViolation] = {}
    for v in violations:
        key = tuple(sorted({i for i in v.offending_item_ids if i}))
        if not key:
            continue
        existing = best.get(key)
        if existing is None or _BUCKET_ORDER.get(v.severity, 0) > _BUCKET_ORDER.get(existing.severity, 0):
            best[key] = v
    return list(best.values())


def _exception_trigger_matches(exc: Dict[str, Any], occasion: str, query: str) -> bool:
    # Deliberately requires the user's own free text, never the occasion
    # bucket alone: the pack's own override_principle is
    # "if the user explicitly requests an unconventional... interpretation"
    # (occasion_incompatibilities.json). Every exception's `applies_to` is
    # silhouette-pattern vocabulary (a deferred P0 family) that this adapter
    # has no reliable way to match against garment/footwear violations, so
    # trusting occasion-only triggers would blanket-suppress unrelated
    # formality violations any time the occasion happens to appear in an
    # exception's trigger list (e.g. nearly every business_casual board).
    query_text = f" {str(query or '').lower()} ".replace("_", " ")
    if not query_text.strip():
        return False
    trigger = exc.get("trigger") if isinstance(exc.get("trigger"), dict) else {}
    aes_triggers = [str(a).replace("_", " ").lower() for a in trigger.get("aesthetic") or []]
    if any(a and a in query_text for a in aes_triggers):
        return True
    if any(term in query_text for term in _BOLD_INTENT_TERMS):
        return True
    return False


def _apply_exceptions(
    violations: List[CompatibilityViolation],
    exceptions_pack: Optional[Dict[str, Any]],
    *,
    occasion: str,
    query: str,
) -> List[CompatibilityViolation]:
    if not violations or not exceptions_pack:
        return violations
    matching = [
        exc for exc in exceptions_pack.get("exceptions") or []
        if isinstance(exc, dict) and _exception_trigger_matches(exc, occasion, query)
    ]
    if not matching:
        return violations

    strongest_override = ""
    for exc in matching:
        override = str(exc.get("override") or "")
        if override == "full_override":
            strongest_override = "full_override"
            break
        if override == "strong_reduction" and strongest_override != "full_override":
            strongest_override = "strong_reduction"
        elif not strongest_override:
            strongest_override = override

    out: List[CompatibilityViolation] = []
    for v in violations:
        if v.severity == SEVERITY_HARD:
            # Exceptions never override hard dress-code / structural violations.
            out.append(v)
            continue
        if strongest_override in ("full_override", "strong_reduction"):
            if v.severity == SEVERITY_STRONG and strongest_override == "strong_reduction":
                v.severity = SEVERITY_SOFT
                v.exception_applied = strongest_override
                out.append(v)
            # full_override, or strong_reduction downgrading SOFT -> dropped entirely.
            elif strongest_override == "full_override":
                continue
            else:
                continue
        else:
            out.append(v)
    return out


def evaluate_outfit(
    items: List[Dict[str, Any]],
    *,
    occasion: str = "",
    query: str = "",
    exceptions_enabled: bool = True,
) -> List[CompatibilityViolation]:
    """Evaluate a candidate outfit's items against the active P0 rule
    families. Never raises - any internal failure logs
    STYLE_COMPAT_EVALUATION_FAILED and returns an empty list (existing AHVI
    checks remain authoritative)."""
    try:
        packs = _load_all()
        if not packs:
            return []
        clean_items = [i for i in (items or []) if isinstance(i, dict)]
        if not clean_items:
            return []
        candidate_keys = _candidate_occasion_keys(occasion)
        explicit_dress_code = _detect_explicit_dress_code(occasion, query)

        violations: List[CompatibilityViolation] = []

        g2g = packs.get("garment_to_garment")
        if g2g and candidate_keys:
            garment_roles = _CATEGORY_ROLE_MAP["garment"]
            for rule in g2g.get("rules") or []:
                v = _eval_pair_rule(
                    rule, clean_items, "garment_1", "garment_2",
                    "garment_to_garment", candidate_keys, occasion, garment_roles, garment_roles,
                )
                if v:
                    violations.append(v)

        g2f = packs.get("garment_to_footwear")
        if g2f and candidate_keys:
            for rule in g2f.get("rules") or []:
                v = _eval_pair_rule(
                    rule, clean_items, "garment", "footwear",
                    "garment_to_footwear", candidate_keys, occasion,
                    _CATEGORY_ROLE_MAP["garment"], _CATEGORY_ROLE_MAP["footwear"],
                )
                if v:
                    violations.append(v)

        oi = packs.get("occasion_incompatibilities")
        if oi and candidate_keys:
            for rule in oi.get("rules") or []:
                v = _eval_occasion_incompatibility_rule(rule, clean_items, candidate_keys, occasion)
                if v:
                    violations.append(v)

        if explicit_dress_code:
            dcv = packs.get("dress_code_violations")
            if dcv:
                for rule in dcv.get("rules") or []:
                    v = _eval_dress_code_rule(rule, clean_items, explicit_dress_code)
                    if v:
                        violations.append(v)

        violations = _dedupe_same_conflict(violations)

        if exceptions_enabled and violations:
            violations = _apply_exceptions(
                violations, packs.get("occasion_based_exceptions"),
                occasion=occasion, query=query,
            )

        for v in violations:
            logger.info(
                "STYLE_COMPAT_SHADOW rule_id=%s family=%s severity=%s occasion=%s "
                "offending_role=%s repairable=%s",
                v.rule_id, v.family, v.severity, occasion,
                ",".join(v.offending_roles), v.repairable,
            )
        return violations
    except Exception:
        logger.warning("STYLE_COMPAT_EVALUATION_FAILED stage=evaluate", exc_info=True)
        return []
