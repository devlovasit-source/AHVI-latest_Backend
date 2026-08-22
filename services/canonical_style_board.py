"""Pure canonical Style Board lifecycle primitives (Batch 1).

This module deliberately imports no routers, generators, provider clients, or
persistence services.  Shadow canonicalization therefore operates only on the
already-generated candidates and response context supplied by its caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from services.style_item_contract import (
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
    normalize_style_item,
)

logger = logging.getLogger("ahvi.canonical_style_board")

CANONICAL_SCHEMA_VERSION = "1.0"
CONTEXT_SCHEMA_VERSION = "1.0"
REQUEST_FINGERPRINT_VERSION = "1.0"
FINALIZER_VERSION = "1.0"
SERIALIZER_VERSION = "1.0"
GUARD_VERSION = "1.0"
SHADOW_FLAG_VERSION = "1.0"
LEGACY_PROJECTION_VERSION = "1.0"

SCENARIOS = frozenset({"wardrobe", "fixed_anchor", "capsule", "visual_inspiration"})
SOURCE_POLICIES = frozenset({"wardrobe", "mixed", "style_asset", "unknown"})
FLOWS = frozenset({"fixed_anchor", "wardrobe", "capsule", "visual_reasoning"})

_BOARD_NAMESPACE = uuid.UUID("5d59275b-b694-5d8b-bf66-c49f49713878")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_VOLATILE_CONTEXT_KEYS = frozenset(
    {
        "timestamp", "created_at", "updated_at", "fetched_at", "request_time",
        "request_timestamp", "current_time", "now", "latency_ms", "trace_id",
    }
)
_MATERIAL_CONTEXT_KEYS = frozenset(
    {
        "canonical_occasion", "occasion", "occasion_brief", "gender", "style_dna",
        "style_preferences", "weather", "weather_context", "event_context",
        "wardrobe", "wardrobe_items", "wardrobe_summary", "recently_worn_ids",
        "underworn_ids", "wear_counts", "saved_item_ids", "favorite_colors",
        "profile_version", "wardrobe_version", "memory_version", "weather_version",
        "event_context_version", "style_dna_version", "context_provenance",
    }
)
_WARDROBE_LIST_KEYS = frozenset({"wardrobe", "wardrobe_items"})
_UNORDERED_ID_LIST_KEYS = frozenset(
    {"recently_worn_ids", "underworn_ids", "saved_item_ids", "anchor_ids"}
)
_LEGACY_ALIAS_KEYS = frozenset(
    {
        "id", "sourcePolicy", "can_save", "can_feedback", "can_share", "can_wear",
        "can_lock", "can_shuffle", "read_only",
    }
)


class CanonicalStyleBoardError(ValueError):
    """Typed invalid-input failure at the pure contract boundary."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class Capability:
    allowed: bool
    reason: str = ""
    retryable: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"allowed": self.allowed}
        if self.reason:
            result["reason"] = self.reason
        if self.retryable is not None:
            result["retryable"] = self.retryable
        return result


@dataclass(frozen=True)
class ScenarioPolicy:
    scenario: str
    require_complete_outfit: bool
    require_owned_items: bool = False
    require_anchor_preservation: bool = False
    require_capsule_completeness: bool = False
    require_style_assets: bool = False
    mutable: bool = True
    checks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalStyleBoardDraft:
    """Persistence-free canonical board produced by the Batch 1 finalizer.

    Registered boards intentionally have no Batch 1 model or factory.  The
    lifecycle fields are explicit so a draft can never be mistaken for an
    immutable revision merely because it has been canonicalized in memory.
    """

    schema_version: str
    board_id: None
    revision: None
    lifecycle_status: str
    persisted: bool
    feedback_context: None
    generation_request_id: str
    board_variant_key: str
    created_at: str
    scenario: str
    source_policy: str
    items: List[Dict[str, Any]]
    request_fingerprint: str
    request_fingerprint_version: str
    candidate_snapshot_hash: str
    finalizer_version: str
    serializer_version: str
    guard_version: str
    snapshot_hash: str
    provenance: Dict[str, Any]
    capabilities: Dict[str, Dict[str, Any]]
    extensions: Dict[str, Dict[str, Any]]

    def __post_init__(self) -> None:
        if (
            self.lifecycle_status != "draft"
            or self.persisted is not False
            or self.board_id is not None
            or self.revision is not None
            or self.feedback_context is not None
        ):
            raise CanonicalStyleBoardError(
                "INVALID_DRAFT_LIFECYCLE",
                "A canonical draft cannot claim registered lifecycle state",
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "board_id": self.board_id,
            "revision": self.revision,
            "lifecycle_status": self.lifecycle_status,
            "persisted": self.persisted,
            "feedback_context": self.feedback_context,
            "generation_request_id": self.generation_request_id,
            "board_variant_key": self.board_variant_key,
            "created_at": self.created_at,
            "scenario": self.scenario,
            "source_policy": self.source_policy,
            "items": [dict(item) for item in self.items],
            "request_fingerprint": self.request_fingerprint,
            "request_fingerprint_version": self.request_fingerprint_version,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "finalizer_version": self.finalizer_version,
            "serializer_version": self.serializer_version,
            "guard_version": self.guard_version,
            "snapshot_hash": self.snapshot_hash,
            "provenance": dict(self.provenance),
            "capabilities": {
                action: dict(capability)
                for action, capability in self.capabilities.items()
            },
            "extensions": {
                namespace: dict(value)
                for namespace, value in self.extensions.items()
            },
        }


SCENARIO_POLICIES: Dict[str, ScenarioPolicy] = {
    "wardrobe": ScenarioPolicy(
        scenario="wardrobe",
        require_complete_outfit=True,
        require_owned_items=True,
        checks=("outfit", "ownership", "source_policy", "duplication", "occasion", "weather", "gender", "private_wear"),
    ),
    "fixed_anchor": ScenarioPolicy(
        scenario="fixed_anchor",
        require_complete_outfit=True,
        require_anchor_preservation=True,
        checks=("outfit", "ownership", "source_policy", "duplication", "occasion", "weather", "gender", "private_wear", "anchor"),
    ),
    "capsule": ScenarioPolicy(
        scenario="capsule",
        require_complete_outfit=True,
        require_capsule_completeness=True,
        checks=("outfit", "ownership", "source_policy", "duplication", "occasion", "weather", "gender", "private_wear", "capsule_completeness"),
    ),
    "visual_inspiration": ScenarioPolicy(
        scenario="visual_inspiration",
        require_complete_outfit=False,
        require_style_assets=True,
        mutable=False,
        checks=("asset", "source_policy", "duplication", "occasion", "weather", "gender", "private_wear"),
    ),
}


@dataclass(frozen=True)
class FinalizationIssue:
    candidate_key: str
    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"candidate_key": self.candidate_key, "code": self.code, "message": self.message}


@dataclass
class FinalizationResult:
    boards: List[CanonicalStyleBoardDraft] = field(default_factory=list)
    rejected: List[FinalizationIssue] = field(default_factory=list)
    candidate_snapshot_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boards": [board.to_dict() for board in self.boards],
            "rejected": [issue.to_dict() for issue in self.rejected],
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
        }


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _stable_item_sort_key(value: Any) -> Tuple[str, str]:
    if isinstance(value, Mapping):
        item_id = canonical_item_id(dict(value))
        return (item_id, stable_hash(value))
    return (str(value), stable_hash(value))


def normalize_for_hash(
    value: Any, *, parent_key: str = "", exclude_volatile: bool = False,
) -> Any:
    """Return a JSON-safe deterministic representation for hashing.

    Dict keys are ordered by serialization, null/empty dict values are omitted,
    and wardrobe and known ID sets are stably ordered. Volatile time/trace keys
    are excluded only for context hashing. Other lists retain order because
    order can affect generation.
    """
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda key: str(key)):
            key = str(raw_key)
            if exclude_volatile and key.lower() in _VOLATILE_CONTEXT_KEYS:
                continue
            child = normalize_for_hash(
                value[raw_key], parent_key=key, exclude_volatile=exclude_volatile,
            )
            if not _is_empty(child):
                normalized[key] = child
        return normalized
    if isinstance(value, (set, frozenset)):
        children = [
            normalize_for_hash(item, parent_key=parent_key, exclude_volatile=exclude_volatile)
            for item in value
        ]
        return sorted((item for item in children if not _is_empty(item)), key=canonical_json)
    if isinstance(value, (list, tuple)):
        children = [
            normalize_for_hash(item, parent_key=parent_key, exclude_volatile=exclude_volatile)
            for item in value
        ]
        children = [item for item in children if not _is_empty(item)]
        if parent_key in _WARDROBE_LIST_KEYS:
            children.sort(key=_stable_item_sort_key)
        elif parent_key in _UNORDERED_ID_LIST_KEYS:
            children.sort(key=canonical_json)
        return children
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (bool, int, float)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_for_hash(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def material_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    source = context if isinstance(context, Mapping) else {}
    selected = {key: source[key] for key in _MATERIAL_CONTEXT_KEYS if key in source}
    return {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "context": normalize_for_hash(selected, exclude_volatile=True),
    }


def context_hash(context: Optional[Mapping[str, Any]]) -> str:
    return stable_hash(material_context(context))


def request_fingerprint(
    *,
    intent: str,
    scenario: str,
    canonical_context: Optional[Mapping[str, Any]],
    source_policy: str,
    anchor_ids: Iterable[Any] = (),
    user_context_versions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    scenario = validate_scenario(scenario)
    source_policy = validate_source_policy(source_policy)
    normalized_intent = re.sub(r"\s+", " ", str(intent or "")).strip().lower()
    ctx_hash = context_hash(canonical_context)
    payload = {
        "version": REQUEST_FINGERPRINT_VERSION,
        "intent": normalized_intent,
        "scenario": scenario,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "context_hash": ctx_hash,
        "source_policy": source_policy,
        "anchor_ids": sorted({str(value).strip() for value in anchor_ids if str(value).strip()}),
        "user_context_versions": normalize_for_hash(user_context_versions or {}),
    }
    return {
        "request_fingerprint": stable_hash(payload),
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "context_hash": ctx_hash,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
    }


def candidate_snapshot_hash(candidates: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash([dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)])


def validate_scenario(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in SCENARIOS:
        raise CanonicalStyleBoardError("INVALID_SCENARIO", f"Unsupported Style Board scenario: {normalized or '<empty>'}")
    return normalized


def validate_source_policy(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in SOURCE_POLICIES:
        raise CanonicalStyleBoardError("INVALID_SOURCE_POLICY", f"Unsupported source policy: {normalized or '<empty>'}")
    return normalized


def _actionable_items(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(item) for item in items if isinstance(item, Mapping) and item.get("actionable", True) is not False]


def derive_source_policy(items: Iterable[Mapping[str, Any]]) -> str:
    """Derive board policy strictly from actionable item provenance."""
    actionable = _actionable_items(items)
    if not actionable:
        return "unknown"
    sources = {canonical_item_source(item) for item in actionable}
    if "unknown" in sources:
        return "unknown"
    if sources == {"wardrobe"}:
        return "wardrobe"
    if sources == {"style_asset"}:
        return "style_asset"
    if "wardrobe" in sources and len(sources) > 1:
        return "mixed"
    return "unknown"


def _candidate_items(candidate: Mapping[str, Any]) -> List[Dict[str, Any]]:
    for key in ("items", "board_items", "outfit_items", "products"):
        raw = candidate.get(key)
        if isinstance(raw, list) and any(isinstance(item, Mapping) for item in raw):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _variant_signature(items: Sequence[Mapping[str, Any]], scenario: str) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "items": [
            {
                "item_id": canonical_item_id(dict(item)),
                "role": canonical_item_role(dict(item)),
                "source": canonical_item_source(dict(item)),
                "anchor": bool(item.get("locked") or item.get("is_anchor") or item.get("anchor")),
            }
            for item in items
            if isinstance(item, Mapping)
        ],
    }


def board_variant_key(candidate: Mapping[str, Any], scenario: str) -> str:
    scenario = validate_scenario(scenario)
    items = _candidate_items(candidate)
    return stable_hash(_variant_signature(items, scenario))


def derive_registered_board_id(
    *, user_id: str, flow: str, generation_request_id: str, variant_key: str,
) -> str:
    """Calculate a future registered identity without assigning it to a draft.

    Persistence and registered-board construction belong to Batch 2A.  This
    helper is intentionally not called by ``finalize_candidates``.
    """
    flow = str(flow or "").strip().lower()
    if flow not in FLOWS:
        raise CanonicalStyleBoardError("INVALID_FLOW", f"Unsupported Style Board flow: {flow or '<empty>'}")
    values = (str(user_id).strip(), flow, str(generation_request_id).strip(), str(variant_key).strip())
    if not all(values):
        raise CanonicalStyleBoardError("INVALID_BOARD_ID_INPUT", "Board identity inputs cannot be empty")
    return str(uuid.uuid5(_BOARD_NAMESPACE, "|".join(values)))


def normalize_created_at(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CanonicalStyleBoardError("CREATED_AT_REQUIRED", "created_at must be supplied by the caller")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CanonicalStyleBoardError("CREATED_AT_INVALID", "created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CanonicalStyleBoardError("CREATED_AT_INVALID", "created_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def registration_outage_capabilities() -> Dict[str, Dict[str, Any]]:
    return _capabilities_dict(
        save=Capability(False, "revision_registration_unavailable", True),
        feedback=Capability(False, "durable_revision_required", True),
        share=Capability(True),
        wear=Capability(False, "source_verification_unavailable", True),
        lock=Capability(False, "durable_revision_required", True),
        shuffle=Capability(False, "durable_revision_required", True),
    )


def unknown_provenance_capabilities() -> Dict[str, Dict[str, Any]]:
    return _capabilities_dict(
        save=Capability(True), feedback=Capability(True), share=Capability(True),
        wear=Capability(False, "source_provenance_unknown", False),
        lock=Capability(False, "source_provenance_unknown", False),
        shuffle=Capability(False, "source_provenance_unknown", False),
    )


def _capabilities_dict(**values: Capability) -> Dict[str, Dict[str, Any]]:
    return {name: capability.to_dict() for name, capability in values.items()}


def _candidate_key(candidate: Mapping[str, Any], index: int, scenario: str) -> str:
    explicit = candidate.get("id") or candidate.get("board_id") or candidate.get("title")
    return str(explicit or board_variant_key(candidate, scenario) or index)


def _validate_candidate(
    candidate: Mapping[str, Any], *, scenario: str, anchor_ids: Sequence[str], source_policy: str,
) -> Optional[Tuple[str, str]]:
    policy = SCENARIO_POLICIES[scenario]
    items = _candidate_items(candidate)
    actionable = _actionable_items(items)
    if not actionable:
        return ("BOARD_ITEMS_REQUIRED", "Candidate has no structured actionable items")
    ids = [canonical_item_id(item) for item in actionable]
    if any(not item_id for item_id in ids):
        return ("ITEM_ID_REQUIRED", "Every actionable item must have a stable identifier")
    if len(ids) != len(set(ids)):
        return ("DUPLICATE_ITEM", "A board cannot contain the same actionable item more than once")
    if derive_source_policy(actionable) != source_policy:
        return ("SOURCE_POLICY_MISMATCH", "Derived item provenance does not match the board source policy")
    if source_policy == "unknown":
        return ("SOURCE_PROVENANCE_UNKNOWN", "Generated boards require proven actionable-item provenance")
    if policy.require_owned_items and source_policy != "wardrobe":
        return ("OWNERSHIP_REQUIRED", "Wardrobe boards require proven user-owned items")
    roles = {canonical_item_role(item) for item in actionable}
    if policy.require_complete_outfit:
        has_base = "dress" in roles or {"top", "bottom"} <= roles
        if not has_base or "footwear" not in roles:
            return ("OUTFIT_INCOMPLETE", "Wearable boards require a base outfit and footwear")
    if policy.require_anchor_preservation:
        missing = sorted(set(anchor_ids) - set(ids))
        if missing:
            return ("ANCHOR_MISSING", "Fixed-anchor board does not preserve every requested anchor")
    if policy.require_style_assets and source_policy != "style_asset":
        return ("STYLE_ASSET_REQUIRED", "Visual inspiration requires proven style assets")
    return None


def _canonical_snapshot_payload(board: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in board.items()
        if key != "snapshot_hash" and key not in _LEGACY_ALIAS_KEYS
    }


def finalize_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    user_id: str,
    flow: str,
    generation_request_id: str,
    scenario: str,
    request_fingerprint_value: str,
    created_at: str,
    anchor_ids: Iterable[Any] = (),
    provenance: Optional[Mapping[str, Any]] = None,
    extensions: Optional[Mapping[str, Any]] = None,
) -> FinalizationResult:
    """Purely validate and canonicalize an already-generated candidate pool."""
    scenario = validate_scenario(scenario)
    created_at = normalize_created_at(created_at)
    flow = str(flow or "").strip().lower()
    generation_request_id = str(generation_request_id or "").strip()
    if not generation_request_id:
        raise CanonicalStyleBoardError("GENERATION_REQUEST_ID_REQUIRED", "generation_request_id is required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(request_fingerprint_value or "")):
        raise CanonicalStyleBoardError("REQUEST_FINGERPRINT_INVALID", "request_fingerprint must be a SHA-256 hex digest")
    candidate_dicts = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
    pool_hash = candidate_snapshot_hash(candidate_dicts)
    anchors = sorted({str(value).strip() for value in anchor_ids if str(value).strip()})
    result = FinalizationResult(candidate_snapshot_hash=pool_hash)
    seen_variants = set()
    extension_input = extensions if isinstance(extensions, Mapping) else {}
    extension_envelope = {
        name: dict(extension_input.get(name) or {})
        for name in ("capsule", "visual_inspiration", "wardrobe", "fixed_anchor")
    }

    for index, candidate in enumerate(candidate_dicts):
        items = _candidate_items(candidate)
        source_policy = derive_source_policy(items)
        key = _candidate_key(candidate, index, scenario)
        issue = _validate_candidate(
            candidate, scenario=scenario, anchor_ids=anchors, source_policy=source_policy,
        )
        if issue:
            result.rejected.append(FinalizationIssue(key, issue[0], issue[1]))
            continue
        variant_key = board_variant_key(candidate, scenario)
        if variant_key in seen_variants:
            result.rejected.append(FinalizationIssue(key, "DUPLICATE_VARIANT", "Equivalent board variant already accepted"))
            continue
        seen_variants.add(variant_key)
        normalized_items = [normalize_style_item(item) for item in items]
        capabilities = registration_outage_capabilities()
        board: Dict[str, Any] = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "board_id": None,
            "revision": None,
            "lifecycle_status": "draft",
            "persisted": False,
            "feedback_context": None,
            "generation_request_id": generation_request_id,
            "board_variant_key": variant_key,
            "created_at": created_at,
            "scenario": scenario,
            "source_policy": source_policy,
            "items": normalized_items,
            "request_fingerprint": request_fingerprint_value,
            "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
            "candidate_snapshot_hash": pool_hash,
            "finalizer_version": FINALIZER_VERSION,
            "serializer_version": SERIALIZER_VERSION,
            "guard_version": GUARD_VERSION,
            "provenance": {
                **dict(provenance or {}),
                "request_fingerprint_algorithm": "sha256",
                "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
                "context_schema_version": CONTEXT_SCHEMA_VERSION,
            },
            "capabilities": capabilities,
            "extensions": extension_envelope,
        }
        board["snapshot_hash"] = stable_hash(_canonical_snapshot_payload(board))
        result.boards.append(CanonicalStyleBoardDraft(**board))
    return result


def project_legacy_aliases(
    board: Mapping[str, Any] | CanonicalStyleBoardDraft,
) -> Dict[str, Any]:
    """Add temporary legacy aliases without changing canonical semantics."""
    source = board.to_dict() if isinstance(board, CanonicalStyleBoardDraft) else dict(board)
    projected = dict(source)
    capabilities = source.get("capabilities") if isinstance(source.get("capabilities"), Mapping) else {}
    projected["id"] = source.get("board_id")
    projected["sourcePolicy"] = str(source.get("source_policy") or "")
    for action in ("save", "feedback", "share", "wear", "lock", "shuffle"):
        setting = capabilities.get(action) if isinstance(capabilities, Mapping) else None
        projected[f"can_{action}"] = bool(setting.get("allowed")) if isinstance(setting, Mapping) else False
    projected["read_only"] = not (
        projected["can_wear"] or projected["can_lock"] or projected["can_shuffle"]
    )
    return projected


def _flag_name(flow: str, mode: str) -> str:
    normalized_flow = str(flow or "").strip().lower()
    if normalized_flow not in FLOWS:
        raise CanonicalStyleBoardError("INVALID_FLOW", f"Unsupported Style Board flow: {normalized_flow or '<empty>'}")
    if mode not in {"SHADOW", "SERVE"}:
        raise CanonicalStyleBoardError("INVALID_FLAG_MODE", "Flag mode must be SHADOW or SERVE")
    return f"CANONICAL_STYLE_{normalized_flow.upper()}_{mode}"


def shadow_enabled(flow: str) -> bool:
    return str(os.getenv(_flag_name(flow, "SHADOW"), "false")).strip().lower() in _TRUE_VALUES


def serve_enabled(flow: str) -> bool:
    return str(os.getenv(_flag_name(flow, "SERVE"), "false")).strip().lower() in _TRUE_VALUES


def cohort_bucket(user_id: str, flow: str, *, buckets: int = 10_000) -> int:
    if buckets <= 0:
        raise CanonicalStyleBoardError("INVALID_BUCKET_COUNT", "buckets must be positive")
    flow = str(flow or "").strip().lower()
    if flow not in FLOWS:
        raise CanonicalStyleBoardError("INVALID_FLOW", f"Unsupported Style Board flow: {flow or '<empty>'}")
    digest = hashlib.sha256(f"{str(user_id).strip()}|{flow}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % buckets


def run_shadow_canonicalization(
    *,
    flow: str,
    user_id: str,
    candidates: Sequence[Mapping[str, Any]],
    generation_request_id: str,
    scenario: str,
    request_fingerprint_value: str,
    created_at: str,
    legacy_path_version: str,
    anchor_ids: Iterable[Any] = (),
    provenance: Optional[Mapping[str, Any]] = None,
    extensions: Optional[Mapping[str, Any]] = None,
    metric_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[FinalizationResult]:
    """Run zero-provider shadow finalization and emit a versioned comparison event."""
    if not shadow_enabled(flow):
        return None
    result = finalize_candidates(
        candidates,
        user_id=user_id,
        flow=flow,
        generation_request_id=generation_request_id,
        scenario=scenario,
        request_fingerprint_value=request_fingerprint_value,
        created_at=created_at,
        anchor_ids=anchor_ids,
        provenance=provenance,
        extensions=extensions,
    )
    event = {
        "event": "style_board.canonical_shadow_comparison",
        "flow": flow,
        "shadow_flag_version": SHADOW_FLAG_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "finalizer_version": FINALIZER_VERSION,
        "serializer_version": SERIALIZER_VERSION,
        "guard_version": GUARD_VERSION,
        "legacy_path_version": str(legacy_path_version or "unknown"),
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "cohort_bucket": cohort_bucket(user_id, flow),
        "legacy_candidate_count": len([c for c in candidates if isinstance(c, Mapping)]),
        "canonical_board_count": len(result.boards),
        "rejected_count": len(result.rejected),
        "rejection_codes": sorted({issue.code for issue in result.rejected}),
    }
    if metric_sink is not None:
        metric_sink(dict(event))
    else:
        logger.info("%s", json.dumps(event, sort_keys=True, separators=(",", ":")))
    return result
