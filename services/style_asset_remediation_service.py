"""Pure safety contract for reviewed Style asset remediation batches.

This module performs no I/O. It converts a refreshed Appwrite snapshot plus a
reviewed proposal into exact-$id update and field-scoped rollback plans.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping

from services.style_asset_metadata_contract import normalize_style_asset_metadata


DERIVED_FIELDS = {
    "metadata_score", "metadata_status", "missing_metadata_fields",
    "metadata_version",
}
MUTABLE_METADATA_FIELDS = {
    "role", "category", "sub_category",
    "gender_fit", "colors", "pattern", "material", "finish", "visual_noise",
    "statement_level", "archetypes", "occasion_families", "formality",
    "energy", "movement", "traits", "weather_tags", "cultural_context",
    "temperature_min_c", "temperature_max_c", "fabric_weight",
    "layering_suitability", "rain_suitable", "wind_suitable",
    "professional_safe", "professionalism_score", "client_meeting_score",
    "boardroom_score", "safety_tags", *DERIVED_FIELDS,
}
PROPOSAL_INPUT_FIELDS = MUTABLE_METADATA_FIELDS | {"occasions", "subcategory"}
CHECKSUM_FIELDS = MUTABLE_METADATA_FIELDS | {
    "occasions", "subcategory", "gender",
}
MEDIA_FIELDS = {
    "image_url", "board_image_url", "normalized_url", "cutout_url",
    "transparent_image_url", "rmbg_url", "catalog_image_url", "asset_url",
    "asset_path", "r2_key", "board_r2_key",
}


class RemediationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _contains_base64(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower().startswith("data:") or "base64," in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_base64(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_base64(item) for item in value)
    return False


def safe_field_checksum(document: Mapping[str, Any]) -> str:
    payload = {
        key: document.get(key)
        for key in sorted(CHECKSUM_FIELDS)
        if key in document
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_asset_id_document_id_map(
    snapshot: Iterable[Mapping[str, Any]],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    document_ids: set[str] = set()
    for row in snapshot:
        asset_id = str(row.get("asset_id") or "").strip()
        document_id = str(row.get("$id") or "").strip()
        if not asset_id or not document_id:
            raise RemediationError(
                "SNAPSHOT_IDENTITY_INVALID",
                "snapshot rows require both asset_id and Appwrite $id",
            )
        if asset_id in mapping or document_id in document_ids:
            raise RemediationError(
                "SNAPSHOT_IDENTITY_INVALID",
                "snapshot asset_id and $id values must be unique",
            )
        mapping[asset_id] = document_id
        document_ids.add(document_id)
    return mapping


def sanitize_metadata_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only batch-owned canonical metadata fields.

    Appwrite system fields, identifiers, URLs and media/storage fields cannot
    enter either an apply or rollback request.
    """
    return {
        key: value
        for key, value in payload.items()
        if key in MUTABLE_METADATA_FIELDS
        and not key.startswith("$")
        and key not in MEDIA_FIELDS
        and not key.lower().endswith("_url")
    }


def validate_proposal_payload(payload: Mapping[str, Any]) -> None:
    forbidden = sorted(key for key in payload if key not in PROPOSAL_INPUT_FIELDS)
    if forbidden:
        raise RemediationError(
            "PAYLOAD_FIELD_FORBIDDEN",
            f"proposal contains forbidden or unknown fields: {forbidden}",
        )
    if _contains_base64(payload):
        raise RemediationError(
            "PAYLOAD_CONTENT_FORBIDDEN", "proposal contains base64 or data content"
        )


def _canonical_payload(raw: Mapping[str, Any], proposed: Mapping[str, Any]) -> Dict[str, Any]:
    validate_proposal_payload(proposed)
    evidence = sanitize_metadata_payload({
        key: value for key, value in proposed.items() if key not in DERIVED_FIELDS
    })
    if "occasion_families" not in evidence and proposed.get("occasions") is not None:
        evidence["occasion_families"] = proposed.get("occasions")
    if "sub_category" not in evidence and proposed.get("subcategory") is not None:
        evidence["sub_category"] = proposed.get("subcategory")
    normalized = normalize_style_asset_metadata(
        {**dict(raw), **evidence}, trusted_style_asset_source=True
    )
    invalid = set(normalized.get("missing_metadata_fields") or []).intersection({
        "invalid_occasion_families", "invalid_safety_tags",
    })
    if invalid:
        raise RemediationError(
            "VOCABULARY_INVALID",
            f"proposal contains unknown canonical vocabulary: {sorted(invalid)}",
        )

    for field in DERIVED_FIELDS:
        if field in proposed and proposed[field] != normalized.get(field):
            raise RemediationError(
                "METADATA_RECOMPUTE_MISMATCH",
                f"proposal {field} disagrees with canonical contract: "
                f"{proposed[field]!r} != {normalized.get(field)!r}"
            )
    canonical = {
        field: normalized.get(field)
        for field in MUTABLE_METADATA_FIELDS
        if field in normalized or field in DERIVED_FIELDS
    }
    return sanitize_metadata_payload(canonical)


def build_remediation_plan(
    snapshot: Iterable[Mapping[str, Any]],
    proposals: Mapping[str, Mapping[str, Any]],
    *,
    approved_asset_ids: Iterable[str] | None = None,
    rejected_asset_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    rows = [dict(row) for row in snapshot]
    mapping = build_asset_id_document_id_map(rows)
    by_asset_id = {str(row["asset_id"]): row for row in rows}
    unknown_assets = sorted(set(proposals).difference(mapping))
    if unknown_assets:
        raise RemediationError(
            "APPROVED_ASSET_MISSING",
            f"proposal assets missing from refreshed snapshot: {unknown_assets}",
        )
    approved = set(approved_asset_ids if approved_asset_ids is not None else proposals)
    rejected = set(rejected_asset_ids)
    if approved != set(proposals):
        raise RemediationError(
            "APPROVED_ALLOWLIST_MISMATCH",
            "approved asset allow-list must exactly match proposal assets",
        )
    overlap = sorted(approved.intersection(rejected))
    if overlap:
        raise RemediationError(
            "REJECTED_CONTROL_INCLUDED",
            f"rejected controls cannot enter the update allow-list: {overlap}",
        )

    updates = []
    rollbacks = []
    for asset_id, proposed in proposals.items():
        raw = by_asset_id[asset_id]
        canonical = _canonical_payload(raw, proposed)
        changes = {
            key: value for key, value in canonical.items() if raw.get(key) != value
        }
        rollback = {
            key: raw[key] if key in raw else None
            for key in changes
        }
        updates.append({
            "asset_id": asset_id,
            "document_id": mapping[asset_id],
            "snapshot_updated_at": raw.get("$updatedAt"),
            "snapshot_safe_checksum": safe_field_checksum(raw),
            "changes": changes,
            "prior_values": sanitize_metadata_payload(rollback),
        })
        rollbacks.append({
            "asset_id": asset_id,
            "document_id": mapping[asset_id],
            "changes": sanitize_metadata_payload(rollback),
        })
    return {
        "asset_id_to_document_id": mapping,
        "updates": updates,
        "rollbacks": rollbacks,
    }


def verify_snapshot_precondition(
    planned: Mapping[str, Any], live_document: Mapping[str, Any]
) -> None:
    if (
        str(live_document.get("$id") or "") != str(planned.get("document_id") or "")
        or live_document.get("$updatedAt") != planned.get("snapshot_updated_at")
        or safe_field_checksum(live_document) != planned.get("snapshot_safe_checksum")
    ):
        raise RemediationError(
            "DOCUMENT_CHANGED_SINCE_REVIEW",
            f"document {planned.get('document_id')} no longer matches reviewed snapshot",
        )


def execute_remediation_plan(
    plan: Mapping[str, Any],
    *,
    fetch_document: Any,
    update_document: Any,
    journal: list[Dict[str, Any]] | None = None,
    now: Any = None,
) -> Dict[str, Any]:
    """Execute an already reviewed plan through injected document functions.

    The caller owns persistence. This deterministic loop stops on the first
    failure and records only completed documents.
    """
    entries = journal if journal is not None else []
    clock = now or (lambda: datetime.now(timezone.utc).isoformat())
    for planned in plan.get("updates", []):
        try:
            live = fetch_document(planned["document_id"])
            verify_snapshot_precondition(planned, live)
            result = update_document(planned["document_id"], dict(planned["changes"]))
            resulting = result if isinstance(result, Mapping) else fetch_document(planned["document_id"])
            entries.append({
                "document_id": planned["document_id"],
                "asset_id": planned["asset_id"],
                "changed_fields": sorted(planned["changes"]),
                "prior_values": dict(planned["prior_values"]),
                "resulting_checksum": safe_field_checksum(resulting),
                "timestamp": clock(),
            })
        except RemediationError as exc:
            return {
                "success": False,
                "failure_code": exc.code,
                "completed_document_ids": [entry["document_id"] for entry in entries],
                "journal": entries,
            }
        except Exception:
            return {
                "success": False,
                "failure_code": "DOCUMENT_UPDATE_FAILED",
                "completed_document_ids": [entry["document_id"] for entry in entries],
                "journal": entries,
            }
    return {
        "success": True,
        "failure_code": None,
        "completed_document_ids": [entry["document_id"] for entry in entries],
        "journal": entries,
    }


def rollback_journal(
    journal: Iterable[Mapping[str, Any]],
    *,
    fetch_document: Any,
    update_document: Any,
) -> Dict[str, Any]:
    completed: list[str] = []
    for entry in reversed([dict(item) for item in journal]):
        document_id = str(entry.get("document_id") or "")
        live = fetch_document(document_id)
        if (
            str(live.get("$id") or "") != document_id
            or safe_field_checksum(live) != entry.get("resulting_checksum")
        ):
            return {
                "success": False,
                "failure_code": "ROLLBACK_CONCURRENCY_CONFLICT",
                "completed_document_ids": completed,
            }
        update_document(document_id, sanitize_metadata_payload(entry.get("prior_values") or {}))
        completed.append(document_id)
    return {
        "success": True,
        "failure_code": None,
        "completed_document_ids": completed,
    }
