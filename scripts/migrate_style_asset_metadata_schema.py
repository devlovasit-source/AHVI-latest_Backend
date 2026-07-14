"""Diff and optionally create only missing canonical Style metadata fields.

Dry-run is the default. Apply requires both ``--apply`` and
``--confirm-apply``. Created attributes are polled until Appwrite reports
``available``; mismatched live definitions fail closed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from scripts import create_style_assets_collection as appwrite


ATTRIBUTE_SPECS = (
    {"key": "source_origin", "type": "string", "size": 128},
    {"key": "role", "type": "string", "size": 32},
    {"key": "sub_category", "type": "string", "size": 128},
    {"key": "gender_fit", "type": "string", "size": 32},
    {"key": "board_image_url", "type": "string", "size": 2048},
    {"key": "normalized_url", "type": "string", "size": 2048},
    {"key": "cutout_url", "type": "string", "size": 2048},
    {"key": "pattern", "type": "string", "size": 96},
    {"key": "material", "type": "string", "size": 96},
    {"key": "finish", "type": "string", "size": 96},
    {"key": "occasion_families", "type": "string", "size": 96, "array": True},
    {"key": "traits", "type": "string", "size": 96, "array": True},
    {"key": "weather_tags", "type": "string", "size": 64, "array": True},
    {"key": "cultural_context", "type": "string", "size": 96, "array": True},
    *(
        {"key": key, "type": "float"}
        for key in (
            "visual_noise", "statement_level", "formality", "energy", "movement",
            "metadata_score", "professionalism_score", "client_meeting_score",
            "boardroom_score", "temperature_min_c", "temperature_max_c",
            "layering_suitability",
        )
    ),
    {"key": "metadata_version", "type": "string", "size": 32},
    {"key": "metadata_status", "type": "string", "size": 32},
    {"key": "missing_metadata_fields", "type": "string", "size": 96, "array": True},
    {"key": "metadata_updated_at", "type": "datetime"},
    {"key": "professional_safe", "type": "boolean"},
    {"key": "safety_tags", "type": "string", "size": 96, "array": True},
    {"key": "fabric_weight", "type": "string", "size": 16},
    {"key": "rain_suitable", "type": "boolean"},
    {"key": "wind_suitable", "type": "boolean"},
)


class SchemaMigrationError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = dict(details or {})


def _canonical_live_type(value: Any) -> str:
    live_type = str(value or "").lower()
    return "float" if live_type == "double" else live_type


def _list_live_attributes() -> list[Dict[str, Any]]:
    response = appwrite._request(
        "GET", f"{appwrite._base()}/{appwrite.COLLECTION_ID}"
    )
    if response.status_code != 200:
        raise RuntimeError(f"collection schema read failed {response.status_code}")
    body = response.json()
    return [row for row in body.get("attributes", []) if isinstance(row, dict)]


def _collection_capacity() -> Dict[str, int]:
    response = appwrite._request(
        "GET", f"{appwrite._base()}/{appwrite.COLLECTION_ID}"
    )
    if response.status_code != 200:
        raise RuntimeError(f"collection capacity read failed {response.status_code}")
    body = response.json()
    return {
        "bytes_used": int(body.get("bytesUsed") or 0),
        "bytes_max": int(body.get("bytesMax") or 0),
        "attribute_count": len(body.get("attributes") or []),
    }


def _estimated_attribute_bytes(spec: Mapping[str, Any]) -> int:
    """Conservative upper bound for Appwrite's row-capacity preflight.

    String capacity assumes four UTF-8 bytes per declared character. A fixed
    per-column reserve covers database metadata and nullable-column overhead.
    """
    value_bytes = int(spec.get("size") or 8) * (4 if spec["type"] == "string" else 1)
    return value_bytes + 256


def verify_capacity(
    capacity: Mapping[str, Any], missing: Iterable[Mapping[str, Any]]
) -> Dict[str, int]:
    additions = [dict(spec) for spec in missing]
    bytes_used = int(capacity.get("bytes_used") or 0)
    bytes_max = int(capacity.get("bytes_max") or 0)
    current_count = int(capacity.get("attribute_count") or 0)
    estimated_addition = sum(_estimated_attribute_bytes(spec) for spec in additions)
    if bytes_max <= 0 or bytes_used + estimated_addition > bytes_max:
        raise SchemaMigrationError(
            "SCHEMA_CAPACITY_EXCEEDED",
            "live collection does not expose enough row capacity for the additive schema",
        )
    return {
        "current_attribute_count": current_count,
        "resulting_attribute_count": current_count + len(additions),
        "bytes_used": bytes_used,
        "bytes_max": bytes_max,
        "estimated_addition_bytes": estimated_addition,
        "estimated_remaining_bytes": bytes_max - bytes_used - estimated_addition,
    }


def compute_schema_diff(
    live_attributes: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    live = {str(row.get("key") or ""): dict(row) for row in live_attributes}
    missing = []
    mismatched = []
    unavailable = []
    existing = []
    for spec in ATTRIBUTE_SPECS:
        key = spec["key"]
        actual = live.get(key)
        if actual is None:
            missing.append(dict(spec))
            continue
        expected_type = spec["type"]
        actual_type = _canonical_live_type(actual.get("type"))
        type_ok = actual_type == expected_type
        array_ok = bool(actual.get("array")) == bool(spec.get("array"))
        required_ok = bool(actual.get("required")) == bool(spec.get("required"))
        default_ok = actual.get("default") == spec.get("default")
        size_ok = expected_type != "string" or int(actual.get("size") or 0) >= int(spec["size"])
        if not (type_ok and array_ok and required_ok and default_ok and size_ok):
            mismatched.append({
                "key": key,
                "expected": dict(spec),
                "actual": {
                    "type": actual_type,
                    "array": bool(actual.get("array")),
                    "required": bool(actual.get("required")),
                    "default": actual.get("default"),
                    "size": actual.get("size"),
                    "status": actual.get("status"),
                },
            })
        else:
            existing.append(key)
        if str(actual.get("status") or "available").lower() != "available":
            unavailable.append(key)
    return {
        "missing": missing,
        "mismatched": mismatched,
        "unavailable": sorted(unavailable),
        "existing": sorted(existing),
        "unexpected": sorted(set(live).difference(spec["key"] for spec in ATTRIBUTE_SPECS)),
    }


def _create(spec: Mapping[str, Any]) -> str:
    kind = spec["type"]
    payload: Dict[str, Any] = {"key": spec["key"], "required": False}
    if kind == "string":
        payload.update({"size": spec["size"], "array": bool(spec.get("array"))})
    else:
        payload["array"] = bool(spec.get("array"))
    response = appwrite._request("POST", appwrite._attribute_url(kind), payload)
    if response.status_code == 409:
        return "conflict"
    if response.status_code not in {200, 201, 202}:
        raise SchemaMigrationError(
            "SCHEMA_CREATE_FAILED",
            f"attribute {spec['key']} failed with status {response.status_code}",
        )
    return "created"


def _verify_exact_attribute(spec: Mapping[str, Any]) -> Dict[str, Any]:
    live = {
        str(row.get("key") or ""): row for row in _list_live_attributes()
    }
    actual = live.get(str(spec["key"]))
    compatible = bool(actual)
    if actual:
        compatible = (
            _canonical_live_type(actual.get("type")) == spec["type"]
            and bool(actual.get("array")) == bool(spec.get("array"))
            and bool(actual.get("required")) == bool(spec.get("required"))
            and actual.get("default") == spec.get("default")
            and (
                spec["type"] != "string"
                or int(actual.get("size") or 0) == int(spec["size"])
            )
            and str(actual.get("status") or "").lower() == "available"
        )
    if not compatible:
        raise SchemaMigrationError(
            "SCHEMA_CONFLICT",
            f"attribute {spec['key']} is not exactly compatible after creation",
        )
    return {
        "key": spec["key"], "type": spec["type"], "size": spec.get("size"),
        "array": bool(spec.get("array")), "required": bool(spec.get("required")),
        "default": spec.get("default"), "status": "available",
    }


def _wait_until_available(keys: set[str], *, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending = set(keys)
    while pending:
        live = {str(row.get("key") or ""): row for row in _list_live_attributes()}
        pending = {
            key for key in pending
            if str((live.get(key) or {}).get("status") or "").lower() != "available"
        }
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise SchemaMigrationError(
                "SCHEMA_MIGRATION_INCOMPLETE",
                f"attributes did not become available: {sorted(pending)}",
            )
        time.sleep(0.5)


def _recovery_error(
    error: Exception,
    *,
    journal: Iterable[Mapping[str, Any]],
    created: Iterable[str],
    remaining_not_attempted: Iterable[str],
    failed_attribute: str | None,
) -> SchemaMigrationError:
    entries = [dict(item) for item in journal]
    typed = error if isinstance(error, SchemaMigrationError) else SchemaMigrationError(
        "SCHEMA_OPERATION_FAILED", "schema migration operation failed"
    )
    verified = [
        str(item["key"]) for item in entries if item.get("status") == "verified"
    ]
    pending = [
        str(item["key"])
        for item in entries
        if item.get("status") == "pending_verification"
    ]
    remaining = list(remaining_not_attempted)
    typed.details.update({
        "completed": [item for item in entries if item.get("status") == "verified"],
        "journal": entries,
        "created": list(created),
        "verified": verified,
        "pending_verification": pending,
        "remaining": remaining,
        "remaining_not_attempted": remaining,
        "failed_attribute": failed_attribute,
    })
    return typed


def migrate(*, apply: bool = False, confirm_apply: bool = False) -> Dict[str, Any]:
    if apply and not confirm_apply:
        raise ValueError("--apply requires --confirm-apply")
    before = compute_schema_diff(_list_live_attributes())
    if before["mismatched"]:
        raise SchemaMigrationError(
            "SCHEMA_CONFLICT", "live schema contains incompatible metadata attributes"
        )
    capacity = verify_capacity(_collection_capacity(), before["missing"])
    if not apply:
        return {"dry_run": True, **before, "created": [], "journal": [], "capacity": capacity}

    created = []
    journal = []
    missing_specs = list(before["missing"])
    for index, spec in enumerate(missing_specs):
        try:
            outcome = _create(spec)
            entry: Dict[str, Any] = {
                "key": spec["key"],
                "outcome": outcome,
                "status": "pending_verification",
            }
            journal.append(entry)
            if outcome == "created":
                created.append(spec["key"])
            _wait_until_available({spec["key"]})
            verified = _verify_exact_attribute(spec)
            entry["verified"] = verified
            entry["status"] = "verified"
        except Exception as exc:
            wrapped = _recovery_error(
                exc,
                journal=journal,
                created=created,
                remaining_not_attempted=(
                    item["key"] for item in missing_specs[index + 1:]
                ),
                failed_attribute=spec["key"],
            )
            if wrapped is exc:
                raise
            raise wrapped from exc
    try:
        after = compute_schema_diff(_list_live_attributes())
    except Exception as exc:
        wrapped = _recovery_error(
            exc,
            journal=journal,
            created=created,
            remaining_not_attempted=(),
            failed_attribute=None,
        )
        if wrapped is exc:
            raise
        raise wrapped from exc
    if after["missing"] or after["mismatched"] or after["unavailable"]:
        raise _recovery_error(
            SchemaMigrationError(
                "SCHEMA_MIGRATION_INCOMPLETE",
                "schema migration did not converge to available attributes",
            ),
            journal=journal,
            created=created,
            remaining_not_attempted=(),
            failed_attribute=None,
        )
    return {
        "dry_run": False, **after, "created": created, "journal": journal,
        "capacity": capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-apply", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.confirm_apply and not args.apply:
        parser.error("--confirm-apply requires --apply")
    try:
        result = {"success": True, **migrate(
            apply=args.apply, confirm_apply=args.confirm_apply
        )}
    except SchemaMigrationError as exc:
        result = {
            "success": False,
            "error_code": exc.code,
            "completed": exc.details.get("completed", []),
            "created": exc.details.get("created", []),
            "verified": exc.details.get("verified", []),
            "pending_verification": exc.details.get("pending_verification", []),
            "remaining_not_attempted": exc.details.get("remaining_not_attempted", []),
            "failed_attribute": exc.details.get("failed_attribute"),
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        raise SystemExit(1)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
