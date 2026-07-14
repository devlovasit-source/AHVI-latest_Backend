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
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _list_live_attributes() -> list[Dict[str, Any]]:
    response = appwrite._request(
        "GET", f"{appwrite._base()}/{appwrite.COLLECTION_ID}/attributes"
    )
    if response.status_code != 200:
        raise RuntimeError(f"attribute list failed {response.status_code}: {response.text}")
    body = response.json()
    return [row for row in body.get("attributes", []) if isinstance(row, dict)]


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
        actual_type = str(actual.get("type") or "").lower()
        type_ok = actual_type == expected_type
        array_ok = bool(actual.get("array")) == bool(spec.get("array"))
        size_ok = expected_type != "string" or int(actual.get("size") or 0) >= int(spec["size"])
        if not (type_ok and array_ok and size_ok):
            mismatched.append({
                "key": key,
                "expected": dict(spec),
                "actual": {
                    "type": actual_type,
                    "array": bool(actual.get("array")),
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
    }


def _create(spec: Mapping[str, Any]) -> None:
    kind = spec["type"]
    payload: Dict[str, Any] = {"key": spec["key"], "required": False}
    if kind == "string":
        payload.update({"size": spec["size"], "array": bool(spec.get("array"))})
    response = appwrite._request("POST", appwrite._attribute_url(kind), payload)
    if response.status_code not in {200, 201, 202}:
        raise RuntimeError(
            f"attribute {spec['key']} failed {response.status_code}: {response.text}"
        )


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


def migrate(*, apply: bool = False, confirm_apply: bool = False) -> Dict[str, Any]:
    if apply and not confirm_apply:
        raise ValueError("--apply requires --confirm-apply")
    before = compute_schema_diff(_list_live_attributes())
    if before["mismatched"]:
        raise SchemaMigrationError(
            "SCHEMA_CONFLICT", "live schema contains incompatible metadata attributes"
        )
    if not apply:
        return {"dry_run": True, **before, "created": []}

    created = []
    for spec in before["missing"]:
        _create(spec)
        created.append(spec["key"])
    _wait_until_available(set(created).union(before["unavailable"]))
    after = compute_schema_diff(_list_live_attributes())
    if after["missing"] or after["mismatched"] or after["unavailable"]:
        raise SchemaMigrationError(
            "SCHEMA_MIGRATION_INCOMPLETE",
            "schema migration did not converge to available attributes",
        )
    return {"dry_run": False, **after, "created": created}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-apply", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.confirm_apply and not args.apply:
        parser.error("--confirm-apply requires --apply")
    result = migrate(apply=args.apply, confirm_apply=args.confirm_apply)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
