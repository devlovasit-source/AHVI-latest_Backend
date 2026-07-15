"""Safely provision the Appwrite schema used by durable medicine reminders.

This script never deletes collections, attributes, indexes, or documents. It is
an inspection-only command unless both --apply and --confirm-apply are present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests


class MigrationError(RuntimeError):
    pass


class MigrationOutageError(MigrationError):
    """Appwrite could not serve a non-conflict schema request."""


class MigrationConflictError(MigrationError):
    """An existing Appwrite resource is incompatible with the contract."""


class MigrationCapacityError(MigrationError):
    pass


# These fields include the older generic reminder writer because it shares the
# notification_reminders collection with the durable medicine state machine.
SCHEMA: Dict[str, Dict[str, Any]] = {
    "notification_reminders": {
        "attributes": [
            ("userId", "string", {"size": 128, "required": True}),
            ("eventId", "string", {"size": 128, "required": True}),
            ("notificationKey", "string", {"size": 256, "required": False}),
            ("occurrenceId", "string", {"size": 64, "required": False}),
            ("kind", "string", {"size": 32, "required": False}),
            ("status", "string", {"size": 30, "required": False}),
            ("source", "string", {"size": 64, "required": True}),
            ("medId", "string", {"size": 128, "required": False}),
            ("sendAtISO", "string", {"size": 64, "required": True}),
            ("scheduledFor", "string", {"size": 64, "required": False}),
            ("timezone", "string", {"size": 128, "required": False}),
            ("attemptCount", "integer", {"required": False, "min": 0, "max": 2147483647}),
            ("updatedAtISO", "string", {"size": 64, "required": True}),
            ("claimedAtISO", "string", {"size": 64, "required": False}),
            ("dispatchedAtISO", "string", {"size": 64, "required": False}),
            ("lastError", "string", {"size": 600, "required": True}),
            ("priority", "string", {"size": 32, "required": False}),
            ("toneProfile", "string", {"size": 64, "required": False}),
            ("offsetMinutes", "integer", {"required": False, "min": -9223372036854775808, "max": 9223372036854775807}),
            ("message", "string", {"size": 512, "required": True}),
        ],
        "indexes": [
            ("reminder_due", "key", ["kind", "status", "sendAtISO"], ["ASC", "ASC", "ASC"]),
            ("reminder_event", "key", ["userId", "eventId"], ["ASC", "ASC"]),
            ("reminder_notification_key", "key", ["notificationKey"], ["ASC"]),
            ("reminder_occurrence", "key", ["userId", "occurrenceId"], ["ASC", "ASC"]),
        ],
    },
    "notification_devices": {
        "attributes": [
            ("userId", "string", {"size": 128, "required": True}),
            ("platform", "string", {"size": 32, "required": True}),
            ("token", "string", {"size": 512, "required": True}),
            ("updatedAtISO", "string", {"size": 64, "required": True}),
        ],
        "indexes": [("device_user", "key", ["userId"], ["ASC"])],
    },
    "med_logs": {
        "attributes": [
            ("userId", "string", {"size": 50, "required": True}),
            ("medId", "string", {"size": 50, "required": True}),
            ("medName", "string", {"size": 255, "required": True}),
            ("dose", "string", {"size": 50, "required": True}),
            ("time", "datetime", {"required": True}),
            ("status", "string", {"size": 50, "required": True}),
            ("occurrenceId", "string", {"size": 64, "required": False}),
        ],
        "indexes": [
            ("med_log_time", "key", ["time"], ["ASC"]),
            ("med_log_occurrence", "key", ["userId", "occurrenceId"], ["ASC", "ASC"]),
            ("med_log_dose", "key", ["userId", "medId", "time"], ["ASC", "ASC", "ASC"]),
            ("med_log_status_dose", "key", ["userId", "medId", "status", "time"], ["ASC", "ASC", "ASC", "ASC"]),
        ],
    },
    "meds": {
        "attributes": [
            ("userId", "string", {"size": 50, "required": True}),
        ],
        "indexes": [("med_user", "key", ["userId"], ["ASC"])],
    },
}


def _env(*names: str) -> str:
    return next((str(os.getenv(name) or "").strip() for name in names if os.getenv(name)), "")


def configured_base() -> str:
    # Local audit runs may opt into the repository's existing Appwrite proxy
    # loader. This never prints or persists loaded credentials.
    if str(os.getenv("APPWRITE_PROXY_LOAD_LOCAL_ENV", "")).lower() in {"1", "true", "yes", "on"}:
        from services.appwrite_proxy import _load_local_env

        _load_local_env()
    endpoint = _env("APPWRITE_ENDPOINT", "EXPO_PUBLIC_APPWRITE_ENDPOINT").rstrip("/")
    database = _env("APPWRITE_DATABASE_ID", "EXPO_PUBLIC_APPWRITE_DATABASE_ID")
    project = _env("APPWRITE_PROJECT_ID", "EXPO_PUBLIC_APPWRITE_PROJECT_ID")
    key = _env("APPWRITE_API_KEY", "APPWRITE_KEY")
    if not all((endpoint, database, project, key)):
        raise MigrationError("Missing Appwrite endpoint, database, project, or API key configuration.")
    return f"{endpoint}/databases/{database}/collections"


class RequestsTransport:
    def __init__(self, base: Optional[str] = None) -> None:
        self.base = base or configured_base()
        self.headers = {"X-Appwrite-Project": _env("APPWRITE_PROJECT_ID", "EXPO_PUBLIC_APPWRITE_PROJECT_ID"), "X-Appwrite-Key": _env("APPWRITE_API_KEY", "APPWRITE_KEY"), "Content-Type": "application/json"}

    def request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> tuple[int, Any]:
        try:
            response = requests.request(method, f"{self.base}{path}", headers=self.headers, json=payload, timeout=20)
        except requests.RequestException as exc:
            raise MigrationOutageError(f"Appwrite request failed: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = {"message": response.text[:500]}
        return response.status_code, body


def _items(body: Any, name: str) -> List[Dict[str, Any]]:
    return [dict(item) for item in (body.get(name, []) if isinstance(body, dict) else []) if isinstance(item, dict)]


def _same_attribute(actual: Mapping[str, Any], kind: str, expected: Mapping[str, Any]) -> bool:
    if str(actual.get("type", "")).lower() != kind:
        return False
    for key in ("required", "array", "size", "min", "max"):
        if key in expected and actual.get(key) != expected[key]:
            return False
    return True


def _same_index(actual: Mapping[str, Any], expected: tuple[str, str, List[str], List[str]]) -> bool:
    _, index_type, attributes, orders = expected
    return str(actual.get("type", "")).lower() == index_type and list(actual.get("attributes") or []) == attributes and [str(item).upper() for item in actual.get("orders") or []] == orders


class DurableMedicineSchemaMigrator:
    def __init__(self, transport: Any, *, journal_path: Path, poll_seconds: float = 0.1, poll_attempts: int = 30, attribute_capacity: int = 100, index_capacity: int = 100) -> None:
        self.transport = transport
        self.journal_path = journal_path
        self.poll_seconds = poll_seconds
        self.poll_attempts = poll_attempts
        self.attribute_capacity = attribute_capacity
        self.index_capacity = index_capacity

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        status, body = self.transport.request(method, path, payload)
        if 200 <= status < 300:
            return body
        if status == 409:
            raise MigrationConflictError(f"Appwrite conflict for {path}")
        raise MigrationOutageError(f"Appwrite {method} {path} failed with HTTP {status}")

    def _journal(self) -> Dict[str, Any]:
        if not self.journal_path.exists():
            return {"version": 1, "completed": []}
        try:
            value = json.loads(self.journal_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) and isinstance(value.get("completed"), list) else {"version": 1, "completed": []}
        except (OSError, ValueError) as exc:
            raise MigrationError(f"Invalid migration journal: {self.journal_path}") from exc

    def _complete(self, journal: Dict[str, Any], action: str) -> None:
        if action not in journal["completed"]:
            journal["completed"].append(action)
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            self.journal_path.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _collection(self, collection: str) -> None:
        status, _ = self.transport.request("GET", f"/{collection}")
        if 200 <= status < 300:
            return
        if status == 404:
            raise MigrationConflictError(f"Required collection is missing: {collection}")
        raise MigrationOutageError(f"Appwrite GET /{collection} failed with HTTP {status}")

    def _attributes(self, collection: str) -> Dict[str, Dict[str, Any]]:
        return {str(item.get("key")): item for item in _items(self._request("GET", f"/{collection}/attributes"), "attributes") if item.get("key")}

    def _indexes(self, collection: str) -> Dict[str, Dict[str, Any]]:
        return {str(item.get("key")): item for item in _items(self._request("GET", f"/{collection}/indexes"), "indexes") if item.get("key")}

    @staticmethod
    def _has_equivalent_index(indexes: Mapping[str, Mapping[str, Any]], expected: tuple[str, str, List[str], List[str]]) -> bool:
        return any(_same_index(index, expected) for index in indexes.values())

    def _wait_attributes(self, collection: str, keys: Iterable[str]) -> None:
        pending = set(keys)
        for _ in range(self.poll_attempts):
            attributes = self._attributes(collection)
            pending = {key for key in pending if str(attributes.get(key, {}).get("status", "available")).lower() not in {"available", ""}}
            if not pending:
                return
            time.sleep(self.poll_seconds)
        raise MigrationError(f"Timed out waiting for attributes in {collection}: {', '.join(sorted(pending))}")

    def audit(self) -> Dict[str, Any]:
        """Return names and compatibility-relevant metadata only, never data."""
        report: Dict[str, Any] = {"collections": {}}
        for collection in SCHEMA:
            self._collection(collection)
            attributes = self._attributes(collection)
            indexes = self._indexes(collection)
            report["collections"][collection] = {
                "attributes": {
                    key: {name: value for name, value in attribute.items() if name in {"type", "required", "array", "size", "min", "max", "status"}}
                    for key, attribute in sorted(attributes.items())
                },
                "indexes": {
                    key: {name: value for name, value in index.items() if name in {"type", "attributes", "orders", "status"}}
                    for key, index in sorted(indexes.items())
                },
            }
        return report

    def migrate(self, *, apply: bool = False) -> Dict[str, Any]:
        journal = self._journal() if apply else {"completed": []}
        report: Dict[str, Any] = {"dry_run": not apply, "collections": {}, "resumed_actions": list(journal["completed"])}
        for collection, contract in SCHEMA.items():
            self._collection(collection)
            attributes = self._attributes(collection)
            missing = [(key, kind, options) for key, kind, options in contract["attributes"] if key not in attributes]
            incompatible = [key for key, kind, options in contract["attributes"] if key in attributes and not _same_attribute(attributes[key], kind, {"array": False, **options})]
            if incompatible:
                raise MigrationConflictError(f"Incompatible attributes in {collection}: {', '.join(incompatible)}")
            if len(attributes) + len(missing) > self.attribute_capacity:
                raise MigrationCapacityError(f"Attribute capacity exceeded for {collection}")
            entry = {"collection": "ready", "attributes": {key: "missing" for key, _, _ in missing}, "indexes": {}}
            created_attrs: List[str] = []
            if apply:
                for key, kind, options in missing:
                    action = f"attribute:{collection}:{key}"
                    try:
                        self._request("POST", f"/{collection}/attributes/{kind}", {"key": key, "array": False, **options})
                    except MigrationConflictError:
                        current = self._attributes(collection).get(key)
                        if not current or not _same_attribute(current, kind, {"array": False, **options}):
                            raise MigrationConflictError(f"Incompatible concurrent attribute {collection}.{key}")
                    self._complete(journal, action)
                    entry["attributes"][key] = "created"
                    created_attrs.append(key)
                self._wait_attributes(collection, created_attrs)
                attributes = self._attributes(collection)
                # Appwrite may still be building a pre-existing attribute when
                # this migration starts; indexes cannot safely proceed until all
                # fields they reference are queryable.
                self._wait_attributes(collection, [key for key, _, _ in contract["attributes"]])
            indexes = self._indexes(collection)
            missing_indexes = [
                item for item in contract["indexes"]
                if item[0] not in indexes and not self._has_equivalent_index(indexes, item)
            ]
            incompatible_indexes = [item[0] for item in contract["indexes"] if item[0] in indexes and not _same_index(indexes[item[0]], item)]
            if incompatible_indexes:
                raise MigrationConflictError(f"Incompatible indexes in {collection}: {', '.join(incompatible_indexes)}")
            if len(indexes) + len(missing_indexes) > self.index_capacity:
                raise MigrationCapacityError(f"Index capacity exceeded for {collection}")
            for key, index_type, attribute_keys, orders in missing_indexes:
                entry["indexes"][key] = "missing"
                if apply:
                    try:
                        self._request("POST", f"/{collection}/indexes", {"key": key, "type": index_type, "attributes": attribute_keys, "orders": orders})
                    except MigrationConflictError:
                        current = self._indexes(collection).get(key)
                        if not current or not _same_index(current, (key, index_type, attribute_keys, orders)):
                            raise MigrationConflictError(f"Incompatible concurrent index {collection}.{key}")
                    self._complete(journal, f"index:{collection}:{key}")
                    entry["indexes"][key] = "created"
            report["collections"][collection] = entry
        return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Provision durable medicine reminder Appwrite schema.")
    parser.add_argument("--apply", action="store_true", help="Allow schema creates (requires --confirm-apply).")
    parser.add_argument("--confirm-apply", action="store_true", help="Confirm schema creates.")
    parser.add_argument("--audit", action="store_true", help="Read and print sanitized live schema metadata only.")
    parser.add_argument("--journal", type=Path, default=ROOT / ".durable_medicine_schema_journal.json")
    args = parser.parse_args(argv)
    if args.apply != args.confirm_apply:
        parser.error("--apply and --confirm-apply must be supplied together")
    migrator = DurableMedicineSchemaMigrator(RequestsTransport(), journal_path=args.journal)
    result = migrator.audit() if args.audit else migrator.migrate(apply=args.apply)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
