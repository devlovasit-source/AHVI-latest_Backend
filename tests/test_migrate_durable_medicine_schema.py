import json
from pathlib import Path

import pytest

from scripts.migrate_durable_medicine_schema import (
    SCHEMA,
    DurableMedicineSchemaMigrator,
    MigrationCapacityError,
    MigrationConflictError,
    MigrationError,
    MigrationOutageError,
    main,
)


class FakeTransport:
    def __init__(self, schema=None):
        self.schema = schema or {}
        self.calls = []
        self.attribute_statuses = {}
        self.fail_path = None

    @staticmethod
    def _empty_collection():
        return {"attributes": {}, "indexes": {}}

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if path == self.fail_path:
            return 503, {}
        parts = [part for part in path.split("/") if part]
        collection = parts[0] if parts else ""
        if method == "GET" and len(parts) == 1:
            return (200, {}) if collection in self.schema else (404, {})
        if method == "POST" and not parts:
            collection = payload["collectionId"]
            if collection in self.schema:
                return 409, {}
            self.schema[collection] = self._empty_collection()
            return 201, {}
        if collection not in self.schema:
            return 404, {}
        if method == "GET" and parts[-1] == "attributes":
            attributes = []
            for key, value in self.schema[collection]["attributes"].items():
                item = {"key": key, **value}
                statuses = self.attribute_statuses.get((collection, key))
                if statuses:
                    item["status"] = statuses.pop(0) if len(statuses) > 1 else statuses[0]
                attributes.append(item)
            return 200, {"attributes": attributes}
        if method == "GET" and parts[-1] == "indexes":
            return 200, {"indexes": [{"key": key, **value} for key, value in self.schema[collection]["indexes"].items()]}
        if method == "POST" and len(parts) == 3 and parts[1] == "attributes":
            key = payload["key"]
            if key in self.schema[collection]["attributes"]:
                return 409, {}
            self.schema[collection]["attributes"][key] = {"type": parts[2], **payload}
            return 202, {}
        if method == "POST" and parts[-1] == "indexes":
            key = payload["key"]
            if key in self.schema[collection]["indexes"]:
                return 409, {}
            self.schema[collection]["indexes"][key] = dict(payload)
            return 202, {}
        raise AssertionError((method, path, payload))


def complete_schema():
    result = {}
    for collection, contract in SCHEMA.items():
        result[collection] = {"attributes": {}, "indexes": {}}
        for key, kind, options in contract["attributes"]:
            result[collection]["attributes"][key] = {"type": kind, "array": False, **options}
        for key, index_type, attributes, orders in contract["indexes"]:
            result[collection]["indexes"][key] = {"type": index_type, "attributes": attributes, "orders": orders}
    return result


def migrator(tmp_path, transport, **kwargs):
    return DurableMedicineSchemaMigrator(transport, journal_path=tmp_path / "journal.json", poll_seconds=0, **kwargs)


def test_dry_run_noop_makes_no_writes(tmp_path):
    transport = FakeTransport(complete_schema())
    report = migrator(tmp_path, transport).migrate()
    assert report["dry_run"] is True
    assert not [call for call in transport.calls if call[0] == "POST"]
    assert all(not value["attributes"] and not value["indexes"] for value in report["collections"].values())


def test_apply_creates_partial_schema_and_indexes(tmp_path):
    schema = complete_schema()
    del schema["med_logs"]["attributes"]["occurrenceId"]
    del schema["med_logs"]["indexes"]["med_log_occurrence"]
    transport = FakeTransport(schema)
    report = migrator(tmp_path, transport).migrate(apply=True)
    assert report["collections"]["med_logs"]["attributes"]["occurrenceId"] == "created"
    assert report["collections"]["med_logs"]["indexes"]["med_log_occurrence"] == "created"


@pytest.mark.parametrize("target", ["attribute", "index"])
def test_incompatible_existing_schema_fails_without_writes(tmp_path, target):
    schema = complete_schema()
    if target == "attribute":
        schema["med_logs"]["attributes"]["occurrenceId"]["size"] = 255
    else:
        schema["med_logs"]["indexes"]["med_log_occurrence"]["attributes"] = ["occurrenceId", "userId"]
    transport = FakeTransport(schema)
    with pytest.raises(MigrationConflictError):
        migrator(tmp_path, transport).migrate(apply=True)
    assert not [call for call in transport.calls if call[0] == "POST"]


def test_capacity_is_checked_before_creating(tmp_path):
    schema = complete_schema()
    del schema["meds"]["attributes"]["userId"]
    transport = FakeTransport(schema)
    with pytest.raises(MigrationCapacityError):
        migrator(tmp_path, transport, attribute_capacity=len(schema["meds"]["attributes"])).migrate(apply=True)
    assert not [call for call in transport.calls if call[0] == "POST"]


def test_waits_for_asynchronous_attribute_availability(tmp_path):
    schema = complete_schema()
    del schema["med_logs"]["attributes"]["occurrenceId"]
    del schema["med_logs"]["indexes"]["med_log_occurrence"]
    transport = FakeTransport(schema)
    transport.attribute_statuses[("med_logs", "occurrenceId")] = ["processing", "available"]
    migrator(tmp_path, transport).migrate(apply=True)
    posts = [call for call in transport.calls if call[0] == "POST"]
    assert posts[-1][1].endswith("/indexes")
    assert any(call[0] == "GET" and call[1] == "/med_logs/attributes" for call in transport.calls)


def test_attribute_availability_timeout_prevents_index_creation(tmp_path):
    schema = complete_schema()
    del schema["med_logs"]["attributes"]["occurrenceId"]
    del schema["med_logs"]["indexes"]["med_log_occurrence"]
    transport = FakeTransport(schema)
    transport.attribute_statuses[("med_logs", "occurrenceId")] = ["processing"]
    with pytest.raises(MigrationError, match="Timed out"):
        migrator(tmp_path, transport, poll_attempts=2).migrate(apply=True)
    assert not any(call[0] == "POST" and call[1].endswith("/indexes") for call in transport.calls)


def test_journal_resume_reports_prior_action_and_remains_idempotent(tmp_path):
    schema = complete_schema()
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({"version": 1, "completed": ["attribute:med_logs:occurrenceId"]}), encoding="utf-8")
    transport = FakeTransport(schema)
    report = migrator(tmp_path, transport).migrate(apply=True)
    assert "attribute:med_logs:occurrenceId" in report["resumed_actions"]
    assert not [call for call in transport.calls if call[0] == "POST"]


def test_http_409_is_conflict_not_outage(tmp_path):
    transport = FakeTransport(complete_schema())
    transport.schema["meds"]["attributes"]["userId"]["size"] = 32
    with pytest.raises(MigrationConflictError) as error:
        migrator(tmp_path, transport).migrate()
    assert not isinstance(error.value, MigrationOutageError)


def test_apply_requires_explicit_confirmation():
    with pytest.raises(SystemExit) as error:
        main(["--apply"])
    assert error.value.code == 2
