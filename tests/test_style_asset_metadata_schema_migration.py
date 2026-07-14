import pytest

from scripts import migrate_style_asset_metadata_schema as migration


def _live(spec, *, status="available"):
    return {
        "key": spec["key"],
        "type": spec["type"],
        "size": spec.get("size"),
        "array": bool(spec.get("array")),
        "status": status,
    }


def test_schema_diff_creates_only_genuinely_missing_attributes():
    live = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-2]]

    diff = migration.compute_schema_diff(live)

    assert [item["key"] for item in diff["missing"]] == [
        migration.ATTRIBUTE_SPECS[-2]["key"],
        migration.ATTRIBUTE_SPECS[-1]["key"],
    ]
    assert diff["mismatched"] == []
    assert diff["unavailable"] == []


def test_schema_diff_fails_closed_on_incompatible_existing_attribute():
    spec = migration.ATTRIBUTE_SPECS[0]
    diff = migration.compute_schema_diff([{
        **_live(spec),
        "type": "integer",
    }])

    assert diff["mismatched"][0]["key"] == spec["key"]


def test_noop_migration_reports_no_missing_attributes(monkeypatch):
    monkeypatch.setattr(
        migration,
        "_list_live_attributes",
        lambda: [_live(spec) for spec in migration.ATTRIBUTE_SPECS],
    )

    result = migration.migrate()

    assert result["dry_run"] is True
    assert result["missing"] == []
    assert result["created"] == []


def test_confirmed_apply_creates_only_partial_diff_and_waits(monkeypatch):
    missing_spec = migration.ATTRIBUTE_SPECS[-1]
    before = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-1]]
    after = [*before, _live(missing_spec)]
    reads = iter([before, after])
    created = []
    waited = []
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: next(reads))
    monkeypatch.setattr(migration, "_create", lambda spec: created.append(spec["key"]))
    monkeypatch.setattr(
        migration, "_wait_until_available", lambda keys: waited.extend(sorted(keys))
    )

    result = migration.migrate(apply=True, confirm_apply=True)

    assert created == [missing_spec["key"]]
    assert waited == [missing_spec["key"]]
    assert result["created"] == [missing_spec["key"]]


def test_incompatible_schema_returns_typed_conflict(monkeypatch):
    first = migration.ATTRIBUTE_SPECS[0]
    monkeypatch.setattr(
        migration,
        "_list_live_attributes",
        lambda: [{**_live(first), "type": "integer"}],
    )

    with pytest.raises(migration.SchemaMigrationError) as exc:
        migration.migrate()

    assert exc.value.code == "SCHEMA_CONFLICT"


def test_apply_requires_double_confirmation():
    with pytest.raises(ValueError, match="confirm-apply"):
        migration.migrate(apply=True, confirm_apply=False)


def test_wait_polls_until_every_attribute_is_available(monkeypatch):
    calls = iter([
        [{"key": "role", "status": "processing"}],
        [{"key": "role", "status": "available"}],
    ])
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: next(calls))
    monkeypatch.setattr(migration.time, "sleep", lambda _seconds: None)

    migration._wait_until_available({"role"}, timeout_seconds=5)


def test_wait_timeout_is_typed_incomplete_failure(monkeypatch):
    monkeypatch.setattr(
        migration,
        "_list_live_attributes",
        lambda: [{"key": "role", "status": "processing"}],
    )
    times = iter([0.0, 10.0])
    monkeypatch.setattr(migration.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(migration.time, "sleep", lambda _seconds: None)

    with pytest.raises(migration.SchemaMigrationError) as exc:
        migration._wait_until_available({"role"}, timeout_seconds=5)

    assert exc.value.code == "SCHEMA_MIGRATION_INCOMPLETE"
