import pytest

from scripts import migrate_style_asset_metadata_schema as migration


def _live(spec, *, status="available"):
    return {
        "key": spec["key"],
        "type": spec["type"],
        "size": spec.get("size"),
        "array": bool(spec.get("array")),
        "required": bool(spec.get("required")),
        "default": spec.get("default"),
        "status": status,
    }


@pytest.fixture(autouse=True)
def _safe_capacity(monkeypatch):
    monkeypatch.setattr(migration, "_collection_capacity", lambda: {
        "bytes_used": 1000, "bytes_max": 65535, "attribute_count": 19,
    })


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


@pytest.mark.parametrize(
    "live_type,compatible",
    [
        ("double", True),
        ("float", True),
        ("integer", False),
        ("string", False),
        ("boolean", False),
    ],
)
def test_float_type_compatibility_is_explicit(live_type, compatible):
    spec = next(item for item in migration.ATTRIBUTE_SPECS if item["type"] == "float")

    diff = migration.compute_schema_diff([{**_live(spec), "type": live_type}])

    assert (diff["mismatched"] == []) is compatible
    assert (diff["existing"] == [spec["key"]]) is compatible


@pytest.mark.parametrize("field,value", [("required", True), ("default", "unsafe")])
def test_schema_diff_checks_required_and_default(field, value):
    spec = migration.ATTRIBUTE_SPECS[0]

    diff = migration.compute_schema_diff([{**_live(spec), field: value}])

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
    monkeypatch.setattr(
        migration, "_create", lambda spec: created.append(spec["key"]) or "created"
    )
    monkeypatch.setattr(
        migration, "_wait_until_available", lambda keys: waited.extend(sorted(keys))
    )
    monkeypatch.setattr(migration, "_verify_exact_attribute", lambda spec: _live(spec))

    result = migration.migrate(apply=True, confirm_apply=True)

    assert created == [missing_spec["key"]]
    assert waited == [missing_spec["key"]]
    assert result["created"] == [missing_spec["key"]]


def test_apply_waits_and_verifies_each_attribute_before_next(monkeypatch):
    missing = list(migration.ATTRIBUTE_SPECS[-2:])
    before = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-2]]
    after = [*before, *(_live(spec) for spec in missing)]
    reads = iter([before, after])
    events = []
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: next(reads))
    monkeypatch.setattr(
        migration, "_create", lambda spec: events.append(("create", spec["key"])) or "created"
    )
    monkeypatch.setattr(
        migration, "_wait_until_available", lambda keys: events.append(("wait", next(iter(keys))))
    )
    monkeypatch.setattr(
        migration, "_verify_exact_attribute", lambda spec: events.append(("verify", spec["key"])) or _live(spec)
    )

    result = migration.migrate(apply=True, confirm_apply=True)

    assert events == [
        ("create", missing[0]["key"]), ("wait", missing[0]["key"]),
        ("verify", missing[0]["key"]), ("create", missing[1]["key"]),
        ("wait", missing[1]["key"]), ("verify", missing[1]["key"]),
    ]
    assert len(result["journal"]) == 2
    assert all(item["status"] == "verified" for item in result["journal"])


def test_create_failure_excludes_already_attempted_attribute_from_remaining(monkeypatch):
    missing = list(migration.ATTRIBUTE_SPECS[-2:])
    before = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-2]]
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: before)
    calls = 0

    def create(spec):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise migration.SchemaMigrationError("SCHEMA_CREATE_FAILED", "controlled")
        return "created"

    monkeypatch.setattr(migration, "_create", create)
    monkeypatch.setattr(migration, "_wait_until_available", lambda _keys: None)
    monkeypatch.setattr(migration, "_verify_exact_attribute", lambda spec: _live(spec))

    with pytest.raises(migration.SchemaMigrationError) as exc:
        migration.migrate(apply=True, confirm_apply=True)

    assert exc.value.details["created"] == [missing[0]["key"]]
    assert exc.value.details["verified"] == [missing[0]["key"]]
    assert exc.value.details["pending_verification"] == []
    assert exc.value.details["remaining_not_attempted"] == []
    assert [item["key"] for item in exc.value.details["completed"]] == [missing[0]["key"]]


def test_post_succeeds_wait_timeout_records_current_as_pending(monkeypatch):
    missing = migration.ATTRIBUTE_SPECS[-1]
    before = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-1]]
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: before)
    monkeypatch.setattr(migration, "_create", lambda _spec: "created")
    monkeypatch.setattr(
        migration,
        "_wait_until_available",
        lambda _keys: (_ for _ in ()).throw(
            migration.SchemaMigrationError("SCHEMA_MIGRATION_INCOMPLETE", "controlled")
        ),
    )

    with pytest.raises(migration.SchemaMigrationError) as exc:
        migration.migrate(apply=True, confirm_apply=True)

    assert exc.value.details["created"] == [missing["key"]]
    assert exc.value.details["verified"] == []
    assert exc.value.details["pending_verification"] == [missing["key"]]
    assert exc.value.details["remaining_not_attempted"] == []


def test_post_succeeds_verification_failure_records_current_as_pending(monkeypatch):
    missing = migration.ATTRIBUTE_SPECS[-1]
    before = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-1]]
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: before)
    monkeypatch.setattr(migration, "_create", lambda _spec: "created")
    monkeypatch.setattr(migration, "_wait_until_available", lambda _keys: None)
    monkeypatch.setattr(
        migration,
        "_verify_exact_attribute",
        lambda _spec: (_ for _ in ()).throw(
            migration.SchemaMigrationError("SCHEMA_CONFLICT", "controlled")
        ),
    )

    with pytest.raises(migration.SchemaMigrationError) as exc:
        migration.migrate(apply=True, confirm_apply=True)

    assert exc.value.details["created"] == [missing["key"]]
    assert exc.value.details["verified"] == []
    assert exc.value.details["pending_verification"] == [missing["key"]]
    assert exc.value.details["remaining_not_attempted"] == []


def test_unexpected_connection_error_preserves_recovery_subsets(monkeypatch):
    missing = list(migration.ATTRIBUTE_SPECS[-3:])
    before = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-3]]
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: before)
    monkeypatch.setattr(migration, "_create", lambda _spec: "created")
    waits = 0

    def wait(_keys):
        nonlocal waits
        waits += 1
        if waits == 2:
            raise ConnectionError("must not enter recovery output")

    monkeypatch.setattr(migration, "_wait_until_available", wait)
    monkeypatch.setattr(migration, "_verify_exact_attribute", lambda spec: _live(spec))

    with pytest.raises(migration.SchemaMigrationError) as exc:
        migration.migrate(apply=True, confirm_apply=True)

    assert exc.value.code == "SCHEMA_OPERATION_FAILED"
    assert exc.value.details["created"] == [missing[0]["key"], missing[1]["key"]]
    assert exc.value.details["verified"] == [missing[0]["key"]]
    assert exc.value.details["pending_verification"] == [missing[1]["key"]]
    assert exc.value.details["remaining_not_attempted"] == [missing[2]["key"]]
    assert "must not enter" not in str(exc.value)


def test_resume_conflict_verifies_compatible_existing_attribute(monkeypatch):
    missing = migration.ATTRIBUTE_SPECS[-1]
    before = [_live(spec) for spec in migration.ATTRIBUTE_SPECS[:-1]]
    after = [*before, _live(missing)]
    reads = iter([before, after])
    monkeypatch.setattr(migration, "_list_live_attributes", lambda: next(reads))
    monkeypatch.setattr(migration, "_create", lambda _spec: "conflict")
    monkeypatch.setattr(migration, "_wait_until_available", lambda _keys: None)
    monkeypatch.setattr(migration, "_verify_exact_attribute", lambda spec: _live(spec))

    result = migration.migrate(apply=True, confirm_apply=True)

    assert result["created"] == []
    assert result["journal"] == [{
        "key": missing["key"],
        "outcome": "conflict",
        "status": "verified",
        "verified": _live(missing),
    }]


def test_capacity_preflight_accepts_54_attribute_result():
    missing = [
        spec for spec in migration.ATTRIBUTE_SPECS
        if spec["key"] != "board_image_url"
    ]
    result = migration.verify_capacity(
        {"bytes_used": 33528, "bytes_max": 65535, "attribute_count": 20},
        missing,
    )

    assert result["resulting_attribute_count"] == 54
    assert result["estimated_remaining_bytes"] > 0


def test_capacity_preflight_fails_before_create_when_insufficient():
    with pytest.raises(migration.SchemaMigrationError) as exc:
        migration.verify_capacity(
            {"bytes_used": 65000, "bytes_max": 65535, "attribute_count": 19},
            [spec for spec in migration.ATTRIBUTE_SPECS if spec["key"] != "board_image_url"],
        )

    assert exc.value.code == "SCHEMA_CAPACITY_EXCEEDED"


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
