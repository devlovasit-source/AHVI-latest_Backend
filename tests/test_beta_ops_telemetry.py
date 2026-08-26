from datetime import datetime

import pytest

from services import beta_ops_telemetry as telemetry
from scripts import create_beta_ops_events_collection as schema


class FakeAppwriteProxyError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class FakeTelemetryProxy:
    def __init__(self):
        self.docs = {}
        self.create_calls = 0

    def create_document(self, resource, data, document_id="unique()"):
        assert resource == telemetry.BETA_OPS_EVENTS_RESOURCE
        self.create_calls += 1
        if document_id in self.docs:
            raise FakeAppwriteProxyError("duplicate", status_code=409)
        doc = {"$id": document_id, **data}
        self.docs[document_id] = doc
        return doc

    def get_document(self, resource, document_id):
        assert resource == telemetry.BETA_OPS_EVENTS_RESOURCE
        return self.docs[document_id]

    def list_documents(self, resource, **kwargs):
        assert resource == telemetry.BETA_OPS_EVENTS_RESOURCE
        user_id = kwargs.get("user_id")
        return [
            doc for doc in self.docs.values()
            if not user_id or doc.get("user_id") == user_id
        ]

    def find_by_attribute(self, resource, attribute, value, **kwargs):
        assert resource == telemetry.BETA_OPS_EVENTS_RESOURCE
        return [doc for doc in self.docs.values() if doc.get(attribute) == value]


def test_required_fields_and_event_type_are_validated():
    with pytest.raises(ValueError, match="user_id is required"):
        telemetry.normalize_event(event_type="user.activity", user_id="")
    with pytest.raises(ValueError, match="unsupported"):
        telemetry.normalize_event(event_type="not.allowed", user_id="u1")


def test_optional_numeric_fields_are_non_negative_integers():
    event = telemetry.normalize_event(
        event_type="llm.usage_attempt",
        user_id="u1",
        attempt=1,
        duration_ms=25,
        input_tokens=10,
        output_tokens=5,
        cached_tokens=2,
    )
    assert event["attempt"] == 1
    assert event["cached_tokens"] == 2
    with pytest.raises(ValueError):
        telemetry.normalize_event(event_type="llm.usage_attempt", user_id="u1", attempt=-1)
    with pytest.raises(ValueError):
        telemetry.normalize_event(event_type="llm.usage_attempt", user_id="u1", attempt=True)


def test_datetime_is_normalized_to_utc():
    event = telemetry.normalize_event(
        event_type="user.activity",
        user_id="u1",
        occurred_at="2026-08-26T12:00:00+05:30",
    )
    assert event["occurred_at"] == "2026-08-26T06:30:00+00:00"
    naive = telemetry.normalize_datetime(datetime(2026, 8, 26, 12, 0, 0))
    assert naive.endswith("+00:00")


def test_deterministic_id_is_existing_sha256_shape():
    first = telemetry.deterministic_appwrite_id("beta_ops", "u1", "op1", "1")
    second = telemetry.deterministic_appwrite_id("beta_ops", "u1", "op1", "1")
    assert first == second
    assert len(first) == 36
    assert all(char in "0123456789abcdef" for char in first)


def test_metadata_allowlist_and_reason_counts():
    event = telemetry.normalize_event(
        event_type="style.request_outcome",
        user_id="u1",
        metadata={
            "source": "stylist",
            "requested_count": 3,
            "reason_counts": {"occasion_mismatch": 2},
        },
    )
    assert '"requested_count":3' in event["metadata_json"]
    with pytest.raises(ValueError, match="not allowed"):
        telemetry.normalize_metadata({"prompt": "do not store"})
    with pytest.raises(ValueError, match="reason_counts"):
        telemetry.normalize_metadata({"reason_counts": {"bad": -1}})


def test_metadata_is_bounded():
    reason_counts = {
        f"reason_{index:02d}_{'x' * 50}": 1
        for index in range(40)
    }
    with pytest.raises(ValueError, match="2048"):
        telemetry.normalize_metadata({"reason_counts": reason_counts})


def test_metadata_scalar_string_boundary():
    assert telemetry.normalize_metadata({"flow": "x" * 256}) is not None
    with pytest.raises(ValueError):
        telemetry.normalize_metadata({"flow": "x" * 257})


def test_prohibited_metadata_content_is_rejected():
    with pytest.raises(ValueError, match="prohibited"):
        telemetry.normalize_metadata({"source": "https://service.test/?token=secret"})
    with pytest.raises(ValueError, match="not allowed"):
        telemetry.normalize_metadata({"image_url": "https://images.test/a.png"})
    with pytest.raises(ValueError, match="prohibited"):
        telemetry.normalize_event(event_type="product.event", user_id="u1", usecase="raw provider response")
    with pytest.raises(ValueError, match="prohibited"):
        telemetry.normalize_event(event_type="product.event", user_id="u1", operation_id='{"items": []}')
    with pytest.raises(ValueError, match="prohibited"):
        telemetry.normalize_metadata({"source": "https://user:password@example.test/path"})


def test_duplicate_idempotent_insert_returns_existing_document(monkeypatch):
    fake = FakeTelemetryProxy()
    monkeypatch.setattr(telemetry, "AppwriteProxy", lambda: fake)
    fields = {
        "event_type": "user.activity",
        "user_id": "u1",
        "operation_id": "op1",
    }
    first = telemetry.record_event(idempotency_key="u1|activity|op1", **fields)
    second = telemetry.record_event(idempotency_key="u1|activity|op1", **fields)
    assert first["persisted"] is True
    assert second["persisted"] is True
    assert second["duplicate"] is True
    assert first["document_id"] == second["document_id"]
    assert len(fake.docs) == 1


def test_appwrite_unavailable_is_observational(monkeypatch):
    def unavailable():
        raise RuntimeError("not configured")

    monkeypatch.setattr(telemetry, "AppwriteProxy", unavailable)
    result = telemetry.record_event(event_type="user.activity", user_id="u1")
    assert result["persisted"] is False
    assert result["event"] is None


def test_duplicate_lookup_failure_is_not_reported_as_persisted(monkeypatch):
    class BrokenDuplicateProxy(FakeTelemetryProxy):
        def get_document(self, resource, document_id):
            raise RuntimeError("lookup failed")

    fake = BrokenDuplicateProxy()
    monkeypatch.setattr(telemetry, "AppwriteProxy", lambda: fake)
    first = telemetry.record_event(idempotency_key="same", event_type="user.activity", user_id="u1")
    second = telemetry.record_event(idempotency_key="same", event_type="user.activity", user_id="u1")
    assert first["persisted"] is True
    assert second["persisted"] is False
    assert second["duplicate"] is False


def test_validation_failure_does_not_propagate_to_user_operation():
    result = telemetry.record_event(
        event_type="product.event",
        user_id="u1",
        metadata={"chat_contents": "never persist"},
    )
    assert result["persisted"] is False


def test_list_events_scopes_user_and_event_type(monkeypatch):
    fake = FakeTelemetryProxy()
    monkeypatch.setattr(telemetry, "AppwriteProxy", lambda: fake)
    telemetry.record_event(event_type="user.activity", user_id="u1")
    telemetry.record_event(event_type="user.activity", user_id="u2")
    rows = telemetry.list_events(user_id="u1", event_type="user.activity")
    assert len(rows) == 1
    assert rows[0]["user_id"] == "u1"


def test_schema_has_exact_contract():
    assert len(schema.ATTRIBUTE_DEFINITIONS) == 16
    assert sum(bool(definition.get("required")) for _, definition in schema.ATTRIBUTE_DEFINITIONS) == 3
    assert len(schema.INDEX_DEFINITIONS) == 2
    assert all(attribute_type in {"string", "integer", "datetime"} for attribute_type, _ in schema.ATTRIBUTE_DEFINITIONS)
    assert schema.ATTRIBUTE_DEFINITIONS[2] == (
        "datetime", {"key": "occurred_at", "required": True}
    )


def test_repeated_events_without_key_can_have_distinct_ids():
    first = telemetry.normalize_event(
        event_type="product.event", user_id="u1", occurred_at="2026-08-26T10:00:00Z"
    )
    second = telemetry.normalize_event(
        event_type="product.event", user_id="u1", occurred_at="2026-08-26T10:00:01Z"
    )
    assert telemetry.event_document_id(first) != telemetry.event_document_id(second)


def test_query_time_bounds_are_utc_and_exclusive_at_upper_bound(monkeypatch):
    fake = FakeTelemetryProxy()
    monkeypatch.setattr(telemetry, "AppwriteProxy", lambda: fake)
    telemetry.record_event(event_type="user.activity", user_id="u1", occurred_at="2026-08-26T10:00:00Z")
    telemetry.record_event(event_type="user.activity", user_id="u1", occurred_at="2026-08-26T11:00:00Z")
    rows = telemetry.list_events(
        user_id="u1",
        occurred_after="2026-08-26T05:00:00-05:00",
        occurred_before="2026-08-26T11:00:00Z",
    )
    assert len(rows) == 1
    assert rows[0]["occurred_at"] == "2026-08-26T10:00:00+00:00"
