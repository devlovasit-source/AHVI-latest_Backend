from datetime import datetime

import pytest

from services import beta_ops_telemetry as telemetry


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
    with pytest.raises(ValueError, match="2048"):
        telemetry.normalize_metadata({"flow": "x" * 2100})


def test_prohibited_metadata_content_is_rejected():
    with pytest.raises(ValueError, match="prohibited"):
        telemetry.normalize_metadata({"source": "https://service.test/?token=secret"})
    with pytest.raises(ValueError, match="not allowed"):
        telemetry.normalize_metadata({"image_url": "https://images.test/a.png"})


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
