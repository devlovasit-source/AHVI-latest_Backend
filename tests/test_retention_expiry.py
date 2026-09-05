"""Unit tests for configurable retention worker and TTL settings."""

import os
from datetime import datetime, timedelta, timezone
from brain.temporal.opportunity_models import Opportunity
from brain.temporal.opportunity_store import opportunity_store
from brain.temporal.retention_worker import get_retention_settings, retention_worker


def test_retention_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TIMELINE_ITEM_RETENTION_PAST_DAYS", "45")
    monkeypatch.setenv("TIMELINE_ITEM_RETENTION_FUTURE_DAYS", "120")
    monkeypatch.setenv("TEMPORAL_SIGNAL_TTL_HOURS", "72")
    monkeypatch.setenv("OPPORTUNITY_RETENTION_POST_RESOLUTION_DAYS", "90")

    settings = get_retention_settings()

    assert settings["TIMELINE_ITEM_RETENTION_PAST_DAYS"] == 45
    assert settings["TIMELINE_ITEM_RETENTION_FUTURE_DAYS"] == 120
    assert settings["TEMPORAL_SIGNAL_TTL_HOURS"] == 72
    assert settings["OPPORTUNITY_RETENTION_POST_RESOLUTION_DAYS"] == 90


def test_retention_worker_cleanup() -> None:
    user_id = "usr_retention_test"
    now = datetime.now(timezone.utc)
    expired_time = now - timedelta(hours=1)

    opp = Opportunity.create(
        user_id=user_id,
        opportunity_type="test_retention",
        timeline_item_id="cal_evt_ret",
        trigger_window="window_1",
        expires_at=expired_time,
    )

    opportunity_store.save_opportunity(opp)

    metrics = retention_worker.run_cleanup(user_id)
    assert metrics["opportunities_cleaned"] >= 1

    fetched = opportunity_store.get_opportunity(opp.id)
    assert fetched is not None
    assert fetched.status.value == "EXPIRED"
