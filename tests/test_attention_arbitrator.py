"""Unit tests for Attention Arbitrator, candidate action ranking, suppression, and durable deferrals."""

from datetime import datetime, timedelta, timezone
from brain.temporal.attention_arbitrator import AttentionArbitrator, attention_arbitrator
from brain.temporal.attention_models import CandidateAction
from brain.temporal.candidate_action_store import candidate_action_store


def test_candidate_action_deliverability_and_expiry() -> None:
    now = datetime.now(timezone.utc)
    future = now + timedelta(hours=2)
    past = now - timedelta(hours=1)

    action_deliverable = CandidateAction(
        id="act_1",
        user_id="usr_att_1",
        source_opportunity_id="opp_1",
        source_module="calendar",
        action_type="meeting_reminder",
        priority=4,
        urgency=0.8,
        attention_cost=0.3,
        deliver_after=past,
        expires_at=future,
        payload={"title": "Board Meeting"},
    )

    assert action_deliverable.is_expired is False
    assert action_deliverable.is_deliverable is True

    action_expired = CandidateAction(
        id="act_2",
        user_id="usr_att_1",
        source_opportunity_id="opp_2",
        source_module="calendar",
        action_type="meeting_reminder",
        priority=2,
        urgency=0.2,
        attention_cost=0.8,
        deliver_after=past,
        expires_at=past,
    )

    assert action_expired.is_expired is True
    assert action_expired.is_deliverable is False

    action_deferred = CandidateAction(
        id="act_3",
        user_id="usr_att_1",
        source_opportunity_id="opp_3",
        source_module="calendar",
        action_type="meeting_reminder",
        priority=3,
        urgency=0.5,
        attention_cost=0.5,
        deliver_after=future,
        expires_at=future + timedelta(hours=2),
    )

    assert action_deferred.is_expired is False
    assert action_deferred.is_deliverable is False


def test_attention_arbitrator_pipeline() -> None:
    arbitrator = AttentionArbitrator()
    now = datetime.now(timezone.utc)

    act_high = CandidateAction(
        id="act_high",
        user_id="usr_att_2",
        source_opportunity_id="opp_10",
        source_module="calendar",
        action_type="board_meeting",
        priority=5,
        urgency=0.9,
        attention_cost=0.2,
        deliver_after=now - timedelta(minutes=5),
    )

    act_dup = CandidateAction(
        id="act_dup",
        user_id="usr_att_2",
        source_opportunity_id="opp_11",
        source_module="calendar",
        action_type="board_meeting",
        priority=3,
        urgency=0.4,
        attention_cost=0.5,
        deliver_after=now - timedelta(minutes=5),
    )

    act_workout = CandidateAction(
        id="act_workout",
        user_id="usr_att_2",
        source_opportunity_id="opp_12",
        source_module="workout",
        action_type="hiit_reminder",
        priority=4,
        urgency=0.7,
        attention_cost=0.4,
        deliver_after=now - timedelta(minutes=5),
    )

    actions = [act_high, act_dup, act_workout]
    arbitrated = arbitrator.arbitrate(actions, max_delivery=2)

    assert len(arbitrated) == 2
    assert arbitrated[0].id == "act_high"
    assert arbitrated[1].id == "act_workout"


def test_persistent_candidate_action_deferral_and_resume() -> None:
    """Test that CandidateActions deferred by AttentionArbitrator are written to CandidateActionStore and survive process restarts."""
    user_id = "usr_defer_test"
    now = datetime.now(timezone.utc)

    action = CandidateAction(
        id="act_def_99",
        user_id=user_id,
        source_opportunity_id="opp_def_99",
        source_module="skincare",
        action_type="night_routine",
        priority=3,
        urgency=0.6,
        attention_cost=0.3,
    )

    # Defer action by 15 minutes
    deferred = attention_arbitrator.defer_action(action, timedelta(minutes=15))
    assert deferred.deliver_after > now

    # Verify action is in durable CandidateActionStore
    stored = candidate_action_store.get_action("act_def_99")
    assert stored is not None
    assert stored.id == "act_def_99"
    assert stored.deliver_after == deferred.deliver_after

    # Simulate process restart by clearing memory cache
    candidate_action_store.clear_cache()

    # Re-query user actions; must survive process restart
    user_actions = candidate_action_store.query_user_actions(user_id)
    assert len(user_actions) >= 1
    assert any(a.id == "act_def_99" for a in user_actions)


def test_deferred_candidate_action_redelivery_sweep() -> None:
    """Verify that deferred actions (deliver_after <= now) are scanned by AttentionArbitrator and routed through DeliveryRouter."""
    user_id = "usr_redelivery_test"
    now = datetime.now(timezone.utc)
    past = now - timedelta(minutes=5)

    action = CandidateAction(
        id="act_due_101",
        user_id=user_id,
        source_opportunity_id="opp_due_101",
        source_module="calendar",
        action_type="due_meeting",
        priority=4,
        urgency=0.8,
        attention_cost=0.3,
        deliver_after=past,
        expires_at=now + timedelta(hours=1),
        payload={"title": "Important Sync"},
    )
    candidate_action_store.save_action(action)

    redelivered = attention_arbitrator.scan_and_deliver_due_deferred_actions(user_id)
    assert len(redelivered) == 1
    assert redelivered[0]["action_id"] == "act_due_101"
    assert redelivered[0]["channel"] == "SYSTEM_REMINDER"
    assert redelivered[0]["target_surface"] == "home_ui"

    updated = candidate_action_store.get_action("act_due_101")
    assert updated is not None
    assert updated.status == "DELIVERED"
    assert updated.deliver_after is None

    # A subsequent sweep run must find 0 actions to redeliver (preventing infinite duplicate redeliveries)
    second_run = attention_arbitrator.scan_and_deliver_due_deferred_actions(user_id)
    assert len(second_run) == 0


def test_deferred_candidate_action_routing_failure_preserves_pending_status(monkeypatch) -> None:
    """Verify that if DeliveryRouter returns None or success=False, status remains PENDING for future retry."""
    user_id = "usr_routing_fail_test"
    now = datetime.now(timezone.utc)
    past = now - timedelta(minutes=5)

    action = CandidateAction(
        id="act_fail_202",
        user_id=user_id,
        source_opportunity_id="opp_fail_202",
        source_module="calendar",
        action_type="failing_delivery",
        priority=4,
        urgency=0.8,
        attention_cost=0.3,
        deliver_after=past,
        expires_at=now + timedelta(hours=1),
    )
    candidate_action_store.save_action(action)

    monkeypatch.setattr(
        "brain.temporal.delivery_router.delivery_router.route_candidate_action",
        lambda act: None,
    )

    redelivered = attention_arbitrator.scan_and_deliver_due_deferred_actions(user_id)
    assert len(redelivered) == 0

    updated = candidate_action_store.get_action("act_fail_202")
    assert updated is not None
    assert updated.status == "PENDING"
    assert updated.deliver_after is not None



