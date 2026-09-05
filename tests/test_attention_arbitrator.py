"""Unit tests for Attention Arbitrator, candidate action ranking, suppression, and deferrals."""

from datetime import datetime, timedelta, timezone
from brain.temporal.attention_arbitrator import AttentionArbitrator
from brain.temporal.attention_models import CandidateAction


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
