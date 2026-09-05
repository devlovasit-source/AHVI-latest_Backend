"""Unit tests for DeliveryRouter channel categorization and payload routing."""

from datetime import datetime, timezone
from brain.temporal.attention_models import CandidateAction, DeliveryChannel
from brain.temporal.delivery_router import DeliveryRouter


def test_delivery_router_categorization_and_routing() -> None:
    router = DeliveryRouter()
    now = datetime.now(timezone.utc)

    sys_action = CandidateAction(
        id="act_sys",
        user_id="usr_del_1",
        source_opportunity_id="opp_sys",
        source_module="calendar",
        action_type="meeting_reminder",
        priority=4,
        urgency=0.8,
        attention_cost=0.3,
        deliver_after=now,
        payload={"title": "Client Board Meeting"},
    )

    mod_action = CandidateAction(
        id="act_mod",
        user_id="usr_del_1",
        source_opportunity_id="opp_mod",
        source_module="workout",
        action_type="gym_session",
        priority=3,
        urgency=0.6,
        attention_cost=0.4,
        deliver_after=now,
        payload={"title": "Leg Day Workout"},
    )

    opp_action = CandidateAction(
        id="act_opp",
        user_id="usr_del_1",
        source_opportunity_id="opp_style",
        source_module="style",
        action_type="outfit_prep",
        priority=4,
        urgency=0.7,
        attention_cost=0.2,
        deliver_after=now,
        payload={"title": "Blazer Suggestion"},
    )

    assert router.categorize_channel(sys_action) == DeliveryChannel.SYSTEM_REMINDER
    assert router.categorize_channel(mod_action) == DeliveryChannel.MODULE_REMINDER
    assert router.categorize_channel(opp_action) == DeliveryChannel.AHVI_OPPORTUNITY

    sys_payload = router.route_action(sys_action)
    assert sys_payload["channel"] == "SYSTEM_REMINDER"
    assert sys_payload["target_surface"] == "home_ui"

    mod_payload = router.route_action(mod_action)
    assert mod_payload["channel"] == "MODULE_REMINDER"
    assert mod_payload["target_surface"] == "module_dashboard"

    opp_payload = router.route_action(opp_action)
    assert opp_payload["channel"] == "AHVI_OPPORTUNITY"
    assert opp_payload["target_surface"] == "chat_or_push"
