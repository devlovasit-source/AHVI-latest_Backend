"""Unit tests for OpportunityStore persistence and query operations."""

from brain.temporal.opportunity_models import Opportunity, OpportunityStatus
from brain.temporal.opportunity_store import OpportunityStore


def test_opportunity_store_lifecycle_and_queries() -> None:
    store = OpportunityStore()

    opp = Opportunity.create(
        user_id="usr_store_test",
        opportunity_type="style_preparation",
        timeline_item_id="cal_evt_100",
        trigger_window="2h_prep",
        payload={"suggestion": "Formal Blazer"},
    )

    assert store.has_idempotency_key(opp.idempotency_key) is False

    saved = store.save_opportunity(opp)
    assert saved is True
    assert store.has_idempotency_key(opp.idempotency_key) is True

    fetched = store.get_opportunity(opp.id)
    assert fetched is not None
    assert fetched.id == opp.id
    assert fetched.user_id == "usr_store_test"
    assert fetched.status == OpportunityStatus.CREATED

    user_opps = store.query_user_opportunities("usr_store_test", status=OpportunityStatus.CREATED)
    assert len(user_opps) == 1
    assert user_opps[0].id == opp.id

    updated = store.update_status(opp.id, OpportunityStatus.AVAILABLE)
    assert updated is True

    fetched_updated = store.get_opportunity(opp.id)
    assert fetched_updated is not None
    assert fetched_updated.status == OpportunityStatus.AVAILABLE

    deleted = store.delete_opportunity(opp.id)
    assert deleted is True
    assert store.get_opportunity(opp.id) is None
    assert store.has_idempotency_key(opp.idempotency_key) is False
