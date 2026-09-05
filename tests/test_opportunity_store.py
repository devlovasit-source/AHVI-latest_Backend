"""Unit tests for OpportunityStore persistence, database-layer atomicity, distinct key isolation, and claim leases."""

import concurrent.futures
from datetime import datetime, timedelta, timezone
from brain.temporal.opportunity_models import Opportunity, OpportunityStatus
from brain.temporal.opportunity_store import OpportunityStore, opportunity_store


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


def test_concurrent_idempotency_race() -> None:
    """Test 10 concurrent threads attempting to save opportunities with identical idempotency keys."""
    store = OpportunityStore()
    user_id = "usr_race_test"

    results = []

    def attempt_save(idx: int) -> bool:
        opp = Opportunity.create(
            user_id=user_id,
            opportunity_type="race_condition_type",
            timeline_item_id="cal_race_999",
            trigger_window="logical_win_prep",
            rule_version="v1",
        )
        return store.save_opportunity(opp)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(attempt_save, i) for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Exactly 1 save call should succeed (return True), the other 9 should be deduplicated (return False)
    successes = [r for r in results if r is True]
    assert len(successes) == 1

    # Verify only 1 opportunity exists in store for user
    opps = store.query_user_opportunities(user_id)
    assert len(opps) == 1


def test_distinct_opportunities_do_not_collide() -> None:
    """Assert that distinct logical opportunities for the same user and type stay distinct and do not collide."""
    store = OpportunityStore()
    user_id = "usr_distinct_test"

    opp1 = Opportunity.create(
        user_id=user_id,
        opportunity_type="style_preparation",
        timeline_item_id="cal_evt_101",
        trigger_window="logical_win_prep",
    )
    opp2 = Opportunity.create(
        user_id=user_id,
        opportunity_type="style_preparation",
        timeline_item_id="cal_evt_102",
        trigger_window="logical_win_prep",
    )

    assert opp1.id != opp2.id
    assert opp1.idempotency_key != opp2.idempotency_key

    saved1 = store.save_opportunity(opp1)
    saved2 = store.save_opportunity(opp2)

    assert saved1 is True
    assert saved2 is True

    opps = store.query_user_opportunities(user_id)
    assert len(opps) == 2


def test_claim_lease_and_reclaim() -> None:
    """Test consumer claim lease creation, active lease rejection, and lease expiration reclamation."""
    store = OpportunityStore()
    user_id = "usr_lease_test"

    opp = Opportunity.create(
        user_id=user_id,
        opportunity_type="test_lease",
        timeline_item_id="cal_lease_1",
        trigger_window="win_prep",
    )
    opp.status = OpportunityStatus.AVAILABLE
    store.save_opportunity(opp)

    # 1. Claim opportunity with 2-second lease
    claimed = store.claim_opportunity(opp.id, consumer_id="consumer_A", lease_seconds=2)
    assert claimed is not None
    assert claimed.status == OpportunityStatus.CLAIMED
    assert claimed.claimed_at is not None

    # 2. Immediate second claim while lease is active must be rejected
    second_claim = store.claim_opportunity(opp.id, consumer_id="consumer_B", lease_seconds=2)
    assert second_claim is None

    # 3. Simulate expired lease by overriding lease_expires_at to past
    claimed.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    store._memory_cache[opp.id] = claimed
    if opp.id in store._persistent_db_store:
        store._persistent_db_store[opp.id]["lease_expires_at"] = claimed.lease_expires_at.isoformat()

    # 4. Reclaim expired leases
    reclaimed = store.reclaim_expired_leases(user_id)
    assert len(reclaimed) == 1
    assert reclaimed[0].id == opp.id
    assert reclaimed[0].status == OpportunityStatus.AVAILABLE


def test_database_409_conflict_atomic_rejection(monkeypatch) -> None:
    """Verify that when Appwrite returns a 409 Conflict / duplicate ID error, save_opportunity atomically rejects duplicate."""
    store = OpportunityStore()
    opp = Opportunity.create(
        user_id="usr_409_test",
        opportunity_type="test_type",
        timeline_item_id="item_409",
        trigger_window="win_409",
    )

    def mock_create_document(*args, **kwargs):
        raise Exception("AppwriteProxyError: 409 Document already exists with ID")

    monkeypatch.setattr("services.appwrite_proxy.AppwriteProxy.create_document", mock_create_document)

    saved = store.save_opportunity(opp)
    assert saved is False

