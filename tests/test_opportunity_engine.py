"""Unit tests for Phase 3 Signals, OpportunityEngine, and OpportunityNotifier."""

from datetime import datetime, timedelta, timezone

from brain.temporal.models import TimelineItem, TimelineSourceType
from brain.temporal.opportunity_engine import opportunity_engine
from brain.temporal.opportunity_models import OpportunityStatus
from brain.temporal.opportunity_notifier import opportunity_notifier
from brain.temporal.opportunity_store import opportunity_store
from brain.temporal.signals import TemporalSignal, TemporalSignalType, signal_emitter


def test_signal_emitter() -> None:
    now = datetime.now(timezone.utc)
    item = TimelineItem(
        id="cal_signal_1",
        user_id="usr_sig_test",
        source=TimelineSourceType.CALENDAR,
        source_id="c_sig_1",
        title="Strategy Meeting",
        start_time=now + timedelta(minutes=15),
        end_time=now + timedelta(minutes=60),
        preparation_required=True,
        preparation_start=now + timedelta(minutes=5),
    )

    signals = signal_emitter.emit_signals_for_items([item], current_time=now + timedelta(minutes=10))
    assert len(signals) >= 1
    signal_types = [s.signal_type for s in signals]
    assert TemporalSignalType.STARTING_SOON in signal_types or TemporalSignalType.PREPARATION_WINDOW_OPEN in signal_types


def test_opportunity_engine_idempotency_and_notifier() -> None:
    user_id = "usr_opp_engine_test"
    now = datetime.now(timezone.utc)

    notified_opps = []

    def handle_opp(opp):
        notified_opps.append(opp)

    opportunity_notifier.subscribe(handle_opp)

    sig = TemporalSignal(
        id="sig_test_1",
        user_id=user_id,
        signal_type=TemporalSignalType.PREPARATION_WINDOW_OPEN,
        timeline_item_id="cal_prep_123",
        timestamp=now,
        metadata={"title": "Board Meeting"},
    )

    # First evaluation -> Creates opportunity and notifies subscriber
    opp1 = opportunity_engine.evaluate_signal(sig)
    assert opp1 is not None
    assert opp1.status == OpportunityStatus.AVAILABLE
    assert len(notified_opps) == 1
    assert notified_opps[0].id == opp1.id

    # Second evaluation with identical signal -> Deduplicated via idempotency key
    opp2 = opportunity_engine.evaluate_signal(sig)
    assert opp2 is None
    assert len(notified_opps) == 1  # No duplicate notification

    # Verify store retrieval
    stored = opportunity_store.get_opportunity(opp1.id)
    assert stored is not None
    assert stored.idempotency_key == opp1.idempotency_key

    opportunity_notifier.unsubscribe(handle_opp)
