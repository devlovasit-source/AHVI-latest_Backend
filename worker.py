import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
import logging
from services.job_tracker import job_tracker
from services.request_context import set_request_id
from services.settings import settings

try:
    from sentry_sdk.integrations.redis import RedisIntegration
except Exception:
    RedisIntegration = None


# =========================
# SENTRY SETUP
# =========================
def _has_redis_client() -> bool:
    try:
        import redis  # noqa

        return True
    except Exception:
        return False


_sentry_integrations = [CeleryIntegration()]
if RedisIntegration is not None and _has_redis_client():
    _sentry_integrations.append(RedisIntegration())

_sentry_dsn = os.getenv("SENTRY_DSN")
_sentry_client_ready = False
try:
    _sentry_client_ready = bool(getattr(sentry_sdk.Hub.current, "client", None))
except Exception:
    _sentry_client_ready = False
if _sentry_dsn and not _sentry_client_ready:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=1.0,
        integrations=_sentry_integrations,
    )


# =========================
# CELERY INIT
# =========================
redis_url = str(
    getattr(settings, "redis_url", "")
    or os.getenv("REDIS_URL", "redis://localhost:6379/0")
)
logger = logging.getLogger("ahvi.worker")

celery_app = Celery(
    "ahvi_tasks",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _retry_or_fail(
    task, exc: Exception, *, max_retries: int = 2, request_id: str | None = None
):
    retries = int(getattr(task.request, "retries", 0))
    if request_id:
        set_request_id(request_id)
    if retries >= max_retries:
        try:
            job_tracker.mark_failed(
                str(getattr(task.request, "id", "")), error=str(exc)
            )
        except Exception:
            pass
        raise exc
    countdown = 2 ** (retries + 1)
    try:
        job_tracker.mark_retrying(
            str(getattr(task.request, "id", "")),
            error=str(exc),
            attempt=retries + 1,
            max_retries=max_retries,
        )
    except Exception:
        pass
    raise task.retry(exc=exc, countdown=countdown, max_retries=max_retries)


def _mark_started(
    task, *, request_id: str | None = None, user_id: str | None = None
) -> None:
    try:
        if request_id:
            set_request_id(request_id)
        fields = {}
        if request_id:
            fields["request_id"] = str(request_id)
        if user_id:
            fields["user_id"] = str(user_id)
        if fields:
            job_tracker.update(str(getattr(task.request, "id", "")), **fields)
        attempt = int(getattr(task.request, "retries", 0)) + 1
        job_tracker.mark_started(str(getattr(task.request, "id", "")), attempt=attempt)
    except Exception:
        pass


def _mark_succeeded(
    task, result_meta: dict | None = None, *, request_id: str | None = None
) -> None:
    try:
        if request_id:
            set_request_id(request_id)
        job_tracker.mark_succeeded(
            str(getattr(task.request, "id", "")), result_meta=result_meta or {}
        )
    except Exception:
        pass


# =========================
# AUDIO TASK
# =========================
@celery_app.task(name="generate_audio", bind=True)
def run_heavy_audio_task(self, text_to_clone, lang, request_id: str = ""):
    from services import audio_service

    _mark_started(self, request_id=request_id)
    try:
        audio_base64 = audio_service.generate_cloned_audio(text_to_clone, lang)
        _mark_succeeded(
            self,
            {"task": "generate_audio", "request_id": request_id},
            request_id=request_id,
        )
        return {"status": "success", "audio_base64": audio_base64}
    except Exception as e:
        logger.exception("AUDIO TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


# =========================
# IMAGE TASKS
# =========================
@celery_app.task(name="bg_remove_task", bind=True)
def bg_remove_task(self, image_base64: str, request_id: str = ""):
    from routers.bg_remover import remove_background_sync

    _mark_started(self, request_id=request_id)
    try:
        result = remove_background_sync(image_base64)
        _mark_succeeded(
            self,
            {"task": "bg_remove_task", "request_id": request_id},
            request_id=request_id,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("BG TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="calendar_runtime_task", bind=True)
def calendar_runtime_task(self, event: dict, user_id: str = "", request_id: str = ""):
    """
    Runs calendar intelligence off the main request thread.
    Payload is a CalendarEventInput-like dict.
    """
    from models.calendar_models import CalendarEventInput
    from brain.engines.calendar_runtime import run_calendar_runtime

    _mark_started(self, request_id=request_id, user_id=user_id)
    try:
        parsed = CalendarEventInput.model_validate(event or {})
        result = run_calendar_runtime(parsed, user_id=user_id)
        _mark_succeeded(
            self,
            {"task": "calendar_runtime_task", "request_id": request_id},
            request_id=request_id,
        )
        return {"status": "success", "result": result.model_dump()}
    except Exception as e:
        logger.exception("CALENDAR TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="calendar_daily_task", bind=True)
def calendar_daily_task(self, payload: dict, user_id: str = "", request_id: str = ""):
    """
    Batch calendar runtime for a list of events.
    Payload shape: {"events": [CalendarEventInput-like dict, ...]}
    """
    from models.calendar_models import CalendarEventInput
    from brain.engines.calendar_runtime import run_calendar_runtime

    _mark_started(self, request_id=request_id, user_id=user_id)
    try:
        events = (payload or {}).get("events") or []
        results = []
        for raw in events:
            parsed = CalendarEventInput.model_validate(raw or {})
            results.append(run_calendar_runtime(parsed, user_id=user_id).model_dump())
        _mark_succeeded(
            self,
            {
                "task": "calendar_daily_task",
                "request_id": request_id,
                "events": len(results),
            },
            request_id=request_id,
        )
        return {"status": "success", "result": results}
    except Exception as e:
        logger.exception("CALENDAR DAILY TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="dispatch_due_reminders_task", bind=True)
def dispatch_due_reminders_task(self, window_seconds: int = 60, request_id: str = ""):
    """
    Dispatches due reminders via Firebase push.
    Designed to be triggered by a cron hitting /api/notifications/dispatch-due/async.
    """
    from services.firebase_push_service import firebase_push_service
    from services.notification_store import notification_store

    _mark_started(self, request_id=request_id, user_id="system")
    try:
        due = notification_store.list_due_reminders(window_seconds=int(window_seconds))
        processed = 0
        sent = 0
        failed = 0
        suppressed = 0
        duplicate_prevented = 0

        for rem in due:
            processed += 1
            doc_id = str(rem.get("$id") or rem.get("id") or "")
            user_id = str(rem.get("userId") or "")
            message = str(rem.get("message") or "")
            title = "AHVI"
            category = str(rem.get("source") or "").strip().lower()

            if not doc_id:
                failed += 1
                continue

            claim = notification_store.try_claim_reminder(reminder_doc_id=doc_id)
            if not claim.get("claimed"):
                if claim.get("reason") in ("already_sent", "already_processing"):
                    duplicate_prevented += 1
                continue

            enabled, pref_status = (
                notification_store.get_notification_preference_for_dispatch(
                    user_id=user_id, category=category
                )
            )

            if pref_status == "lookup_failed":
                notification_store.release_claim(reminder_doc_id=doc_id)
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="scheduled",
                    error="preference_lookup_failed",
                )
                continue

            if pref_status in ("invalid_category", "invalid_user"):
                suppressed += 1
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="suppressed"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="suppressed",
                    error=pref_status,
                )
                continue

            if not enabled:
                suppressed += 1
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="suppressed"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="suppressed",
                    error="user_preference",
                )
                continue

            devices = notification_store.list_devices(user_id=user_id)
            tokens = [
                str(d.get("token") or "").strip()
                for d in devices
                if str(d.get("token") or "").strip()
            ]
            resp = firebase_push_service.send_to_tokens(
                tokens=tokens, title=title, body=message, data={"type": "reminder"}
            )
            if resp.get("success") and int(resp.get("sent") or 0) > 0:
                sent += int(resp.get("sent") or 0)
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="sent"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id, status="sent"
                )
            else:
                failed += int(resp.get("failed") or 1)
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="failed"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="failed",
                    error=str(resp.get("error") or ""),
                )

        _mark_succeeded(
            self,
            {
                "task": "dispatch_due_reminders_task",
                "processed": processed,
                "sent": sent,
                "failed": failed,
                "suppressed": suppressed,
                "duplicate_prevented": duplicate_prevented,
            },
            request_id=request_id,
        )
        return {
            "status": "success",
            "processed": processed,
            "sent": sent,
            "failed": failed,
            "suppressed": suppressed,
            "duplicate_prevented": duplicate_prevented,
        }
    except Exception as e:
        logger.exception("NOTIFICATIONS DISPATCH ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="temporal_all_users_background_sweep_task", bind=True)
def temporal_all_users_background_sweep_task(self, request_id: str = ""):
    """
    System-wide periodic cron sweep task that enumerates active user IDs from the system
    user profile index ('users') and dispatches isolated background sweeps per user.
    Guarantees domain collections are NEVER queried in an unscoped cross-tenant manner.
    """
    from services.data_access_service import list_active_user_ids

    _mark_started(self, request_id=request_id)
    try:
        user_ids = list_active_user_ids()
        dispatched = 0
        for uid in user_ids:
            temporal_background_sweep_task.delay(user_id=uid, request_id=request_id)
            dispatched += 1

        _mark_succeeded(
            self,
            {
                "task": "temporal_all_users_background_sweep_task",
                "users_enumerated": len(user_ids),
                "tasks_dispatched": dispatched,
            },
            request_id=request_id,
        )
        return {
            "status": "success",
            "users_enumerated": len(user_ids),
            "tasks_dispatched": dispatched,
        }
    except Exception as e:
        logger.exception("TEMPORAL ALL USERS BACKGROUND SWEEP TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="temporal_background_sweep_task", bind=True)
def temporal_background_sweep_task(self, user_id: str = "", request_id: str = ""):
    """
    Executes background temporal intelligence evaluation, cleanup, crash recovery, and
    deferred action redelivery sweep for a specific user_id.
    Explicitly user-bound in background context — no un-scoped global reads.
    """
    from brain.temporal.aggregation_service import aggregation_service
    from brain.temporal.attention_arbitrator import attention_arbitrator
    from brain.temporal.opportunity_engine import opportunity_engine
    from brain.temporal.retention_worker import retention_worker
    from brain.temporal.signals import signal_emitter

    uid = str(user_id or "").strip()
    if not uid:
        return {"status": "skipped", "reason": "missing_user_id"}

    _mark_started(self, request_id=request_id, user_id=uid)
    try:
        # Step 1: Cleanup and auto-reclaim leases / replay unconsumed opportunities
        cleanup_metrics = retention_worker.run_cleanup(uid)

        # Step 2: Fetch user-bound unified timeline
        items = aggregation_service.fetch_unified_timeline(uid)

        # Step 3: Emit signals
        signals = signal_emitter.emit_signals_for_items(items)

        # Step 4: Process signals with logical idempotency
        opps = opportunity_engine.process_signals(signals)

        # Step 5: Scan and deliver due deferred candidate actions
        redelivered_actions = attention_arbitrator.scan_and_deliver_due_deferred_actions(uid)

        _mark_succeeded(
            self,
            {
                "task": "temporal_background_sweep_task",
                "user_id": uid,
                "items_count": len(items),
                "signals_count": len(signals),
                "opportunities_created": len(opps),
                "redelivered_actions_count": len(redelivered_actions),
                "cleanup_metrics": cleanup_metrics,
            },
            request_id=request_id,
        )
        return {
            "status": "success",
            "user_id": uid,
            "items_count": len(items),
            "signals_count": len(signals),
            "opportunities_created": len(opps),
            "redelivered_actions_count": len(redelivered_actions),
            "cleanup_metrics": cleanup_metrics,
        }
    except Exception as e:
        logger.exception("TEMPORAL BACKGROUND SWEEP TASK ERROR user_id=%s", uid)
        _retry_or_fail(self, e, request_id=request_id)
