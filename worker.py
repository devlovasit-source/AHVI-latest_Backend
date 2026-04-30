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
redis_url = str(getattr(settings, "redis_url", "") or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
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


def _retry_or_fail(task, exc: Exception, *, max_retries: int = 2, request_id: str | None = None):
    retries = int(getattr(task.request, "retries", 0))
    if request_id:
        set_request_id(request_id)
    if retries >= max_retries:
        try:
            job_tracker.mark_failed(str(getattr(task.request, "id", "")), error=str(exc))
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


def _mark_started(task, *, request_id: str | None = None, user_id: str | None = None) -> None:
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


def _mark_succeeded(task, result_meta: dict | None = None, *, request_id: str | None = None) -> None:
    try:
        if request_id:
            set_request_id(request_id)
        job_tracker.mark_succeeded(str(getattr(task.request, "id", "")), result_meta=result_meta or {})
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
        _mark_succeeded(self, {"task": "generate_audio", "request_id": request_id}, request_id=request_id)
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
        _mark_succeeded(self, {"task": "bg_remove_task", "request_id": request_id}, request_id=request_id)
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
            {"task": "calendar_daily_task", "request_id": request_id, "events": len(results)},
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

        for rem in due:
            processed += 1
            doc_id = str(rem.get("$id") or rem.get("id") or "")
            user_id = str(rem.get("userId") or "")
            message = str(rem.get("message") or "")
            title = "AHVI"

            devices = notification_store.list_devices(user_id=user_id)
            tokens = [str(d.get("token") or "").strip() for d in devices if str(d.get("token") or "").strip()]
            resp = firebase_push_service.send_to_tokens(tokens=tokens, title=title, body=message, data={"type": "reminder"})
            if resp.get("success") and int(resp.get("sent") or 0) > 0:
                sent += int(resp.get("sent") or 0)
                if doc_id:
                    notification_store.mark_reminder(reminder_doc_id=doc_id, status="sent")
            else:
                failed += int(resp.get("failed") or 1)
                if doc_id:
                    notification_store.mark_reminder(reminder_doc_id=doc_id, status="failed", error=str(resp.get("error") or ""))

        _mark_succeeded(
            self,
            {"task": "dispatch_due_reminders_task", "processed": processed, "sent": sent, "failed": failed},
            request_id=request_id,
        )
        return {"status": "success", "processed": processed, "sent": sent, "failed": failed}
    except Exception as e:
        logger.exception("NOTIFICATIONS DISPATCH ERROR")
        _retry_or_fail(self, e, request_id=request_id)


