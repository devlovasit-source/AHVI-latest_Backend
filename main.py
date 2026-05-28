import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Callable
from typing import Any, Dict
from uuid import uuid4
from time import perf_counter

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import importlib.util
import os


def _is_production() -> bool:
    env = (
        str(os.getenv("ENV") or os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "")
        .strip()
        .lower()
    )
    return env in {"prod", "production"}


# ?? QDRANT SERVICE
from services.qdrant_service import qdrant_service
from services.security_limits import (
    check_rate_limit,
    extract_client_ip,
    is_redis_rate_limit_ready,
)
from services.settings import settings
from middleware.auth_middleware import get_current_user
from services.job_tracker import job_tracker
from services.request_context import set_request_id

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)
# Make our app loggers actually emit alongside uvicorn's access log.
for _ln in ("ahvi", "ahvi.main", "ahvi.routers.chat"):
    logging.getLogger(_ln).setLevel(_LOG_LEVEL)

logger = logging.getLogger("ahvi.main")
ROUTER_LOAD_STATUS: dict[str, dict[str, Any]] = {}
REQUIRED_ROUTERS = set(settings.required_routers or [])
SERVICE_TAG = str(os.getenv("AHVI_SERVICE_TAG") or "board-intel-gap-msg").strip()
SERVICE_REVISION = str(
    os.getenv("K_REVISION")
    or os.getenv("GIT_SHA")
    or os.getenv("APP_REVISION")
    or os.getenv("APP_RELEASE")
    or ""
).strip()


def _mark_router_skipped(module_name: str, reason: str):
    required = module_name in REQUIRED_ROUTERS
    ROUTER_LOAD_STATUS[module_name] = {
        "status": "skipped",
        "required": required,
        "error": reason,
    }
    logger.info(
        "router skipped module=%s reason=%s required=%s", module_name, reason, required
    )
    if required and settings.strict_router_loading:
        raise RuntimeError(f"required router skipped: {module_name} ({reason})")


# -------------------------
# OPTIONAL ROUTER LOADER
# -------------------------
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_optional_router(module_name: str, attr: str = "router"):
    required = module_name in REQUIRED_ROUTERS
    if not _has_module(module_name):
        status = {"status": "not_found", "required": required, "error": None}
        ROUTER_LOAD_STATUS[module_name] = status
        logger.info(
            "router skipped module=%s reason=not_found required=%s",
            module_name,
            required,
        )
        if required and settings.strict_router_loading:
            raise RuntimeError(f"required router not found: {module_name}")
        return None
    try:
        module = __import__(module_name, fromlist=[attr])
        router = getattr(module, attr)
        ROUTER_LOAD_STATUS[module_name] = {
            "status": "loaded",
            "required": required,
            "error": None,
        }
        return router
    except Exception as exc:
        ROUTER_LOAD_STATUS[module_name] = {
            "status": "failed",
            "required": required,
            "error": str(exc),
        }
        logger.exception(
            "router load failed module=%s required=%s", module_name, required
        )
        if required and settings.strict_router_loading:
            raise RuntimeError(
                f"required router failed to load: {module_name}"
            ) from exc
        return None


# -------------------------
# LOAD ALL ROUTERS (SAFE)
# -------------------------
chat_router = _load_optional_router("routers.chat")
data_router = _load_optional_router("routers.data")
utilities_router = _load_optional_router("routers.utilities")
boards_router = _load_optional_router("routers.boards")
feedback_router = _load_optional_router("routers.feedback")
ops_router = _load_optional_router("routers.ops")
calendar_router = _load_optional_router("routers.calendar")
med_logs_router = _load_optional_router("routers.med_logs")
medi_router = _load_optional_router("routers.medi")
skincare_adherence_router = _load_optional_router("routers.skincare_adherence")
notifications_router = _load_optional_router("routers.notifications")
workouts_router = _load_optional_router("routers.workouts")
bills_router = _load_optional_router("routers.bills")
lens_similar_router = _load_optional_router("routers.lens_similar")
contacts_router = _load_optional_router("routers.ahvi_contacts")

# AI
ai_router = _load_optional_router("api.ai")

# Optional
stylist_router = _load_optional_router("routers.stylist")
reddit_router = _load_optional_router("routers.reddit")

# Feature-based
bg_router = None
if os.getenv("ENABLE_BG_REMOVER", "false").lower() in ("1", "true", "yes"):
    # bg_service routes uploads to the GCE-hosted RMBG service via
    # RMBG_SERVICE_URL (httpx + redis only). HuggingFace Inference is the
    # last-resort fallback when HF_TOKEN is set.
    if _has_module("httpx"):
        bg_router = _load_optional_router("routers.bg_router")
    else:
        _mark_router_skipped("routers.bg_router", "missing_dependency")
else:
    _mark_router_skipped("routers.bg_router", "feature_flag_disabled")

vision_router = None
if os.getenv("ENABLE_VISION", "false").lower() in ("1", "true", "yes"):
    if all(_has_module(m) for m in ["cv2", "sklearn", "numpy"]):
        vision_router = _load_optional_router("routers.vision")
    else:
        _mark_router_skipped("routers.vision", "missing_dependency")
else:
    _mark_router_skipped("routers.vision", "feature_flag_disabled")

wardrobe_capture_router = None
if os.getenv("ENABLE_WARDROBE_CAPTURE", "true").lower() in ("1", "true", "yes"):
    if all(_has_module(m) for m in ["numpy", "PIL"]):
        wardrobe_capture_router = _load_optional_router("routers.wardrobe_capture")
    else:
        _mark_router_skipped("routers.wardrobe_capture", "missing_dependency")
else:
    _mark_router_skipped("routers.wardrobe_capture", "feature_flag_disabled")

garment_router = None
if os.getenv("ENABLE_GARMENT_ANALYZER", "false").lower() in ("1", "true", "yes"):
    if all(_has_module(m) for m in ["transformers", "PIL", "cv2", "sklearn", "numpy"]):
        garment_router = _load_optional_router("routers.garment_analyzer")
    else:
        _mark_router_skipped("routers.garment_analyzer", "missing_dependency")
else:
    _mark_router_skipped("routers.garment_analyzer", "feature_flag_disabled")


# -------------------------
# OPTIONAL IMPORTS
# -------------------------
try:
    from celery.result import AsyncResult
except Exception:
    AsyncResult = None

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except Exception:
    sentry_sdk = None
    FastApiIntegration = None

try:
    from worker import celery_app
except Exception:
    celery_app = None


# -------------------------
# SENTRY
# -------------------------
_sentry_dsn = str(os.getenv("SENTRY_DSN") or "").strip()
_sentry_client_ready = False
if sentry_sdk:
    try:
        _sentry_client_ready = bool(getattr(sentry_sdk.Hub.current, "client", None))
    except Exception:
        _sentry_client_ready = False

def _looks_like_cloud_run() -> bool:
    # Google Cloud Run injects K_SERVICE / K_REVISION at runtime.
    return bool(os.getenv("K_SERVICE") or os.getenv("K_REVISION"))


# Sentry is only mandatory when ENV is explicitly production. On Cloud Run we
# warn but do not block startup so a missing DSN cannot keep the service from
# coming up.
if _is_production():
    if not _sentry_dsn:
        raise RuntimeError(
            "SENTRY_DSN is required when ENV=production. "
            "Set SENTRY_DSN or unset ENV/APP_ENV to run without it."
        )
    if not (sentry_sdk and FastApiIntegration):
        raise RuntimeError(
            "sentry-sdk is not installed but ENV=production. "
            "Add sentry-sdk to requirements.txt."
        )
elif _looks_like_cloud_run() and not _sentry_dsn:
    logger.warning(
        "running on Cloud Run without SENTRY_DSN — error tracking disabled. "
        "Set SENTRY_DSN to enable."
    )

if _sentry_dsn and sentry_sdk and FastApiIntegration and not _sentry_client_ready:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        integrations=[FastApiIntegration()],
        environment=str(os.getenv("ENV") or os.getenv("APP_ENV") or "development"),
        release=str(os.getenv("APP_RELEASE") or os.getenv("GIT_SHA") or ""),
    )
    _sentry_client_ready = True


# -------------------------
# APP INIT (with lifespan for graceful startup/shutdown)
# -------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("startup begin")
    try:
        await asyncio.to_thread(qdrant_service.init)
    except Exception as e:
        logger.exception("qdrant startup failed: %s", e)

    try:
        yield
    finally:
        logger.info("shutdown begin")

        # Drain in-flight chat orchestrator work first.
        try:
            from routers.chat import shutdown_chat_resources

            await asyncio.to_thread(shutdown_chat_resources)
        except Exception:
            logger.exception("chat orchestrator shutdown failed")

        # Close Qdrant.
        try:
            client = getattr(qdrant_service, "client", None)
            if client is not None and hasattr(client, "close"):
                await asyncio.to_thread(client.close)
        except Exception as e:
            logger.exception("qdrant shutdown failed: %s", e)

        # Close Redis pool used for rate-limit + auth cache.
        try:
            from services.security_limits import get_redis_client

            redis_client = await get_redis_client()
            if redis_client is not None and hasattr(redis_client, "aclose"):
                await redis_client.aclose()
            elif redis_client is not None and hasattr(redis_client, "close"):
                await redis_client.close()
        except Exception:
            logger.exception("redis shutdown failed")

        # Close shared httpx async client used by AppwriteProxy.
        try:
            from services.appwrite_proxy import AppwriteProxy

            client = AppwriteProxy._shared_async_client
            if client is not None and hasattr(client, "aclose"):
                await client.aclose()
                AppwriteProxy._shared_async_client = None
        except Exception:
            logger.exception("appwrite async client shutdown failed")

        # Close Appwrite admin client.
        try:
            from services import appwrite_service

            appwrite_client = getattr(appwrite_service, "client", None)
            if appwrite_client is not None and hasattr(appwrite_client, "close"):
                await asyncio.to_thread(appwrite_client.close)
            else:
                logger.info("appwrite shutdown skip: client.close() unavailable")
        except Exception as e:
            logger.warning("appwrite shutdown skip error=%s", e)

        logger.info("shutdown complete")


app = FastAPI(
    title="AHVI AI Master Brain API",
    version="2.2.0",
    lifespan=lifespan,
)

logger.info("AHVI Backend Started")


class PayloadTooLargeError(Exception):
    pass


class StreamBodyLimitMiddleware:
    def __init__(self, app: Callable, max_bytes: int):
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "")).upper()
        if method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = {}
        for k, v in scope.get("headers", []):
            try:
                headers[k.decode("latin-1").lower()] = v.decode("latin-1")
            except Exception:
                continue

        content_length = headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "success": False,
                            "error": {
                                "code": "PAYLOAD_TOO_LARGE",
                                "message": f"Upload exceeds max size {self.max_bytes} bytes",
                            },
                        },
                    )
                    await response(scope, receive, send)
                    return
            except Exception:
                pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                received += len(chunk)
                if received > self.max_bytes:
                    raise PayloadTooLargeError()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except PayloadTooLargeError:
            response = JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error": {
                        "code": "PAYLOAD_TOO_LARGE",
                        "message": f"Upload exceeds max size {self.max_bytes} bytes",
                    },
                },
            )
            await response(scope, receive, send)


# -------------------------
# ERROR HANDLERS
# -------------------------
_HTTP_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = str(getattr(request.state, "request_id", "") or "")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "request_id": request_id,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request",
                "details": exc.errors(),
            },
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = str(getattr(request.state, "request_id", "") or "")
    code = _HTTP_ERROR_CODES.get(exc.status_code, "HTTP_ERROR")
    detail = exc.detail
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("detail") or code)
        details = detail
    else:
        message = str(detail or code)
        details = None
    body: Dict[str, Any] = {
        "success": False,
        "request_id": request_id,
        "error": {"code": code, "message": message},
    }
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(getattr(request.state, "request_id", "") or "")
    logger.exception("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "request_id": request_id,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
            },
        },
    )


# -------------------------
# MIDDLEWARE
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=(
        settings.cors_allow_credentials and not ("*" in settings.cors_allowed_origins)
    ),
    allow_methods=settings.cors_allowed_methods,
    allow_headers=settings.cors_allowed_headers,
)
if settings.cors_allow_credentials and "*" in settings.cors_allowed_origins:
    logger.warning(
        "CORS_ALLOW_CREDENTIALS ignored because CORS_ALLOWED_ORIGINS contains '*'"
    )

app.add_middleware(StreamBodyLimitMiddleware, max_bytes=settings.upload_max_bytes)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID")
    request_id = str(incoming or "").strip() or str(uuid4())
    set_request_id(request_id)
    request.state.request_id = request_id
    if sentry_sdk and _sentry_client_ready:
        try:
            sentry_sdk.set_tag("request_id", request_id)
            sentry_sdk.set_tag("path", request.url.path)
        except Exception:
            pass
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(getattr(response, "status_code", 500))
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception(
            "request failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    finally:
        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "request request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )


@app.middleware("http")
async def auth_guard_middleware(request: Request, call_next):
    if not settings.auth_required:
        return await call_next(request)
    # CORS preflights never carry an Authorization header. Let the CORS
    # middleware below answer them.
    if str(request.method or "").upper() == "OPTIONS":
        return await call_next(request)
    path = str(request.url.path or "")
    if (
        path == "/"
        or path.startswith("/health")
        or path == "/api/health"
        or path == "/api/notifications/health"
        or path.startswith("/api/notifications/dispatch-due")
        or path.startswith("/docs")
        or path.startswith("/openapi")
        or path == "/api/wardrobe/diagnostics"
    ):
        # Note: dispatch-due is intentionally bypassed here — the route enforces
        # its own NOTIFICATIONS_DISPATCH_SECRET.
        return await call_next(request)
    try:
        request.state.user = await get_current_user(request)
    except HTTPException as exc:
        request_id = str(getattr(request.state, "request_id", "") or "")
        code = _HTTP_ERROR_CODES.get(exc.status_code, "HTTP_ERROR")
        has_auth_header = bool(request.headers.get("authorization"))
        logger.warning(
            "auth_guard_reject path=%s status=%s reason=%r has_auth_header=%s request_id=%s",
            path,
            exc.status_code,
            str(exc.detail),
            has_auth_header,
            request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "request_id": request_id,
                "error": {"code": code, "message": str(exc.detail or code)},
            },
        )
    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not settings.rate_limit_enabled:
        return await call_next(request)
    if str(request.method or "").upper() == "OPTIONS":
        return await call_next(request)
    redis_ready = await is_redis_rate_limit_ready()
    if settings.rate_limit_require_redis and not redis_ready:
        status_code = 429 if settings.rate_limit_fail_closed else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "request_id": str(getattr(request.state, "request_id", "") or ""),
                "error": {
                    "code": "RATE_LIMIT_BACKEND_UNAVAILABLE",
                    "message": "Rate-limit backend unavailable",
                },
            },
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )
    request_id = str(getattr(request.state, "request_id", "") or "")
    ip = extract_client_ip(
        request.headers, request.client.host if request.client else None
    )
    user_id = ""
    path = str(request.url.path or "")
    if (
        settings.auth_required
        and not isinstance(getattr(request.state, "user", None), dict)
        and path != "/"
        and not path.startswith("/health")
        and path != "/api/health"
        and path != "/api/notifications/health"
        and not path.startswith("/api/notifications/dispatch-due")
        and not path.startswith("/docs")
        and not path.startswith("/openapi")
    ):
        try:
            request.state.user = await get_current_user(request)
        except HTTPException:
            pass
    if isinstance(getattr(request.state, "user", None), dict):
        user_id = str(
            request.state.user.get("user_id")
            or request.state.user.get("$id")
            or request.state.user.get("id")
            or ""
        )
    identity = user_id or ip
    allowed, remaining = await check_rate_limit(
        bucket_key=f"{identity}:{request.url.path}",
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "request_id": request_id,
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "Too many requests. Please retry later.",
                },
            },
            headers={
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Limit": str(settings.rate_limit_max_requests),
                "X-RateLimit-Window": str(settings.rate_limit_window_seconds),
                "X-RateLimit-Backend": "redis" if redis_ready else "local",
                "Retry-After": str(settings.rate_limit_window_seconds),
            },
        )
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_max_requests)
    response.headers["X-RateLimit-Window"] = str(settings.rate_limit_window_seconds)
    response.headers["X-RateLimit-Backend"] = "redis" if redis_ready else "local"
    return response


# -------------------------
# ROUTER REGISTRATION
# -------------------------
if chat_router:
    app.include_router(chat_router, prefix="/api", tags=["Chat"])

if data_router:
    app.include_router(data_router)

if utilities_router:
    app.include_router(utilities_router)

if boards_router:
    app.include_router(boards_router)

if ai_router:
    app.include_router(ai_router, prefix="/api", tags=["AI"])

if feedback_router:
    app.include_router(feedback_router, tags=["Feedback"])

if ops_router:
    app.include_router(ops_router, prefix="/api/ops", tags=["Ops"])

if calendar_router:
    app.include_router(calendar_router, prefix="/api")

if med_logs_router:
    app.include_router(med_logs_router, prefix="/api")

if medi_router:
    app.include_router(medi_router, prefix="/api")

if skincare_adherence_router:
    app.include_router(skincare_adherence_router, prefix="/api")

if notifications_router:
    app.include_router(notifications_router)

if workouts_router:
    app.include_router(workouts_router, prefix="/api")

if bills_router:
    app.include_router(bills_router)

if lens_similar_router:
    app.include_router(lens_similar_router)

if contacts_router:
    app.include_router(contacts_router, prefix="/api/contacts", tags=["contacts"])

if stylist_router:
    app.include_router(stylist_router, prefix="/api/stylist")

if reddit_router:
    app.include_router(reddit_router)

if vision_router:
    app.include_router(vision_router, prefix="/api/vision")

if wardrobe_capture_router:
    app.include_router(wardrobe_capture_router)
    try:
        from routers.wardrobe_capture import wardrobe_router as _wardrobe_admin_router
        app.include_router(_wardrobe_admin_router)
    except Exception:
        _mark_router_skipped("routers.wardrobe_capture.wardrobe_router", "import_failed")

if bg_router:
    app.include_router(bg_router, prefix="/api/background")

if garment_router:
    app.include_router(garment_router, prefix="/api")

if not chat_router:

    class _FallbackMessage(BaseModel):
        role: str = Field(default="user", min_length=1, max_length=24)
        content: str = Field(default="", max_length=4000)

    class _FallbackChatRequest(BaseModel):
        messages: list[_FallbackMessage] = Field(default_factory=list, max_length=30)
        user_id: str | None = Field(default=None, max_length=128)
        userID: str | None = Field(default=None, max_length=128)

    @app.post("/api/text")
    def fallback_text_chat(payload: _FallbackChatRequest):
        prompt = ""
        if payload.messages:
            prompt = str(payload.messages[-1].content or "").strip()
        lower = prompt.lower()
        if "joke" in lower:
            message = "Here is a tiny one: Why did the shirt get promoted? Because it had outstanding style."
        elif "how are you" in lower or lower in {"hi", "hello", "hey"}:
            message = "I am here and ready. Ask me for an outfit, a capsule wardrobe, or just talk to me."
        elif any(
            k in lower
            for k in ["outfit", "wear", "style", "wardrobe", "date", "casual"]
        ):
            message = "I will assume smart casual for now: choose one clean hero piece, pair it with a neutral base, and finish with footwear or an accessory that matches the occasion."
        else:
            message = "I can help with that. Tell me a little more, or ask me to style an outfit, plan your day, or build a capsule wardrobe."
        # Canonical AHVI chat response shape — kept in sync with
        # routers/chat.py:_module_llm_response so clients don't need to
        # special-case the fallback path.
        return {
            "success": True,
            "type": "fallback_chat",
            "module": "fallback",
            "response": message,
            "message_text": message,
            "message": {"role": "assistant", "content": message},
            "cards": [],
            "style_boards": [],
            "chips": [],
            "data": {
                "module": "fallback",
                "rendered_boards": [],
                "outfits": [],
            },
            "meta": {
                "mode": "fallback",
                "chat_router_loaded": False,
                "board_count": 0,
            },
        }

    @app.post("/api/module-chat")
    @app.post("/api/chat/module-chat")
    def fallback_module_chat(payload: Dict[str, Any]):
        message = str(payload.get("message") or "").strip()
        if not message:
            messages = payload.get("messages") or []
            if messages and isinstance(messages, list):
                last = messages[-1] or {}
                if isinstance(last, dict):
                    message = str(last.get("content") or "").strip()
        response = fallback_text_chat(
            _FallbackChatRequest(messages=[_FallbackMessage(role="user", content=message)])
        )
        response["type"] = "module_response"
        response["module"] = str(
            payload.get("domain") or payload.get("module") or "chat"
        ).strip().lower()
        return response


# -------------------------
# HEALTH
# -------------------------


# -------------------------
# BG REMOVE COMPAT ROUTES
# -------------------------
class BgCompatRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)


@app.post("/api/background/remove-bg")
@app.post("/api/remove-bg")
async def remove_bg_compat(payload: BgCompatRequest):
    try:
        from services.bg_service import remove_bg_bytes
        import base64
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"BG remover unavailable: {exc}")

    try:
        image_bytes = base64.b64decode(payload.image_base64.split(",")[-1])
        result_bytes = await remove_bg_bytes(image_bytes)
        return {
            "success": True,
            "bg_removed": True,
            "image_base64": base64.b64encode(result_bytes).decode(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Background removal failed: {exc}")


class VisionCompatRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    user_id: str | None = None
    userId: str | None = None


if not vision_router:

    @app.post("/api/analyze-image")
    @app.post("/api/vision/analyze-image")
    @app.post("/api/vision/analyze")
    @app.post("/api/analyze")
    def analyze_compat(payload: VisionCompatRequest):
        raise HTTPException(
            status_code=503,
            detail="Vision analyzer is currently disabled on this server.",
        )


@app.get("/")
def root():
    return {"message": "AHVI backend running"}


def _health_identity() -> Dict[str, str]:
    return {
        "service": "ahvi-backend",
        "tag": SERVICE_TAG,
        "revision": SERVICE_REVISION,
    }


@app.get("/health")
async def health_check():
    required_router_failures = [
        name
        for name, row in ROUTER_LOAD_STATUS.items()
        if bool((row or {}).get("required"))
        and str((row or {}).get("status")) != "loaded"
    ]
    redis_ready = await is_redis_rate_limit_ready()
    qdrant_ready = bool(getattr(qdrant_service, "client", None))
    appwrite_endpoint = str(
        os.getenv("APPWRITE_ENDPOINT", "")
        or os.getenv("EXPO_PUBLIC_APPWRITE_ENDPOINT", "")
    ).strip()
    appwrite_project = str(
        os.getenv("APPWRITE_PROJECT_ID", "")
        or os.getenv("APPWRITE_PROJECT", "")
        or os.getenv("EXPO_PUBLIC_APPWRITE_PROJECT_ID", "")
    ).strip()
    appwrite_database = str(
        os.getenv("APPWRITE_DATABASE_ID", "")
        or os.getenv("EXPO_PUBLIC_APPWRITE_DATABASE_ID", "")
    ).strip()
    appwrite_configured = all((appwrite_endpoint, appwrite_project, appwrite_database))
    celery_ready = bool(celery_app and AsyncResult is not None)

    ready = not required_router_failures
    status_text = "online" if ready else "degraded"
    return {
        **_health_identity(),
        "status": status_text,
        "ready": ready,
        "checks": {
            "required_routers_ok": not required_router_failures,
            "required_router_failures": required_router_failures,
            "redis_ready": redis_ready,
            "qdrant_configured": qdrant_ready,
            "appwrite_configured": appwrite_configured,
            "celery_configured": celery_ready,
        },
    }


@app.get("/api/health")
async def api_health_check():
    return await health_check()


async def _probe_redis(timeout: float) -> Dict[str, Any]:
    try:
        from services.security_limits import get_redis_client

        client = await asyncio.wait_for(get_redis_client(), timeout=timeout)
        if client is None:
            return {"ok": False, "error": "not_configured"}
        await asyncio.wait_for(client.ping(), timeout=timeout)
        return {"ok": True}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


async def _probe_qdrant(timeout: float) -> Dict[str, Any]:
    client = getattr(qdrant_service, "client", None)
    if client is None:
        return {"ok": False, "error": "not_configured"}
    try:
        await asyncio.wait_for(
            asyncio.to_thread(client.get_collections), timeout=timeout
        )
        return {"ok": True}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


async def _probe_appwrite(timeout: float) -> Dict[str, Any]:
    appwrite_endpoint = str(
        os.getenv("APPWRITE_ENDPOINT", "")
        or os.getenv("EXPO_PUBLIC_APPWRITE_ENDPOINT", "")
    ).strip()
    appwrite_project = str(
        os.getenv("APPWRITE_PROJECT_ID", "")
        or os.getenv("APPWRITE_PROJECT", "")
        or os.getenv("EXPO_PUBLIC_APPWRITE_PROJECT_ID", "")
    ).strip()
    if not (appwrite_endpoint and appwrite_project):
        return {"ok": False, "error": "not_configured"}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(
                f"{appwrite_endpoint.rstrip('/')}/health",
                headers={"X-Appwrite-Project": appwrite_project},
            )
            if resp.status_code < 500:
                return {"ok": True, "status": resp.status_code}
            return {"ok": False, "error": f"http_{resp.status_code}"}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


@app.get("/health/ready")
async def health_ready():
    timeout = float(os.getenv("READINESS_PROBE_TIMEOUT_SECONDS", "1.5"))
    request_id = ""

    required_router_failures = [
        name
        for name, row in ROUTER_LOAD_STATUS.items()
        if bool((row or {}).get("required"))
        and str((row or {}).get("status")) != "loaded"
    ]

    redis_probe, qdrant_probe, appwrite_probe = await asyncio.gather(
        _probe_redis(timeout),
        _probe_qdrant(timeout),
        _probe_appwrite(timeout),
    )

    # Critical: appwrite (auth + data) must be reachable.
    # Redis: critical only if rate_limit_require_redis OR auth cache is required.
    # Qdrant: degraded-but-not-critical (wardrobe vector search).
    redis_critical = bool(settings.rate_limit_require_redis)
    appwrite_critical = bool(settings.auth_required)

    critical_failures = []
    if required_router_failures:
        critical_failures.append(
            {"check": "required_routers", "failed": required_router_failures}
        )
    if appwrite_critical and not appwrite_probe["ok"]:
        critical_failures.append(
            {"check": "appwrite", "error": appwrite_probe.get("error")}
        )
    if redis_critical and not redis_probe["ok"]:
        critical_failures.append({"check": "redis", "error": redis_probe.get("error")})

    ready = not critical_failures
    body = {
        "ready": ready,
        "status": "online" if ready else "degraded",
        "checks": {
            "required_routers_ok": not required_router_failures,
            "required_router_failures": required_router_failures,
            "redis": redis_probe,
            "qdrant": qdrant_probe,
            "appwrite": appwrite_probe,
        },
        "critical_failures": critical_failures,
    }
    if not ready:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "request_id": request_id,
                "error": {
                    "code": "NOT_READY",
                    "message": "One or more critical dependencies are unavailable.",
                    "details": body,
                },
            },
        )
    return body


@app.get("/health/routes")
def health_routes():
    required_router_failures = [
        name
        for name, row in ROUTER_LOAD_STATUS.items()
        if bool((row or {}).get("required"))
        and str((row or {}).get("status")) != "loaded"
    ]
    return {
        "status": "online" if not required_router_failures else "degraded",
        "strict_router_loading": settings.strict_router_loading,
        "required_routers": sorted(REQUIRED_ROUTERS),
        "required_router_failures": required_router_failures,
        "routers": ROUTER_LOAD_STATUS,
    }


# -------------------------
# CELERY STATUS
# -------------------------
@app.get("/api/tasks/{job_id}")
def get_task_status(job_id: str, request: Request):
    from services.auth_helpers import require_user

    authed_user = require_user(request)
    request_id = str(getattr(request.state, "request_id", "") or "")
    tracker_data = job_tracker.get(job_id) or {}
    job_owner = str((tracker_data or {}).get("user_id") or "").strip()
    if job_owner and job_owner != authed_user:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not celery_app or AsyncResult is None:
        if tracker_data:
            return {
                "status": tracker_data.get("status", "queued"),
                "state": tracker_data.get("state", "PENDING"),
                "job": tracker_data,
                "request_id": request_id,
            }
        return {"status": "celery not configured", "request_id": request_id}

    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.state == "PENDING":
        return {
            "status": str(tracker_data.get("status") or "queued"),
            "state": "PENDING",
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "STARTED":
        return {
            "status": "processing",
            "state": "STARTED",
            "meta": task_result.info if isinstance(task_result.info, dict) else {},
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "SUCCESS":
        return {
            "status": "completed",
            "state": "SUCCESS",
            "result": task_result.result,
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "FAILURE":
        return {
            "status": "failed",
            "state": "FAILURE",
            "error": str(task_result.info),
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "RETRY":
        return {
            "status": "retrying",
            "state": "RETRY",
            "error": str(task_result.info),
            "job": tracker_data,
            "request_id": request_id,
        }

    return {
        "status": str(tracker_data.get("status") or "processing"),
        "state": task_result.state,
        "job": tracker_data,
        "request_id": request_id,
    }


@app.get("/api/jobs/recent")
def list_recent_jobs(
    request: Request,
    limit: int = 25,
    user_id: str | None = None,
    request_id: str | None = None,
):
    from services.auth_helpers import enforce_owner

    user_id = enforce_owner(request, user_id)
    return {
        "success": True,
        "jobs": job_tracker.list_recent(
            limit=limit, user_id=user_id, request_id=request_id
        ),
    }
