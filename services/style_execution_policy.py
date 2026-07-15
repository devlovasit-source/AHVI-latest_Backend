"""Server-owned execution controls for Style recommendation work.

The active policy lives in a context variable so nested orchestration, ranking,
and curation stages share one model-call budget.  HTTP payloads never select a
policy or budget; route handlers create the production policy server-side.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, TypeVar

logger = logging.getLogger("ahvi.style_execution")

# Measured from the existing successful module-chat Style path: one agent call,
# one curation call, and at most one application-level truncation retry.
PRODUCTION_MODEL_CALL_LIMIT = 3
READ_ONLY_EVALUATION_MODEL_CALL_LIMIT = 1


class UnknownStyleExecutionPolicy(ValueError):
    pass


class ModelCallBudgetExceeded(RuntimeError):
    code = "MODEL_CALL_BUDGET_EXCEEDED"


@dataclass
class ModelCallBudget:
    limit: int
    _count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.count)

    def consume(self, *, stage: str, model_alias: str) -> int:
        with self._lock:
            if self._count >= self.limit:
                logger.info(
                    "style_model_call blocked count=%d stage=%s model=%s",
                    self._count,
                    _safe_label(stage),
                    _safe_label(model_alias),
                )
                raise ModelCallBudgetExceeded(ModelCallBudgetExceeded.code)
            self._count += 1
            count = self._count
        logger.info(
            "style_model_call allowed count=%d stage=%s model=%s",
            count,
            _safe_label(stage),
            _safe_label(model_alias),
        )
        return count


@dataclass(frozen=True)
class StyleExecutionPolicy:
    name: str
    allow_preference_learning: bool
    allow_board_registration: bool
    allow_cache_writes: bool
    allow_image_generation: bool
    model_call_limit: int


@dataclass
class StyleExecutionSinks:
    preference_memory: Optional[Callable[..., Any]] = None
    learning_vector: Optional[Callable[..., Any]] = None
    board_state: Optional[Callable[..., Any]] = None


@dataclass
class StyleExecutionSession:
    policy: StyleExecutionPolicy
    budget: ModelCallBudget
    sinks: StyleExecutionSinks = field(default_factory=StyleExecutionSinks)


_ACTIVE: contextvars.ContextVar[Optional[StyleExecutionSession]] = contextvars.ContextVar(
    "active_style_execution", default=None
)


def _safe_label(value: Any) -> str:
    text = "".join(ch for ch in str(value or "unknown").strip() if ch.isalnum() or ch in "._-")
    return text[:64] or "unknown"


def create_style_execution_session(
    policy_name: str = "production",
    *,
    sinks: Optional[StyleExecutionSinks] = None,
) -> StyleExecutionSession:
    name = str(policy_name or "").strip().lower()
    if name == "production":
        policy = StyleExecutionPolicy(
            name="production",
            allow_preference_learning=False,
            allow_board_registration=True,
            allow_cache_writes=True,
            allow_image_generation=True,
            model_call_limit=PRODUCTION_MODEL_CALL_LIMIT,
        )
    elif name == "read_only_evaluation":
        policy = StyleExecutionPolicy(
            name="read_only_evaluation",
            allow_preference_learning=False,
            allow_board_registration=True,
            allow_cache_writes=False,
            allow_image_generation=False,
            model_call_limit=READ_ONLY_EVALUATION_MODEL_CALL_LIMIT,
        )
    else:
        raise UnknownStyleExecutionPolicy("unknown Style execution policy")
    return StyleExecutionSession(
        policy=policy,
        budget=ModelCallBudget(policy.model_call_limit),
        sinks=sinks or StyleExecutionSinks(),
    )


def get_style_execution_session() -> Optional[StyleExecutionSession]:
    return _ACTIVE.get()


@contextmanager
def activate_style_execution(session: StyleExecutionSession) -> Iterator[StyleExecutionSession]:
    if not isinstance(session, StyleExecutionSession):
        raise UnknownStyleExecutionPolicy("invalid Style execution session")
    if session.policy.name not in {"production", "read_only_evaluation"}:
        raise UnknownStyleExecutionPolicy("unknown Style execution policy")
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)


@contextmanager
def production_style_execution() -> Iterator[StyleExecutionSession]:
    existing = get_style_execution_session()
    if existing is not None:
        yield existing
        return
    with activate_style_execution(create_style_execution_session("production")) as session:
        yield session


def consume_model_call(*, stage: str, model_alias: str) -> int:
    session = get_style_execution_session()
    if session is None:
        return 0
    return session.budget.consume(stage=stage, model_alias=model_alias)


def record_model_latency(*, stage: str, model_alias: str, started: float) -> None:
    session = get_style_execution_session()
    if session is None:
        return
    logger.info(
        "style_model_call complete count=%d stage=%s model=%s latency_ms=%d",
        session.budget.count,
        _safe_label(stage),
        _safe_label(model_alias),
        max(0, int((time.perf_counter() - started) * 1000)),
    )


def preference_learning_allowed() -> bool:
    session = get_style_execution_session()
    return session is None or bool(session.policy.allow_preference_learning)


def board_registration_allowed() -> bool:
    session = get_style_execution_session()
    return session is None or bool(session.policy.allow_board_registration)


def image_generation_allowed() -> bool:
    session = get_style_execution_session()
    return session is None or bool(session.policy.allow_image_generation)


def cache_writes_allowed() -> bool:
    session = get_style_execution_session()
    return session is None or bool(session.policy.allow_cache_writes)


def run_preference_memory_write(default_sink: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    session = get_style_execution_session()
    if session is not None and not session.policy.allow_preference_learning:
        return False
    sink = session.sinks.preference_memory if session and session.sinks.preference_memory else default_sink
    return sink(*args, **kwargs)


def run_learning_vector_upsert(default_sink: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    session = get_style_execution_session()
    if session is not None and not session.policy.allow_preference_learning:
        return None
    sink = session.sinks.learning_vector if session and session.sinks.learning_vector else default_sink
    return sink(*args, **kwargs)


def run_board_registration(default_sink: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    session = get_style_execution_session()
    if session is not None and not session.policy.allow_board_registration:
        return {
            "ok": False,
            "error": {"code": "BOARD_REGISTRATION_NOT_ALLOWED", "message": "Board registration is unavailable."},
        }
    sink = session.sinks.board_state if session and session.sinks.board_state else default_sink
    return sink(*args, **kwargs)


F = TypeVar("F", bound=Callable[..., Any])


def server_style_execution(func: F) -> F:
    """Wrap an internal/route function in a server-created production policy."""
    if getattr(func, "__style_execution_wrapped__", False):
        return func

    if __import__("inspect").iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with production_style_execution():
                return await func(*args, **kwargs)

        async_wrapper.__style_execution_wrapped__ = True
        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        with production_style_execution():
            return func(*args, **kwargs)

    sync_wrapper.__style_execution_wrapped__ = True
    return sync_wrapper  # type: ignore[return-value]


__all__ = [
    "ModelCallBudget",
    "ModelCallBudgetExceeded",
    "PRODUCTION_MODEL_CALL_LIMIT",
    "READ_ONLY_EVALUATION_MODEL_CALL_LIMIT",
    "StyleExecutionPolicy",
    "StyleExecutionSession",
    "StyleExecutionSinks",
    "UnknownStyleExecutionPolicy",
    "activate_style_execution",
    "board_registration_allowed",
    "cache_writes_allowed",
    "consume_model_call",
    "create_style_execution_session",
    "get_style_execution_session",
    "image_generation_allowed",
    "preference_learning_allowed",
    "production_style_execution",
    "record_model_latency",
    "run_board_registration",
    "run_learning_vector_upsert",
    "run_preference_memory_write",
    "server_style_execution",
]
