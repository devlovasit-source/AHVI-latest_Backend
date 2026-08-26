import uuid
from contextvars import ContextVar

_request_id_ctx: ContextVar[str] = ContextVar("ahvi_request_id", default="")
_authenticated_user_id_ctx: ContextVar[str] = ContextVar(
    "ahvi_authenticated_user_id", default=""
)


def new_request_id() -> str:
    return str(uuid.uuid4())


def set_request_id(request_id: str | None) -> str:
    rid = str(request_id or "").strip() or new_request_id()
    _request_id_ctx.set(rid)
    return rid


def get_request_id(default: str = "") -> str:
    rid = _request_id_ctx.get(default)
    return str(rid or "").strip()


def set_authenticated_user_id(user_id: str | None):
    return _authenticated_user_id_ctx.set(str(user_id or "").strip())


def reset_authenticated_user_id(token) -> None:
    _authenticated_user_id_ctx.reset(token)


def get_authenticated_user_id(default: str = "") -> str:
    user_id = _authenticated_user_id_ctx.get(default)
    return str(user_id or "").strip()
