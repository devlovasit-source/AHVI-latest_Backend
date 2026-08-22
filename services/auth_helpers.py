"""Shared authn/authz helpers and outbound-URL safety checks."""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Iterable, Optional
from urllib.parse import urlparse

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


def get_authed_user_id(request: Request) -> str:
    """Extract authenticated user id from request.state.user, set by auth middleware."""
    user = getattr(request.state, "user", None)
    if not isinstance(user, dict):
        return ""
    return str(user.get("user_id") or user.get("$id") or "").strip()


def require_user(request: Request) -> str:
    """Return authed user id or raise 401."""
    uid = get_authed_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication required")
    return uid


def enforce_owner(request: Request, supplied_user_id: Optional[str]) -> str:
    """Verify the supplied user_id matches the authed user. Returns the canonical id.

    If no supplied user_id, returns the authed user's id (used as the implicit owner).
    Raises 403 if mismatch.
    """
    authed = require_user(request)
    supplied = (supplied_user_id or "").strip()
    if supplied and supplied != authed:
        logger.warning(
            "ownership_mismatch authed=%s supplied=%s path=%s",
            authed,
            supplied,
            request.url.path,
        )
        raise HTTPException(status_code=403, detail="Forbidden")
    return authed


# ---------------------------------------------------------------------------
# Outbound-URL safety (SSRF guard)
# ---------------------------------------------------------------------------

def _split_csv_env(name: str) -> list[str]:
    raw = os.getenv(name) or ""
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


# Defaults cover the project's R2 public host shapes; operators can extend via env.
_DEFAULT_OUTBOUND_HOST_SUFFIXES: tuple[str, ...] = (
    ".r2.cloudflarestorage.com",
    ".r2.dev",
    ".cloudfront.net",
    ".amazonaws.com",  # presigned S3 URLs if used
)


def _allowed_host_suffixes() -> tuple[str, ...]:
    extra = tuple(_split_csv_env("OUTBOUND_URL_ALLOWED_SUFFIXES"))
    return _DEFAULT_OUTBOUND_HOST_SUFFIXES + extra


def _allowed_exact_hosts() -> tuple[str, ...]:
    return tuple(_split_csv_env("OUTBOUND_URL_ALLOWED_HOSTS"))


_PRIVATE_NETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local incl. cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _is_private_ip(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return any(ip in net for net in _PRIVATE_NETS)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return True
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except (ValueError, IndexError):
                continue
            if any(ip in net for net in _PRIVATE_NETS):
                return True
        return False


def is_safe_outbound_url(url: str) -> bool:
    """Whitelist + private-IP block for user-controlled outbound URLs."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if _is_private_ip(host):
        return False
    suffixes = _allowed_host_suffixes()
    exact = _allowed_exact_hosts()
    if host in exact:
        return True
    return any(host == s.lstrip(".") or host.endswith(s) for s in suffixes)


def assert_safe_outbound_url(url: str) -> str:
    if not is_safe_outbound_url(url):
        logger.warning("outbound_url_blocked url=%s", url)
        raise HTTPException(status_code=400, detail="URL not allowed")
    return url


__all__: Iterable[str] = (
    "get_authed_user_id",
    "require_user",
    "enforce_owner",
    "is_safe_outbound_url",
    "assert_safe_outbound_url",
)
