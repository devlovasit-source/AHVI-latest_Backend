"""Test-suite network tripwire.

Hard-fails any test that attempts a real outbound connection to a
non-loopback address. Unit tests must mock external services (Qdrant,
Appwrite, R2, Gemini/Vertex, Firebase, plain HTTP). Loopback and AF_UNIX
sockets stay allowed so in-process fixtures (ASGI TestClient, local test
servers) keep working.

This exists to guarantee "zero external calls" during the suite. If a test
newly fails here, it was relying on a real network/SDK call and is not
hermetic -- fix the test, do not weaken this guard.
"""
from __future__ import annotations

import ipaddress
import socket

import pytest

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection

_ALLOWED_HOSTNAMES = {"localhost", ""}


def _is_allowed(address) -> bool:
    # AF_UNIX (str path) or anything non-(host, port) -> allow.
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if host in _ALLOWED_HOSTNAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Unresolved external hostname -> block.
        return False
    return ip.is_loopback


def _block(address):
    raise RuntimeError(
        f"Blocked external network connection to {address!r}. Tests must mock "
        "external services (no real Qdrant/Appwrite/R2/Gemini/Firebase/HTTP calls)."
    )


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch):
    def guard_connect(self, address, *args, **kwargs):
        if not _is_allowed(address):
            _block(address)
        return _real_connect(self, address, *args, **kwargs)

    def guard_connect_ex(self, address, *args, **kwargs):
        if not _is_allowed(address):
            _block(address)
        return _real_connect_ex(self, address, *args, **kwargs)

    def guard_create_connection(address, *args, **kwargs):
        if not _is_allowed(address):
            _block(address)
        return _real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guard_connect, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", guard_connect_ex, raising=True)
    monkeypatch.setattr(socket, "create_connection", guard_create_connection, raising=True)
    yield
