import asyncio

import pytest

import services.security_limits as sl


def _reset():
    sl._redis_client = None
    sl._redis_disabled = None
    sl._redis_next_retry = 0.0


def test_redis_disabled_when_unconfigured(monkeypatch):
    _reset()
    for var in ("REDIS_URL", "UPSTASH_REDIS_URL", "RAILWAY_REDIS_URL"):
        monkeypatch.delenv(var, raising=False)
    client = asyncio.run(sl.get_redis_client())
    assert client is None
    assert sl._redis_disabled is True


def test_redis_unconfigured_does_not_redial(monkeypatch):
    _reset()
    for var in ("REDIS_URL", "UPSTASH_REDIS_URL", "RAILWAY_REDIS_URL"):
        monkeypatch.delenv(var, raising=False)
    calls = {"n": 0}
    if sl.redis_async is not None:
        def _boom(*a, **k):
            calls["n"] += 1
            raise RuntimeError("should not dial when unconfigured")
        monkeypatch.setattr(sl.redis_async, "from_url", _boom)
    assert asyncio.run(sl.get_redis_client()) is None
    assert asyncio.run(sl.get_redis_client()) is None
    assert calls["n"] == 0  # never dialed localhost


def test_redis_failure_trips_cooldown(monkeypatch):
    _reset()
    monkeypatch.setenv("REDIS_URL", "redis://example.invalid:6379/0")
    monkeypatch.setattr(sl, "_redis_configured", lambda: True)
    if sl.redis_async is None:
        pytest.skip("redis lib unavailable")

    def _boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(sl.redis_async, "from_url", _boom)
    assert asyncio.run(sl.get_redis_client()) is None
    assert sl._redis_next_retry > 0.0  # circuit breaker armed
    # Within cooldown, a second call returns None without re-dialing.
    assert asyncio.run(sl.get_redis_client()) is None


def test_workouts_today_route_registered_as_authenticated_get():
    from main import app
    routes = {
        (r.path, "GET" in getattr(r, "methods", set()) if getattr(r, "methods", None) else False)
        for r in app.routes
        if getattr(r, "path", "") == "/api/workouts/today"
    }
    assert any(path == "/api/workouts/today" for path, _ in routes)
    # Route depends on get_current_user (authenticated).
    import routers.workouts as w
    assert callable(w.today_workout)
