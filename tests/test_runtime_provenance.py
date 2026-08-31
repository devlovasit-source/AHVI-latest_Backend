import main
from fastapi.testclient import TestClient


def test_cloud_run_identity_is_surfaced_from_runtime_environment(monkeypatch):
    monkeypatch.setenv("K_REVISION", "ahvi-backend-00042-abc")
    monkeypatch.setenv("K_SERVICE", "ahvi-backend")
    monkeypatch.setenv("K_CONFIGURATION", "ahvi-backend")

    identity = main._runtime_identity()

    assert identity == {
        "revision": "ahvi-backend-00042-abc",
        "cloud_run_service": "ahvi-backend",
        "cloud_run_configuration": "ahvi-backend",
    }

    async def redis_not_configured():
        return False

    monkeypatch.setattr(main, "is_redis_rate_limit_ready", redis_not_configured)
    health = TestClient(main.app).get("/health")
    assert health.status_code == 200
    assert health.json()["revision"] == "ahvi-backend-00042-abc"
    assert health.json()["cloud_run_service"] == "ahvi-backend"
    assert health.headers["X-AHVI-Revision"] == "ahvi-backend-00042-abc"


def test_cloud_run_identity_has_safe_local_fallback(monkeypatch):
    for name in ("K_REVISION", "K_SERVICE", "K_CONFIGURATION"):
        monkeypatch.delenv(name, raising=False)

    assert main._runtime_identity() == {
        "revision": "unknown",
        "cloud_run_service": "local",
        "cloud_run_configuration": "local",
    }


def test_revision_header_correlates_a_request_without_exposing_secrets(monkeypatch):
    monkeypatch.setenv("K_REVISION", "ahvi-backend-00042-abc")
    monkeypatch.setenv("SENTRY_DSN", "super-secret-dsn")
    response = TestClient(main.app).get("/")

    assert response.status_code == 200
    assert response.headers["X-AHVI-Revision"] == "ahvi-backend-00042-abc"
    assert "super-secret-dsn" not in response.text
    assert set(main._health_identity()) == {
        "service",
        "tag",
        "revision",
        "cloud_run_service",
        "cloud_run_configuration",
    }
