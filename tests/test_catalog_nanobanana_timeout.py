"""Nano Banana catalog provider must BOUND the Vertex image-gen call.

Regression guard for the save-selected ~120s tail: the provider used to
`del timeout`, leaving the SDK call unbounded. It must now pass a positive
timeout (seconds) into the genai client, defaulting to 75s and overridable via
NANO_BANANA_TIMEOUT_SECONDS.
"""
import services.catalog_png_generation_service as svc


def _capture_timeout(monkeypatch, env_value=None):
    if env_value is None:
        monkeypatch.delenv("NANO_BANANA_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("NANO_BANANA_TIMEOUT_SECONDS", env_value)
    provider = svc.CatalogProviderNanoBanana()
    seen = {}

    def fake_client(self, timeout_s=None):
        seen["timeout_s"] = timeout_s
        return None  # short-circuits generate() after recording the timeout

    monkeypatch.setattr(svc.CatalogProviderNanoBanana, "_client", fake_client)
    provider.generate(
        cutout_bytes=b"x", prompt="p", item_metadata={}, timeout=30
    )
    return seen["timeout_s"]


def test_default_timeout_is_75s(monkeypatch):
    assert _capture_timeout(monkeypatch) == 75


def test_env_overrides_timeout(monkeypatch):
    assert _capture_timeout(monkeypatch, "90") == 90


def test_generic_timeout_never_shrinks_below_nanobanana_floor(monkeypatch):
    # A generic CATALOG_TIMEOUT_SECONDS of 30s must NOT be applied to
    # nanobanana (normal success is ~37s) — the 75s floor wins.
    assert _capture_timeout(monkeypatch) >= 75


def test_bad_env_falls_back_to_default(monkeypatch):
    assert _capture_timeout(monkeypatch, "not-a-number") == 75
