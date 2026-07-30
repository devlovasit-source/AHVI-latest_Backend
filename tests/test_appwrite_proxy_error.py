from types import SimpleNamespace

import pytest

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError


def test_legacy_error_construction_preserves_message():
    error = AppwriteProxyError("legacy failure")

    assert str(error) == "legacy failure"
    assert error.status_code is None


def test_error_accepts_explicit_status_code():
    error = AppwriteProxyError("not found", status_code=404)

    assert str(error) == "not found"
    assert error.status_code == 404


def test_known_http_failure_preserves_status_code(monkeypatch):
    response = SimpleNamespace(status_code=409, text="conflict")
    session = SimpleNamespace(request=lambda **kwargs: response)
    monkeypatch.setattr(AppwriteProxy, "_get_session", classmethod(lambda cls: session))
    proxy = AppwriteProxy()
    monkeypatch.setattr(proxy, "_ensure_config", lambda: None)

    with pytest.raises(AppwriteProxyError) as raised:
        proxy._request("POST", "https://example.test")

    assert raised.value.status_code == 409
    assert str(raised.value) == "Appwrite request failed (409): conflict"


def test_unknown_failure_has_no_status_code():
    error = AppwriteProxyError("connection failed")

    assert error.status_code is None
