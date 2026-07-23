"""Regression tests for the /api/calendar/today 500.

Root cause: AppwriteProxy._list_documents_page had an UNGUARDED final fallback
_request; a missing/misconfigured collection or unreachable backend raised out
of list_documents and became an uncaught 500 for read endpoints. Reads must
degrade to empty, not 500.
"""
import pytest
from fastapi import HTTPException

import routers.calendar as cal
from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError


def test_list_documents_degrades_to_empty_when_backend_unreachable(monkeypatch):
    proxy = AppwriteProxy()

    def boom(*args, **kwargs):
        raise AppwriteProxyError("unreachable / unconfigured collection")

    monkeypatch.setattr(proxy, "_request", boom)
    # Every query form + the final fallback fail; list_documents must return []
    # rather than propagate the error (which previously caused HTTP 500).
    assert proxy.list_documents("calendar_events", limit=50) == []


def test_calendar_today_empty_returns_200_empty(monkeypatch):
    monkeypatch.setattr(cal, "list_today_calendar_events", lambda uid, date=None: [])
    resp = cal.today_events(user={"$id": "u1"}, date=None)
    assert resp == {"success": True, "events": [], "count": 0}


def test_calendar_today_with_events_returns_count(monkeypatch):
    rows = [{"id": "e1", "start_time": "2026-07-25T09:00:00"}]
    monkeypatch.setattr(cal, "list_today_calendar_events", lambda uid, date=None: rows)
    resp = cal.today_events(user={"$id": "u1"}, date=None)
    assert resp["success"] is True
    assert resp["count"] == 1
    assert resp["events"] == rows


def test_calendar_today_bad_date_is_400_not_500():
    with pytest.raises(HTTPException) as excinfo:
        cal.today_events(user={"$id": "u1"}, date="not-a-real-date")
    assert excinfo.value.status_code == 400
