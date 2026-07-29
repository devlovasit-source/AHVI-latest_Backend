"""list_today_calendar_events must resolve "today" in the calendar timezone.

A naive server-local now() (UTC on Cloud Run) resolved to the previous day
between 00:00 and 05:30 IST, so an early-morning IST event was missing from the
Calendar/Home "today" list. No network.
"""
from datetime import datetime, timedelta

import services.calendar_service as cs


def _events(store):
    def _list(user_id, *, start_time=None, end_time=None, limit=200):
        s = cs._as_utc(cs._parse_iso(start_time))
        e = cs._as_utc(cs._parse_iso(end_time))
        out = []
        for ev in store:
            if ev.get("user_id") != user_id:
                continue
            edt = cs._as_utc(cs._parse_iso(ev.get("start_time")))
            if s and edt and edt < s:
                continue
            if e and edt and edt > e:
                continue
            out.append(ev)
        return out

    return _list


def _ev(title, dt):
    return {"id": title, "title": title, "user_id": "u1", "start_time": dt.isoformat()}


class _FrozenDatetime(datetime):
    """datetime.now(tz) frozen at 00:30 IST — the UTC date is still yesterday."""

    _frozen = datetime(2026, 7, 26, 0, 30, tzinfo=cs._CALENDAR_TZ)

    @classmethod
    def now(cls, tz=None):
        if tz:
            return cls._frozen.astimezone(tz)
        # No tz => server-local wall clock. Cloud Run runs UTC, so this is the
        # PREVIOUS day at 19:00 — exactly the bug being guarded.
        return cls._frozen.astimezone(cs.timezone.utc).replace(tzinfo=None)


def test_0030_ist_uses_the_local_date_not_the_utc_date(monkeypatch):
    # 00:30 IST on the 26th == 19:00 UTC on the 25th.
    assert _FrozenDatetime._frozen.astimezone(cs.timezone.utc).day == 25
    today_evt = _ev("Gym", datetime(2026, 7, 26, 7, 0, tzinfo=cs._CALENDAR_TZ))
    yesterday_evt = _ev("Old", datetime(2026, 7, 25, 7, 0, tzinfo=cs._CALENDAR_TZ))
    store = [today_evt, yesterday_evt]
    monkeypatch.setattr(cs, "list_calendar_events", _events(store))
    monkeypatch.setattr(cs, "datetime", _FrozenDatetime)
    out = cs.list_today_calendar_events("u1")
    titles = [e["title"] for e in out]
    assert titles == ["Gym"]  # local day, not the UTC day


def test_end_of_day_boundary_included_and_next_day_excluded(monkeypatch):
    day = datetime(2026, 7, 26, tzinfo=cs._CALENDAR_TZ)
    store = [
        _ev("LateTonight", day + timedelta(hours=23, minutes=59)),
        _ev("Midnight", day + timedelta(days=1)),  # boundary end == next day start
        _ev("Tomorrow", day + timedelta(days=1, hours=1)),
    ]
    monkeypatch.setattr(cs, "list_calendar_events", _events(store))
    out = cs.list_today_calendar_events("u1", date="2026-07-26")
    titles = [e["title"] for e in out]
    assert "LateTonight" in titles
    assert "Tomorrow" not in titles


def test_event_just_before_local_midnight_is_excluded(monkeypatch):
    day = datetime(2026, 7, 26, tzinfo=cs._CALENDAR_TZ)
    store = [
        _ev("JustBefore", day - timedelta(minutes=1)),
        _ev("Today", day + timedelta(hours=9)),
    ]
    monkeypatch.setattr(cs, "list_calendar_events", _events(store))
    out = cs.list_today_calendar_events("u1", date="2026-07-26")
    assert [e["title"] for e in out] == ["Today"]


def test_window_is_timezone_aware(monkeypatch):
    captured = {}

    def _list(user_id, *, start_time=None, end_time=None, limit=200):
        captured["start"] = start_time
        captured["end"] = end_time
        return []

    monkeypatch.setattr(cs, "list_calendar_events", _list)
    cs.list_today_calendar_events("u1", date="2026-07-26")
    assert cs._parse_iso(captured["start"]).utcoffset() is not None
    assert cs._parse_iso(captured["end"]).utcoffset() is not None
