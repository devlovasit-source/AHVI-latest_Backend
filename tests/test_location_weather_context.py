from services.location_weather_context import (
    clear_location_weather_cache,
    resolve_location_weather_context,
)
from services.weather_service import WeatherProviderError
from services.weather_service import WeatherEngine


def _weather(condition="clear", temperature=24):
    return {
        "condition": condition,
        "temperature": temperature,
        "time_of_day": "afternoon",
    }


def test_direct_weather_overrides_coordinates_and_normalizes_aliases(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})

    def fail_provider(**kwargs):
        raise AssertionError("direct weather must bypass provider lookup")

    result = resolve_location_weather_context(
        user_id="u1",
        request_data={
            "latitude": 10,
            "longitude": 20,
            "weatherData": {"weatherType": "Rain", "tempC": 19},
        },
        provider=fail_provider,
    )

    assert result["location"]["source"] == "direct_coordinates"
    assert result["weather"]["status"] == "available"
    assert result["weather"]["condition"] == "rain"
    assert result["weather"]["weather_type"] == "rain"
    assert result["weather"]["temperature"] == 19
    assert result["weather"]["temp_c"] == 19
    assert result["context_usage"]["weather"]["source"] == "direct_weather"


def test_location_precedence_request_then_profile_then_saved(monkeypatch):
    monkeypatch.setattr(
        "services.location_weather_context.get_user_profile",
        lambda **_: {
            "location": {"lat": 30, "lng": 31},
            "savedLocation": {"latitude": 40, "longitude": 41},
        },
    )
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return _weather()

    result = resolve_location_weather_context(
        user_id="u1",
        request_data={"location": {"latitude": 20, "longitude": 21}},
        provider=provider,
    )
    assert result["location"]["source"] == "request.location"
    assert calls == [{"lat": 20.0, "lon": 21.0}]

    clear_location_weather_cache()
    result = resolve_location_weather_context(user_id="u1", provider=provider)
    assert result["location"]["source"] == "profile.location"


def test_saved_current_home_and_legacy_aliases(monkeypatch):
    profiles = [
        ({"saved_location": {"lat": 1, "lon": 2}}, "profile.saved_location"),
        ({"currentLocation": {"latitude": 3, "longitude": 4}}, "profile.currentLocation"),
        ({"homeLocation": {"lat": 5, "lng": 6}}, "profile.homeLocation"),
        ({"latitude": 7, "longitude": 8}, "profile.legacy_coordinates"),
    ]
    for profile, expected in profiles:
        clear_location_weather_cache()
        monkeypatch.setattr(
            "services.location_weather_context.get_user_profile",
            (lambda value: lambda **_: value)(profile),
        )
        result = resolve_location_weather_context(
            user_id="u1", provider=lambda **_: _weather()
        )
        assert result["location"]["source"] == expected


def test_provider_failure_is_typed_and_does_not_fabricate_weather(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})

    def provider(**kwargs):
        raise WeatherProviderError("provider_timeout")

    result = resolve_location_weather_context(
        user_id="u1",
        request_data={"coordinates": {"lat": 10, "lng": 20}},
        provider=provider,
    )
    assert result["weather"] == {
        "status": "unavailable",
        "reason": "weather_provider_unavailable",
        "source": "provider",
        "error": {"type": "weather_provider_error", "code": "provider_timeout"},
    }
    assert "temperature" not in result["weather"]


def test_cache_is_bounded_by_user_and_location(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})
    clear_location_weather_cache()
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return _weather()

    request = {"lat": 10.123456, "lon": 20.654321}
    first = resolve_location_weather_context(user_id="u1", request_data=request, provider=provider)
    second = resolve_location_weather_context(user_id="u1", request_data=request, provider=provider)
    third = resolve_location_weather_context(user_id="u2", request_data=request, provider=provider)
    fourth = resolve_location_weather_context(
        user_id="u1", request_data={"lat": 11, "lon": 20.654321}, provider=provider
    )

    assert first["context_usage"]["weather"]["cache_hit"] is False
    assert second["context_usage"]["weather"]["cache_hit"] is True
    assert third["context_usage"]["weather"]["cache_hit"] is False
    assert fourth["context_usage"]["weather"]["cache_hit"] is False
    assert len(calls) == 3


def test_missing_location_and_weather_are_canonically_unavailable(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})
    result = resolve_location_weather_context(user_id="u1")
    assert result["location"]["status"] == "unavailable"
    assert result["weather"] == {
        "status": "unavailable",
        "reason": "weather_location_missing",
        "source": "none",
    }


def test_frontend_location_context_shape_is_resolved(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})
    result = resolve_location_weather_context(
        user_id="u1",
        request_data={
            "location_context": {
                "lat": 17.385,
                "lon": 78.487,
                "timezone": "Asia/Kolkata",
                "captured_at": "2026-07-29T07:00:00Z",
                "source": "device",
                "permission": "granted",
                "accuracy_m": 12,
            }
        },
        provider=lambda **_: _weather("rain", 28),
    )
    assert result["location"]["source"] == "request.location_context"
    assert result["location"]["timezone"] == "Asia/Kolkata"
    assert result["weather"]["status"] == "available"
    assert result["context_usage"]["location_used"] is True
    assert result["context_usage"]["weather_used"] is True


def test_weather_producer_uses_provider_local_time_and_consumer_aliases(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "current": {
                    "time": "2026-07-29T19:00",
                    "temperature_2m": 27,
                    "weather_code": 61,
                    "wind_speed_10m": 14,
                }
            }

    monkeypatch.setattr("services.weather_service.requests.get", lambda *args, **kwargs: Response())
    weather = WeatherEngine().get_weather_context(12, 77)
    assert weather["time_of_day"] == "evening"
    assert weather["condition"] == weather["weather_type"] == "rain"
    assert weather["temperature"] == weather["temp_c"] == weather["temperature_c"] == 27


def test_weather_producer_raises_typed_failure(monkeypatch):
    import requests

    def timeout(*args, **kwargs):
        raise requests.Timeout("offline")

    monkeypatch.setattr("services.weather_service.requests.get", timeout)
    try:
        WeatherEngine().get_weather_context(12, 77)
    except WeatherProviderError as exc:
        assert exc.code == "provider_timeout"
    else:
        raise AssertionError("expected typed provider failure")
