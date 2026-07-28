from __future__ import annotations

from collections import OrderedDict
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from services.data_access_service import get_user_profile, merge_user_profiles
from services.weather_service import WeatherProviderError, get_hourly_weather

logger = logging.getLogger("ahvi.location_weather_context")

_CACHE_MAX = max(16, int(os.getenv("LOCATION_WEATHER_CACHE_MAX_ITEMS", "256")))
_CACHE_TTL = max(30, int(os.getenv("LOCATION_WEATHER_CACHE_TTL_SECONDS", "600")))
_cache: "OrderedDict[Tuple[str, float, float], Tuple[float, Dict[str, Any]]]" = OrderedDict()
_cache_lock = threading.Lock()

_WEATHER_KEYS = ("weather", "weather_context", "weatherContext", "weather_data", "weatherData")
_LOCATION_KEYS = (
    "location_context", "locationContext", "saved_location", "savedLocation",
    "current_location", "currentLocation", "home_location", "homeLocation",
)


def clear_location_weather_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coordinates(value: Any) -> Optional[Tuple[float, float]]:
    row = _dict(value)
    if not row:
        return None
    nested = _dict(row.get("coordinates") or row.get("coords"))
    source = {**nested, **row}
    lat = _number(source.get("lat") if source.get("lat") is not None else source.get("latitude"))
    lon_value = source.get("lon")
    if lon_value is None:
        lon_value = source.get("lng")
    if lon_value is None:
        lon_value = source.get("longitude")
    lon = _number(lon_value)
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    return lat, lon


def _location(value: Any, source: str) -> Optional[Dict[str, Any]]:
    coords = _coordinates(value)
    if not coords:
        return None
    row = _dict(value)
    lat, lon = coords
    label = str(row.get("label") or row.get("name") or row.get("city") or row.get("locationLabel") or "").strip()
    return {
        "status": "available",
        "source": source,
        "latitude": lat,
        "longitude": lon,
        "lat": lat,
        "lon": lon,
        "lng": lon,
        "label": label or None,
        "timezone": row.get("timezone"),
        "captured_at": row.get("captured_at") or row.get("capturedAt"),
        "permission": row.get("permission"),
        "accuracy_m": row.get("accuracy_m") or row.get("accuracyM"),
    }


def _normalize_weather(value: Any, source: str) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        row: Dict[str, Any] = {"condition": value}
    else:
        row = _dict(value)
    if not row:
        return None
    explicit_status = str(row.get("status") or "").strip().lower()
    condition = str(
        row.get("condition") or row.get("weather_type") or row.get("weatherType")
        or row.get("summary") or row.get("description") or ""
    ).strip().lower()
    temperature = None
    for key in ("temperature", "temp_c", "temperature_c", "tempC", "temperatureC"):
        if row.get(key) is not None:
            temperature = _number(row.get(key))
            break
    meaningful = bool(condition) or temperature is not None or bool(row.get("signals"))
    if explicit_status == "unavailable" or not meaningful:
        if explicit_status == "unavailable":
            return {"status": "unavailable", "source": source}
        return None
    out = dict(row)
    out.update({
        "status": "available",
        "source": source,
        "condition": condition or "unknown",
        "weather_type": condition or "unknown",
        "temperature": temperature,
        "temp_c": temperature,
        "temperature_c": temperature,
    })
    return out


def _direct_weather(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in _WEATHER_KEYS:
        if key in request_data:
            weather = _normalize_weather(request_data.get(key), "direct_weather")
            if weather and weather.get("status") == "available":
                return weather
    for container_key in ("context", "context_data", "style_context"):
        container = _dict(request_data.get(container_key))
        for key in _WEATHER_KEYS:
            if key in container:
                weather = _normalize_weather(container.get(key), "direct_weather")
                if weather and weather.get("status") == "available":
                    return weather
    return None


def _resolve_location(request_data: Dict[str, Any], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direct = _location(request_data, "direct_coordinates")
    if direct:
        return direct
    for key in ("coordinates", "coords"):
        direct = _location(request_data.get(key), "direct_coordinates")
        if direct:
            return direct
    request_location = _location(request_data.get("location"), "request.location")
    if request_location:
        return request_location
    for container_key in ("context", "context_data", "style_context"):
        container = _dict(request_data.get(container_key))
        for key in ("location", "location_context", "locationContext"):
            found = _location(container.get(key), f"request.{container_key}.{key}")
            if found:
                return found
    profile_location = _location(profile.get("location"), "profile.location")
    if profile_location:
        return profile_location
    for key in _LOCATION_KEYS:
        for container_name, container in (("request", request_data), ("profile", profile)):
            found = _location(container.get(key), f"{container_name}.{key}")
            if found:
                return found
    return _location(profile, "profile.legacy_coordinates")


def _cached_provider_weather(
    user_id: str,
    lat: float,
    lon: float,
    provider: Callable[..., Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool]:
    key = (user_id, round(lat, 4), round(lon, 4))
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and now < cached[0]:
            _cache.move_to_end(key)
            return dict(cached[1]), True
        if cached:
            _cache.pop(key, None)
    weather = _normalize_weather(provider(lat=lat, lon=lon), "provider")
    if not weather or weather.get("status") != "available":
        raise WeatherProviderError("incomplete_provider_response")
    with _cache_lock:
        _cache[key] = (now + _CACHE_TTL, dict(weather))
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return weather, False


def resolve_location_weather_context(
    *,
    user_id: str,
    request_data: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    provider: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    request_row = _dict(request_data)
    persisted = get_user_profile(user_id=uid) if uid else {}
    effective_profile = merge_user_profiles(persisted, _dict(profile))
    location = _resolve_location(request_row, effective_profile)
    weather = _direct_weather(request_row)
    cache_hit = False
    provider_error = None

    if weather is None and location is not None:
        try:
            weather_provider = provider or get_hourly_weather
            weather, cache_hit = _cached_provider_weather(
                uid or "anonymous", location["latitude"], location["longitude"], weather_provider
            )
        except WeatherProviderError as exc:
            provider_error = {"type": "weather_provider_error", "code": exc.code}
            weather = {
                "status": "unavailable",
                "reason": "weather_provider_unavailable",
                "source": "provider",
                "error": provider_error,
            }
        except Exception:
            provider_error = {"type": "weather_provider_error", "code": "unexpected_provider_failure"}
            weather = {
                "status": "unavailable",
                "reason": "weather_provider_unavailable",
                "source": "provider",
                "error": provider_error,
            }
    if weather is None:
        weather = {
            "status": "unavailable",
            "reason": "weather_location_missing",
            "source": "none",
        }
    if weather.get("status") == "available":
        weather.setdefault(
            "provider",
            "open-meteo" if weather.get("source") == "provider" else weather.get("source"),
        )
        weather.setdefault("weather_timestamp", weather.get("observed_at"))

    usage = {
        "location_used": bool(location),
        "location_source": location.get("source") if location else "none",
        "weather_used": weather.get("status") == "available",
        "weather_status": weather["status"],
        "weather_reason": weather.get("reason"),
        "weather_timestamp": weather.get("weather_timestamp"),
        "weather_provider": weather.get("provider"),
        "calendar_used": False,
        "location": {
            "status": "available" if location else "unavailable",
            "source": location.get("source") if location else "none",
        },
        "weather": {
            "status": weather["status"],
            "source": weather.get("source") or "none",
            "cache_hit": cache_hit,
        },
    }
    if provider_error:
        usage["weather"]["error"] = provider_error
    event = "context.weather_resolved" if weather["status"] == "available" else "context.weather_unavailable"
    logger.info(
        "%s user_id=%s location_status=%s location_source=%s weather_status=%s weather_reason=%s weather_source=%s cache_hit=%s",
        event,
        uid or "anonymous", usage["location"]["status"], usage["location"]["source"],
        usage["weather"]["status"], weather.get("reason"), usage["weather"]["source"], cache_hit,
    )
    return {
        "location": location or {"status": "unavailable", "source": "none"},
        "weather": weather,
        "context_usage": usage,
        "profile": effective_profile,
    }
