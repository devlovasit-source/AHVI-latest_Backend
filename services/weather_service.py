from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import requests


class WeatherProviderError(RuntimeError):
    """A stable, non-provider-specific weather lookup failure."""

    def __init__(self, code: str, message: str = "Weather provider unavailable"):
        super().__init__(message)
        self.code = code


def _weather_type(code: int) -> str:
    if code == 0:
        return "clear"
    if code in {1, 2}:
        return "partly_cloudy"
    if code == 3:
        return "cloudy"
    if code in {45, 48}:
        return "fog"
    if code in {51, 53, 55, 61, 63, 65, 80, 81, 82}:
        return "rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if code in {95, 96, 99}:
        return "storm"
    return "unknown"


def _time_of_day(value: Any) -> str:
    try:
        hour = datetime.fromisoformat(str(value)).hour
    except (TypeError, ValueError):
        raise WeatherProviderError("invalid_provider_time")
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


class WeatherEngine:
    def get_weather_context(self, lat: float, lon: float) -> Dict[str, Any]:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,rain,weather_code,wind_speed_10m"
            "&hourly=precipitation_probability"
            "&timezone=auto"
        )
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise WeatherProviderError("provider_timeout") from exc
        except requests.RequestException as exc:
            raise WeatherProviderError("provider_request_failed") from exc
        except ValueError as exc:
            raise WeatherProviderError("invalid_provider_response") from exc

        current = data.get("current") if isinstance(data, dict) else None
        if not isinstance(current, dict):
            raise WeatherProviderError("incomplete_provider_response")
        try:
            temperature = float(current["temperature_2m"])
            code = int(current["weather_code"])
            wind = float(current["wind_speed_10m"])
            observed_at = str(current["time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherProviderError("incomplete_provider_response") from exc

        condition = _weather_type(code)
        if temperature >= 35:
            temp_level, sweat_risk = "extreme_heat", "very_high"
        elif temperature >= 30:
            temp_level, sweat_risk = "very_hot", "high"
        elif temperature >= 26:
            temp_level, sweat_risk = "hot", "medium"
        elif temperature >= 18:
            temp_level, sweat_risk = "mild", "low"
        else:
            temp_level, sweat_risk = "cold", "low"
        wind_level = "strong" if wind >= 25 else "moderate" if wind >= 12 else "light"
        signals = {
            "layering_needed": temperature < 20 or condition in {"rain", "storm", "snow"},
            "breathable_required": temperature >= 28,
            "waterproof_required": condition in {"rain", "storm", "snow"},
            "avoid_loose_flow": wind_level == "strong",
            "prefer_light_colors": temperature >= 30,
            "prefer_dark_colors": condition in {"cloudy", "storm"},
            "outdoor_friendly": condition in {"clear", "partly_cloudy"},
            "sweat_risk": sweat_risk,
        }
        feels_like = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        hourly = data.get("hourly") if isinstance(data.get("hourly"), dict) else {}
        hourly_times = hourly.get("time") if isinstance(hourly.get("time"), list) else []
        rain_probabilities = (
            hourly.get("precipitation_probability")
            if isinstance(hourly.get("precipitation_probability"), list)
            else []
        )
        rain_probability = None
        if observed_at in hourly_times:
            index = hourly_times.index(observed_at)
            if index < len(rain_probabilities):
                rain_probability = rain_probabilities[index]
        return {
            "status": "available",
            "temperature": temperature,
            "temp_c": temperature,
            "temperature_c": temperature,
            "feels_like_c": float(feels_like) if feels_like is not None else temperature,
            "humidity": float(humidity) if humidity is not None else None,
            "rain_probability": rain_probability,
            "temp_level": temp_level,
            "condition": condition,
            "weather_type": condition,
            "wind_speed": wind,
            "wind_level": wind_level,
            "time_of_day": _time_of_day(observed_at),
            "observed_at": observed_at,
            "weather_timestamp": observed_at,
            "provider": "open-meteo",
            "signals": signals,
            "raw": {"code": code, "wind_speed": wind},
        }


weather_engine = WeatherEngine()


def get_hourly_weather(lat: float, lon: float) -> Dict[str, Any]:
    return weather_engine.get_weather_context(lat, lon)
