import requests
from datetime import datetime


class WeatherEngine:
    """
    🔥 ELITE WEATHER INTELLIGENCE ENGINE

    Converts raw weather → styling signals

    Outputs:
    - temperature intelligence
    - condition intelligence
    - styling signals
    """

    def get_weather_context(self, lat: float, lon: float):

        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m,weathercode,wind_speed_10m"
                f"&timezone=auto"
            )

            res = requests.get(url, timeout=5)
            res.raise_for_status()
            data = res.json()

            hourly = data.get("hourly") or {}
            times = hourly.get("time") or []
            temps = hourly.get("temperature_2m") or []
            codes = hourly.get("weathercode") or []
            winds = hourly.get("wind_speed_10m") or []
            if not (times and temps and codes and winds):
                raise ValueError("incomplete hourly weather data")

            now = datetime.now()
            now_str = now.strftime("%Y-%m-%dT%H:00")

            # -------------------------
            # SAFE INDEX MATCH
            # -------------------------
            if now_str in times:
                idx = times.index(now_str)
            else:
                now_epoch = now.timestamp()

                def _hour_distance(index: int) -> float:
                    try:
                        return abs(datetime.fromisoformat(str(times[index])).timestamp() - now_epoch)
                    except Exception:
                        return float("inf")

                idx = min(range(len(times)), key=_hour_distance)
                if idx >= len(temps) or idx >= len(codes) or idx >= len(winds):
                    idx = 0

            temp = temps[idx]
            code = codes[idx]
            wind = winds[idx]

            # -------------------------
            # 🌡️ TEMPERATURE INTELLIGENCE
            # -------------------------
            if temp >= 35:
                temp_level = "extreme_heat"
                sweat_risk = "very_high"
            elif temp >= 30:
                temp_level = "very_hot"
                sweat_risk = "high"
            elif temp >= 26:
                temp_level = "hot"
                sweat_risk = "medium"
            elif temp >= 18:
                temp_level = "mild"
                sweat_risk = "low"
            else:
                temp_level = "cold"
                sweat_risk = "low"

            # -------------------------
            # 🌧️ WEATHER TYPE
            # -------------------------
            if code == 0:
                weather_type = "clear"
            elif code in [1, 2]:
                weather_type = "partly_cloudy"
            elif code == 3:
                weather_type = "cloudy"
            elif code in [45, 48]:
                weather_type = "fog"
            elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                weather_type = "rain"
            elif code in [95, 96, 99]:
                weather_type = "storm"
            else:
                weather_type = "unknown"

            # -------------------------
            # 🌬️ WIND INTELLIGENCE
            # -------------------------
            if wind >= 25:
                wind_level = "strong"
            elif wind >= 12:
                wind_level = "moderate"
            else:
                wind_level = "light"

            # -------------------------
            # 🌅 TIME OF DAY
            # -------------------------
            hour = now.hour

            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

            # -------------------------
            # 🔥 STYLE SIGNALS (THE REAL MAGIC)
            # -------------------------
            signals = {
                "layering_needed": temp < 20 or weather_type in ["rain", "storm"],
                "breathable_required": temp >= 28,
                "waterproof_required": weather_type in ["rain", "storm"],
                "avoid_loose_flow": wind_level == "strong",
                "prefer_light_colors": temp >= 30,
                "prefer_dark_colors": weather_type in ["cloudy", "storm"],
                "outdoor_friendly": weather_type in ["clear", "partly_cloudy"],
                "sweat_risk": sweat_risk
            }

            return {
                "temperature": temp,
                "temp_level": temp_level,
                "weather_type": weather_type,
                "wind_level": wind_level,
                "time_of_day": time_of_day,
                "signals": signals,
                "raw": {
                    "code": code,
                    "wind_speed": wind
                }
            }

        except Exception as e:
            print("Weather engine failed:", str(e))

            return {
                "temperature": 25,
                "temp_level": "mild",
                "weather_type": "clear",
                "wind_level": "light",
                "time_of_day": "day",
                "signals": {
                    "layering_needed": False,
                    "breathable_required": True,
                    "waterproof_required": False,
                    "avoid_loose_flow": False,
                    "prefer_light_colors": True,
                    "prefer_dark_colors": False,
                    "outdoor_friendly": True,
                    "sweat_risk": "low"
                },
                "raw": {}
            }


weather_engine = WeatherEngine()

def get_hourly_weather(lat: float, lon: float):
    return weather_engine.get_weather_context(lat, lon)
