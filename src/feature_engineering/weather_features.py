from typing import Dict, Any
import pandas as pd


def _get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def compute_time_based_features_unix(ts_unix: int) -> Dict[str, Any]:
    dt = pd.to_datetime(ts_unix, unit="s", utc=True).tz_convert("UTC")
    return {
        "timestamp": dt,
        "hour": int(dt.hour),
        "day": int(dt.day),
        "month": int(dt.month),
        "year": int(dt.year),
        "weekday": int(dt.weekday()),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "season": _get_season(int(dt.month)),
    }


def compute_weather_features(raw_weather_data: Dict[str, Any]) -> Dict[str, Any]:
    ts = raw_weather_data.get("timestamp")
    features: Dict[str, Any] = {}
    if ts is not None:
        features.update(compute_time_based_features_unix(ts))

    features.update(
        {
            "temperature": raw_weather_data.get("temperature"),
            "feels_like": raw_weather_data.get("feels_like"),
            "temp_min": raw_weather_data.get("temp_min"),
            "temp_max": raw_weather_data.get("temp_max"),
            "pressure": raw_weather_data.get("pressure"),
            "humidity": raw_weather_data.get("humidity"),
            "visibility": raw_weather_data.get("visibility"),
            "wind_speed": raw_weather_data.get("wind_speed"),
            "wind_deg": raw_weather_data.get("wind_deg"),
            "clouds": raw_weather_data.get("clouds"),
            "weather_main": raw_weather_data.get("weather_main"),
            "weather_description": raw_weather_data.get("weather_description"),
            "city_name": raw_weather_data.get("city_name"),
            "country": raw_weather_data.get("country"),
        }
    )

    tmin, tmax = features.get("temp_min"), features.get("temp_max")
    if tmin is not None and tmax is not None:
        features["temp_range"] = tmax - tmin

    return features