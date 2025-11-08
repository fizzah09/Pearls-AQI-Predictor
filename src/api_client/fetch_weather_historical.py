from typing import Dict, Any, Optional
import logging
import requests

log = logging.getLogger(__name__)


def fetch_weather_historical(
    api_key: str,
    base_url: str,
    lat: float,
    lon: float,
    dt_unix: int,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Fetch historical weather near a unix timestamp using One Call timemachine.

    Keeps this as a standalone function (separate from WeatherClient) per request.
    Returns a simplified dict similar to WeatherClient.fetch_weather_data.
    """
    sess = session or requests
    url = f"{base_url}/onecall/timemachine"
    params = {"lat": lat, "lon": lon, "dt": int(dt_unix), "appid": api_key, "units": "metric"}
    resp = sess.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    record = raw.get("current")
    if not record:
        hours = raw.get("hourly") or []
        if hours:
            record = min(hours, key=lambda h: abs(h.get("dt", 0) - int(dt_unix)))
        else:
            record = {}

    log.info("Fetched historical weather for ts=%s", dt_unix)

    return {
        "timestamp": record.get("dt"),
        "temperature": record.get("temp"),
        "feels_like": record.get("feels_like"),
        "temp_min": None,
        "temp_max": None,
        "pressure": record.get("pressure"),
        "humidity": record.get("humidity"),
        "visibility": record.get("visibility"),
        "wind_speed": record.get("wind_speed") or (record.get("wind") or {}).get("speed"),
        "wind_deg": record.get("wind_deg") or (record.get("wind") or {}).get("deg"),
        "clouds": record.get("clouds"),
        "weather_main": (record.get("weather") or [{}])[0].get("main"),
        "weather_description": (record.get("weather") or [{}])[0].get("description"),
        "city_name": None,
        "country": None,
    }