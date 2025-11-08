import requests
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


class WeatherClient:
    """Client for OpenWeather current weather API."""

    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url

    def fetch_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch current weather for given coordinates.

        Returns a simplified dict we use downstream.
        """
        url = f"{self.base_url}/weather"
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        log.info("Fetched weather: %s", raw.get("name"))

        return {
            "timestamp": raw.get("dt"),
            "temperature": raw.get("main", {}).get("temp"),
            "feels_like": raw.get("main", {}).get("feels_like"),
            "temp_min": raw.get("main", {}).get("temp_min"),
            "temp_max": raw.get("main", {}).get("temp_max"),
            "pressure": raw.get("main", {}).get("pressure"),
            "humidity": raw.get("main", {}).get("humidity"),
            "visibility": raw.get("visibility"),
            "wind_speed": raw.get("wind", {}).get("speed"),
            "wind_deg": raw.get("wind", {}).get("deg"),
            "clouds": raw.get("clouds", {}).get("all"),
            "weather_main": (raw.get("weather") or [{}])[0].get("main"),
            "weather_description": (raw.get("weather") or [{}])[0].get("description"),
            "city_name": raw.get("name"),
            "country": raw.get("sys", {}).get("country"),
        }