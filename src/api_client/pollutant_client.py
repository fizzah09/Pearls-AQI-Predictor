import requests
from typing import Dict, Any
import logging

log = logging.getLogger(__name__)


class PollutantClient:
    """Client for OpenWeather Air Pollution API."""

    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org/data/2.5/air_pollution"):
        self.api_key = api_key
        self.base_url = base_url

    def fetch_pollutant_data(self, lat: float, lon: float) -> Dict[str, Any]:
        params = {"lat": lat, "lon": lon, "appid": self.api_key}
        resp = requests.get(self.base_url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        lst = (raw.get("list") or [{}])[0]
        comp = lst.get("components", {})
        log.info("Fetched pollutant payload")
        return {
            "timestamp": lst.get("dt"),
            "aqi": lst.get("main", {}).get("aqi"),
            "co": comp.get("co"),
            "no": comp.get("no"),
            "no2": comp.get("no2"),
            "o3": comp.get("o3"),
            "so2": comp.get("so2"),
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "nh3": comp.get("nh3"),
        }