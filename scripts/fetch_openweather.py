                      
"""
Fetch current OpenWeather data and append a single row to data/openweather_daily.csv

This is the standalone version of the inline script used in the GitHub Actions workflow.

Usage examples:
  # Use env vars stored in the GitHub Secrets or local environment
  OPENWEATHER_API_KEY=xxx python scripts/fetch_openweather.py

  # Override latitude/longitude on the command line
  python scripts/fetch_openweather.py --lat 51.5074 --lon -0.1278

Environment variables:
  OPENWEATHER_API_KEY (required)
  OPENWEATHER_LAT (optional)
  OPENWEATHER_LON (optional)

Outputs:
  Appends a CSV row to data/openweather_daily.csv (creates the file with header if missing).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import sys
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


import requests

LOG = logging.getLogger("fetch_openweather")


def get_current_weather(api_key: str, lat: str, lon: str, timeout: int = 30) -> Dict:
    """Call OpenWeather One Call API and return the 'current' block as a dict."""
                                                                      
    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "minutely,hourly,daily,alerts",
        "appid": api_key,
        "units": "metric",
    }
    try:
        LOG.debug("Requesting OpenWeather One Call (v2.5) current data for %s,%s", lat, lon)
        resp = requests.get("https://api.openweathermap.org/data/2.5/onecall", params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("current", {})
    except requests.exceptions.HTTPError as http_err:
                                                                                                                        
        status = getattr(http_err.response, "status_code", None)
        LOG.warning("One Call request failed with status %s - falling back to Current Weather endpoint", status)
    except Exception:
        LOG.exception("One Call request failed - falling back to Current Weather endpoint")

                                                                           
    try:
        LOG.debug("Requesting OpenWeather Current Weather for %s,%s", lat, lon)
        params_simple = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
        resp = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params_simple, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

                                                                                          
        now = {}
        now["dt"] = payload.get("dt")
        main = payload.get("main", {})
        now["temp"] = main.get("temp")
        now["feels_like"] = main.get("feels_like")
        now["pressure"] = main.get("pressure")
        now["humidity"] = main.get("humidity")
        now["dew_point"] = None
        now["uvi"] = None
        now["clouds"] = payload.get("clouds", {}).get("all")
        now["visibility"] = payload.get("visibility")
        wind = payload.get("wind", {})
        now["wind_speed"] = wind.get("speed")
        now["wind_deg"] = wind.get("deg")
        now["weather"] = payload.get("weather")
        return now
    except Exception:
        LOG.exception("Fallback Current Weather request failed")
        raise


def build_row(now: Dict, lat: str, lon: str) -> Dict[str, Optional[object]]:
    ts = now.get("dt")
    if ts:
        ts = datetime.datetime.utcfromtimestamp(ts).isoformat()
    else:
        ts = datetime.datetime.utcnow().isoformat()

    row = {
        "timestamp": ts,
        "lat": lat,
        "lon": lon,
        "temp_c": now.get("temp"),
        "feels_like_c": now.get("feels_like"),
        "pressure_hpa": now.get("pressure"),
        "humidity_pct": now.get("humidity"),
        "dew_point_c": now.get("dew_point"),
        "uvi": now.get("uvi"),
        "clouds_pct": now.get("clouds"),
        "visibility_m": now.get("visibility"),
        "wind_speed_m_s": now.get("wind_speed"),
        "wind_deg": now.get("wind_deg"),
        "weather_main": (now.get("weather") or [{}])[0].get("main", ""),
        "weather_desc": (now.get("weather") or [{}])[0].get("description", ""),
    }
    return row


def append_row_to_csv(csv_path: str, row: Dict[str, Optional[object]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    existed = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not existed:
            writer.writeheader()
        writer.writerow(row)
    LOG.info("Appended OpenWeather row to %s", csv_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch OpenWeather and append to CSV")
    parser.add_argument("--lat", type=str, help="Latitude (overrides OPENWEATHER_LAT)")
    parser.add_argument("--lon", type=str, help="Longitude (overrides OPENWEATHER_LON)")
    parser.add_argument("--csv", type=str, default="data/openweather_daily.csv", help="CSV path to append")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        LOG.error("OPENWEATHER_API_KEY is not set in environment")
        return 2

    lat = args.lat or os.getenv("OPENWEATHER_LAT") or "0"
    lon = args.lon or os.getenv("OPENWEATHER_LON") or "0"

    try:
        now = get_current_weather(api_key=api_key, lat=lat, lon=lon)
    except Exception as exc:                                      
        LOG.exception("Failed to fetch OpenWeather data: %s", exc)
        return 3

    row = build_row(now, lat, lon)

    try:
        append_row_to_csv(args.csv, row)
    except Exception as exc:
        LOG.exception("Failed to append row to CSV: %s", exc)
        return 4

                                                                 
    LOG.debug("Appended row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
