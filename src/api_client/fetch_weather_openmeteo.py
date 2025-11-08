"""
Fetch historical weather data from Open-Meteo (FREE, no API key needed).
Open-Meteo provides weather archive data from 1940 onwards.
"""
from typing import Dict, Any, List
import logging
import requests
import pandas as pd

log = logging.getLogger(__name__)


def fetch_openmeteo_bulk(lat: float, lon: float, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch bulk historical weather data from Open-Meteo for date range.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        List of daily weather records matching your feature schema
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_mean",
            "pressure_msl_mean",
            "relative_humidity_2m_mean",
            "windspeed_10m_max",
            "winddirection_10m_dominant",
            "cloudcover_mean",
            "precipitation_sum",
        ]),
        "timezone": "UTC"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        
        records = []
        for i, date_str in enumerate(dates):
            dt = pd.to_datetime(date_str + " 12:00:00", utc=True)
            
            record = {
                "timestamp": int(dt.timestamp()),
                "temperature": daily.get("temperature_2m_mean", [None])[i],
                "feels_like": daily.get("apparent_temperature_mean", [None])[i],
                "temp_min": daily.get("temperature_2m_min", [None])[i],
                "temp_max": daily.get("temperature_2m_max", [None])[i],
                "pressure": daily.get("pressure_msl_mean", [None])[i],
                "humidity": daily.get("relative_humidity_2m_mean", [None])[i],
                "visibility": 10000,
                "wind_speed": daily.get("windspeed_10m_max", [None])[i],
                "wind_deg": daily.get("winddirection_10m_dominant", [None])[i],
                "clouds": daily.get("cloudcover_mean", [None])[i],
                "weather_main": _get_weather_condition(
                    daily.get("precipitation_sum", [None])[i],
                    daily.get("cloudcover_mean", [None])[i]
                ),
                "weather_description": "open-meteo historical data",
                "city_name": None,
                "country": None,
            }
            records.append(record)
        
        log.info("Fetched %d days of weather data from Open-Meteo", len(records))
        return records
        
    except Exception as e:
        log.error("Open-Meteo fetch failed: %s", e)
        raise


def _get_weather_condition(precip: float, clouds: float) -> str:
    """Simple weather condition inference from precipitation and clouds."""
    if precip is None or clouds is None:
        return "Clear"
    if precip > 0.5:
        return "Rain"
    if clouds > 70:
        return "Clouds"
    return "Clear"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    records = fetch_openmeteo_bulk(24.8607, 67.0011, "2024-10-14", "2024-10-20")
    print(f"Got {len(records)} records")
    for r in records[:3]:
        print(r)
