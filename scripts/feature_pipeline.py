from __future__ import annotations

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

LOG = logging.getLogger("feature_pipeline")

# explicit exports when imported by other scripts
__all__ = ["run_for_timestamp", "main"]

try:
    from src.api_client.fetch_weather_openmeteo import fetch_openmeteo_bulk
except Exception:
    fetch_openmeteo_bulk = None

try:
    from src.api_client.weather_client import WeatherClient
except Exception:
    WeatherClient = None

try:
    from src.api_client.fetch_pollutant_historical import fetch_pollutant_historical
except Exception:
    fetch_pollutant_historical = None

try:
    from src.feature_engineering.pollutant_features import compute_pollutant_features
except Exception:
    compute_pollutant_features = None

try:
    from src.feature_store.store_manager import StoreManager  # type: ignore
except Exception:
    StoreManager = None  # type: ignore


def _append_to_local_csv(features: Dict[str, Any], csv_path: Path):
    """
    Utility to append features to a local CSV (used as fallback when Hopsworks is not requested).
    Not used when --use-hopsworks is set, but kept for safety.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([features])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        LOG.info("Created local features CSV: %s", csv_path)
        return
    try:
        df.to_csv(csv_path, mode="a", index=False, header=False)
        LOG.info("Appended row to local CSV: %s", csv_path)
    except Exception:
        LOG.exception("Failed to append to local CSV")


def run_for_timestamp(
    ts_iso: str,
    lat: float,
    lon: float,
    use_hopsworks: bool = False,
    openweather_api_key: Optional[str] = None,
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5",
    hopsworks_api_key: Optional[str] = None,
    hopsworks_project: Optional[str] = None,
    features_fg_name: str = "pollutant_features",
    weather_fg_name: str = "weather_features",
    local_csv: str = "data/features_local.csv",
) -> Dict[str, Any]:
    """
    Build features for a single timestamp and store them.

    Requirements/behavior:
    - If use_hopsworks is True (or HOPSWORKS env vars present), the function will
      push pollutant and weather rows to Hopsworks feature groups (requires StoreManager).
    - The function will also append the same row to data/ml_training_data_1year.csv to
      allow immediate retraining locally.
    - If required helper functions are missing it will raise a descriptive error.
    """

    # Basic validation & timezone normalization
    ts = pd.to_datetime(ts_iso, utc=True)
    ts_unix = int(ts.timestamp())
    date_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure compute function exists
    if compute_pollutant_features is None:
        raise RuntimeError("compute_pollutant_features is not available. Fix imports in src.feature_engineering.pollutant_features")

    # Fetch pollutant historical data for this timestamp (best-effort)
    pollutant_raw: Optional[Dict[str, Any]] = None
    try:
        if fetch_pollutant_historical is not None:
            api_key = openweather_api_key or os.getenv("OPENWEATHER_API_KEY")
            try:
                pollutant_raw = fetch_pollutant_historical(api_key, openweather_base_url, lat, lon, ts_unix)  # type: ignore
            except TypeError:
                # If signature mismatch, try alternative orders (best effort)
                try:
                    pollutant_raw = fetch_pollutant_historical(openweather_base_url, lat, lon, ts_unix)  # type: ignore
                except Exception:
                    pollutant_raw = None
            LOG.info("Fetched pollutant historical record: %s", bool(pollutant_raw))
    except Exception:
        LOG.exception("Failed to fetch pollutant historical data; proceeding with None (compute may handle)")

    # Get weather features: prefer OpenWeather via WeatherClient if API key provided, else Open-Meteo
    weather: Dict[str, Any] = {}
    open_key = openweather_api_key or os.getenv("OPENWEATHER_API_KEY")
    try:
        if open_key and WeatherClient is not None:
            try:
                wc = WeatherClient(api_key=open_key, base_url=openweather_base_url)
                weather = wc.get_weather_for_timestamp(ts_iso, lat, lon)  # type: ignore
            except Exception:
                LOG.exception("WeatherClient failed, will try Open-Meteo")
        if not weather and fetch_openmeteo_bulk is not None:
            try:
                # fetch_openmeteo_bulk expects (lat, lon, start_date, end_date)
                date_only = ts.strftime("%Y-%m-%d")
                records = fetch_openmeteo_bulk(lat, lon, date_only, date_only)  # type: ignore
                if records:
                    # pick the first (only) day's record
                    weather = records[0]
                else:
                    LOG.warning("Open-Meteo returned no records for %s", date_only)
            except Exception:
                LOG.exception("fetch_openmeteo_bulk failed; weather features empty")
    except Exception:
        LOG.exception("Weather fetch failed entirely")

    # Compute pollutant-derived features
    try:
        # compute_pollutant_features expects a raw pollutant dict first, and optional historical_data list
        if pollutant_raw:
            pollutant_feats = compute_pollutant_features(pollutant_raw, historical_data=None)  # type: ignore
        else:
            # If no pollutant record available, call with an empty dict to let implementation decide
            pollutant_feats = compute_pollutant_features({}, historical_data=None)  # type: ignore
    except Exception:
        LOG.exception("Failed computing pollutant features")
        raise

    # Merge features
    features: Dict[str, Any] = {}
    features.update({k: v for k, v in (weather or {}).items()})
    features.update({k: v for k, v in (pollutant_feats or {}).items()})
    # basic time/location fields
    features["timestamp"] = int(ts_unix)
    features["iso_date"] = date_str
    features["year"] = int(ts.year)
    features["month"] = int(ts.month)
    features["day"] = int(ts.day)
    features["hour"] = int(ts.hour)
    features["weekday"] = int(ts.weekday())
    features["lat"] = float(lat)
    features["lon"] = float(lon)

    # Require Hopsworks for storage when requested
    want_hs = use_hopsworks or (os.getenv("HOPSWORKS_API_KEY") and os.getenv("HOPSWORKS_PROJECT_NAME"))
    if not want_hs:
        raise RuntimeError("Hopsworks storage required. Call with --use-hopsworks or set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME env vars")

    api_key = hopsworks_api_key or os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks_project or os.getenv("HOPSWORKS_PROJECT_NAME")
    if not api_key or not project:
        raise RuntimeError("Hopsworks credentials missing (api key / project)")

    # Import StoreManager now and push pollutant & weather feature groups
    try:
        from src.feature_store.store_manager import StoreManager  # type: ignore
    except Exception:
        LOG.exception("Failed to import StoreManager from src.feature_store.store_manager")
        raise

    sm = StoreManager(api_key=api_key, project_name=project)

    # Split payloads by naive heuristic: pollutant fields often include 'pm2' 'pm10' 'aqi' etc.
    pollutant_keys = {k for k in features.keys() if any(s in k.lower() for s in ("pm", "aqi", "co", "no2", "no", "o3", "so2", "nh3", "nox"))}
    # Build payloads: include timestamp and location in both payloads
    pollutant_payload = {k: features[k] for k in pollutant_keys if k in features}
    weather_payload = {k: v for k, v in features.items() if k not in pollutant_keys}
    # Ensure timestamp and location are present in both payloads
    for p in (pollutant_payload, weather_payload):
        if "timestamp" in features:
            p.setdefault("timestamp", features["timestamp"])
        if "lat" in features:
            p.setdefault("lat", features["lat"])
        if "lon" in features:
            p.setdefault("lon", features["lon"])

    # Store to Hopsworks feature groups (best-effort; will raise on failure)
    sm.store_features(pollutant_payload, name=features_fg_name, version=2, primary_key=["timestamp"], event_time="timestamp")
    sm.store_features(weather_payload, name=weather_fg_name, version=2, primary_key=["timestamp"], event_time="timestamp")

    # Append to local ml training CSV to keep local training data in sync
    ml_path = Path("data/ml_training_data_1year.csv")
    df_new = pd.DataFrame([features])
    if ml_path.exists():
        try:
            df_ml = pd.read_csv(ml_path)
        except Exception:
            LOG.warning("Reading existing ml training CSV failed, recreating")
            df_ml = pd.DataFrame(columns=df_new.columns)
        df_combined = pd.concat([df_ml, df_new], ignore_index=True, sort=False)
    else:
        df_combined = df_new
    if "timestamp" in df_combined.columns:
        df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df_combined.to_csv(ml_path, index=False)
    LOG.info("Appended features to training CSV: %s", ml_path)

    return features


def main(argv=None):
    parser = argparse.ArgumentParser(description="Feature pipeline for a single timestamp")
    parser.add_argument("--ts", required=True, help="ISO timestamp (e.g., 2024-10-14T12:00:00Z)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--use-hopsworks", action="store_true")
    parser.add_argument("--local-csv", default="data/features_local.csv")
    parser.add_argument("--openweather-api-key", default=None)
    parser.add_argument("--hopsworks-api-key", default=None)
    parser.add_argument("--hopsworks-project", default=None)
    args = parser.parse_args(argv)

    features = run_for_timestamp(
        ts_iso=args.ts,
        lat=args.lat,
        lon=args.lon,
        use_hopsworks=args.use_hopsworks,
        openweather_api_key=args.openweather_api_key,
        hopsworks_api_key=args.hopsworks_api_key,
        hopsworks_project=args.hopsworks_project,
        local_csv=args.local_csv,
    )
    print("Stored features for", args.ts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
