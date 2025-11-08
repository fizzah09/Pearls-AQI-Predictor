from __future__ import annotations
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import time

LOG = logging.getLogger("feature_pipeline")

try:
    from src.api_client.fetch_weather_openmeteo import fetch_openmeteo_bulk
    from src.api_client.fetch_pollutant_historical import fetch_pollutant_historical
    from src.feature_engineering.pollutant_features import compute_pollutant_features
    from src.feature_store.store_manager import StoreManager
except Exception:
    # Allow imports to fail in environments where package paths are not configured
    # We'll raise if these are actually needed at runtime.
    fetch_openmeteo_bulk = None  # type: ignore
    fetch_pollutant_historical = None  # type: ignore
    compute_pollutant_features = None  # type: ignore
    StoreManager = None  # type: ignore


def _append_to_local_csv(features: Dict[str, Any], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([features])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        LOG.info("Created local features CSV: %s", csv_path)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)
        LOG.info("Appended row to local features CSV: %s", csv_path)


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
):
    """Run feature extraction for a single timestamp.

    Args:
        ts_iso: ISO timestamp string (e.g., 2024-10-14T12:00:00Z)
        lat, lon: location
        use_hopsworks: if True attempt to store to Hopsworks, otherwise write local csv
    Returns:
        features dict
    """
    if compute_pollutant_features is None:
        raise RuntimeError("Feature compute/imports not available in this environment")

    ts = pd.to_datetime(ts_iso, utc=True)
    ts_unix = int(ts.timestamp())

    # 1) Fetch pollutant data (closest to timestamp)
    poll = fetch_pollutant_historical(
        api_key=openweather_api_key or os.getenv("OPENWEATHER_API_KEY"),
        base_url=openweather_base_url,
        lat=lat,
        lon=lon,
        dt_unix=ts_unix,
    )

    # 2) Fetch weather data for that date (Open-Meteo daily archive, use same day)
    date_str = ts.strftime("%Y-%m-%d")
    try:
        weather_list = fetch_openmeteo_bulk(lat=lat, lon=lon, start_date=date_str, end_date=date_str)
        weather = weather_list[0] if weather_list else {}
    except Exception:
        LOG.exception("Weather fetch failed, continuing with empty weather")
        weather = {}

    # Combine raw records into a single raw dict (pollutant fields + weather)
    raw = {**poll, **weather}

    # Load recent historical local file if exists to compute rolling stats
    hist = []
    local_path = Path(local_csv)
    if local_path.exists():
        try:
            df_hist = pd.read_csv(local_path)
            if 'timestamp' in df_hist.columns:
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True, errors='coerce')
                df_hist = df_hist[df_hist['timestamp'] < ts]
                # keep last 7 days
                df_hist = df_hist.sort_values('timestamp').tail(7 * 24)
                hist = df_hist.to_dict('records')
        except Exception:
            LOG.exception("Failed to read local history for rolling features")

    # Compute pollutant features (includes change rates if historical provided)
    features = compute_pollutant_features(raw, historical_data=hist)

    # Add time features
    features['timestamp'] = int(ts_unix)
    features['year'] = int(ts.year)
    features['month'] = int(ts.month)
    features['day'] = int(ts.day)
    features['hour'] = int(ts.hour)
    features['weekday'] = int(ts.weekday())
    features['lat'] = float(lat)
    features['lon'] = float(lon)

    # Simple rolling aggregates from hist (local fallback)
    try:
        if hist:
            dfh = pd.DataFrame(hist)
            if 'pm2_5' in dfh.columns:
                features['pm2_5_7d_mean'] = float(dfh['pm2_5'].astype(float).mean())
            if 'pm10' in dfh.columns:
                features['pm10_7d_mean'] = float(dfh['pm10'].astype(float).mean())
    except Exception:
        LOG.exception("Failed computing rolling aggregates")

    # Store features
    if use_hopsworks or (os.getenv('HOPSWORKS_API_KEY') and os.getenv('HOPSWORKS_PROJECT_NAME')):
        api_key = hopsworks_api_key or os.getenv('HOPSWORKS_API_KEY')
        project = hopsworks_project or os.getenv('HOPSWORKS_PROJECT_NAME')
        try:
            sm = StoreManager(api_key=api_key, project_name=project)
            sm.store_features(features=features, name=features_fg_name, version=2, primary_key=['timestamp'], event_time='timestamp')
            LOG.info("Stored features to Hopsworks feature group %s", features_fg_name)
        except Exception:
            LOG.exception("Failed to store to Hopsworks, writing local CSV instead")
            _append_to_local_csv(features, local_path)
    else:
        _append_to_local_csv(features, local_path)

    return features


def main(argv=None):
    parser = argparse.ArgumentParser(description="Feature pipeline for a single timestamp")
    parser.add_argument("--ts", required=True, help="ISO timestamp (e.g., 2024-10-14T12:00:00Z)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--use-hopsworks", action='store_true')
    parser.add_argument("--local-csv", default="data/features_local.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    LOG.info("Running feature pipeline ts=%s lat=%s lon=%s", args.ts, args.lat, args.lon)

    features = run_for_timestamp(
        ts_iso=args.ts,
        lat=args.lat,
        lon=args.lon,
        use_hopsworks=args.use_hopsworks,
        local_csv=args.local_csv,
    )

    LOG.info("Feature extraction complete. Keys: %s", list(features.keys()))


if __name__ == "__main__":
    main()
from typing import Optional, Dict, Any
import pandas as pd
import time

LOG = logging.getLogger("feature_pipeline")

try:
    from src.api_client.fetch_weather_openmeteo import fetch_openmeteo_bulk
    from src.api_client.fetch_pollutant_historical import fetch_pollutant_historical
    from src.feature_engineering.pollutant_features import compute_pollutant_features
    from src.feature_store.store_manager import StoreManager
except Exception:
    # Allow imports to fail in environments where package paths are not configured
    # We'll raise if these are actually needed at runtime.
    fetch_openmeteo_bulk = None  # type: ignore
    fetch_pollutant_historical = None  # type: ignore
    compute_pollutant_features = None  # type: ignore
    StoreManager = None  # type: ignore


def _append_to_local_csv(features: Dict[str, Any], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([features])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        LOG.info("Created local features CSV: %s", csv_path)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)
        LOG.info("Appended row to local features CSV: %s", csv_path)


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
):
    """Run feature extraction for a single timestamp.

    Args:
        ts_iso: ISO timestamp string (e.g., 2024-10-14T12:00:00Z)
        lat, lon: location
        use_hopsworks: if True attempt to store to Hopsworks, otherwise write local csv
    Returns:
        features dict
    """
    if compute_pollutant_features is None:
        raise RuntimeError("Feature compute/imports not available in this environment")

    ts = pd.to_datetime(ts_iso, utc=True)
    ts_unix = int(ts.timestamp())

    # 1) Fetch pollutant data (closest to timestamp)
    poll = fetch_pollutant_historical(
        api_key=openweather_api_key or os.getenv("OPENWEATHER_API_KEY"),
        base_url=openweather_base_url,
        lat=lat,
        lon=lon,
        dt_unix=ts_unix,
    )

    # 2) Fetch weather data for that date (Open-Meteo daily archive, use same day)
    date_str = ts.strftime("%Y-%m-%d")
    try:
        weather_list = fetch_openmeteo_bulk(lat=lat, lon=lon, start_date=date_str, end_date=date_str)
        weather = weather_list[0] if weather_list else {}
    except Exception:
        LOG.exception("Weather fetch failed, continuing with empty weather")
        weather = {}

    # Combine raw records into a single raw dict (pollutant fields + weather)
    raw = {**poll, **weather}

    # Load recent historical local file if exists to compute rolling stats
    hist = []
    local_path = Path(local_csv)
    if local_path.exists():
        try:
            df_hist = pd.read_csv(local_path)
            if 'timestamp' in df_hist.columns:
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True, errors='coerce')
                df_hist = df_hist[df_hist['timestamp'] < ts]
                # keep last 7 days
                df_hist = df_hist.sort_values('timestamp').tail(7 * 24)
                hist = df_hist.to_dict('records')
        except Exception:
            LOG.exception("Failed to read local history for rolling features")

    # Compute pollutant features (includes change rates if historical provided)
    features = compute_pollutant_features(raw, historical_data=hist)

    # Add time features
    features['timestamp'] = int(ts_unix)
    features['year'] = int(ts.year)
    features['month'] = int(ts.month)
    features['day'] = int(ts.day)
    features['hour'] = int(ts.hour)
    features['weekday'] = int(ts.weekday())
    features['lat'] = float(lat)
    features['lon'] = float(lon)

    # Simple rolling aggregates from hist (local fallback)
    try:
        if hist:
            dfh = pd.DataFrame(hist)
            if 'pm2_5' in dfh.columns:
                features['pm2_5_7d_mean'] = float(dfh['pm2_5'].astype(float).mean())
            if 'pm10' in dfh.columns:
                features['pm10_7d_mean'] = float(dfh['pm10'].astype(float).mean())
    except Exception:
        LOG.exception("Failed computing rolling aggregates")

    # Store features
    if use_hopsworks or (os.getenv('HOPSWORKS_API_KEY') and os.getenv('HOPSWORKS_PROJECT_NAME')):
        api_key = hopsworks_api_key or os.getenv('HOPSWORKS_API_KEY')
        project = hopsworks_project or os.getenv('HOPSWORKS_PROJECT_NAME')
        try:
            sm = StoreManager(api_key=api_key, project_name=project)
            sm.store_features(features=features, name=features_fg_name, version=2, primary_key=['timestamp'], event_time='timestamp')
            LOG.info("Stored features to Hopsworks feature group %s", features_fg_name)
        except Exception:
            LOG.exception("Failed to store to Hopsworks, writing local CSV instead")
            _append_to_local_csv(features, local_path)
    else:
        _append_to_local_csv(features, local_path)

    return features


def main(argv=None):
    parser = argparse.ArgumentParser(description="Feature pipeline for a single timestamp")
    parser.add_argument("--ts", required=True, help="ISO timestamp (e.g., 2024-10-14T12:00:00Z)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--use-hopsworks", action='store_true')
    parser.add_argument("--local-csv", default="data/features_local.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    LOG.info("Running feature pipeline ts=%s lat=%s lon=%s", args.ts, args.lat, args.lon)

    features = run_for_timestamp(
        ts_iso=args.ts,
        lat=args.lat,
        lon=args.lon,
        use_hopsworks=args.use_hopsworks,
        local_csv=args.local_csv,
    )

    LOG.info("Feature extraction complete. Keys: %s", list(features.keys()))


if __name__ == "__main__":
    main()
