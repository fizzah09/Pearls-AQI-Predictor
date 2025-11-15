from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ensure repo root is importable same as feature_pipeline does
_REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# If a .env file exists at repo root, load simple KEY=VAL pairs into environment
env_path = _REPO_ROOT.joinpath(".env")
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        LOG = logging.getLogger("daily_store_features")
        LOG.debug("Failed to load .env file, continuing without it")

from scripts.feature_pipeline import run_for_timestamp  # type: ignore

LOG = logging.getLogger("daily_store_features")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def store_for_date(dt: datetime, lat: float, lon: float):
    # use midday UTC to avoid missing daily archives
    ts_iso = dt.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    LOG.info("Storing features for %s (lat=%s lon=%s)", ts_iso, lat, lon)

    # Validate Hopsworks credentials exist before attempting store
    hs_key = os.getenv("HOPSWORKS_API_KEY")
    hs_project = os.getenv("HOPSWORKS_PROJECT_NAME")
    if not hs_key or not hs_project:
        LOG.error(
            "Hopsworks credentials missing. Set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME to store to Hopsworks."
        )
        return

    # warn if OpenWeather key is missing (we can still use Open-Meteo fallback)
    if not os.getenv("OPENWEATHER_API_KEY"):
        LOG.warning("OPENWEATHER_API_KEY not set; pollutant data via OpenWeather may be unavailable. Open-Meteo weather will be used.")

    features = run_for_timestamp(
        ts_iso=ts_iso,
        lat=lat,
        lon=lon,
        use_hopsworks=True,
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        hopsworks_api_key=hs_key,
        hopsworks_project=hs_project,
    )
    LOG.info("Stored features keys: %s", list(features.keys()))


def main():
    # configure location and range via env or defaults
    lat = float(os.getenv("LOCATION_LAT", os.getenv("LAT", "24.8607")))
    lon = float(os.getenv("LOCATION_LON", os.getenv("LON", "67.0011")))
    # by default store for yesterday (so daily cron at e.g. 01:00 UTC stores previous day)
    days_back = int(os.getenv("DAYS_BACK", "1"))
    base_date = datetime.utcnow().date() - timedelta(days=1)
    for i in range(days_back):
        dt = datetime.combine(base_date - timedelta(days=i), datetime.min.time()).replace(tzinfo=timezone.utc)
        try:
            store_for_date(dt, lat, lon)
        except Exception:
            LOG.exception("Failed to store features for %s", dt.isoformat())


if __name__ == "__main__":
    main()
