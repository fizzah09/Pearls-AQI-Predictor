from __future__ import annotations
import argparse
import logging
import os
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from pathlib import Path

LOG = logging.getLogger("backfill_features")
run_for_timestamp = None
run_for_range = None


def backfill(start_iso: str, end_iso: str, freq: str = "daily", lat: float = 0.0, lon: float = 0.0, use_hopsworks: bool = False):
    """Backfill features.

    - If run_for_timestamp is available this behaves as before (iterates timestamps).
    - Otherwise, if a single-call range backfill (run_for_range) is found it will be called once.
    - If neither is available it falls back to per-timestamp loop but will log warnings.
    """

    if run_for_timestamp is None and run_for_range is not None:
        LOG.info("Using range backfill function instead of per-timestamp iteration")
        try:
            run_for_range(start_iso, end_iso, lat=lat, lon=lon, use_hopsworks=use_hopsworks)
            return
        except TypeError:
            try:
                run_for_range(start_iso, end_iso, freq=freq, lat=lat, lon=lon, use_hopsworks=use_hopsworks)
                return
            except Exception:
                LOG.exception("Range backfill function failed; falling back to per-timestamp loop")

    start = date_parser.isoparse(start_iso)
    end = date_parser.isoparse(end_iso)
    cur = start
    delta = timedelta(days=1) if freq == "daily" else timedelta(hours=1)

    while cur <= end:
        ts_iso = cur.isoformat()
        LOG.info("Backfilling features for %s", ts_iso)
        try:
            if run_for_timestamp is None:
                raise RuntimeError("No run_for_timestamp available to handle single timestamp")
            run_for_timestamp(ts_iso, lat=lat, lon=lon, use_hopsworks=use_hopsworks)
        except Exception:
            LOG.exception("Failed for %s", ts_iso)
        cur = cur + delta


def main(argv=None):
    parser = argparse.ArgumentParser("Backfill feature store")
    parser.add_argument("--start", help="Start ISO date e.g. 2024-01-01T00:00:00Z")
    parser.add_argument("--end", help="End ISO date")
    parser.add_argument("--hours", type=int, help="Backfill the last N hours (alternative to --start/--end)")
    parser.add_argument("--freq", choices=["daily", "hourly"], default="daily")

    lat_env = os.getenv("LAT")
    lon_env = os.getenv("LON")
    parser.add_argument("--lat", type=float, default=float(lat_env) if lat_env else 24.86, help="Latitude (or set LAT in env/.env")
    parser.add_argument("--lon", type=float, default=float(lon_env) if lon_env else 67.00, help="Longitude (or set LON in env/.env")
    parser.add_argument("--use-hopsworks", action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    # If --hours is provided, compute start/end based on current UTC time
    if args.hours is not None:
        now = datetime.utcnow()
        if args.freq == "daily":
            # Interpret hours as days when freq=daily: backfill last N days
            end_dt = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(days=(args.hours - 1))
        else:
            # hourly: backfill last N hours up to the previous completed hour
            end_dt = (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(hours=(args.hours - 1))

        start_iso = start_dt.replace(tzinfo=None).isoformat() + "Z"
        end_iso = end_dt.replace(tzinfo=None).isoformat() + "Z"
        LOG.info("Computed start=%s end=%s from --hours=%s freq=%s", start_iso, end_iso, args.hours, args.freq)
        backfill(start_iso, end_iso, args.freq, lat=args.lat, lon=args.lon, use_hopsworks=args.use_hopsworks)
        return

    if not args.start or not args.end:
        parser.error("You must provide either --hours or both --start and --end")

    backfill(args.start, args.end, args.freq, lat=args.lat, lon=args.lon, use_hopsworks=args.use_hopsworks)


if __name__ == "__main__":
    main()



