from typing import Dict, Any, List, Optional

import pandas as pd


def _change_rate(curr: Optional[float], prev: Optional[float]) -> float:
    if curr is None or prev is None or prev == 0:
        return 0.0
    return round(((curr - prev) / prev) * 100.0, 2)


def compute_pollutant_features(raw: Optional[Dict[str, Any]] = None, historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compute pollutant-derived features from a raw pollutant record.

    This function is defensive: `raw` may be None or missing keys when upstream
    fetches returned no data. In that case we populate features with sensible
    None/zero defaults so downstream storage and ML pipelines remain stable.
    """

    features: Dict[str, Any] = {}

    if raw is None:
        raw = {}

    ts = raw.get("timestamp")
    if ts is not None:
        try:
            features["timestamp"] = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC")
        except Exception:
            features["timestamp"] = None

    for k in ("aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"):
        # ensure numeric fields are either float or None
        val = raw.get(k)
        try:
            features[k] = float(val) if val is not None else None
        except Exception:
            features[k] = None

    pm25, pm10 = features.get("pm2_5"), features.get("pm10")
    if pm10 not in (None, 0) and pm25 is not None:
        try:
            features["pm_ratio"] = round(pm25 / pm10, 3)
        except Exception:
            features["pm_ratio"] = None

    no, no2 = features.get("no"), features.get("no2")
    if no is not None and no2 is not None:
        try:
            features["nox_total"] = round(no + no2, 3)
        except Exception:
            features["nox_total"] = None

    cat_map = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    aqi = features.get("aqi")
    try:
        aqi_key = int(aqi) if aqi is not None and str(aqi).isdigit() else aqi
    except Exception:
        aqi_key = aqi
    features["aqi_category"] = cat_map.get(aqi_key, "Unknown")

    if historical_data:
        try:
            prev = historical_data[-1]
            features["aqi_change_rate"] = _change_rate(aqi, prev.get("aqi"))
            features["pm2_5_change_rate"] = _change_rate(pm25, prev.get("pm2_5"))
        except Exception:
            features["aqi_change_rate"] = 0.0
            features["pm2_5_change_rate"] = 0.0
    else:
        features["aqi_change_rate"] = 0.0
        features["pm2_5_change_rate"] = 0.0

    return features