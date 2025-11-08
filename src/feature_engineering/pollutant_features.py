from typing import Dict, Any, List, Optional
import pandas as pd


def _change_rate(curr: Optional[float], prev: Optional[float]) -> float:
    if curr is None or prev is None or prev == 0:
        return 0.0
    return round(((curr - prev) / prev) * 100.0, 2)


def compute_pollutant_features(raw: Dict[str, Any], historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    ts = raw.get("timestamp")
    if ts is not None:
        features["timestamp"] = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC")

    for k in ("aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"):
        features[k] = raw.get(k)

    pm25, pm10 = features.get("pm2_5"), features.get("pm10")
    if pm10 not in (None, 0) and pm25 is not None:
        features["pm_ratio"] = round(pm25 / pm10, 3)

    no, no2 = features.get("no"), features.get("no2")
    if no is not None and no2 is not None:
        features["nox_total"] = round(no + no2, 3)

    cat_map = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    aqi = features.get("aqi")
    features["aqi_category"] = cat_map.get(aqi, "Unknown")

    if historical_data:
        prev = historical_data[-1]
        features["aqi_change_rate"] = _change_rate(aqi, prev.get("aqi"))
        features["pm2_5_change_rate"] = _change_rate(pm25, prev.get("pm2_5"))
    else:
        features["aqi_change_rate"] = 0.0
        features["pm2_5_change_rate"] = 0.0

    return features