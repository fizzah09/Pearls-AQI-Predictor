from typing import Dict, Any
import math


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def clean_weather_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    h = out.get("humidity")
    if h is not None:
        try:
            out["humidity"] = max(0, min(100, float(h)))
        except Exception:
            out["humidity"] = None

    vis = out.get("visibility")
    if vis is not None:
        try:
            out["visibility"] = max(0.0, float(vis))
        except Exception:
            out["visibility"] = None

    for k, v in list(out.items()):
        if _is_nan(v):
            out[k] = None
    return out


def clean_pollutant_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    
    for k, v in list(out.items()):
        if _is_nan(v):
            out[k] = None
    
    for k in ("co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"):
        v = out.get(k)
        if v is not None:
            try:
                out[k] = max(0.0, float(v))
            except Exception:
                out[k] = None

    aqi = out.get("aqi")
    if aqi is not None:
        try:
            aqi_i = int(aqi)
            out["aqi"] = aqi_i if 1 <= aqi_i <= 5 else None
        except Exception:
            out["aqi"] = None

    return out
