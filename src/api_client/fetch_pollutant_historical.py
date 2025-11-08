from typing import Dict, Any, Optional
import logging
import requests

log = logging.getLogger(__name__)


def fetch_pollutant_historical(
    api_key: str,
    base_url: str,
    lat: float,
    lon: float,
    dt_unix: int,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Fetch historical air pollution for a unix timestamp.

    OpenWeather air pollution endpoint supports a history path at /air_pollution/history
    with start and end unix parameters. We query a narrow window [dt-3600, dt+3600]
    and pick the record closest to dt.
    """
    sess = session or requests
    if not base_url.endswith("/air_pollution"):
        base_url = base_url.rstrip("/") + "/air_pollution"
    url = f"{base_url}/history"
    window = 3600
    params = {
        "lat": lat,
        "lon": lon,
        "start": int(dt_unix) - window,
        "end": int(dt_unix) + window,
        "appid": api_key,
    }
    resp = sess.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    items = raw.get("list") or []
    if not items:
        log.warning("No pollutant history returned for ts=%s", dt_unix)
        lst = {}
        comp = {}
    else:
        lst = min(items, key=lambda x: abs(x.get("dt", 0) - int(dt_unix)))
        comp = lst.get("components", {})

    log.info("Fetched historical pollutant for ts=%s", dt_unix)
    return {
        "timestamp": lst.get("dt"),
        "aqi": (lst.get("main") or {}).get("aqi"),
        "co": comp.get("co"),
        "no": comp.get("no"),
        "no2": comp.get("no2"),
        "o3": comp.get("o3"),
        "so2": comp.get("so2"),
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "nh3": comp.get("nh3"),
    }