                      
"""
Predict AQI for the next N days using Open-Meteo daily forecast + latest local pollutant persistence.
Save results to CSV for the dashboard to consume.
"""
from pathlib import Path
import argparse
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
import sys
import os
import joblib
try:
                                                            
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

                                                      
parser = argparse.ArgumentParser("perdict 3_days.py")
parser.add_argument("--lat", type=float, default=os.getenv("LAT"), required=(os.getenv("LAT") is None), help="Latitude (or set LAT in env/.env)")
parser.add_argument("--lon", type=float, default=os.getenv("LON"), required=(os.getenv("LON") is None), help="Longitude (or set LON in env/.env)")
parser.add_argument("--days", type=int, default=3)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--out", type=str, default="data/predictions_3day_openweather.csv")
parser.add_argument("--blend", action="store_true", help="Enable blending of model prediction with last observed AQI (persistence)")
parser.add_argument("--blend-weight", type=float, default=0.8, help="Weight for model prediction when blending (0-1). persistence weight = 1 - this")
parser.add_argument("--blend-threshold", type=float, default=20.0, help="Minimum absolute difference (AQI points) between model and persistence to apply blending")
parser.add_argument("--use-live", action="store_true", help="If set and SENSOR_API_URL is configured, prefer live sensor API for persistence AQI")
parser.add_argument("--nn-k", type=int, default=5, help="Number of nearest neighbors to consider for near-extreme check")
parser.add_argument("--nn-extreme-threshold", type=float, default=200.0, help="AQI threshold to label a neighbor as extreme")
parser.add_argument("--persistence-value", type=float, default=None, help="Override the last-observed persistence AQI value (for testing)")
args = parser.parse_args()

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from inference.predictor import AQIInferenceEngine, categorize_aqi

LOG = logging.getLogger("predict_3day_openmeteo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_latest_model_from_models_dir(models_dir: Path = None):
    models_dir = Path(models_dir or (REPO_ROOT / "modeling" / "models"))
    pkls = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not pkls:
        raise FileNotFoundError(f"No .pkl models found in {models_dir}")
    chosen = pkls[-1]
    print("Using model:", chosen)
    m = joblib.load(chosen)
                                                                                   
    if isinstance(m, dict):
        return chosen, m
    return chosen, {"model": m}

def find_latest_model(models_dir: Path) -> Path:
    models = list(models_dir.glob("*_xgboost.pkl")) + list(models_dir.glob("*.pkl"))
    if not models:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    return max(models, key=lambda p: p.stat().st_mtime)

def fetch_openmeteo_daily(lat: float, lon: float, days: int):
    url = "https://api.open-meteo.com/v1/forecast"
    start = date.today()
    end = start + timedelta(days=days-1)

                                                                       
    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "windspeed_10m_max",
        "winddirection_10m_dominant",
        "cloudcover_mean",
        "weathercode"
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(daily_vars),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC"
    }

    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        LOG.error("Open-Meteo request failed: %s", e)
        raise

                                                           
    if r.status_code != 200:
        LOG.error("Open-Meteo HTTP %s: %s", r.status_code, r.text)
    r.raise_for_status()

    data = r.json().get("daily", {})
    records = []
    times = data.get("time", [])
    for i, ts_date in enumerate(times):
                                    
        dt = datetime.fromisoformat(ts_date + "T12:00:00+00:00")
        rec = {
            "timestamp": int(dt.timestamp()),
            "temperature": None,
            "temp_min": data.get("temperature_2m_min", [None])[i],
            "temp_max": data.get("temperature_2m_max", [None])[i],
            "feels_like": None,
            "pressure": None,
            "humidity": None,
            "wind_speed": data.get("windspeed_10m_max", [None])[i],
            "wind_deg": data.get("winddirection_10m_dominant", [None])[i],
            "clouds": data.get("cloudcover_mean", [None])[i],
            "precipitation": data.get("precipitation_sum", [0])[i],
            "weather_main": f"code_{data.get('weathercode', [None])[i]}",
            "weather_description": "open-meteo forecast",
            "visibility": None
        }
                                                                      
        try:
            tmin = rec["temp_min"]
            tmax = rec["temp_max"]
            rec["temperature"] = None if (tmin is None or tmax is None) else (tmin + tmax) / 2.0
        except Exception:
            rec["temperature"] = rec["temp_max"] or rec["temp_min"]
        records.append(rec)
    return records

def load_latest_pollutant(local_paths):
    for p in local_paths:
        p = Path(p)
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                if len(df) == 0:
                    continue
                last = df.iloc[-1].to_dict()
                return last
            except Exception as e:
                LOG.warning("Failed to read %s: %s", p, e)
    LOG.warning("No local pollutant CSV found; pollutant fields will be empty")
    return {}


def fetch_live_aqi():
    """Simple adapter to fetch last observed AQI from a configured sensor/API.

    Expects environment variable SENSOR_API_URL and optional SENSOR_API_KEY.
    The API is expected to return JSON with an 'aqi' or 'value' numeric field. Returns None on failure.
    """
    url = os.getenv('SENSOR_API_URL')
    if not url:
        return None, None
    headers = {}
    key = os.getenv('SENSOR_API_KEY')
    if key:
        headers['Authorization'] = f"Bearer {key}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict):
            if 'aqi' in j:
                return float(j['aqi']), url
            if 'value' in j:
                return float(j['value']), url
                                                             
            if 'data' in j and isinstance(j['data'], dict):
                d = j['data']
                if 'aqi' in d:
                    return float(d['aqi']), url
                if 'value' in d:
                    return float(d['value']), url
        LOG.warning('Live sensor API did not return expected AQI field (aqi/value)')
    except Exception as e:
        LOG.warning('Failed to fetch live sensor AQI from %s: %s', url, e)
    return None, url

def build_feature_row(weather_rec, pollutant_last, extra_lat=None, extra_lon=None):
    ts = weather_rec.get("timestamp")
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    row = {
        "timestamp": int(ts),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "weekday": dt.weekday(),
        "temperature": weather_rec.get("temperature"),
        "temp_min": weather_rec.get("temp_min"),
        "temp_max": weather_rec.get("temp_max"),
        "feels_like": weather_rec.get("feels_like"),
        "pressure": weather_rec.get("pressure"),
        "humidity": weather_rec.get("humidity"),
        "wind_speed": weather_rec.get("wind_speed"),
        "wind_deg": weather_rec.get("wind_deg"),
        "clouds": weather_rec.get("clouds"),
        "precipitation": weather_rec.get("precipitation"),
        "weather_main": weather_rec.get("weather_main") or "Unknown",
        "weather_description": weather_rec.get("weather_description") or "",
        "lat": extra_lat,
        "lon": extra_lon
    }
                                                                           
                                                                           
    for k in ("aqi","co","no","no2","o3","so2","pm2_5","pm10","nh3"):
        pref = f"pollutant_{k}" if k != "aqi" else "pollutant_aqi"
        row[pref] = pollutant_last.get(k)

                                                                          
    row.setdefault("pollutant_pm2_5_7d_mean", pollutant_last.get("pm2_5_7d_mean"))
    row.setdefault("pollutant_pm10_7d_mean", pollutant_last.get("pm10_7d_mean"))
                                                        
    row.setdefault("pollutant_pm_ratio", pollutant_last.get("pm_ratio"))
    row.setdefault("pollutant_nox_total", pollutant_last.get("nox_total"))
    row.setdefault("pollutant_aqi_change_rate", pollutant_last.get("aqi_change_rate"))
    row.setdefault("pollutant_pm2_5_change_rate", pollutant_last.get("pm2_5_change_rate"))
    return row

def align_features_to_model(row_dict, feature_names):
    out = {}
    for fn in feature_names:
        if fn in row_dict:
            out[fn] = row_dict[fn]
        else:
            if fn.startswith("weather_main_"):
                cat = fn.split("weather_main_",1)[1]
                out[fn] = 1 if str(row_dict.get("weather_main","")).lower() == cat.lower() else 0
            else:
                out[fn] = 0
    return pd.DataFrame([out], columns=feature_names)

def main():
    models_dir = Path("modeling/models")
                                                                      
    engine = None
    if args.model:
        model_path = Path(args.model)
        LOG.info("Using model: %s", model_path)
        engine = AQIInferenceEngine(str(model_path))
    else:
                                                                                          
        preferred_candidates = list(models_dir.glob("*pollutant*aqui*")) + list(models_dir.glob("*pollutant*aqi*"))
                                            
        xgb_candidate = models_dir / "pollutant_aqi_regression_xgboost.pkl"
        chosen = None
        loaded = None
        if xgb_candidate.exists():
            try:
                chosen = xgb_candidate
                loaded = joblib.load(str(chosen))
                LOG.info("Using preferred model: %s", chosen)
            except Exception:
                chosen = None
                loaded = None
        if chosen is None:
            try:
                chosen, loaded = load_latest_model_from_models_dir(models_dir)
                LOG.info("Using model: %s", chosen)
            except Exception as e:
                LOG.warning("Failed to load latest model: %s", e)
                chosen = find_latest_model(models_dir)
                loaded = joblib.load(str(chosen))
                LOG.info("Falling back to model: %s", chosen)

                                                                       
        if isinstance(loaded, dict):
            engine = AQIInferenceEngine()
            engine.model = loaded.get("model")
            engine.scaler = loaded.get("scaler")
            engine.feature_names = loaded.get("feature_names")
            engine.task = loaded.get("task", "regression")
            LOG.info("Loaded model from dictionary format: %s", chosen)
        else:
            engine = AQIInferenceEngine(str(chosen))

    weather_recs = fetch_openmeteo_daily(args.lat, args.lon, args.days)
    pollutant_last = load_latest_pollutant(["data/features_local.csv", "data/ml_training_data_1year.csv"])

                                                                                       
    persistence_aqi = None
    persistence_source = None

                                          
    if args.use_live or os.getenv('USE_LIVE_SENSOR') == '1':
        live_aqi, live_url = fetch_live_aqi()
        if live_aqi is not None:
            persistence_aqi = live_aqi
            persistence_source = live_url or 'live_sensor'
            LOG.info('Using live sensor AQI from %s = %s', persistence_source, persistence_aqi)

                                                                 
    if persistence_aqi is None:
        try:
            p = Path("data/ml_training_data_1year.csv")
            if p.exists():
                import pandas as _pd
                df = _pd.read_csv(p, low_memory=False)
                if 'pollutant_aqi' in df.columns:
                    s = df['pollutant_aqi'].dropna()
                    if len(s) > 0:
                        persistence_aqi = float(s.iloc[-1])
                        persistence_source = str(p)
        except Exception:
            persistence_aqi = None
            persistence_source = None

    if persistence_aqi is None and pollutant_last:
        persistence_aqi = pollutant_last.get('pollutant_aqi') or pollutant_last.get('aqi')
        persistence_source = persistence_source or 'features_local'

    if persistence_aqi is not None:
        LOG.info("Using persistence AQI from %s = %s", persistence_source, persistence_aqi)
    else:
        LOG.info("No persistence AQI found; blending will be disabled if requested")

                                                                            
    if getattr(args, 'persistence_value', None) is not None:
        try:
            persistence_aqi = float(args.persistence_value)
            persistence_source = 'cli_override'
            LOG.info('Overriding persistence AQI from CLI: %s', persistence_aqi)
        except Exception:
            LOG.warning('Invalid persistence_value provided; ignoring override')

    results = []
    aligned_rows = []
    raw_feature_rows = []
    nn_flags = []
    nn_distances = []

                                                                                            
    train_X_scaled = None
    train_y_vals = None
    if engine and getattr(engine, 'scaler', None) and getattr(engine, 'feature_names', None):
        train_csv = Path('data/ml_training_data_1year.csv')
        if train_csv.exists():
            try:
                from modeling.data_loader import load_training_data, prepare_features_targets
                train_df = load_training_data(str(train_csv))
                X_train, y_train = prepare_features_targets(train_df, target_col='pollutant_aqi')
                                                                                                 
                for fn in engine.feature_names:
                    if fn not in X_train.columns:
                        X_train[fn] = 0
                X_train = X_train.reindex(columns=engine.feature_names, fill_value=0)
                                                                                            
                X_train = X_train.fillna(0).infer_objects(copy=False)
                            
                train_X_scaled = engine.scaler.transform(X_train)
                train_y_vals = y_train.values
                LOG.info('Prepared training matrix for NN checks: %d rows, %d cols', train_X_scaled.shape[0], train_X_scaled.shape[1])
            except Exception:
                LOG.warning('Failed to prepare training NN matrix; NN checks will be skipped', exc_info=True)
    for rec in weather_recs:
        feat = build_feature_row(rec, pollutant_last, extra_lat=args.lat, extra_lon=args.lon)
        raw_feature_rows.append(feat.copy())
        if engine.feature_names:
            X = align_features_to_model(feat, engine.feature_names)
            aligned_rows.append(X.iloc[0].to_dict())
        else:
            X = pd.DataFrame([feat])
            aligned_rows.append(X.iloc[0].to_dict())
                                                                        
        X = X.fillna(0).infer_objects(copy=False)
                                      
        try:
            LOG.debug("Engine feature_names: %s", getattr(engine, 'feature_names', None))
            LOG.debug("Input X columns: %s", list(X.columns))
        except Exception:
            pass
        preds = engine.predict(X)
        pred = float(preds[0])
                                                                                   
        near_extreme = False
        nn_min_dist = None
        try:
            if train_X_scaled is not None and train_y_vals is not None:
                                     
                x_in = engine.scaler.transform(X)
                                                                                             
                dists = np.linalg.norm(train_X_scaled - x_in, axis=1)
                nn_idx = np.argsort(dists)[: int(max(1, args.nn_k))]
                nn_min_dist = float(dists[nn_idx[0]])
                nn_vals = train_y_vals[nn_idx]
                if any([float(v) >= float(args.nn_extreme_threshold) for v in nn_vals]):
                    near_extreme = True
        except Exception:
            LOG.debug('Nearest-neighbor check failed; continuing without flag', exc_info=True)
                                                                         
        if getattr(args, 'blend', False) and (persistence_aqi is not None):
            bw = float(getattr(args, 'blend_weight', 0.8))
            thresh = float(getattr(args, 'blend_threshold', 20.0))
            try:
                pers = float(persistence_aqi)
                if abs(pred - pers) >= thresh:
                    blended = bw * pred + (1.0 - bw) * pers
                    LOG.info("Applying blending (threshold %.1f): model %.2f, persistence %.2f -> blended %.2f", thresh, pred, pers, blended)
                    pred = blended
                else:
                    LOG.info("Skipping blending because abs(model - persistence) < %.1f (%.2f)", thresh, abs(pred - pers))
            except Exception:
                LOG.warning("Failed to apply persistence blending; using model prediction")
        info = categorize_aqi(pred)
        out = {
            "date": datetime.fromtimestamp(rec["timestamp"], tz=timezone.utc).date().isoformat(),
            "aqi_pred": round(pred,2),
            "category": info["category"],
            "level": info["level"],
            "color": info["color"],
            "message": info["message"]
        }
        out['near_extreme_neighbor'] = bool(near_extreme)
        out['nn_min_distance'] = (None if nn_min_dist is None else float(nn_min_dist))
        results.append(out)
        nn_flags.append(bool(near_extreme))
        nn_distances.append(nn_min_dist)
        LOG.info("Pred %s -> AQI %.2f (%s)", out["date"], out["aqi_pred"], out["category"])

    df_out = pd.DataFrame(results)
    if len(aligned_rows) > 0:
        df_features = pd.DataFrame(aligned_rows)
        df_features.to_csv("data/predictions_3day_features.csv", index=False)
    if len(raw_feature_rows) > 0:
        pd.DataFrame(raw_feature_rows).to_csv("data/predictions_3day_raw_features.csv", index=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
                                                                                          
    dashboard_path = REPO_ROOT / "data" / "predictions_3day_openweather.csv"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_out.to_csv(dashboard_path, index=False)
        LOG.info("Saved dashboard predictions to: %s", dashboard_path)
    except Exception:
        LOG.exception("Failed to save dashboard predictions to %s", dashboard_path)
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()