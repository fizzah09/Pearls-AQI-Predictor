from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import AQIInferenceEngine, categorize_aqi, generate_aqi_alert
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
app = FastAPI(
    title="AQI Prediction API",
    description="API for predicting Air Quality Index (AQI) and providing health recommendations",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
inference_engine = None

class FeatureInput(BaseModel):
    weather_temp: float = Field(..., description="Temperature in Celsius")
    weather_humidity: float = Field(..., description="Humidity percentage")
    weather_pressure: float = Field(..., description="Atmospheric pressure")
    weather_wind_speed: float = Field(..., description="Wind speed")
    pollutant_pm2_5: Optional[float] = Field(None, description="PM2.5 concentration")
    pollutant_pm10: Optional[float] = Field(None, description="PM10 concentration")
    pollutant_no2: Optional[float] = Field(None, description="NO2 concentration")
    pollutant_o3: Optional[float] = Field(None, description="O3 concentration")
    pollutant_so2: Optional[float] = Field(None, description="SO2 concentration")
    pollutant_co: Optional[float] = Field(None, description="CO concentration")


class PredictionResponse(BaseModel):
    aqi_prediction: float
    category: str
    level: int
    color: str
    health_message: str
    is_hazardous: bool
    alert: Optional[str]
    timestamp: str


class BatchPredictionRequest(BaseModel):
    features: List[Dict[str, float]]


class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


@app.on_event("startup")
async def startup_event():
    global inference_engine
    
    try:
        log.info("Starting AQI Prediction API...")
        
        model_dir = Path(__file__).parent.parent / "modeling" / "models"
        model_files = list(model_dir.glob("*_xgboost.pkl"))
        
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            inference_engine = AQIInferenceEngine(str(latest_model))
            log.info(f" Model loaded: {latest_model.name}")
        else:
            log.warning("No model found. Please train a model first.")
            
    except Exception as e:
        log.error(f"Failed to initialize: {e}")


@app.get("/", response_model=HealthStatus)
async def root():
    return {
        "status": "healthy" if inference_engine else "no model loaded",
        "model_loaded": inference_engine is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    return {
        "status": "healthy" if inference_engine else "no model loaded",
        "model_loaded": inference_engine is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: FeatureInput):
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_dict = features.dict()
        
        aqi_pred = inference_engine.predict_single(feature_dict)
        
        aqi_info = categorize_aqi(aqi_pred)
        
        alert = generate_aqi_alert(aqi_pred)
        
        return {
            "aqi_prediction": round(aqi_pred, 2),
            "category": aqi_info['category'],
            "level": aqi_info['level'],
            "color": aqi_info['color'],
            "health_message": aqi_info['message'],
            "is_hazardous": aqi_info['is_hazardous'],
            "alert": alert,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame(request.features)
        
        predictions = inference_engine.predict(df)
        
        results = []
        for pred in predictions:
            aqi_info = categorize_aqi(pred)
            results.append({
                "aqi_prediction": round(pred, 2),
                "category": aqi_info['category'],
                "level": aqi_info['level'],
                "is_hazardous": aqi_info['is_hazardous']
            })
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain")
async def explain(top_k: int = 10):
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        try:
            fi_df = inference_engine.get_feature_importance()
        except Exception:
            fi_df = None

        if fi_df is None:
            raise HTTPException(status_code=501, detail="Feature importance not available for this model")

        fi_df = fi_df.sort_values('importance', ascending=False).head(top_k)
        result = fi_df.to_dict('records')
        return {"top_k": len(result), "features": result}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to compute explanation: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def get_aqi_categories():
    return {
        "categories": [
            {"range": "0-50", "category": "Good", "color": "#00e400"},
            {"range": "51-100", "category": "Moderate", "color": "#ffff00"},
            {"range": "101-150", "category": "Unhealthy for Sensitive Groups", "color": "#ff7e00"},
            {"range": "151-200", "category": "Unhealthy", "color": "#ff0000"},
            {"range": "201-300", "category": "Very Unhealthy", "color": "#8f3f97"},
            {"range": "300+", "category": "Hazardous", "color": "#7e0023"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
