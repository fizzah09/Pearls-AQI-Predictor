
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
log = logging.getLogger(__name__)
class AQIInferenceEngine:

    
    def __init__(self, model_path: Optional[str] = None):

        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.feature_names = None
        self.task = 'regression'
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        try:
            log.info(f"Loading model from: {model_path}")
            
            loaded_obj = joblib.load(model_path)
            
            if isinstance(loaded_obj, dict):
                self.model = loaded_obj.get('model')
                self.scaler = loaded_obj.get('scaler')
                self.feature_names = loaded_obj.get('feature_names')
                self.task = loaded_obj.get('task', 'regression')
                log.info(f" Loaded model from dictionary format")
            else:
                self.model = loaded_obj
                if hasattr(self.model, 'feature_names_'):
                    self.feature_names = self.model.feature_names_
                log.info(f" Loaded model directly")
            
            if self.feature_names:
                log.info(f" Model has {len(self.feature_names)} features")
            
            log.info(f" Model loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise
    
    def load_from_registry(self, model_name: str, version: Optional[int] = None):
        try:
            from modeling.model_registry import load_model_from_registry
            
            log.info(f"Loading model from registry: {model_name}")
            self.model, metadata = load_model_from_registry(model_name, version)
            
            log.info(f" Model loaded from registry: v{metadata.version}")
            
        except Exception as e:
            log.error(f"Failed to load from registry: {e}")
            raise
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
                predictions = self.model.predict(features_scaled)
            else:
                predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            log.error(f"Prediction failed: {e}")
            raise
    
    def predict_single(self, feature_dict: Dict[str, Any]) -> float:
        
        df = pd.DataFrame([feature_dict])
        prediction = self.predict(df)
        return float(prediction[0])
    
    def predict_with_uncertainty(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = self.predict(features)
        
        uncertainty = np.abs(predictions) * 0.1
        
        return {
            'predictions': predictions,
            'lower_bound': predictions - uncertainty,
            'upper_bound': predictions + uncertainty,
            'uncertainty': uncertainty
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importance))]
            
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            df = df.sort_values('importance', ascending=False).reset_index(drop=True)
            return df
        elif hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance()
        else:
            raise NotImplementedError("Model doesn't support feature importance")


def categorize_aqi(aqi_value: float) -> Dict[str, Any]:
    
    if aqi_value <= 50:
        return {
            'category': 'Good',
            'level': 1,
            'color': '#00e400',
            'message': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'is_hazardous': False
        }
    elif aqi_value <= 100:
        return {
            'category': 'Moderate',
            'level': 2,
            'color': '#ffff00',
            'message': 'Air quality is acceptable. However, there may be a risk for some people.',
            'is_hazardous': False
        }
    elif aqi_value <= 150:
        return {
            'category': 'Unhealthy for Sensitive Groups',
            'level': 3,
            'color': '#ff7e00',
            'message': 'Members of sensitive groups may experience health effects.',
            'is_hazardous': False
        }
    elif aqi_value <= 200:
        return {
            'category': 'Unhealthy',
            'level': 4,
            'color': '#ff0000',
            'message': 'Some members of the general public may experience health effects.',
            'is_hazardous': True
        }
    elif aqi_value <= 300:
        return {
            'category': 'Very Unhealthy',
            'level': 5,
            'color': '#8f3f97',
            'message': 'Health alert: The risk of health effects is increased for everyone.',
            'is_hazardous': True
        }
    else:
        return {
            'category': 'Hazardous',
            'level': 6,
            'color': '#7e0023',
            'message': ' HEALTH WARNING: Everyone may experience serious health effects.',
            'is_hazardous': True
        }


def generate_aqi_alert(aqi_value: float) -> Optional[str]:
    info = categorize_aqi(aqi_value)
    
    if info['is_hazardous']:
        return f" ALERT: Air quality is {info['category']} (AQI: {aqi_value:.0f}). {info['message']}"
    
    return None
