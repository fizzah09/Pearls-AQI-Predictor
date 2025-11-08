
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.predictor import AQIInferenceEngine
@st.cache_resource
def load_model():

    try:
        model_dir = Path(__file__).parent.parent.parent / "modeling" / "models"
        model_files = list(model_dir.glob("*_xgboost.pkl"))
        
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            engine = AQIInferenceEngine(str(latest_model))
            return engine
        else:
            st.error("No trained model found. Please train a model first.")
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
@st.cache_data
def load_data():
    try:
        data_path = Path(__file__).parent.parent.parent / "data" / "ml_training_data_1year.csv"
        
        if not data_path.exists():
            st.warning("No historical data found.")
            return None
        
        df = pd.read_csv(data_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            df['timestamp'] = pd.date_range(
                end=datetime.now(), 
                periods=len(df), 
                freq='H'
            )
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None
def get_target_column(df):
    if 'pollutant_aqi' in df.columns:
        return 'pollutant_aqi'
    else:
        return df.columns[-1]
