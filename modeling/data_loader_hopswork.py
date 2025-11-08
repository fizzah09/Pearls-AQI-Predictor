"""
Hopsworks Feature Store Data Loader for Training Pipeline
"""
import hopsworks
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data_from_hopsworks(
    api_key: str = None,
    project_name: str = None,
    weather_fg_name: str = "weather_features",
    pollutant_fg_name: str = "pollutant_features",
    version: int = 2,
    days_back: int = 30
) -> pd.DataFrame:
    try:
        api_key = api_key or os.getenv('HOPSWORKS_API_KEY')
        project_name = project_name or os.getenv('HOPSWORKS_PROJECT_NAME')
        
        if not api_key or not project_name:
            raise ValueError("HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME must be set in environment")
        
        logger.info(f" Connecting to Hopsworks project: {project_name}")
        
        project = hopsworks.login(
            api_key_value=api_key,
            project=project_name
        )
        
        fs = project.get_feature_store()
        logger.info(f" Connected to Feature Store")
        
        logger.info(f" Loading Weather Features...")
        weather_fg = fs.get_feature_group(name=weather_fg_name, version=version)
        weather_df = weather_fg.read()
        logger.info(f"   Loaded {len(weather_df)} weather records")
        
        logger.info(f" Loading Pollutant Features...")
        pollutant_fg = fs.get_feature_group(name=pollutant_fg_name, version=version)
        pollutant_df = pollutant_fg.read()
        logger.info(f"   Loaded {len(pollutant_df)} pollutant records")
        
        logger.info(f" Merging datasets...")
        merged_df = pd.merge(
            weather_df,
            pollutant_df,
            on='timestamp',
            how='inner',
            suffixes=('_weather', '_pollutant')
        )
        
        logger.info(f" Merged: {len(merged_df)} records, {len(merged_df.columns)} columns")
        
        if 'timestamp' in merged_df.columns and days_back:
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            merged_df = merged_df[
                (merged_df['timestamp'] >= start_date) & 
                (merged_df['timestamp'] <= end_date)
            ]
            logger.info(f" Filtered to last {days_back} days: {len(merged_df)} records")
        
        logger.info(f" Final dataset shape: {merged_df.shape}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f" Error loading data from Hopsworks: {str(e)}")
        raise


def get_latest_features_for_inference(
    api_key: str = None,
    project_name: str = None,
    weather_fg_name: str = "weather_features",
    pollutant_fg_name: str = "pollutant_features",
    version: int = 2,
    limit: int = 1
) -> pd.DataFrame:
    try:
        api_key = api_key or os.getenv('HOPSWORKS_API_KEY')
        project_name = project_name or os.getenv('HOPSWORKS_PROJECT_NAME')
        
        project = hopsworks.login(
            api_key_value=api_key,
            project=project_name
        )
        
        fs = project.get_feature_store()
        
        weather_fg = fs.get_feature_group(name=weather_fg_name, version=version)
        pollutant_fg = fs.get_feature_group(name=pollutant_fg_name, version=version)
        
        weather_df = weather_fg.read()
        pollutant_df = pollutant_fg.read()
        
        merged_df = pd.merge(
            weather_df,
            pollutant_df,
            on='timestamp',
            how='inner',
            suffixes=('_weather', '_pollutant')
        )
        
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.sort_values('timestamp', ascending=False).head(limit)
        
        logger.info(f" Fetched {len(merged_df)} latest records for inference")
        
        return merged_df
        
    except Exception as e:
        logger.error(f" Error fetching latest features: {str(e)}")
        raise


def verify_feature_store_connection():
    """Verify connection to Hopsworks Feature Store"""
    try:
        api_key = os.getenv('HOPSWORKS_API_KEY')
        project_name = os.getenv('HOPSWORKS_PROJECT_NAME')
        
        if not api_key or not project_name:
            print(" HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME must be set")
            return False
        
        project = hopsworks.login(
            api_key_value=api_key,
            project=project_name
        )
        
        fs = project.get_feature_store()
        
        feature_groups = fs.get_feature_groups()
        
        print(f" Connected to Hopsworks successfully!")
        print(f" Project: {project.name}")
        print(f" Available Feature Groups:")
        for fg in feature_groups:
            print(f"  - {fg.name} (v{fg.version}) - {len(fg.features)} features")
        
        return True
        
    except Exception as e:
        print(f" Connection failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Testing Hopsworks Connection...")
    print("="*70)
    
    verify_feature_store_connection()
    
    print("\n" + "="*70)
    print("Testing Data Loading...")
    print("="*70)
    
    try:
        df = load_training_data_from_hopsworks(days_back=7)
        print(f"\n Training Data Shape: {df.shape}")
        print(f" Columns: {list(df.columns)}")
        print(f"\n Sample Data:")
        print(df.head())
        print(f"\n Data Info:")
        print(df.info())
    except Exception as e:
        print(f"\n Error loading data: {e}")