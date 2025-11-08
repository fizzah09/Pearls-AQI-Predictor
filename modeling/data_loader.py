import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
log = logging.getLogger(__name__)
def load_training_data(csv_path: str) -> pd.DataFrame:
    """
    Load training data from CSV file.
    provide with 
        csv_path: Path to CSV file with historical features return us DataFrame with all features and targets
    """
    log.info("Loading training data from: %s", csv_path)
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    log.info(" Loaded %d rows, %d columns", len(df), len(df.columns))
    log.info("Date range: %s to %s", df['timestamp'].min(), df['timestamp'].max())
    
    return df
def prepare_features_targets(
    df: pd.DataFrame,
    target_col: str = 'pollutant_aqi',
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from target (y) by using data frames and series 
        df: Input DataFrame
        target_col: Column name for prediction target
        drop_cols: Additional columns to drop from features give us (X, y) tuple where X is features DataFrame and y is target Series
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    default_drop = ['timestamp', 'city_name', 'country', 
                   'weather_description', 'pollutant_aqi_category']
    
    if drop_cols:
        default_drop.extend(drop_cols)
    
    cols_to_drop = [col for col in default_drop if col in df.columns]
    cols_to_drop = list(set(cols_to_drop))
    
    X = df.drop(columns=[target_col] + cols_to_drop)
    y = df[target_col]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        log.info("Encoding categorical columns: %s", categorical_cols)
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    if X.isnull().sum().sum() > 0:
        log.warning("Found %d NaN values in features, filling with median", 
                   X.isnull().sum().sum())
        X = X.fillna(X.median())
    
    if y.isnull().sum() > 0:
        log.warning("Found %d NaN values in target, dropping those rows", 
                   y.isnull().sum())
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]
    
    log.info(" Features shape: %s", X.shape)
    log.info(" Target shape: %s", y.shape)
    log.info("Feature columns: %s", list(X.columns))
    
    return X, y


def get_feature_importance_names(X: pd.DataFrame) -> List[str]:
    """Get feature names for importance plotting."""
    return list(X.columns)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    df = load_training_data("../data/ml_training_data_1year.csv")
    X, y = prepare_features_targets(df, target_col='pollutant_aqi')
    
    print(f"\nDataset ready for modeling:")
    print(f"  Features: {X.shape}")
    print(f"  Target: {y.shape}")
    print(f"  Target distribution:\n{y.value_counts().sort_index()}")
