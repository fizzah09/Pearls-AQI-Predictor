
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path

log = logging.getLogger(__name__)


class AQIPredictor:
    def __init__(self, task: str = 'regression', random_state: int = 42, model_params: dict = None):
        """Create AQIPredictor.

        model_params: optional dict to override XGBoost params (n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda, subsample, colsample_bytree, etc.)
        """
        self.task = task
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None

                            
        default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'reg_alpha': 0.0,
            'reg_lambda': 1.0
        }

        if model_params:
            default_params.update(model_params)

        if task == 'regression':
            self.model = xgb.XGBRegressor(**default_params)
        elif task == 'classification':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                objective='multi:softmax',
                num_class=5,
                eval_metric='merror'
            )
        else:
            raise ValueError(f"Unknown task: {task}. Use 'regression' or 'classification'")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        scale_features: bool = True,
        early_stopping_rounds: int = None
    ) -> Dict[str, Any]:

        log.info("Training XGBoost %s model...", self.task)
        self.feature_names = list(X_train.columns)

        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            else:
                X_val_scaled = None
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val

        eval_set = []
        if X_val_scaled is not None and y_val is not None:
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        else:
            eval_set = [(X_train_scaled, y_train)]

        fit_kwargs = {'eval_set': eval_set, 'verbose': False}
        if early_stopping_rounds is not None:
                                                                                                    
            fit_kwargs['callbacks'] = [xgb.callback.EarlyStopping(rounds=int(early_stopping_rounds), save_best=True)]

                                                                                          
        self.model.fit(X_train_scaled, y_train, **fit_kwargs)

        log.info("Model trained with %d features", len(self.feature_names))

        history = {
            'n_features': len(self.feature_names),
            'n_samples': len(X_train),
            'feature_names': self.feature_names
        }

        return history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, model_path: str):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'task': self.task
        }
        
        joblib.dump(model_data, model_path)
        log.info("Model saved to: %s", model_path)
    
    @classmethod
    def load(cls, model_path: str) -> 'AQIPredictor':
        model_data = joblib.load(model_path)
        
        predictor = cls(task=model_data['task'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        
        log.info("Model loaded from: %s", model_path)
        return predictor


def split_train_test(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    log.info("Data split:")
    log.info("  Train: %d samples (%.1f%%)", len(X_train), len(X_train)/len(X)*100)
    log.info("  Val:   %d samples (%.1f%%)", len(X_val), len(X_val)/len(X)*100)
    log.info("  Test:  %d samples (%.1f%%)", len(X_test), len(X_test)/len(X)*100)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from data_loader import load_training_data, prepare_features_targets
    
    df = load_training_data("../data/ml_training_data_1year.csv")
    X, y = prepare_features_targets(df, target_col='pollutant_aqi')
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(X, y)
    
    predictor = AQIPredictor(task='regression')
    predictor.train(X_train, y_train, X_val, y_val)
    
    y_pred = predictor.predict(X_test)
    print(f"\nPrediction sample: {y_pred[:5]}")
    print(f"Actual sample: {y_test.values[:5]}")
