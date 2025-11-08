import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

from inference.predictor import AQIInferenceEngine

@st.cache_data
def _evaluate_model_on_df(model_path: str, df: pd.DataFrame, test_frac: float = 0.2):
    """Evaluate a saved model on the last `test_frac` portion of df.

    Returns a dict of metrics (rmse, mae, r2, mape, n_samples) for the test split.
    """
    try:
                                    
        engine = AQIInferenceEngine(str(model_path))
                                                                                     
        from modeling.data_loader import prepare_features_targets

        X_all, y_all = prepare_features_targets(df.copy(), target_col='pollutant_aqi')

                                                    
        if getattr(engine, 'feature_names', None):
            fn = list(engine.feature_names)
            for c in fn:
                if c not in X_all.columns:
                    X_all[c] = 0
            X_all = X_all.reindex(columns=fn, fill_value=0)

                   
        X_all = X_all.fillna(0)

        n = len(X_all)
        n_test = max(1, int(n * test_frac))
        X_test = X_all.iloc[-n_test:]
        y_test = y_all.iloc[-n_test:]

                     
        preds = engine.predict(X_test)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

        rmse = float(np.sqrt(mean_squared_error(y_test.values, preds)))
        mae = float(mean_absolute_error(y_test.values, preds))
        r2 = float(r2_score(y_test.values, preds))
        mape = float(mean_absolute_percentage_error(y_test.values, preds) * 100)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'n_samples': int(len(y_test))
        }
    except Exception as e:
        return {'error': str(e)}


def show_model_metrics(engine: Optional[AQIInferenceEngine] = None):
    st.header(" Model Metrics & Comparison")

    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / 'modeling' / 'models'
    eval_dir = project_root / 'modeling' / 'evaluation'

                                                                               
    try:
        from dashboard.utils.data_loader import load_data
        df = load_data()
    except Exception:
        df = None

    if df is None:
        st.warning('No historical data available to evaluate models. Place training CSV at data/ml_training_data_1year.csv')
        return

                                                                         
    patterns = ['*xgboost*.pkl', '*ridge*.pkl', '*randomforest*.pkl', '*random_forest*.pkl', '*rf*.pkl']
    model_files = []
    for p in patterns:
        model_files.extend(list(model_dir.glob(p)))

                                  
    model_files = sorted(set(model_files), key=lambda p: p.name)

    if not model_files:
        st.warning('No model artifacts found in modeling/models. Train or drop model files there.')
        return

    st.markdown('### Discovered models')
    for m in model_files:
        st.write(f'- {m.name}')

                                                 
    results = []
    for m in model_files:
        with st.spinner(f'Evaluating {m.name} ...'):
            metrics = _evaluate_model_on_df(str(m), df, test_frac=0.2)
            row = {
                'model_file': m.name,
                'rmse': metrics.get('rmse'),
                'mae': metrics.get('mae'),
                'r2': metrics.get('r2'),
                'mape': metrics.get('mape'),
                'n_samples': metrics.get('n_samples'),
                'error': metrics.get('error')
            }
            results.append(row)

    comp_df = pd.DataFrame(results).set_index('model_file')

                                                       
    if engine is not None and getattr(engine, 'model_path', None):
        try:
            loaded_stem = Path(engine.model_path).name
        except Exception:
            loaded_stem = None
    else:
        loaded_stem = None

    def _format_row(r):
        if pd.notna(r['error']):
            return f"Error: {r['error']}"
        return ''

    st.markdown('### Model comparison (evaluated on last 20% of historical data)')
    display_df = comp_df[['rmse','mae','r2','mape','n_samples','error']]
    st.dataframe(display_df.style.format({'rmse':'{:.3f}','mae':'{:.3f}','r2':'{:.3f}','mape':'{:.2f}','n_samples':'{:.0f}'}))

    if loaded_stem:
        st.markdown('---')
        st.info(f'Currently loaded model in the dashboard: `{loaded_stem}`')

                                               
    comp_img = eval_dir / 'metrics_comparison.png'
    if comp_img.exists():
        st.markdown('---')
        st.subheader(' Saved metrics comparison chart')
        st.image(str(comp_img))

    return
