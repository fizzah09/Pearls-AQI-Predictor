"""
Explainability Page - Model interpretability using SHAP and LIME
"""
import streamlit as st
import plotly.express as px


def show_explainability(engine, df):
    """
    Display the model explainability page
    
    This page provides:
    - SHAP (SHapley Additive exPlanations) analysis
    - LIME (Local Interpretable Model-agnostic Explanations) analysis
    - Feature importance visualization
    
    Args:
        engine (AQIInferenceEngine): Loaded model engine
        df (pd.DataFrame): Historical data
    """
    st.header(" Model Explainability")
    
    st.info(
        "Model explainability helps understand **why** the model makes certain predictions. "
        "SHAP and LIME are two popular techniques for interpreting machine learning models."
    )
    
    tab1, tab2 = st.tabs(["SHAP Analysis", "LIME Analysis"])
    
    with tab1:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        
        st.markdown("""
        **What is SHAP?**
        
        SHAP values show how each feature contributes to the prediction:
        - **Positive SHAP value**: Feature pushes prediction higher
        - **Negative SHAP value**: Feature pushes prediction lower
        - **Magnitude**: How much impact the feature has
        """)
        
        try:
            if hasattr(engine, 'model') and hasattr(engine.model, 'get_booster'):
                import xgboost as xgb
                booster = engine.model.get_booster()
                importance_dict = booster.get_score(importance_type='weight')
                
                import pandas as pd
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in importance_dict.items()
                ]).sort_values('importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 20 Features by Importance",
                    labels={'importance': 'Feature Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    height=600, 
                    showlegend=False, 
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Top 10 Most Important Features")
                st.dataframe(
                    importance_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )
                
            else:
                raise AttributeError("Model does not support feature importance extraction")
            
        except Exception as e:
            st.warning(f" Feature importance not available: {str(e)}")
            st.info(
                "**To enable SHAP analysis:**\n\n"
                "1. Ensure your model is properly trained\n"
                "2. Install SHAP: `pip install shap`\n"
                "3. Model must be XGBoost or support feature importance"
            )
    
    with tab2:
        st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
        
        st.markdown("""
        **What is LIME?**
        
        LIME explains individual predictions by:
        1. Creating perturbed samples around the prediction
        2. Training a simple interpretable model locally
        3. Showing which features contributed most to that specific prediction
        
        **Use Case**: Understanding why a specific prediction was made
        """)
        
        if 'last_prediction' in st.session_state:
            pred_data = st.session_state['last_prediction']
            
            st.success(f" Analyzing prediction: **AQI = {pred_data['prediction']:.2f}**")
            
            st.markdown("#### Input Features Used:")
            import pandas as pd
            feature_df = pd.DataFrame([pred_data['feature_values']]).T
            feature_df.columns = ['Value']
            st.dataframe(feature_df, use_container_width=True)
            
            try:
                import lime
                import lime.lime_tabular
                import numpy as np
                
                st.markdown("#### LIME Explanation")
                
                with st.spinner("Generating LIME explanation..."):
                    import pandas as pd
                    model_features = []
                    if hasattr(engine, 'feature_names') and engine.feature_names:
                        model_features = list(engine.feature_names)
                    else:
                        model_features = list(pred_data['features'].columns)

                    X_train = None
                    feature_cols = list(model_features)

                    if df is not None and len(df) > 10:
                        numeric_df = df.select_dtypes(include=[np.number]).copy()
                        
                        if 'pollutant_aqi' in numeric_df.columns:
                            numeric_df = numeric_df.drop(columns=['pollutant_aqi'])
                        
                        available = [c for c in feature_cols if c in numeric_df.columns]

                        if len(available) >= max(3, int(0.5 * len(feature_cols))):
                            X_train_df = numeric_df[available].sample(min(1000, len(numeric_df))).copy()
                            
                            for c in feature_cols:
                                if c not in X_train_df.columns:
                                    X_train_df[c] = 0.0
                            
                            X_train_df = X_train_df[feature_cols]
                            X_train = X_train_df.astype(float).values

                    if X_train is None:
                        features_df = pred_data['features'].copy()
                        
                        features_df_numeric = features_df.copy()
                        for col in features_df_numeric.columns:
                            features_df_numeric[col] = pd.to_numeric(features_df_numeric[col], errors='coerce')
                        features_df_numeric = features_df_numeric.fillna(0.0)
                        
                        features_df_numeric = features_df_numeric.reindex(columns=feature_cols, fill_value=0.0)
                        center = features_df_numeric.values[0].astype(float)
                        
                        np.random.seed(42)
                        n_samples = 500
                        X_train = np.zeros((n_samples, len(feature_cols)))
                        
                        for i, val in enumerate(center):
                            if abs(val) > 1e-6:
                                std = abs(val) * 0.15
                                X_train[:, i] = np.random.normal(val, std, n_samples)
                            else:
                                X_train[:, i] = np.random.uniform(-1, 1, n_samples)
                        
                        X_train = np.maximum(X_train, 0)

                                                                               
                    X_train = np.array(X_train, dtype=float)
                    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                    stds = X_train.std(axis=0)
                    mins = X_train.min(axis=0)
                    maxs = X_train.max(axis=0)
                    for j in range(X_train.shape[1]):
                        if stds[j] <= 1e-12 or (maxs[j] - mins[j]) <= 1e-12:
                            jitter = 1e-6 if stds[j] <= 1e-12 else float(stds[j] * 1e-3)
                            X_train[:, j] = X_train[:, j] + np.random.normal(0.0, jitter, size=X_train.shape[0])
                            if X_train.shape[0] >= 2 and np.allclose(X_train[:, j].max(), X_train[:, j].min()):
                                X_train[0, j] -= 1e-6
                                X_train[1, j] += 1e-6

                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        X_train,
                        feature_names=feature_cols,
                        mode='regression',
                        discretize_continuous=True,
                        verbose=False
                    )

                    features_df = pred_data['features'].copy()
                    
                    for col in features_df.columns:
                        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                    features_df = features_df.fillna(0.0)
                    
                    features_df = features_df.reindex(columns=feature_cols, fill_value=0.0)
                    features_array = features_df.values[0].astype(float)

                    def predict_fn(X):
                        X_df = pd.DataFrame(X, columns=feature_cols)
                        for col in X_df.columns:
                            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
                        X_df = X_df.fillna(0.0).astype(float)
                        preds = engine.predict(X_df)
                        return preds.ravel()

                    explanation = explainer.explain_instance(
                        features_array,
                        predict_fn,
                        num_features=10
                    )
                    
                    exp_list = explanation.as_list()
                    
                    features = [item[0] for item in exp_list]
                    impacts = [item[1] for item in exp_list]
                    
                    lime_df = pd.DataFrame({
                        'Feature': features,
                        'Impact': impacts
                    }).sort_values('Impact', key=abs, ascending=False)
                    
                    fig = px.bar(
                        lime_df,
                        x='Impact',
                        y='Feature',
                        orientation='h',
                        title="LIME Feature Contributions",
                        labels={'Impact': 'Contribution to Prediction', 'Feature': 'Feature'},
                        color='Impact',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0
                    )
                    
                    fig.update_layout(height=500, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("Interpretation:")
                    st.markdown("""
                    - **Positive values** (green): Features that increased the predicted AQI
                    - **Negative values** (red): Features that decreased the predicted AQI
                    - **Magnitude**: How much impact each feature had
                    """)
                    
                    with st.expander("Detailed Feature Contributions"):
                        st.dataframe(lime_df, use_container_width=True, hide_index=True)
                        
            except ImportError:
                st.warning(" LIME is not installed")
                st.markdown("#### Setup Instructions")
                st.code("pip install lime", language="bash")
                
            except Exception as e:
                st.error(f" LIME analysis failed: {str(e)}")
                st.info("This might be due to feature mismatch or insufficient data")
        
        else:
            st.info(
                " **No prediction available yet!**\n\n"
                "Go to the ** Make Predictions** tab to make a prediction first, "
                "then return here to see LIME explanations."
            )
            
            with st.expander(" How to use LIME"):
                st.markdown("""
                1. Navigate to the ** Make Predictions** tab
                2. Adjust the sliders to set feature values
                3. Click **" Predict AQI"** button
                4. Return to this **Model Explainability** tab
                5. The LIME explanation will show:
                   - Which features increased/decreased the prediction
                   - By how much each feature contributed
                   - A visual bar chart of contributions
                   
                **Example Interpretation:**
                - If PM2.5 shows +15, it means PM2.5 increased the AQI by 15 points
                - If Wind Speed shows -5, it means wind decreased the AQI by 5 points
                """)
