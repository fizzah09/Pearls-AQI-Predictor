import streamlit as st
from dashboard.components.metrics import display_prediction_result


def show_prediction_interface(engine):
    st.header(" Make Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Weather Features")
        temp = st.slider("Temperature (°C)", -20.0, 50.0, 25.0, 0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 1.0)
        pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0, 1.0)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.1)
    
    with col2:
        st.subheader("Pollutant Features")
        pm2_5 = st.slider("PM2.5 (μg/m³)", 0.0, 500.0, 50.0, 1.0)
        pm10 = st.slider("PM10 (μg/m³)", 0.0, 600.0, 80.0, 1.0)
        no2 = st.slider("NO2 (μg/m³)", 0.0, 400.0, 40.0, 1.0)
    
    with col3:
        st.subheader("Additional Pollutants")
        o3 = st.slider("O3 (μg/m³)", 0.0, 300.0, 60.0, 1.0)
        so2 = st.slider("SO2 (μg/m³)", 0.0, 200.0, 20.0, 1.0)
        co = st.slider("CO (μg/m³)", 0.0, 10000.0, 1000.0, 10.0)
    
    if st.button(" Predict AQI", type="primary", use_container_width=True):
        import pandas as pd
        import numpy as np
        
        if hasattr(engine, 'feature_names') and engine.feature_names:
            feature_names = engine.feature_names
            
            features_dict = {}
            for feature in feature_names:
                if 'temperature' in feature or 'temp' in feature:
                    features_dict[feature] = temp
                elif 'humidity' in feature:
                    features_dict[feature] = humidity
                elif 'pressure' in feature:
                    features_dict[feature] = pressure
                elif 'wind_speed' in feature:
                    features_dict[feature] = wind_speed
                elif 'pm2_5' in feature:
                    features_dict[feature] = pm2_5
                elif 'pm10' in feature:
                    features_dict[feature] = pm10
                elif 'no2' in feature:
                    features_dict[feature] = no2
                elif 'o3' in feature:
                    features_dict[feature] = o3
                elif 'so2' in feature:
                    features_dict[feature] = so2
                elif 'co' in feature:
                    features_dict[feature] = co
                elif feature.startswith('season_') or feature.startswith('weather_'):
                    features_dict[feature] = 0
                else:
                    features_dict[feature] = 0
            
            features_df = pd.DataFrame([features_dict])
        else:
            st.error(" Model feature names not available")
            return
        
        try:
            if engine is None:
                st.error(" Model not loaded. Please check model file exists.")
                return
            
            prediction = engine.predict(features_df)[0]
            
            st.session_state['last_prediction'] = {
                'features': features_df,
                'prediction': prediction,
                'feature_values': {
                    'Temperature': temp,
                    'Humidity': humidity,
                    'Pressure': pressure,
                    'Wind Speed': wind_speed,
                    'PM2.5': pm2_5,
                    'PM10': pm10,
                    'NO2': no2,
                    'O3': o3,
                    'SO2': so2,
                    'CO': co
                }
            }
            
            st.markdown("---")
            st.subheader(" Prediction Results")
            display_prediction_result(prediction)
            
            st.success(" Prediction saved! Go to **Model Explainability** tab to see LIME analysis.")
            
        except AttributeError as e:
            st.error(f" Model error: {str(e)}")
            st.info(
                "The model might not be loaded correctly. "
                "Please check:\n"
                "1. Model file exists in modeling/models/\n"
                "2. Model is a valid XGBoost model\n"
                "3. Run `python run_training.py` to retrain"
            )
        except Exception as e:
            st.error(f" Prediction failed: {str(e)}")
            st.info("Please ensure all features are provided correctly.")
