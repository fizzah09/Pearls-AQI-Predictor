import streamlit as st
from inference.predictor import categorize_aqi, generate_aqi_alert
def display_aqi_metrics(df, target_col):
   
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aqi = df[target_col].mean()
        st.metric("Average AQI", f"{avg_aqi:.1f}")
    
    with col2:
        max_aqi = df[target_col].max()
        st.metric("Max AQI", f"{max_aqi:.1f}")
    
    with col3:
        min_aqi = df[target_col].min()
        st.metric("Min AQI", f"{min_aqi:.1f}")
    
    with col4:
        hazardous_count = len(df[df[target_col] > 200])
        st.metric("Hazardous Days", hazardous_count)
def display_prediction_result(prediction):
    aqi_info = categorize_aqi(prediction)
    alert = generate_aqi_alert(prediction)
    
    if alert:
        st.markdown(
            f'<div class="alert-box alert-danger">{alert}</div>',
            unsafe_allow_html=True
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "AQI Value",
            f"{prediction:.1f}",
            delta=None
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-card" style="background-color: {aqi_info["color"]}20; '
            f'border-left: 5px solid {aqi_info["color"]};">'
            f'<h3>{aqi_info["category"]}</h3>'
            f'<p>Level {aqi_info["level"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.info(aqi_info['message'])
    
    with col4:
        if aqi_info['is_hazardous']:
            st.error(" HAZARDOUS")
        else:
            st.success(" SAFE")


def display_sidebar_info():
    st.image("https://img.icons8.com/color/96/000000/air-quality.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        [" Dashboard", " Predictions", " EDA & Trends", " Explainability"]
    )
    
    st.markdown("---")
    st.markdown("About")
    st.info(
        "This dashboard provides real-time AQI predictions, "
        "trend analysis, and model explainability using advanced ML techniques."
    )
    
    return page
