import streamlit as st
from dashboard.components.charts import plot_aqi_trend, plot_pollutant_bars
from dashboard.components.metrics import display_aqi_metrics
from dashboard.utils.data_loader import get_target_column
def show_dashboard_overview(df):
    st.header(" Dashboard Overview")
    
    if df is None:
        st.warning("No data available for display")
        return
    
    target_col = get_target_column(df)
    
    display_aqi_metrics(df, target_col)
    
    st.markdown("---")
    
    st.subheader(" AQI Trend Over Time")
    fig_trend = plot_aqi_trend(df, target_col)
    if fig_trend:
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.subheader(" Pollutant Concentrations")
    fig_bars = plot_pollutant_bars(df)
    if fig_bars:
        st.plotly_chart(fig_bars, use_container_width=True)
