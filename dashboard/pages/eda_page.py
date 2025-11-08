
import streamlit as st
import numpy as np
from dashboard.components.charts import (
    create_correlation_heatmap,
    plot_scatter_analysis,
    plot_distribution
)
from dashboard.utils.data_loader import get_target_column

def show_eda_analysis(df):

    st.header(" Exploratory Data Analysis")
    
    if df is None:
        st.warning("No data available for analysis")
        return
    
    tab1, tab2, tab3 = st.tabs([
        "Correlation Analysis", 
        "Scatter Plots", 
        "Distribution Analysis"
    ])
    
    with tab1:
        st.subheader("Feature Correlation Heatmap")
        st.info(
            "This heatmap shows the correlation between all numeric features. "
            "Values close to +1 indicate strong positive correlation, "
            "values close to -1 indicate strong negative correlation."
        )
        
        fig_corr = create_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.subheader("Scatter Plot Analysis")
        st.info(
            "Explore relationships between features with interactive scatter plots. "
            "The trendline shows the overall relationship direction."
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if 'timestamp' not in col.lower()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, index=0)
        
        with col2:
            y_col = st.selectbox(
                "Y-axis", 
                numeric_cols, 
                index=min(1, len(numeric_cols)-1)
            )
        
        if x_col and y_col:
            fig_scatter = plot_scatter_analysis(df, x_col, y_col)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.subheader("Distribution Analysis")
        st.info(
            "View the distribution of values for the target variable (AQI). "
            "This helps understand the range and frequency of AQI values in the dataset."
        )
        
        target_col = get_target_column(df)
        
        fig_dist = plot_distribution(df, target_col)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("#### Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{df[target_col].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[target_col].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df[target_col].std():.2f}")
        with col4:
            st.metric("Range", f"{df[target_col].max() - df[target_col].min():.2f}")
