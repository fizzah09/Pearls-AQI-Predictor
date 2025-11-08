from pathlib import Path

import streamlit as st
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_CSV = REPO_ROOT / "data" / "predictions_3day_openweather.csv"


def show_three_day_panel():
    st.header("3-day AQI Forecast")

    if not PRED_CSV.exists():
        st.warning("Run the 3-day prediction script first (it writes data/predictions_3day_openweather.csv).")
        return

    df_pred = pd.read_csv(PRED_CSV)
    st.subheader("Daily summary")
    for _, row in df_pred.iterrows():
        date = row.get("date", "")
        aqi = row.get("aqi_pred", "")
        category = row.get("category", "")
        message = row.get("message", "")
        st.markdown(f"**{date}** — AQI: **{aqi}** ({category})")
        if message:
            st.write(message)


if __name__ == "__main__":
    show_three_day_panel()
