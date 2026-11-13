import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Demand Forecasting Dashboard", layout="wide")

st.title("Retail Demand Forecasting Dashboard")

forecast_path = os.path.join("outputs", "future_forecast.csv")
results_path = os.path.join("outputs", "forecast_results.csv")
metrics_path = os.path.join("outputs", "model_metrics.txt")

if not os.path.exists(forecast_path):
    st.error("Future forecast file not found. Run forecast_future.py first.")
    st.stop()

forecast_df = pd.read_csv(forecast_path, parse_dates=["date"])
results_df = pd.read_csv(results_path) if os.path.exists(results_path) else None

store_options = sorted(forecast_df["store_id"].unique())
product_options = sorted(forecast_df["product_id"].unique())

selected_store = st.selectbox("Select Store", store_options)
selected_product = st.selectbox("Select Product", product_options)

filtered = forecast_df[
    (forecast_df["store_id"] == selected_store)
    & (forecast_df["product_id"] == selected_product)
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Next-Day Forecast")
    st.dataframe(filtered, use_container_width=True)

with col2:
    st.subheader("Forecast Summary")
    if not filtered.empty:
        st.metric("Forecast Units", int(filtered["forecast_units"].iloc[0]))
        st.metric("Forecast Date", str(filtered["date"].dt.date.iloc[0]))

if results_df is not None:
    st.subheader("Model Actual vs Predicted Sample")
    sample = results_df.head(200)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample["actual"].values, label="Actual")
    ax.plot(sample["predicted"].values, label="Predicted")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Units Sold")
    ax.legend()
    st.pyplot(fig)

if os.path.exists(metrics_path):
    st.subheader("Model Metrics")
    with open(metrics_path, "r", encoding="utf-8") as f:
        st.text(f.read())