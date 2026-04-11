import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load model
model = joblib.load("team18_model.pkl")

def plot_20hr_accuracy_test(df, model, start_date_str):
    start_date = pd.to_datetime(start_date_str)
    mask = df.index >= start_date
    week_data = df.loc[mask].head(168).copy()

    if len(week_data) < 168:
        st.warning(f"Not enough data for a full 168-hour week starting from {start_date_str}.")
        return

    hour_numbers = np.arange(1, 169)
    y_true = week_data['ac'].values
    last_20_data = week_data.iloc[-20:]

    try:
        required_features = model.feature_names_in_
        X_last_20 = last_20_data[required_features]
    except AttributeError:
        X_last_20 = last_20_data[['season', 'house_id', 'hour', 'day', 'month']]

    y_pred_raw = model.predict(X_last_20)
    y_pred = np.clip(y_pred_raw, 0.1, 1.5)

    mae = mean_absolute_error(last_20_data['ac'], y_pred)
    st.write(f"### 20-Hour Forecast Accuracy Test")
    st.write(f"Average Error (MAE): {mae:.3f} kW")

    plt.figure(figsize=(15, 6))
    plt.plot(hour_numbers, y_true, color='royalblue', linewidth=1.5,
             marker='.', markersize=8, label='True Consumption (168 hours)')
    x_pred = hour_numbers[-20:]
    plt.plot(x_pred, y_pred, color='darkorange', linewidth=2.5,
             linestyle='--', marker='o', markersize=6, label='Predicted Consumption (Last 20 hours)')
    plt.axvspan(148.5, 168.5, color='orange', alpha=0.1, label='Prediction Test Zone')
    plt.title('Model Accuracy: 7-Day Lookback with 20-Hour Prediction Overlap')
    plt.xlabel('Elapsed Time (Hours: 1 to 168)')
    plt.ylabel('AC Usage (kW)')
    plt.ylim(0, 1.6)
    plt.xticks(np.arange(0, 169, 24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    st.pyplot(plt.gcf())

# Streamlit UI
st.title("Team 18 Energy Forecast App")

df = pd.read_csv("20k_energy.csv", parse_dates=['date'], index_col='date')
start_date_str = st.text_input("Enter start date (YYYY-MM-DD)", "2023-01-01")

if st.button("Run Forecast"):
    plot_20hr_accuracy_test(df, model, start_date_str)
