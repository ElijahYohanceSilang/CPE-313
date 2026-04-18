import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import datetime

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # pandas handles decompression automatically
    df = pd.read_pickle("dl_final/dataset.pkl.zip")
    return df

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dl_final/model_1_base_LSTM.keras")

# --- PLOTTING & PREDICTION FUNCTION ---
def plot_battery_soh(df, model, selected_battery, battery_col, date_col, selected_date):
    # 1. Filter data for the selected battery and date
    battery_df = df[df[battery_col] == selected_battery].copy()
    
    # Ensure we are comparing date objects
    battery_df = battery_df[battery_df[date_col].dt.date == selected_date]
    
    # Sort chronologically
    battery_df = battery_df.sort_values(by=date_col)

    window_size = 50 # Model expects 50 timesteps
    
    if len(battery_df) < window_size:
        st.warning(f"Not enough data on {selected_date} to create a 50-step window (Found {len(battery_df)} rows).")
        return

    FEATURE_COLUMNS = [
        'voltage_charger', 'temperature_battery', 
        'voltage_load', 'current_load', 'temperature_mosfet', 
        'temperature_resistor', 'mission_type'
    ]
    
    X, y_true, plot_times = [], [], []
    
    # 2. Group by Hour
    # We use the actual column name passed via date_col
    for hour, group in battery_df.groupby(battery_df[date_col].dt.hour):
        if len(group) >= window_size:
            # Grab the first 50-second window available in this hour
            window_df = group.iloc[:window_size]
            features = window_df[FEATURE_COLUMNS].values
            
            # Actual SOH at the end of this window
            target = window_df['SOH_percent'].iloc[-1] 
            window_time = window_df[date_col].iloc[-1]
            
            X.append(features)
            y_true.append(target)
            plot_times.append(window_time)
            
    if not X:
        st.warning(f"No contiguous 50-step windows found for {selected_date}.")
        return

    X = np.array(X)
    y_true = np.array(y_true)
    
    # 3. Predict
    y_pred = model.predict(X).flatten()
    
    # 4. UI Display
    mae = mean_absolute_error(y_true, y_pred)
    st.write("### SOH Forecast Accuracy (Hourly Sample)")
    st.metric("Mean Absolute Error", f"{mae:.4f}")
    
    # 5. Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_times, y=y_true, mode='lines+markers', 
        name="Actual SOH", line=dict(color='royalblue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=plot_times, y=y_pred, mode='lines+markers', 
        name="Predicted SOH", line=dict(color='darkorange', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"Hourly SOH: Battery {selected_battery} ({selected_date})",
        xaxis_title="Time of Day",
        yaxis_title="SOH %",
        hovermode="x unified",
        xaxis=dict(tickformat="%H:%M")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Battery SOH Tracker", layout="wide")
    st.title("🔋 Battery SOH Prediction App")
    
    try:
        df = load_data()
        model = load_model()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return
        
    # --- UPDATED COLUMN NAMES ---
    battery_col = 'battery_id' 
    date_col = 'start_time'    
    
    if date_col not in df.columns or battery_col not in df.columns:
        st.error(f"Required columns not found. Check if '{date_col}' and '{battery_col}' exist.")
        st.write("Columns in data:", df.columns.tolist())
        return

    # Ensure datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Sidebar Controls
    st.sidebar.header("Filter Settings")
    available_batteries = df[battery_col].unique()
    selected_battery = st.sidebar.selectbox("Select Battery ID", available_batteries)
    
    # Filter dates for selected battery
    battery_data = df[df[battery_col] == selected_battery]
    available_dates = battery_data[date_col].dt.date.unique()
    available_dates.sort()
    
    selected_date = st.sidebar.date_input(
        "Select Date", 
        value=available_dates[0],
        min_value=min(available_dates),
        max_value=max(available_dates)
    )
    
    if st.sidebar.button("Generate Forecast"):
        with st.spinner('Analyzing time-series data...'):
            plot_battery_soh(df, model, selected_battery, battery_col, date_col, selected_date)
    else:
        st.info("Select a battery and date from the sidebar, then click 'Generate Forecast'.")

if __name__ == "__main__":
    main()
