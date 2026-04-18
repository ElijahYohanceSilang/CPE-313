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
    # pandas handles the decompression automatically if you keep the .zip extension
    df = pd.read_pickle("dl_final/dataset.pkl.zip")
    return df

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dl_final/model_1_base_LSTM.keras")

# --- PLOTTING & PREDICTION FUNCTION ---
def plot_battery_soh(df, model, selected_battery, battery_col, start_time, selected_date):
    # 1. Filter data for the selected battery and date
    battery_df = df[df[battery_col] == selected_battery].copy()
    
    # Filter by selected date
    battery_df = battery_df[battery_df[start_time].dt.date == selected_date]
    
    # Sort chronologically just in case
    battery_df = battery_df.sort_values(by=start_time)

    window_size = 50 # 50 windows = 50 seconds
    
    if len(battery_df) <= window_size:
        st.warning(f"Not enough data on {selected_date} to create a 50-second window.")
        return

    FEATURE_COLUMNS = [
        'voltage_charger', 'temperature_battery', 
        'voltage_load', 'current_load', 'temperature_mosfet', 
        'temperature_resistor', 'mission_type'
    ]
    
    X, y_true, plot_times = [], [], []
    
    # 2. Group by Hour to get Hourly SOH predictions
    # We will extract a contiguous 50-second window for each hour
    for hour, group in battery_df.groupby(battery_df[date_col].dt.hour):
        if len(group) >= window_size:
            # Take the first 50 rows (50 seconds) of this hour to form the window
            window_df = group.iloc[:window_size]
            features = window_df[FEATURE_COLUMNS].values
            
            # Target SOH at the end of this 50-second window
            target = window_df['SOH_percent'].iloc[-1] 
            
            # Timestamp for the plot (using the end of the window)
            window_time = window_df[date_col].iloc[-1]
            
            X.append(features)
            y_true.append(target)
            plot_times.append(window_time)
            
    if len(X) == 0:
        st.warning(f"Could not find any full 50-second continuous blocks of data on {selected_date}.")
        return

    X = np.array(X)
    y_true = np.array(y_true)
    
    # 3. Predict SOH for these hourly windows
    y_pred = model.predict(X).flatten()
    
    # 4. Metrics & UI Display
    mae = mean_absolute_error(y_true, y_pred)
    st.write("### SOH Forecast Accuracy Test (Hourly Aggregation)")
    st.write(f"**Average Error (MAE):** {mae:.4f}")
    
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
    
    # Add background shade
    if len(plot_times) > 1:
        fig.add_vrect(
            x0=plot_times[0], x1=plot_times[-1],
            fillcolor="orange", opacity=0.1,
            layer="below", line_width=0,
        )
    
    fig.update_layout(
        title=f"Hourly SOH Prediction - Battery {selected_battery} on {selected_date}",
        xaxis_title="Time of Day",
        yaxis_title="SOH Percent",
        hovermode="x unified",
        xaxis=dict(tickformat="%H:%M") # Format x-axis nicely for hours
    )
    
    st.plotly_chart(fig, use_container_width=True)


# --- MAIN APP ---
def main():
    st.title("Battery SOH Prediction App")
    
    # Load resources
    try:
        df = load_data()
        model = load_model()
    except Exception as e:
        st.error(f"Error loading data or model: {e}")
        return
        
    # --- CONFIGURE COLUMNS ---
    battery_col = 'battery_id' # Update if your column is named differently
    date_col = 'timestamp'     # Update if your timestamp column is named differently
    
    missing_cols = [col for col in [battery_col, date_col] if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}. Please update `battery_col` or `date_col` in the script.")
        st.write("Available columns are:", df.columns.tolist())
        return

    # Ensure date column is proper datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # UI Inputs - Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        available_batteries = df[battery_col].unique()
        selected_battery = st.selectbox("Select a Battery", available_batteries)
    
    # Filter for dates specific to the chosen battery
    battery_dates = df[df[battery_col] == selected_battery][date_col].dt.date
    min_date, max_date = battery_dates.min(), battery_dates.max()
    
    with col2:
        selected_date = st.date_input(
            "Select a Date", 
            value=min_date, 
            min_value=min_date, 
            max_value=max_date
        )
    
    if st.button("Run Hourly Forecast"):
        with st.spinner(f'Calculating predictions for {selected_date}...'):
            plot_battery_soh(df, model, selected_battery, battery_col, date_col, selected_date)

if __name__ == "__main__":
    main()
