import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    df = pd.read_pickle("dl_final/dataset.pkl.zip")
    return df

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dl_final/model_1_base_LSTM.keras")

# --- PLOTTING & PREDICTION FUNCTION ---
def plot_single_window_forecast(window_df, model):
    FEATURE_COLUMNS = [
        'voltage_charger', 'temperature_battery', 
        'voltage_load', 'current_load', 'temperature_mosfet', 
        'temperature_resistor', 'mission_type'
    ]
    
    # 1. Split into 49 inputs and the 50th target
    X_raw = window_df[FEATURE_COLUMNS].iloc[:49].values
    y_true_history = window_df['SOH_percent'].iloc[:49].values
    y_true_target = window_df['SOH_percent'].iloc[49]
    
    # 2. Reshape for LSTM (Batch size: 1, Timesteps: 49, Features: 7)
    X = np.expand_dims(X_raw, axis=0)
    
    # 3. Predict the 50th step
    y_pred = model.predict(X).flatten()[0]
    
    # 4. Metrics display
    error = abs(y_true_target - y_pred)
    st.write(f"### SOH Single Window Forecast")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual SOH (Step 50)", f"{y_true_target:.4f}%")
    col2.metric("Predicted SOH", f"{y_pred:.4f}%")
    col3.metric("Absolute Error", f"{error:.4f}")
    
    # 5. Plotting
    fig = go.Figure()
    
    # X-axis relative steps (1 to 50)
    steps = np.arange(1, 51)
    
    # Plot the 49 steps of history (the input to the model)
    fig.add_trace(go.Scatter(
        x=steps[:49], y=y_true_history, 
        mode='lines+markers', 
        name="Actual History (Input)", 
        line=dict(color='royalblue', width=2)
    ))
    
    # Plot the actual 50th step
    fig.add_trace(go.Scatter(
        x=[50], y=[y_true_target], 
        mode='markers', 
        name="Actual SOH (Target)", 
        marker=dict(color='green', size=10, symbol='circle')
    ))
    
    # Plot the predicted 50th step
    fig.add_trace(go.Scatter(
        x=[50], y=[y_pred], 
        mode='markers', 
        name="Predicted SOH", 
        marker=dict(color='darkorange', size=14, symbol='star')
    ))
    
    # Highlight the final step visually
    fig.add_vrect(
        x0=49.5, x1=50.5,
        fillcolor="orange", opacity=0.1,
        layer="below", line_width=0,
    )
    
    fig.update_layout(
        title="Model Prediction: 49 Steps Lookback -> 1 Step Forecast",
        xaxis_title="Step within Window",
        yaxis_title="SOH Percent",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)


# --- MAIN APP ---
def main():
    st.title("Battery SOH Single-Window Predictor")
    
    try:
        df = load_data()
        model = load_model()
    except Exception as e:
        st.error(f"Error loading data or model: {e}")
        return
        
    # Set to your actual battery identifier column name
    battery_col = 'battery_id' 
    
    if battery_col not in df.columns:
        st.error(f"Column '{battery_col}' not found. Please update the script.")
        return

    # 1. Select Battery
    available_batteries = df[battery_col].unique()
    selected_battery = st.selectbox("Select a Battery", available_batteries)
    
    # 2. Filter dataset
    battery_df = df[df[battery_col] == selected_battery].copy()
    
    window_size = 50
    max_start_idx = len(battery_df) - window_size
    
    if max_start_idx < 0:
        st.warning(f"Not enough data for a full {window_size}-step window for this battery.")
        return
        
    # 3. Select Range (Where the window begins)
    start_idx = st.slider(
        "Slide to select the start of the 50-step window:", 
        min_value=0, 
        max_value=max_start_idx, 
        value=max_start_idx, # Default to the most recent data block
        help="0 is the oldest data. Sliding right moves the window to more recent data."
    )
    
    # 4. Execute
    if st.button("Run Forecast"):
        with st.spinner('Analyzing window and predicting...'):
            # Slice exactly 50 rows based on the slider selection
            window_df = battery_df.iloc[start_idx : start_idx + window_size]
            plot_single_window_forecast(window_df, model)

if __name__ == "__main__":
    main()
