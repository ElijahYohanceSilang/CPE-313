import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timedelta

# 1. Setup and Page Config
st.set_page_config(page_title="AC Power Predictor", layout="wide")
st.title("⚡ Smart AC Energy Monitor & Forecaster")

# 2. Load Model and Data
@st.cache_resource
def load_assets():
    model = joblib.load('your_model.pkl') 
    df = pd.read_csv('your_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    return model, df

model, data = load_assets()

# 3. Sidebar Controls
st.sidebar.header("Simulation Settings")
selected_date = st.sidebar.date_input("Select Simulation Start Date", datetime(2023, 1, 1))
prediction_window = st.sidebar.slider("Forecast Window (Hours)", 1, 24, 6)
simulate_btn = st.sidebar.button("Toggle AC Power")

# Initialize Session State
if "running" not in st.session_state:
    st.session_state.running = False

if simulate_btn:
    st.session_state.running = not st.session_state.running

# 4. Simulation Logic
if st.session_state.running:
    st.success(f"AC is currently RUNNING. Simulating from {selected_date}")
    
    # Placeholder for the dynamic chart
    chart_placeholder = st.empty()
    metric_placeholder = st.columns(2)
    
    # Filter data based on selected date
    sim_data = data[data['date'] >= pd.to_datetime(selected_date)].reset_index(drop=True)

    # Simulation Loop
    for i in range(len(sim_data)):
        if not st.session_state.running:
            break
            
        # Get "Current" state
        current_row = sim_data.iloc[i]
        actual_cons = current_row['ac']
        
        # Prepare Features for Prediction (matching your schema)
        # Assuming features are: hour, month, day_of_week
        features = np.array([[current_row['hour'], current_row['month'], current_row['day']]])
        
        # Make Prediction for the 'next' hour or a window
        prediction = model.predict(features)[0]

        # UI Updates
        with metric_placeholder[0]:
            st.metric("Current Consumption", f"{actual_cons} kWh")
        with metric_placeholder[1]:
            st.metric("Model Predicted", f"{round(prediction, 3)} kWh", 
                      delta=f"{round(prediction - actual_cons, 3)} vs Actual")

        # Visualizing the "Real-time" flow
        st.write(f"Timestamp: {current_row['time']} | Predicting next hour...")
        
        # Artificial delay to simulate real-time (1 second = 1 hour)
        time.sleep(1) 
else:
    st.warning("AC is currently OFF. Click 'Toggle AC Power' to start simulation.")
