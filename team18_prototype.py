import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timedelta

# --- 1. SETUP ---
st.set_page_config(page_title="AI Energy Advisor", layout="wide")
st.title("🌬️ Smart AC Energy Advisor")

@st.cache_resource
def load_assets():
    model = joblib.load('team18_model.pkl') 
    data = pd.read_csv('20k_energy.csv')
    data['date'] = pd.to_datetime(data['date'])
    return model, data

model, data = load_assets()

# --- 2. SIDEBAR & SETTINGS ---
st.sidebar.header("🗓️ Simulation Settings")
selected_date = st.sidebar.date_input("Select Date", datetime(2023, 4, 15)) # Default to a hot month
forecast_hours = st.sidebar.slider("Forecast Window (Hours)", 1, 12, 8)
run_sim = st.sidebar.button("Enable Air Conditioner")

# --- 3. THE "ADVISOR" LOGIC ---
# We calculate the prediction for the selected window immediately
start_time_data = data[data['date'] == pd.to_datetime(selected_date)].iloc[0:forecast_hours]

if not start_time_data.empty:
    # Feature preparation for the window
    window_features = start_time_data[['hour', 'month', 'day']]

    predicted_values = model.predict(window_features)
    total_forecasted_kwh = round(np.sum(predicted_values), 2)
    
    # Advisor UI
    month_name = selected_date.strftime("%B")
    st.info(f"### 💡 Energy Advisor Recommendation")
    st.write(f"Based on historical usage for **{month_name}**, if you turn on the AC now:")
    st.metric("Estimated Consumption (Next {0} hrs)".format(forecast_hours), f"{total_forecasted_kwh} kWh")
    
    if month_name in ['April', 'May', 'June']:
        st.warning("⚠️ High usage expected due to summer heat in April. Consider setting the AC to 25°C to save energy.")
    else:
        st.success("❄️ Lower usage expected. The outdoor temperature is favorable today.")

# --- 4. REAL-TIME MONITORING GRAPH ---
st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live Power Consumption Monitor")
    chart_placeholder = st.empty()

with col2:
    st.subheader("📋 Status Log")
    status_placeholder = st.empty()

# --- 5. RUN SIMULATION ---
if run_sim:
    history = []
    
    for i in range(len(start_time_data)):
        current_row = start_time_data.iloc[i]
        
        # 1. Update History
        history.append({
            "Hour": f"{current_row['hour']}:00",
            "Actual Consumption (kWh)": current_row['ac'],
            "Predicted": predicted_values[i]
        })
        
        # 2. Update Graph
        chart_df = pd.DataFrame(history).set_index("Hour")
        chart_placeholder.area_chart(chart_df)
        
        # 3. Update Status
        status_placeholder.write(f"**Current Hour:** {current_row['hour']}:00")
        status_placeholder.write(f"**Hardware Feed:** {current_row['ac']} kWh")
        
        time.sleep(0.5) # Speed up for demo
    
    st.success("Simulation Complete for the selected window.")
else:
    st.write("Click **Enable Air Conditioner** to see the real-time power draw.")
