import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# Ignore scikit-learn version warnings for cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="AC Smart Monitor", layout="wide")

# --- DATA & MODEL LOADING ---

@st.cache_resource
def load_ml_model():
    """Loads the pre-trained machine learning model."""
    try:
        with open("Team18/team18_model.pkl", "rb") as file:
            model = joblib.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'team18_model.pkl' not found. Please ensure it is in the 'Team18' directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_historical_data():
    """Loads and formats the CSV dataset."""
    try:
        df = pd.read_csv("Team18/20k_energy.csv")
        # Combine date and time into a single Timestamp column for graphing
        df['Timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        # Sort by time just in case the CSV is out of order
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("Dataset '20k_energy.csv' not found. Please ensure it is in the 'Team18' directory.")
        return pd.DataFrame()

model = load_ml_model()
df_source = load_historical_data()


# --- STREAMLIT UI ---

st.title("❄️ Smart AC Energy Monitor & Predictor")
st.markdown("Monitoring historical data from `20k_energy.csv` and predicting future consumption using `team18_model.pkl`.")

# Sidebar Controls
st.sidebar.header("AC Controls & Settings")
ac_enabled = st.sidebar.toggle("Power (AC Enabled)", value=True)

# We use a time input to simulate a "live" environment from the historical data
target_date = st.sidebar.date_input("Select Date", datetime.date(2023, 1, 1))
target_time = st.sidebar.time_input("Select Current Time", datetime.time(12, 0))
ac_mode = st.sidebar.selectbox("AC Mode", ["Normal", "Chilling"])

st.sidebar.markdown("---")
st.sidebar.header("Prediction Settings")
prediction_hours = st.sidebar.slider("Prediction Horizon (Hours)", min_value=1, max_value=12, value=4)
# Updated to PHP with a realistic default Meralco/Local rate
kwh_price = st.sidebar.number_input("Electricity Price per kWh (₱)", min_value=1.00, value=12.00, step=0.50)

# Main Dashboard
if not ac_enabled:
    st.warning("The Air Conditioner is currently turned OFF. Turn it on in the sidebar to view metrics and predictions.")
elif df_source.empty:
    st.warning("Awaiting dataset to render dashboard.")
else:
    # --- 1. Process Historical Data ---
    current_simulated_time = datetime.datetime.combine(target_date, target_time)
    
    mask = df_source['Timestamp'] <= current_simulated_time
    past_data = df_source[mask].tail(12).copy()
    
    if past_data.empty:
        past_data = df_source.tail(12).copy()
        current_simulated_time = past_data.iloc[-1]['Timestamp']
        st.info(f"Selected date not in dataset. Defaulting to latest available data: {current_simulated_time}")

    current_kw = past_data.iloc[-1]['ac'] if not past_data.empty else 0.0

# --- 2. Generate Future Predictions ---
    future_timestamps = [current_simulated_time + datetime.timedelta(hours=i+1) for i in range(prediction_hours)]
    future_df = pd.DataFrame({'Timestamp': future_timestamps})
    
    future_df['hour'] = future_df['Timestamp'].dt.hour
    future_df['month'] = future_df['Timestamp'].dt.month
    future_df['day'] = future_df['Timestamp'].dt.dayofweek
    
    predicted_ac = []
    
    if model is not None:
        try:
            # Prepare features
            X_predict = future_df[['hour', 'month', 'day']]
            
            # Auto-align columns if model has saved feature names
            if hasattr(model, "feature_names_in_"):
                X_predict = future_df[list(model.feature_names_in_)]
                
            raw_predictions = model.predict(X_predict)
            
            # Apply AC Mode multiplier
            mode_multiplier = 1.6 if ac_mode == "Chilling" else 1.0
            predicted_ac = [max(0.01, p * mode_multiplier) for p in raw_predictions]
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            predicted_ac = [0.0] * prediction_hours
    else:
        # This handles the case where the model failed to load
        predicted_ac = [0.0] * prediction_hours

    future_df['ac'] = predicted_ac
    
    # Calculate Metrics
    # Because data is strictly hourly, sum(kW) == total kWh.
    predicted_total_kwh = sum(predicted_ac) 
    predicted_cost = predicted_total_kwh * kwh_price
    
    # --- Top Metrics Row ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Output", f"{current_kw:.2f} kW", f"{ac_mode} Mode")
    col2.metric(f"Predicted Usage ({prediction_hours}h)", f"{predicted_total_kwh:.2f} kWh")
    col3.metric(f"Estimated Cost ({prediction_hours}h)", f"₱{predicted_cost:.2f}", f"@ ₱{kwh_price}/kWh")
    
    st.markdown("---")
    
    # --- Graphing (Matplotlib) ---
    st.subheader("Hourly Energy Consumption Graph")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(
        past_data["Timestamp"], 
        past_data["ac"], 
        marker='o', 
        linestyle='-', 
        color='royalblue', 
        linewidth=2, 
        label='Historical (CSV Data)'
    )
    
    ax.plot(
        future_df["Timestamp"], 
        future_df["ac"], 
        marker='o', 
        linestyle='--', 
        color='tomato', 
        linewidth=2, 
        label='Predicted (team18_model)'
    )
    
    ax.set_title("Historical vs Predicted AC Usage")
    ax.set_xlabel("Time")
    ax.set_ylabel("Kilowatts (kW)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # --- AI Advice Module ---
    st.subheader("💡 Smart Advice")
    
    current_month = current_simulated_time.month
    season_text = "summer" if current_month in [3, 4, 5] else "rainy" if current_month in [6, 7, 8, 9, 10] else "cool"
    
    advice_msg = f"**Analysis:** Based on the current date, you are operating in a **{season_text}** pattern. "
    
    if ac_mode == "Chilling":
        normal_kwh = predicted_total_kwh / 1.6 
        savings = (predicted_total_kwh - normal_kwh) * kwh_price 
        
        advice_msg += f"Because you have the AC set to **Chilling**, your predicted power consumption is significantly higher. "
        advice_msg += f"\n\n**Recommendation:** If you switch back to **Normal** mode, you could save approximately **₱{savings:.2f}** over the next {prediction_hours} hours."
        st.warning(advice_msg)
    else:
        advice_msg += "You are currently running on **Normal** mode, which is optimal for energy savings. "
        advice_msg += f"Your projected cost of **₱{predicted_cost:.2f}** for the next {prediction_hours} hours aligns with standard predictive limits."
        st.success(advice_msg)
