import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # pandas handles the decompression automatically if you keep the .zip extension
    df = pd.read_pickle("dl_final/dataset.pkl.zip")
    return df

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dl_final/model_1_base_LSTM.keras")

def main():
    st.title("SOH Prediction (Pickle Optimized)")
    
    # Loading
    df = load_data()
    model = load_model()
    
    # --- PREDICTION LOGIC (Window 50) ---
    # To keep the app fast, let's look at the last 500 rows 
    # to find our windows of 50
    st.subheader("Recent Predictions")
    
    FEATURE_COLUMNS = [
        'mode', 'voltage_charger', 'temperature_battery', 
        'voltage_load', 'current_load', 'temperature_mosfet', 
        'temperature_resistor', 'mission_type'
    ]
    
    # Prepare the most recent window for a quick demo
    # We take the last 50 rows to predict the 50th value
    recent_data = df.tail(100) # taking 100 to show a small trend
    
    # Scale features (Note: Use the same scaler parameters as training!)
    # For this demo, we'll use a simple min-max on the current view
    features = recent_data[FEATURE_COLUMNS].values
    target = recent_data['SOH_percent'].values
    
    X, y_true = [], []
    window_size = 49 # Steps 1-49
    
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y_true.append(target[i+window_size])
        
    X = np.array(X)
    y_true = np.array(y_true)
    y_pred = model.predict(X).flatten()
    
    # --- METRICS & CHART ---
    mae = mean_absolute_error(y_true, y_pred)
    st.metric("MAE on Recent Data", f"{mae:.4f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, name="Actual SOH", line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=y_pred, name="Predicted SOH", line=dict(color='red', dash='dash')))
    
    fig.update_layout(title="SOH Time Series (Window-based Prediction)")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
