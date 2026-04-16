import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="LSTM SOH Predictor", layout="wide")

# The features your model was trained on. Adjust if needed.
# Excluding 'start_time', 'time', 'battery_id', 'cycle_id', and 'SOH_percent' by default
FEATURE_COLUMNS = [
    'mode', 'voltage_charger', 'temperature_battery', 
    'voltage_load', 'current_load', 'temperature_mosfet', 
    'temperature_resistor', 'mission_type'
]
TARGET_COLUMN = 'SOH_percent'
WINDOW_SIZE = 50
SEQ_LENGTH = WINDOW_SIZE - 1  # 49 steps to predict the 50th

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- SEQUENCE GENERATION ---
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

# --- MAIN APP ---
def main():
    st.title("🔋 Battery SOH Prediction via LSTM")
    st.write("This app uses a sliding window (steps 1-49) to predict the 50th step's State of Health (SOH).")

    # Load Model
    model_path = 'model_1_base_LSTM.keras'
    model = load_keras_model(model_path)
    
    if model is None:
        st.stop()
        
    st.success(f"Model `{model_path}` loaded successfully!")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your battery dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        with st.spinner("Processing data..."):
            # Load Data
            df = pd.read_csv(uploaded_file)
            
            # Ensure chronological order if necessary
            if 'time' in df.columns:
                df = df.sort_values(by=['battery_id', 'cycle_id', 'time']).reset_index(drop=True)

            st.write("### Dataset Preview", df.head())

            # Data Preprocessing
            # Note: If you saved a specific scaler (e.g., scaler.pkl) during training, 
            # you should load and use it here instead of fitting a new one.
            scaler_X = MinMaxScaler()
            
            try:
                features_scaled = scaler_X.fit_transform(df[FEATURE_COLUMNS])
                target = df[TARGET_COLUMN].values
                
                # Create Sequences (Window 50: 1-49 for X, 50th for y)
                X, y_true = create_sequences(features_scaled, target, SEQ_LENGTH)
                
                # Model Prediction
                y_pred = model.predict(X)
                
                # Flatten predictions if necessary
                if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()

                # Calculate Metrics
                mae = mean_absolute_error(y_true, y_pred)

                # --- DASHBOARD UI ---
                st.write("---")
                st.subheader("📊 Evaluation Metrics")
                st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}%")

                # --- VISUALIZATION ---
                st.subheader("📈 Time Series Prediction: Actual vs Predicted SOH")
                
                # Create a Plotly Figure
                fig = go.Figure()
                
                # Add Actual SOH trace
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y_true)), 
                    y=y_true, 
                    mode='lines', 
                    name='Actual SOH',
                    line=dict(color='blue', width=2)
                ))
                
                # Add Predicted SOH trace
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y_pred)), 
                    y=y_pred, 
                    mode='lines', 
                    name='Predicted SOH',
                    line=dict(color='red', width=2, dash='dot')
                ))

                fig.update_layout(
                    title="Actual vs Predicted SOH over Time Steps",
                    xaxis_title="Window Step (Time)",
                    yaxis_title="SOH (%)",
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                st.plotly_chart(fig, use_container_width=True)

            except KeyError as e:
                st.error(f"Missing expected column in dataset: {e}. Please check your CSV.")
            except ValueError as e:
                st.error(f"Shape mismatch error (usually caused by wrong number of features): {e}")

if __name__ == '__main__':
    main()
