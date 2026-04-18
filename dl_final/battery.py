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



# --- PLOTTING & PREDICTION FUNCTION ---

def plot_battery_soh(df, model, selected_battery, battery_col):

    # 1. Filter data for the selected battery

    battery_df = df[df[battery_col] == selected_battery].copy()



    # 2. Validate we have enough data to create a window

    window_size = 49 # Steps 1-49

    if len(battery_df) <= window_size:

        st.warning(f"Not enough data to create a window of 50 for battery '{selected_battery}'.")

        return



    FEATURE_COLUMNS = [

        'voltage_charger', 'temperature_battery', 

        'voltage_load', 'current_load', 'temperature_mosfet', 

        'temperature_resistor', 'mission_type'

    ]

    

    # 3. Grab recent data for this specific battery (last 100 rows to keep it fast)

    recent_data = battery_df.tail(100)

    

    # 4. Prepare sequences

    features = recent_data[FEATURE_COLUMNS].values

    target = recent_data['SOH_percent'].values

    

    X, y_true = [], []

    

    for i in range(len(features) - window_size):

        X.append(features[i:i+window_size])

        y_true.append(target[i+window_size])

        

    X = np.array(X)

    y_true = np.array(y_true)

    

    # 5. Predict

    y_pred = model.predict(X).flatten()

    

    # 6. Metrics & UI Display

    mae = mean_absolute_error(y_true, y_pred)

    st.write(f"### SOH Forecast Accuracy Test")

    st.write(f"**Average Error (MAE):** {mae:.4f}")

    

    # 7. Plotting (Using Plotly to mirror your original style but with the requested UI flow)

    fig = go.Figure()

    

    steps = np.arange(1, len(y_true) + 1)

    

    fig.add_trace(go.Scatter(x=steps, y=y_true, mode='lines+markers', 

                             name="Actual SOH", line=dict(color='royalblue', width=2)))

    

    fig.add_trace(go.Scatter(x=steps, y=y_pred, mode='lines+markers', 

                             name="Predicted SOH", line=dict(color='darkorange', width=2, dash='dash')))

    

    # Add a background shaded region to mimic the "Prediction Test Zone" from your example

    fig.add_vrect(

        x0=steps[0], x1=steps[-1],

        fillcolor="orange", opacity=0.1,

        layer="below", line_width=0,

    )

    

    fig.update_layout(

        title=f"SOH Time Series Prediction - Battery {selected_battery}",

        xaxis_title="Prediction Step",

        yaxis_title="SOH Percent",

        hovermode="x unified"

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

        

    # --- CONFIGURE THIS ---

    # Change 'battery_id' to the actual name of your column that identifies batteries

    battery_col = 'battery_id' 

    

    if battery_col not in df.columns:

        st.error(f"Column '{battery_col}' not found. Please update `battery_col` in the script to match your dataframe's column for battery names/IDs.")

        st.write("Available columns are:", df.columns.tolist())

        return



    # UI Inputs

    available_batteries = df[battery_col].unique()

    selected_battery = st.selectbox("Select a Battery", available_batteries)

    

    if st.button("Run Forecast"):

        with st.spinner('Calculating predictions...'):

            plot_battery_soh(df, model, selected_battery, battery_col)



if __name__ == "__main__":

    main()
