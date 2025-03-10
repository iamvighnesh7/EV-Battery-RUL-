import streamlit as st
import numpy as np
import joblib
import pickle

# Load trained model and scaler
with open("EV Battery Health(RUL).pkl", "rb") as f:
    model_data = pickle.load(f)
    
# Extract the actual model
model = model_data["model"]  # Now this is the trained model
feature_names = model_data["fetures_name"]  # Feature names


scaler = joblib.load("scaler.pkl")

st.title("🔋 EV Battery Health Prediction")
st.subheader("⚡ Predict the Remaining Useful Life (RUL) of an EV Battery")

# User input fields with attractive labels
charge_cycle = st.number_input("🔄 Charge Cycle", min_value=0.0, step=1.0, value=1.0)
discharge_time = st.number_input("⚡ Discharge Duration (s)", min_value=0.0, step=0.1, value=2595.3)
voltage_drop = st.number_input("🔋 Voltage Drop (3.6V-3.4V)", min_value=0.0, step=0.1, value=1151.4885)
max_voltage = st.number_input("🚀 Max Discharge Voltage (V)", min_value=0.0, step=0.01, value=3.67)
min_voltage = st.number_input("🔌 Min Charge Voltage (V)", min_value=0.0, step=0.01, value=3.211)
time_415v = st.number_input("⏳ Time at 4.15V (s)", min_value=0.0, step=0.1, value=5460.001)
constant_current_time = st.number_input("⚙️ Constant Current Time (s)", min_value=0.0, step=0.1, value=6755.01)
charging_time = st.number_input("⏱ Total Charging Time (s)", min_value=0.0, step=0.1, value=10777.82)

# Predict button
if st.button("🔍 Predict RUL"):
    # Prepare input data
    user_data = np.array([[charge_cycle, discharge_time, voltage_drop, max_voltage,
                           min_voltage, time_415v, constant_current_time, charging_time]])

    # Scale input data
    user_data_scaled = scaler.transform(user_data)

    # Predict RUL
    predicted_rul = model.predict(user_data_scaled)

    # Display result
    st.success(f"✅ Predicted Remaining Useful Life: {predicted_rul[0]:.2f} cycles")
