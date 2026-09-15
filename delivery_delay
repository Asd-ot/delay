%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Load the saved logistic regression model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the delivery details below to predict if a delay will occur:')

# Create input fields for all 11 features
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1-3)', min_value=1, max_value=3, value=1)
delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
driver_experience = st.number_input('Driver Experience (Years)', min_value=0, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, value=2)
vehicle_age = st.number_input('Vehicle Age (Years)', min_value=0, value=3)
road_condition_score = st.slider('Road Condition Score (1-5)', min_value=1, max_value=5, value=3)
package_weight = st.number_input('Package Weight', min_value=0.0, value=120.0)
fuel_efficiency = st.number_input('Fuel Efficiency', min_value=0.0, value=12.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (Mins)', min_value=0, value=120)

# Make prediction when the user clicks the button
if st.button('Predict Delay'):
    # Prepare input data as a DataFrame with matching column names
    input_data = pd.DataFrame([[
        delivery_distance, traffic_congestion, weather_condition, delivery_slot,
        driver_experience, num_stops, vehicle_age, road_condition_score,
        package_weight, fuel_efficiency, warehouse_processing_time
    ]], columns=[
        'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
        'Warehouse_Processing_Time'
    ])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f'⚠️ Delivery Delayed (Probability: {probability:.2%})')
    else:
        st.success(f'✅ Delivery On Time (Probability of Delay: {probability:.2%})')
