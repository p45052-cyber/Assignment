import streamlit as st
import joblib
import pandas as pd

# Load the scaler and the model
scaler = joblib.load('scaler_model.pkl')
model = joblib.load('linear_regression_model.pkl')

st.title('Flight Fare Prediction')

st.header('Enter Flight Details:')

duration = st.text_input('Duration (hrs)', '')
base_fare = st.text_input('Base Fare (BDT)', '')
tax_surcharge = st.text_input('Tax & Surcharge (BDT)', '')
days_before_departure = st.text_input('Days Before Departure', '')

if st.button('Predict'):
    try:
        # Convert input strings to numerical types
        duration_val = float(duration)
        base_fare_val = float(base_fare)
        tax_surcharge_val = float(tax_surcharge)
        days_before_departure_val = int(days_before_departure)

        # Create a DataFrame with the same structure as the training data's features (excluding the target)
        input_data = pd.DataFrame([[duration_val, base_fare_val, tax_surcharge_val, days_before_departure_val]],
                                  columns=['Duration (hrs)', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Days Before Departure'])

        # Define the columns that were scaled in the original training data (excluding the target)
        scaled_features = ['Duration (hrs)', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Days Before Departure']

        # Select only the features from the input data for scaling
        input_features = input_data[scaled_features]

        # Scale the input features
        scaled_input_data = scaler.transform(input_features)

        # Make a prediction
        predicted_fare = model.predict(scaled_input_data)

        # Display the prediction
        st.header('Predicted Total Fare (BDT):')
        st.write(f'{predicted_fare[0]:,.2f}')

    except ValueError:
        st.error("Please enter valid numerical values for all input fields.")
