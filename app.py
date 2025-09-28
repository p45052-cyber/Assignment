import streamlit as st
import joblib
import pandas as pd

# Load the scaler and the model
scaler = joblib.load('scaler_model.pkl')
model = joblib.load('linear_regression_model.pkl')

st.title('Flight Fare Prediction')

st.header('Enter Flight Details:')

# Define the custom ranges requested by the user
duration_min = 1
duration_max = 24
base_fare_min = 1500
base_fare_max = 40000
tax_surcharge_min = 200
tax_surcharge_max = 20000
days_before_departure_min = 1
days_before_departure_max = 100


duration = st.text_input(f'Duration (hrs) (Range: {duration_min} - {duration_max})', '')
base_fare = st.text_input(f'Base Fare (BDT) (Range: {base_fare_min} - {base_fare_max})', '')
tax_surcharge = st.text_input(f'Tax & Surcharge (BDT) (Range: {tax_surcharge_min} - {tax_surcharge_max})', '')
days_before_departure = st.text_input(f'Days Before Departure (Range: {days_before_departure_min} - {days_before_departure_max})', '')


if st.button('Predict'):
    # Check if any input field is empty
    if not duration or not base_fare or not tax_surcharge or not days_before_departure:
        st.error("Please fill in all the input fields.")
    else:
        try:
            # Convert input strings to numerical types with individual error handling
            try:
                duration_val = float(duration)
            except ValueError:
                st.error(f"Error converting Duration '{duration}' to a number.")
                st.stop()

            try:
                base_fare_val = float(base_fare)
            except ValueError:
                st.error(f"Error converting Base Fare '{base_fare}' to a number.")
                st.stop()

            try:
                tax_surcharge_val = float(tax_surcharge)
            except ValueError:
                st.error(f"Error converting Tax Surcharge '{tax_surcharge}' to a number.")
                st.stop()

            try:
                days_before_departure_val = int(days_before_departure)
            except ValueError:
                st.error(f"Error converting Days Before Departure '{days_before_departure}' to an integer.")
                st.stop()


            # Create a DataFrame with all original numerical columns, including the target with a dummy value
            input_data = pd.DataFrame([[duration_val, base_fare_val, tax_surcharge_val, 0, days_before_departure_val]],
                                      columns=['Duration (hrs)', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Total Fare (BDT)', 'Days Before Departure'])

            # Define the columns that were scaled in the original training data (including the target)
            # This list is used to ensure the input data has the same columns as the data the scaler was fitted on
            scaled_features_list = ['Duration (hrs)', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Total Fare (BDT)', 'Days Before Departure']


            # Scale the input features (including the dummy Total Fare BDT for scaler consistency)
            # The model will now receive all 5 features as it was trained on them
            scaled_input_data = scaler.transform(input_data[scaled_features_list])

            # Pass all 5 scaled features to the model for prediction
            predicted_fare = model.predict(scaled_input_data)

            # Display the prediction
            st.header('Predicted Total Fare (BDT):')
            st.write(f'{predicted_fare[0]:,.2f}')

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
