import joblib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
import requests
import gdown


# Google Drive direct download link
url = f"https://drive.google.com/uc?export=download&id=1BlMv64th03CnAw_0mrSH2AuE3PBvFReF"

output = "final_rf_model.pkl"
gdown.download(url, output, quiet=False)

# Load the model
model = joblib.load(output)

## App title
st.title("Airline Ticket Price Prediction")

st.markdown("""
Welcome! This app predicts flight ticket prices based on your chosen flight details.  
It demonstrates how machine learning can support **dynamic pricing** and **customer planning**.
""")

## Section 1: Flight Details
st.header("Flight Details")
airline_selected = st.radio("Airline", ['Air India', 'Vistara', 'Indigo', 'SpiceJet', 'GO FIRST', 'AirAsia'], horizontal=True)
source_selected = st.selectbox("Source City", ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Hyderabad'])
destination_selected = st.selectbox("Destination City", ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Hyderabad'])

# Validation: source and destination cannot be the same
invalid_source_dest = source_selected == destination_selected
if invalid_source_dest:
    st.warning("Source and Destination cannot be the same!")

## Section 2: Timing
st.header("Flight Timing")
departure_selected = st.radio("Departure Time", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], horizontal=True)
arrival_selected = st.radio("Arrival Time", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], horizontal=True)

## Section 3: Other Factors
st.header("Other Factors")

# Stops as clickable buttons with persistent state
st.markdown("Number of Stops (Please select ONE)")
stop_options = ["Non-stop", "One stop", "Two or more"]
if "selected_stop" not in st.session_state:
    st.session_state.selected_stop = None

cols = st.columns(len(stop_options))
for i, opt in enumerate(stop_options):
    if cols[i].button(opt, use_container_width=True):
        st.session_state.selected_stop = opt

# Highlight selected stop
if st.session_state.selected_stop:
    st.info(f"Selected Stops: {st.session_state.selected_stop}")

duration_selected = st.slider("Flight Duration (hours)", min_value=1, max_value=50, value=1)
days_left_selected = st.slider("Days Left Before Departure", min_value=1, max_value=50, value=1)

## Prediction
if st.button("Predict Ticket Price"):
    # Validation checks
    if invalid_source_dest:
        st.error("Cannot predict: Source and Destination must be different!")
    elif not st.session_state.selected_stop:
        st.error("Please select the number of stops!")
    else:
        # Calculate days left
        today = date.today()
        days_left = days_left_selected  # already a slider value

        input_data = {
            'airline': airline_selected,
            'source_city': source_selected,
            'destination_city': destination_selected,
            'departure_time': departure_selected,
            'arrival_time': arrival_selected,
            'stops': st.session_state.selected_stop,
            'duration': duration_selected,
            'days_left': days_left
        }

        df_input = pd.DataFrame([input_data])
        df_input = pd.get_dummies(df_input)
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = model.predict(df_input)[0]

        # Modal popup
        @st.dialog("Prediction Result")
        def show_modal():
            st.markdown(f"### Predicted Ticket Price: ₹{y_pred:,.2f}")
            st.write("**Inputs used:**")
            st.json(input_data)

        show_modal()

## Page Design
st.markdown(
    """
    <style>
    .stApp {
        font-family: 'Arial';
    }
    h1, h2 {
        color: #2e7d32;
    }
    </style>
    """,
    unsafe_allow_html=True
)