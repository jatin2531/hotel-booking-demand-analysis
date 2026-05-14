import streamlit as st
import pickle
import numpy as np

# Page settings
st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨",
    layout="centered"
)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>🏨 Hotel Booking Prediction</h1>",
    unsafe_allow_html=True
)

st.write("### Enter Booking Details")

# Inputs
lead_time = st.slider("Lead Time", 0, 365, 30)

adults = st.slider("Adults", 1, 10, 2)

week_nights = st.slider("Week Nights", 0, 15, 2)

adr = st.number_input("Average Daily Rate", min_value=0.0, value=100.0)

special_request = st.slider("Special Requests", 0, 5, 1)

# Predict button
if st.button("Predict Booking Status"):

    # 43 features
    features = np.array([[
        lead_time,                 # Lead Time
        adults,                    # Adults
        0,                         # Childrens
        adults,                    # Total Guests
        0,                         # Weekend Nights
        week_nights,               # Week Nights
        week_nights,               # Total Nights Stay
        0,                         # Customer Cancel Count
        adr,                       # Average Daily Rate
        special_request,           # Special Request
        0,                         # Room Change

        # Customer Country
        0,0,0,1,0,0,0,

        # Hotel
        1,

        # Cities
        0,0,0,0,0,0,0,0,1,0,0,0,0,0,

        # Meal Type
        0,1,

        # Market Segment
        0,0,0,1,

        # Reserved Room Type
        0,1,

        # Assigned Room Type
        0,1
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0]

    cancel_prob = round(probability[1] * 100, 2)

    confirm_prob = round(probability[0] * 100, 2)

    # Output
    if prediction == 1:

        st.error(f"❌ Booking Likely To Be Cancelled\n\nCancellation Probability: {cancel_prob}%")

    else:

        st.success(f"✅ Booking Likely To Be Confirmed\n\nConfirmation Probability: {confirm_prob}%")