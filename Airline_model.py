import streamlit as st
import joblib
import numpy as np


# PAGE CONFIG

st.set_page_config(
    page_title="Airline Satisfaction Prediction",
    page_icon="✈️",
    layout="centered"
)

# APP TITLE

st.title("Airline Satisfaction Prediction")
st.write("Streamlit app is working ✅")


# LOAD MODEL & FILES SAFELY

try:
    model = joblib.load("test.pkl")
    le1 = joblib.load("le1.pkl")   # Gender
    le2 = joblib.load("le2.pkl")   # Customer Type
    le3 = joblib.load("le3.pkl")   # Type of Travel
    le4 = joblib.load("le4.pkl")   # Class
    scaler = joblib.load("scaler.pkl")

    st.success("All model files loaded successfully")

except Exception as e:
    st.error("Error loading model files")
    st.exception(e)
    st.stop()


# USER INPUTS

Gender = st.selectbox("Gender", ["Male", "Female"])
Customer_Type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
Age = st.number_input("Age", min_value=1, max_value=100, value=25)
Type_of_Travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
Class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
Flight_Distance = st.number_input("Flight Distance", min_value=100, value=500)

Wifi = st.slider("Inflight wifi service", 0, 5, 3)
Food = st.slider("Food and drink", 0, 5, 3)
Seat = st.slider("Seat comfort", 0, 5, 3)
Cleanliness = st.slider("Cleanliness", 0, 5, 3)


# ENCODING

Gender = le1.transform([Gender])[0]
Customer_Type = le2.transform([Customer_Type])[0]
Type_of_Travel = le3.transform([Type_of_Travel])[0]
Class = le4.transform([Class])[0]


# INPUT ARRAY

input_data = np.array([[
    Gender,
    Customer_Type,
    Age,
    Type_of_Travel,
    Class,
    Flight_Distance,
    Wifi,
    Food,
    Seat,
    Cleanliness
]])


# SCALING

input_scaled = scaler.transform(input_data)


# PREDICTION

if st.button("Predict Satisfaction"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✈️ Passenger is  SATISFIED")
    else:
        st.error("⚠️ Passenger is NEUTRAL / DISSATISFIED")


        

        





