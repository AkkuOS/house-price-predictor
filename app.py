import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏠 House Price Predictor")

# User inputs
area = st.number_input("Area (sq ft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
stories = st.number_input("Stories")

# Prediction button
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, stories]])
    prediction = model.predict(input_data)
    
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")