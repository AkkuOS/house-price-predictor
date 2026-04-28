import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stNumberInput input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.write("### Enter property details below")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    area = st.slider("Area (sq ft)", 500, 10000, 1500)
    bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5])

with col2:
    bathrooms = st.selectbox("Bathrooms", [1,2,3,4])
    stories = st.selectbox("Stories", [1,2,3])

# Divider
st.markdown("---")

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, stories]])
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")

# Sidebar
st.sidebar.header("📘 About")
st.sidebar.write("""
This app predicts house prices using a Machine Learning model.

**Features used:**
- Area
- Bedrooms
- Bathrooms
- Stories

Built with Streamlit.
""")
