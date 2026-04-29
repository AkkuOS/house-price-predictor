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
.stApp {
    background-image: url("https://images.unsplash.com/photo-1505691938895-1758d7feb511");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Dark overlay to reduce image visibility */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(14, 17, 23, 0.9);  /* 👈 50% opacity */
    z-index: -1;
}

/* Button styling */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.write("### Enter property details below")
st.image("https://images.unsplash.com/photo-1560185127-6ed189bf02f4")
tab1, tab2 = st.tabs(["🏠 Predict", "📊 About"])

with tab1:
    # Layout using columns
    col1, col2 = st.columns(2)

    with col1:
        area = st.slider("Area (sq ft)", 500, 10000, 1500)
        bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5])

    with col2:
        bathrooms = st.selectbox("Bathrooms", [1,2,3,4])
        stories = st.selectbox("Stories", [1,2,3])

    st.markdown("---")

    if st.button("Predict Price"):
        with st.spinner("Predicting..."):
            input_data = np.array([[area, bedrooms, bathrooms, stories]])
            prediction = model.predict(input_data)[0]

        st.metric("💰 Estimated Price", f"₹ {prediction:,.0f}")


with tab2:
    st.header("📊 About This Model")

    st.write("""
    This app predicts house prices using a Machine Learning model.

    **Features used:**
    - Area
    - Bedrooms
    - Bathrooms
    - Stories

    **Model:**
    Linear Regression

    **Author:**
    Abhijith Os
    """)
