import streamlit as st
import requests

st.title("🏠 House Price Prediction in India")

# User Inputs
city = st.selectbox("Select City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad"])
area = st.number_input("Area (sq. ft)", min_value=500, max_value=5000, value=1000)
bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
parking = st.selectbox("Parking Spaces", [0, 1, 2])
age = st.number_input("Age of Property (Years)", min_value=0, max_value=50, value=5)
furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Nearby Amenities
near_metro = st.checkbox("Near Metro Station")
near_mall = st.checkbox("Near Shopping Mall")
near_hospital = st.checkbox("Near Hospital")
near_school = st.checkbox("Near School")

if st.button("Predict Price"):
    data = {
        "city": city,
        "area": area,
        "bhk": bhk,
        "bathrooms": bathrooms,
        "parking": parking,
        "age": age,
        "furnishing": furnishing,
        "near_metro": int(near_metro),
        "near_mall": int(near_mall),
        "near_hospital": int(near_hospital),
        "near_school": int(near_school)
    }

    response = requests.post("http://127.0.0.1:5000/predict", json=data)

    if response.status_code == 200:
        price = response.json()["predicted_price"]
        st.success(f"🏡 Estimated Price: {price}")
    else:
        st.error("Error fetching prediction.")
