import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
import plotly.express as px

# Custom CSS to style the application
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Function to format the price into lakhs, crores, etc.
def format_price(pkr_price):
    if pkr_price >= 1e9:
        return f"{pkr_price / 1e9:.2f} Arab"
    elif pkr_price >= 1e7:
        return f"{pkr_price / 1e7:.2f} Crore"
    elif pkr_price >= 1e5:
        return f"{pkr_price / 1e5:.2f} Lakh"
    elif pkr_price >= 1e3:
        return f"{pkr_price / 1e3:.2f} Thousand"
    else:
        return f"{pkr_price:.2f} PKR"



with open('model_sale.pkl', 'rb') as f:
    model_sale = pickle.load(f)
with open('model_rent.pkl', 'rb') as f:
    model_rent = pickle.load(f)


# Load locations from JSON files
with open('locations_sale.json', 'r') as f:
    locations_sale = json.load(f)

with open('locations_rent.json', 'r') as f:
    locations_rent = json.load(f)

# Manually provide categories
cities_sale = ['Islamabad', 'Karachi', 'Lahore', 'Rawalpindi']
property_types = ['Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion']
purposes = ['For Sale', 'For Rent']
cities_rent = ['Islamabad', 'Karachi', 'Rawalpindi']

# User input
purpose = st.selectbox("Select Purpose", purposes)

if purpose == 'For Sale':
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("Select City", cities_sale)
    with col2:
        location = st.selectbox("Select Location", locations_sale)
    with col3:
        property_type = st.selectbox("Select Property Type", property_types)
    
    area_size = st.number_input("Area Size (in square feet)", min_value=0.0, format="%.2f")
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    baths = st.number_input("Baths", min_value=0, step=1)

    if st.button('Predict Sale Price'):
        columns = ['baths', 'bedrooms', 'Area Size'] + locations_sale + cities_sale + property_types + purposes
        input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
        input_df.at[0, 'baths'] = baths
        input_df.at[0, 'bedrooms'] = bedrooms
        input_df.at[0, 'Area Size'] = area_size
        input_df.at[0, location] = 1
        input_df.at[0, city] = 1
        input_df.at[0, property_type] = 1
        input_df.at[0, 'For Sale'] = 1

        predicted_price = model_sale.predict(input_df)[0]
        predicted_price = np.exp(predicted_price)
        formatted_price = format_price(predicted_price)
        st.write(f"The predicted price is: {formatted_price}")

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("Select City", cities_rent)
    with col2:
        location = st.selectbox("Select Location", locations_rent)
    with col3:
        property_type = st.selectbox("Select Property Type", property_types)
    
    area_size = st.number_input("Area Size (in square feet)", min_value=0.0, format="%.2f")
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    baths = st.number_input("Baths", min_value=0, step=1)

    if st.button('Predict Rent Price'):
        columns = ['baths', 'bedrooms', 'Area Size'] + locations_rent + cities_rent + property_types
        input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
        input_df.at[0, 'baths'] = baths
        input_df.at[0, 'bedrooms'] = bedrooms
        input_df.at[0, 'Area Size'] = area_size
        input_df.at[0, location] = 1
        input_df.at[0, city] = 1
        input_df.at[0, property_type] = 1

        predicted_price = model_rent.predict(input_df)[0]
        predicted_price = np.exp(predicted_price)
        formatted_price = format_price(predicted_price)
        st.write(f"The predicted price is: {formatted_price}")
