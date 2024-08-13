import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np

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

# Load the model from a pickle file
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
    city = st.selectbox("Select City", cities_sale)
    location = st.selectbox("Select Location", locations_sale)
    property_type = st.selectbox("Select Property Type", property_types)
    area_size = st.number_input("Area Size", min_value=0.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    baths = st.number_input("Baths", min_value=0, step=1)

    # Create an empty DataFrame with the required columns
    columns = ['baths', 'bedrooms', 'Area Size'] + locations_sale + cities_sale + property_types + purposes
    model_columns = model_sale.feature_names_in_  # Assuming this attribute is available
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Set user input values in the DataFrame
    input_df['baths'] = baths
    input_df['bedrooms'] = bedrooms
    input_df['Area Size'] = area_size
    input_df[location] = 1
    input_df[city] = 1
    input_df[property_type] = 1
    input_df['For Sale'] = 1

    # Add a button to trigger prediction
    if st.button('Predict Price'):
        # Predict price
        predicted_price = model_sale.predict(input_df)[0]
        predicted_price = np.exp(predicted_price)

        # Format the price
        formatted_price = format_price(predicted_price)
        st.write(f"The predicted price is: {formatted_price}")

else:
    city = st.selectbox("Select City", cities_rent)
    location = st.selectbox("Select Location", locations_rent)
    property_type = st.selectbox("Select Property Type", property_types)
    area_size = st.number_input("Area Size", min_value=0.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    baths = st.number_input("Baths", min_value=0, step=1)

    # Create an empty DataFrame with the required columns
    columns = ['baths', 'bedrooms', 'Area Size'] + locations_rent + cities_rent + property_types
    model_columns = model_rent.feature_names_in_  # Assuming this attribute is available
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Set user input values in the DataFrame
    input_df['baths'] = baths
    input_df['bedrooms'] = bedrooms
    input_df['Area Size'] = area_size
    input_df[location] = 1
    input_df[city] = 1
    input_df[property_type] = 1

    # Add a button to trigger prediction
    if st.button('Predict Price'):
        # Predict price
        predicted_price = model_rent.predict(input_df)[0]
        predicted_price = np.exp(predicted_price)

        # Format the price
        formatted_price = format_price(predicted_price)
        st.write(f"The predicted price is: {formatted_price}")
