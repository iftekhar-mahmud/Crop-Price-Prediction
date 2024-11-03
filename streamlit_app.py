import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime
from datetime import datetime as dt
from geopy.geocoders import Nominatim
import pydeck as pdk

# Load the CSV data
data = pd.read_csv('https://raw.githubusercontent.com/iftekhar-mahmud/Crop-Price-Prediction/refs/heads/master/Data/CombinedDataset.csv')

# Preprocess the data
data = data.dropna()
data.dropna(subset=['R Average Price', 'W Average Price'], inplace=True)
data['R Average Price'] = data['R Average Price'].str.replace(',', '')
data['W Average Price'] = data['W Average Price'].str.replace(',', '')
data['R Average Price'] = pd.to_numeric(data['R Average Price'], errors='coerce')
data['W Average Price'] = pd.to_numeric(data['W Average Price'], errors='coerce')
data['R Average Price'] = data['R Average Price'].interpolate(method='linear')
data['W Average Price'] = data['W Average Price'].interpolate(method='linear')

commodity_names = data['Commodity Group'].unique()

# Outlier removal using IQR
Q1 = data.select_dtypes(include=np.number).quantile(0.25)
Q3 = data.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1
data_cleaned_iqr = data[~((data.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) | (data.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)]

# Streamlit UI
st.title("Crop Price Prediction App")

selected_commodity = st.selectbox("Select Commodity:", commodity_names)
price_type = st.radio("Choose Price Type:", ('Retail', 'Wholesale'))
target = 'R Average Price' if price_type == 'Retail' else 'W Average Price'
predictors = ['Year', 'Month', 'Week', 'Division', 'District', 'Upazila', 'Market Name']
if price_type == 'Retail':
    predictors.insert(0, 'W Average Price')

# Set `OneHotEncoder` to ignore unknown categories
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numeric_transformer = 'passthrough'

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, ['Month', 'Division', 'District', 'Upazila', 'Market Name']),
        ('num', numeric_transformer, [col for col in predictors if col not in ['Month', 'Division', 'District', 'Upazila', 'Market Name']])
    ])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor())
])

model_dict = {}
for commodity in commodity_names:
    commodity_data = data_cleaned_iqr[data_cleaned_iqr['Commodity Group'] == commodity]
    X = commodity_data[predictors]
    y = commodity_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    model_dict[commodity] = model_pipeline

selected_model = model_dict[selected_commodity]

selected_date = st.date_input("Select Date:", value=datetime.date.today())
selected_year = selected_date.year
selected_month = selected_date.month
selected_week = selected_date.isocalendar()[1]

selected_division = st.selectbox('Select Division:', data['Division'].unique())
districts = data[data['Division'] == selected_division]['District'].unique()
selected_district = st.selectbox('Select District:', districts)
upazilas = data[data['District'] == selected_district]['Upazila'].unique()
selected_upazila = st.selectbox('Select Upazila:', upazilas)
markets = data[data['Upazila'] == selected_upazila]['Market Name'].unique()
selected_market_name = st.selectbox('Select Market Name:', markets)

st.write(f"Selected Year: {selected_year}, Month: {selected_month}, Week: {selected_week}")

if st.button('Forecast Price'):
    historical_data = data_cleaned_iqr[
        (data_cleaned_iqr['Commodity Group'] == selected_commodity) &
        (data_cleaned_iqr['Division'] == selected_division) &
        (data_cleaned_iqr['District'] == selected_district) &
        (data_cleaned_iqr['Upazila'] == selected_upazila)
    ]

    if not historical_data.empty:
        w_average_price = historical_data['W Average Price'].mean()
        w_average_price = float(w_average_price)  # Ensure it's a float

        future_data = pd.DataFrame({
            'Year': [int(selected_year)],
            'Month': [int(selected_month)],
            'Week': [int(selected_week)],
            'Division': [selected_division],
            'District': [selected_district],
            'Upazila': [selected_upazila],
            'Market Name': [selected_market_name]
        })

        if price_type == 'Retail':
            future_data['W Average Price'] = w_average_price

        st.write("Future Data for Prediction:")
        st.write(future_data)

        # Check for NaN values in future_data before prediction
        if future_data.isnull().values.any():
            st.error("Error: Future data contains NaN values. Please check your inputs.")
        else:
            try:
                forecast_price = selected_model.predict(future_data)
                st.success(f"Forecasted {price_type} Price: {forecast_price[0]:.2f}")
            except Exception as e:
                st.error(f"Error predicting price: {str(e)}")
    else:
        st.error("No historical data found for the selected location and commodity.")



# Geolocation fix for "Chattogram"
geolocator = Nominatim(user_agent="crop_price_predictor")

division_input = "Chittagong" if selected_division in ["Chattogram", "Chattagram"] else selected_division

try:
    location = geolocator.geocode(f"{division_input}, {selected_district}, {selected_upazila}", exactly_one=True)
    if location:
        st.write(f"Latitude: {location.latitude}, Longitude: {location.longitude}")
        
        df_location = pd.DataFrame({'lat': [location.latitude], 'lon': [location.longitude]})
        layer = pdk.Layer(
            'ScatterplotLayer',
            df_location,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=500,
        )

        view_state = pdk.ViewState(
            longitude=location.longitude,
            latitude=location.latitude,
            zoom=11,
            pitch=50,
        )

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
    else:
        st.error("Location could not be found. Please check your inputs.")

except Exception as e:
    st.error(f"Error occurred while fetching location: {str(e)}")
