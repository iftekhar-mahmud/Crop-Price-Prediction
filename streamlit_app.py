import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime
from datetime import datetime as dt
from geopy.geocoders import Nominatim
import pydeck as pdk

# Load CSV data
data = pd.read_csv('https://raw.githubusercontent.com/iftekhar-mahmud/Crop-Price-Prediction/refs/heads/master/Data/CombinedDataset.csv')

# Clean & preprocess
data = data.dropna()
data.dropna(subset=['R Average Price', 'W Average Price'], inplace=True)
data['R Average Price'] = pd.to_numeric(data['R Average Price'].str.replace(',', ''), errors='coerce')
data['W Average Price'] = pd.to_numeric(data['W Average Price'].str.replace(',', ''), errors='coerce')
data['R Average Price'] = data['R Average Price'].interpolate(method='linear')
data['W Average Price'] = data['W Average Price'].interpolate(method='linear')

commodity_names = data['Commodity Group'].unique()

# Outlier removal (IQR)
Q1 = data.select_dtypes(include=np.number).quantile(0.25)
Q3 = data.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1
data_cleaned_iqr = data[~((data.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) |
                          (data.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)]

# UI
# UI
st.title("Crop Price Prediction App")

st.sidebar.header("About Me")
st.sidebar.write("""**Iftekhar Mahmud**
                 https://iftekhar-mahmud.github.io/""")
st.sidebar.header("Publications")
st.sidebar.write("""
**1. Crop Price Prediction**

I. Mahmud, P. R. Das, M. H. Rahman, A. R. Hasan, K. I. Shahin, and D. M. Farid,  
"Predicting Crop Prices using Machine Learning Algorithms for Sustainable Agriculture,"  
2024 IEEE Region 10 Symposium (TENSYMP), New Delhi, India, 2024, pp. 1–6.  
[DOI: 10.1109/TENSYMP61132.2024.10752263](https://doi.org/10.1109/TENSYMP61132.2024.10752263)

---

**2. Multimodal Emotion Recognition**

I. Mahmud, P. Das, N. Rifa, I. Hossain, R. Rahman, and D. M. Farid,  
"Multimodal Emotion Recognition Using Visual and Thermal Image Fusion: A Deep Learning Approach,"  
27th International Conference on Computer and Information Technology (ICCIT), 2024.  
[DOI: 10.1109/ICCIT64611.2024.11022356](https://doi.org/10.1109/ICCIT64611.2024.11022356)

**Award:** Best Technical Presentation Award, IEEE ICCIT 2024
""")

selected_commodity = st.selectbox("Select Commodity:", commodity_names)
price_type = st.radio("Choose Price Type:", ('Retail', 'Wholesale'))
target = 'R Average Price' if price_type == 'Retail' else 'W Average Price'

# Define predictors based on price type
predictors = ['Year', 'Month', 'Week', 'Division', 'District', 'Upazila', 'Market Name']
if price_type == 'Retail':
    predictors.insert(0, 'W Average Price')  # Retail model needs wholesale price

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

# Train models
model_dict = {}
predictors_dict = {}
for commodity in commodity_names:
    # Define predictors for this commodity
    predictors = ['Year', 'Month', 'Week', 'Division', 'District', 'Upazila', 'Market Name']
    if price_type == 'Retail':
        predictors.insert(0, 'W Average Price')
    predictors_dict[commodity] = predictors

    # Prepare data
    commodity_data = data_cleaned_iqr[data_cleaned_iqr['Commodity Group'] == commodity]
    X = commodity_data[predictors]
    y = commodity_data[target]

    # Build a new pipeline for each commodity
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    model_dict[commodity] = model_pipeline
    predictors_dict[commodity] = predictors  # Store predictors for this commodity

selected_model = model_dict[selected_commodity]
selected_predictors = predictors_dict[selected_commodity]

# Inputs
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

# Prediction
if st.button('Forecast Price'):
    historical_data = data_cleaned_iqr[
        (data_cleaned_iqr['Commodity Group'] == selected_commodity) &
        (data_cleaned_iqr['Division'] == selected_division) &
        (data_cleaned_iqr['District'] == selected_district) &
        (data_cleaned_iqr['Upazila'] == selected_upazila)
    ]

    if not historical_data.empty:
        # Create future_data
        future_data = pd.DataFrame({
            'Year': [int(selected_year)],
            'Month': [int(selected_month)],
            'Week': [int(selected_week)],
            'Division': [selected_division],
            'District': [selected_district],
            'Upazila': [selected_upazila],
            'Market Name': [selected_market_name]
        })

        # Add W Average Price if needed
        if 'W Average Price' in selected_predictors:
            w_avg = historical_data['W Average Price'].mean()
            future_data['W Average Price'] = float(w_avg) if not pd.isna(w_avg) else 0.0

        # Ensure column order matches training
        future_data = future_data[selected_predictors]

        # Ensure correct dtypes for numeric columns
        numeric_cols = ['Year', 'Month', 'Week']
        if 'W Average Price' in future_data.columns:
            numeric_cols.append('W Average Price')
        for col in numeric_cols:
            if col in future_data.columns:
                future_data[col] = pd.to_numeric(future_data[col], errors='coerce')

        # Ensure correct dtypes for categorical columns
        categorical_cols = ['Month', 'Division', 'District', 'Upazila', 'Market Name']
        for col in categorical_cols:
            if col in future_data.columns:
                future_data[col] = future_data[col].astype(str)

        # Debug
        st.write("Future Data for Prediction:")
        st.write(future_data)
        st.write("Data types in future_data:")
        st.write(future_data.dtypes)

        try:
            forecast_price = selected_model.predict(future_data)
            st.success(f"Forecasted {price_type} Price: {forecast_price[0]:.2f}")
        except Exception as e:
            st.error(f"Error predicting price: {str(e)}")
    else:
        st.error("No historical data found for the selected location and commodity.")

# Map
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
