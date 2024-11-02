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

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('Data/Combined Dataset.csv')

# Preprocess the data (e.g., handle missing values, encode categorical variables)
data = data.dropna()

# Preprocess the data
data.dropna(subset=['R Average Price', 'W Average Price'], inplace=True)

# Remove commas from the 'R Average Price' and 'W Average Price' columns
data['R Average Price'] = data['R Average Price'].str.replace(',', '')
data['W Average Price'] = data['W Average Price'].str.replace(',', '')

# Convert the 'R Average Price' and 'W Average Price' columns to numeric
data['R Average Price'] = pd.to_numeric(data['R Average Price'], errors='coerce')
data['W Average Price'] = pd.to_numeric(data['W Average Price'], errors='coerce')

# Handle NaN values using interpolation
data['R Average Price'].interpolate(method='linear', inplace=True)
data['W Average Price'].interpolate(method='linear', inplace=True)

# Get the unique commodity names
commodity_names = data['Commodity Group'].unique()

# Data Cleaning: Detect and remove outliers using Interquartile Range (IQR)
Q1 = data.select_dtypes(include=np.number).quantile(0.25)
Q3 = data.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1
data_cleaned_iqr = data[~((data.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) | (data.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)]

# Define the target variable and predictor variables
target = 'R Average Price'
predictors = ['W Average Price', 'Year', 'Month', 'Week', 'Division', 'District', 'Upazila', 'Market Name']

# Encode categorical variables
categorical_cols = ['Month', 'Division', 'District', 'Upazila', 'Market Name']
numeric_cols = [col for col in predictors if col not in categorical_cols]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numeric_transformer = 'passthrough'

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)
    ])

# Create a pipeline for data preprocessing and model fitting
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor())
])

# Train the model for each commodity
model_dict = {}
for commodity in commodity_names:
    # Filter data for the current commodity
    commodity_data = data_cleaned_iqr[data_cleaned_iqr['Commodity Group'] == commodity]

    # Define the target and predictors
    X = commodity_data[predictors]
    y = commodity_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Store the model
    model_dict[commodity] = model_pipeline

    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{commodity} - Decision Tree Regression: R-squared = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}")

# Streamlit UI
st.title("Crop Price Prediction App")

# User inputs
selected_commodity = st.selectbox("Select Commodity:", commodity_names)

# Get the corresponding model for the selected commodity
selected_model = model_dict[selected_commodity]

# User input for prediction using date input
selected_date = st.date_input("Select Date:", value=datetime.date.today())
selected_year = selected_date.year
selected_month = selected_date.month
selected_week = selected_date.isocalendar()[1]  # ISO week number

# Input for Wheat Average Price
selected_w_price = st.number_input('Enter Retail Average Price:', min_value=0.0, step=0.01)

# Create cascading dropdowns for Division, District, Upazila, and Market Name
division = st.selectbox('Select Division:', data['Division'].unique())
districts = data[data['Division'] == division]['District'].unique()
district = st.selectbox('Select District:', districts)
upazilas = data[data['District'] == district]['Upazila'].unique()
upazila = st.selectbox('Select Upazila:', upazilas)
markets = data[data['Upazila'] == upazila]['Market Name'].unique()
market_name = st.selectbox('Select Market Name:', markets)

# Display selected values
st.write(f"Selected Year: {selected_year}, Month: {selected_month}, Week: {selected_week}")

if st.button('Forecast Price'):
    # Create a DataFrame for the future input
    future_data = pd.DataFrame({
        'W Average Price': [selected_w_price],
        'Year': [selected_year],
        'Month': [selected_month],
        'Week': [selected_week],
        'Division': [division],
        'District': [district],
        'Upazila': [upazila],
        'Market Name': [market_name]
    })

    # Ensure the correct types for the DataFrame
    future_data['W Average Price'] = pd.to_numeric(future_data['W Average Price'], errors='coerce')

    # Predict
    try:
        forecast_price = selected_model.predict(future_data)
        st.success(f"Forecasted Price: {forecast_price[0]:.2f}")
    except Exception as e:
        st.error(f"Error predicting price: {str(e)}")

# Expander for displaying metrics and plots
with st.expander("Model Metrics and Plots"):
    # You can add plots or metrics relevant to the selected commodity here.
    pass
