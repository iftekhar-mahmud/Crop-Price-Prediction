import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('Data/Combined Dataset.csv')

# Preprocess the data
data = data.dropna()
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

# Dictionary of regression models
models = {
    'Linear Regression': Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())]),
    'Ridge Regression': Pipeline([('preprocessor', preprocessor), ('model', Ridge())]),
    'Lasso Regression': Pipeline([('preprocessor', preprocessor), ('model', Lasso())]),
    'Decision Tree Regression': Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor())]),
    'Random Forest Regression': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor())]),
    'Support Vector Regression': Pipeline([('preprocessor', preprocessor), ('model', SVR())])
}

# Streamlit app starts here
st.title('Crop Price Prediction')
st.write('Website Created by Iftekhar Mahmud')
st.markdown("Enter crop and location details to predict prices.")

# Sidebar inputs for user selection
commodity_name = st.sidebar.selectbox('Select Commodity', commodity_names)
district = st.sidebar.selectbox('Select District', data['District'].unique())
division = st.sidebar.selectbox('Select Division', data['Division'].unique())
upazila = st.sidebar.selectbox('Select Upazila', data['Upazila'].unique())

# Filter data for the selected commodity
commodity_data = data_cleaned_iqr[data_cleaned_iqr['Commodity Group'] == commodity_name]

# Define the target and predictors
X = commodity_data[predictors]
y = commodity_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through each model and fit, predict, and evaluate
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Display model metrics
    st.write(f"{name}: R-squared = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}")

    # Plotting Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'{name} - Actual vs Predicted Prices\nR-squared = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}')
    plt.grid(True)
    st.pyplot(plt)

