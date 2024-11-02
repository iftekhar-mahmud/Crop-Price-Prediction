import streamlit as st
from utils.combiner import CropDataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Initialize map with folium and center on Bangladesh
def create_map():
    return folium.Map(location=[23.685, 90.3563], zoom_start=6) 

st.title('Crop Price Prediction')
st.write('Website Created by Iftekhar Mahmud')
st.markdown("Enter crop and location details to predict prices.")

# Sidebar inputs for user selection
commodity_name = st.sidebar.selectbox('Select Commodity', ['Tomato', 'Potato', 'Rice', 'Onion', 'Oil Seed', 'Wheat'])
district = st.sidebar.selectbox('Select District', ['Bandarban', 'Barguna', 'Barisal', 'Chattogram', ...])  # add more districts as needed
division = st.sidebar.selectbox('Select Division', ['Barisal', 'Chattagram', 'Dhaka', 'Khulna', 'Rajshahi', 'Rangpur'])
upazila = st.sidebar.selectbox('Select Upazila', ['Bandarban Sadar', 'Barguna Sadar', 'Barisal Sadar', ...])  # add more upazilas as needed

# Initialize the preprocessor
preprocessor = CropDataPreprocessor(commodity_name=commodity_name, district=district, division=division, upazila=upazila)
data = preprocessor.load_data('data/combined_data.csv')  # Load and preprocess data

# Define predictors and target variable
target = 'R Average Price'
predictors = ['W Average Price', 'Year', 'Month', 'Week', 'Division', 'District', 'Upazila', 'Market Name']
X = data[predictors]
y = data[target]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary of regression models
models = {
    'Linear Regression': Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())]),
    'Ridge Regression': Pipeline([('preprocessor', preprocessor), ('model', Ridge())]),
    'Lasso Regression': Pipeline([('preprocessor', preprocessor), ('model', Lasso())]),
    'Decision Tree Regression': Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor())]),
    'Random Forest Regression': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor())]),
    'Support Vector Regression': Pipeline([('preprocessor', preprocessor), ('model', SVR())])
}

# Display metrics for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Display model metrics
    st.write(f"{name}: R-squared = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}")

    # Plotting Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{name} - Actual vs Predicted')
    st.pyplot(fig)

# Create and display map with folium
map_bd = create_map()
# Add marker for the selected location
folium.Marker(
    location=[23.685, 90.3563],  # Update with actual lat, long if available
    popup=f"{upazila}, {district}, {division}",
    tooltip="Selected Location"
).add_to(map_bd)

# Display the map in Streamlit
folium_static(map_bd)
