import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Crop Data Preprocessor class
class CropDataPreprocessor:
    def __init__(self, commodity_name=None, district=None, division=None, upazila=None):
        self.commodity_name = commodity_name
        self.district = district
        self.division = division
        self.upazila = upazila

    def load_data(self, path):
        data = pd.read_csv(path)
        '''
        if self.commodity_name:
            data = data[data['Commodity Name'] == self.commodity_name]
        if self.district:
            data = data[data['District'] == self.district]
        if self.division:
            data = data[data['Division'] == self.division]
        if self.upazila:
            data = data[data['Upazila'] == self.upazila]
            '''
        return data

# Initialize map with folium and center on Bangladesh
def create_map():
    return folium.Map(location=[23.685, 90.3563], zoom_start=6) 

# Streamlit app starts here
st.title('Crop Price Prediction')
st.write('Website Created by Iftekhar Mahmud')
st.markdown("Enter crop and location details to predict prices.")

# Sidebar inputs for user selection
commodity_name = st.sidebar.selectbox('Select Commodity', ['Tomato', 'Potato', 'Rice', 'Onion', 'Oil Seed', 'Wheat'])
district = st.sidebar.selectbox('Select District', ['Bandarban', 'Barguna', 'Barisal', 'Chattogram', 'Chuadanga', 'Comilla', 'Dinajpur', 'Faridpur', 'Gaibandha', 'Jhalakathi', 'Jhenaidah', 'Khagrachhari', 'Kurigram', 'Lakshmipur', 'Manikganj', 'Narail', 'Patuakhali', 'Rangamati', 'Shariatpur', 'Sirajganj', 'Thakurgaon'])
division = st.sidebar.selectbox('Select Division', ['Barisal', 'Chattagram', 'Dhaka', 'Khulna', 'Rajshahi', 'Rangpur'])
upazila = st.sidebar.selectbox('Select Upazila', ['Bandarban Sadar', 'Barguna Sadar', 'Barisal Sadar', 'Chattogram City Corporation', 'Chuadanga Sadar', 'Comilla Sadar', 'Dinajpur Sadar', 'Faridpur Sadar', 'Gaibandha Sadar', 'Jhalakathi Sadar', 'Jhenaidah Sadar', 'Khagrachhari Sadar', 'Kurigram Sadar', 'Lakshmipur Sadar', 'Manikganj Sadar', 'Narail Sadar', 'Patuakhali Sadar', 'Rangamati Sadar', 'Shariatpur Sadar', 'Sirajganj Sadar', 'Thakurgaon Sadar'])

# Initialize the preprocessor
preprocessor = CropDataPreprocessor(commodity_name=commodity_name, district=district, division=division, upazila=upazila)
data = preprocessor.load_data('Data/Combined Dataset.csv')  # Load and preprocess data

# Troubleshooting
st.write("Data shape:", data.shape)
st.write("First few rows of data:", data.head())

# Ensure that data is not empty
if data.empty:
    st.error("No data available for the selected parameters.")
else:
    target_column_name = 'R Average Price'  # replace with your actual target column name
    predictors = ['W Average Price', 'Year', 'Month', 'Week', 'Division', 'District', 'Upazila', 'Market Name']
    
    # Define predictors and target variable
    X = data[predictors]
    y = data[target_column_name]

    st.write("Shape of X:", X.shape)
    st.write("Shape of y:", y.shape)

    # Check for missing values
    st.write("Missing values in X:", X.isnull().sum())
    st.write("Missing values in y:", y.isnull().sum())

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary of regression models
    models = {
        'Linear Regression': Pipeline([('model', LinearRegression())]),
        'Ridge Regression': Pipeline([('model', Ridge())]),
        'Lasso Regression': Pipeline([('model', Lasso())]),
        'Decision Tree Regression': Pipeline([('model', DecisionTreeRegressor())]),
        'Random Forest Regression': Pipeline([('model', RandomForestRegressor())]),
        'Support Vector Regression': Pipeline([('model', SVR())])
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
