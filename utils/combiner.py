import pandas as pd
import geopandas as gpd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class CropDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, commodity_name=None, district=None, division=None, upazila=None):
        self.commodity_name = commodity_name
        self.district = district
        self.division = division
        self.upazila = upazila

    def load_data(self, path):
        # Load combined dataset
        data = pd.read_csv(path)
        # Apply filtering based on commodity, district, etc.
        if self.commodity_name:
            data = data[data['Commodity Name'] == self.commodity_name]
        if self.district:
            data = data[data['District'] == self.district]
        if self.division:
            data = data[data['Division'] == self.division]
        if self.upazila:
            data = data[data['Upazila'] == self.upazila]
        return data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convert categorical variables, handle missing values, etc.
        categorical_cols = ['Month', 'Division', 'District', 'Upazila', 'Market Name']
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', 'passthrough', numeric_cols)
            ])
        return preprocessor.fit_transform(X)

