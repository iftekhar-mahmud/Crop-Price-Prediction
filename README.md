# 📦 Crop Price Prediction App

A web application for forecasting retail and wholesale crop prices based on historical market data from different regions in Bangladesh.  
Built with Streamlit, scikit-learn, and geospatial visualization using PyDeck.

---
## Disclaimer

If the app has been inactive for a while, Streamlit may put it to sleep.  
If you see a **red button** that says **"Wake the app up"**, please click it — the app will restart in a few seconds.

---
## Features

- Predict retail and wholesale prices for various commodities
- Interactive filters for date, division, district, upazila, and market selection
- Real-time geolocation mapping of selected markets
- Uses machine learning (Decision Tree Regression) with preprocessing pipelines

---

## Demo App

Try it live here:  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://croppriceprediction.streamlit.app/)

---
## ⚙️ App Settings (Theme & Layout)

You can customize how the app looks and feels using Streamlit’s settings menu:

1. Click the **⋮** (three dots) menu in the top-right corner of the app.
2. Go to **Settings**.
3. Choose your preferred **Theme**:
   - **System** (matches your OS theme)
   - **Light**
   - **Dark**
   - **Custom**
4. Toggle **Wide mode** for a more spacious layout.

---
## Installation & Usage

```bash
git clone https://github.com/iftekhar-mahmud/Crop-Price-Prediction.git
cd Crop-Price-Prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
