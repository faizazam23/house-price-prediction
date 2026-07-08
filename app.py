import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# ============ GENERATE FILES IF MISSING ============
@st.cache_resource
def setup_data():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if not os.path.exists('data/X_test.pkl'):
        st.write("⏳ Generating training data...")
        
        # Download dataset
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data['Price'] = housing.target
        
        # Split
        y = data['Price']
        X = data.drop('Price', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save
        with open('data/X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open('data/X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open('data/y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open('data/y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        
        st.write("✅ Data generated!")
    
    if not os.path.exists('models/best_model.pkl'):
        st.write("⏳ Training model...")
        
        # Load data
        with open('data/X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('data/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        
        # Train
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Save
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        st.write("✅ Model trained!")

# Generate files if missing
setup_data()

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_test_data():
    with open('data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test

model = load_model()
X_test, y_test = load_test_data()

# ============ TITLE ============
st.title("🏡 House Price Prediction App")
st.write("California Housing Dataset - Powered by Random Forest ML")

# ============ SIDEBAR ============
st.sidebar.header("📊 House Details")
st.sidebar.write("---")

# Input sliders
medinc = st.sidebar.slider("Income Level (MedInc)", 
                           float(X_test['MedInc'].min()), 
                           float(X_test['MedInc'].max()), 
                           float(X_test['MedInc'].mean()))

houseage = st.sidebar.slider("House Age (Years)",
                             float(X_test['HouseAge'].min()),
                             float(X_test['HouseAge'].max()),
                             float(X_test['HouseAge'].mean()))

averoom = st.sidebar.slider("Average Rooms",
                            float(X_test['AveRooms'].min()),
                            float(X_test['AveRooms'].max()),
                            float(X_test['AveRooms'].mean()))

avebedroom = st.sidebar.slider("Average Bedrooms",
                               float(X_test['AveBedrms'].min()),
                               float(X_test['AveBedrms'].max()),
                               float(X_test['AveBedrms'].mean()))

population = st.sidebar.slider("Population",
                               float(X_test['Population'].min()),
                               float(X_test['Population'].max()),
                               float(X_test['Population'].mean()))

aveoccup = st.sidebar.slider("Average Occupancy",
                             float(X_test['AveOccup'].min()),
                             float(X_test['AveOccup'].max()),
                             float(X_test['AveOccup'].mean()))

latitude = st.sidebar.slider("Latitude",
                             float(X_test['Latitude'].min()),
                             float(X_test['Latitude'].max()),
                             float(X_test['Latitude'].mean()))

longitude = st.sidebar.slider("Longitude",
                              float(X_test['Longitude'].min()),
                              float(X_test['Longitude'].max()),
                              float(X_test['Longitude'].mean()))

# ============ PREDICTION ============
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Prediction Results")
    
    features = np.array([[medinc, houseage, averoom, avebedroom, population, aveoccup, latitude, longitude]])
    predicted_price = model.predict(features)[0]
    
    st.metric(
        label="🏡 Predicted House Price",
        value=f"${predicted_price * 100000:,.0f}"
    )
    
    st.success(f"✅ Predicted Price: ${predicted_price:.2f}00k")

with col2:
    st.subheader("📊 Model Performance")
    
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    st.metric("R² Score", f"{test_r2:.4f}")
    st.metric("RMSE", f"${test_rmse:.4f}00k")

# ============ DETAILS TABLE ============
st.subheader("📋 House Details Summary")

summary_data = {
    'Feature': ['Income', 'House Age', 'Avg Rooms', 'Avg Bedrooms', 'Population', 'Occupancy', 'Latitude', 'Longitude'],
    'Value': [f"{medinc:.2f}", f"{houseage:.2f} yrs", f"{averoom:.2f}", f"{avebedroom:.2f}", 
              f"{population:.2f}", f"{aveoccup:.2f}", f"{latitude:.2f}", f"{longitude:.2f}"]
}

summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

# ============ FOOTER ============
st.write("---")
st.write("💡 **Tip:** Adjust the sliders to see how features affect house prices!")