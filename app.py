import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Assuming you've already loaded your model and scaler correctly
# model = joblib.load('path_to_your_model.joblib')
# scaler = joblib.load('path_to_your_scaler.joblib')

# Streamlit app title
st.title('Vaccination Status Prediction')

# Function to preprocess the data and make predictions
def preprocess_and_predict(data, model, scaler):
    # Define your feature columns exactly as they were during model training
    feature_cols = ['number_of_polio_doses_received', 'number_of_pentavalent_doses_received', 
                    'number_of_pneumococcal_doses_received', 'number_of_rotavirus_doses_received',
                    'number_of_measles_doses_received']
    
    # Ensure the uploaded data contains all necessary columns
    if not set(feature_cols).issubset(data.columns):
        st.error("Uploaded data is missing one or more required columns.")
        return None

    # Scale the features using the scaler object
    features = data[feature_cols]
    features_scaled = scaler.transform(features)
    
    # Make predictions with the scaled features
    predictions = model.predict(features_scaled)
    
    return predictions

# Function to visualize predictions on a map
def visualize_predictions(data, predictions):
    # Map the numeric predictions to their corresponding status labels
    status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
    data['Predicted_Status'] = [status_mapping[pred] for pred in predictions]
    
    # Create a base map
    mean_lat = data['latitude'].mean()
    mean_long = data['longitude'].mean()
    vaccination_map = folium.Map(location=[mean_lat, mean_long], zoom_start=6)

    # Add points to the map
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=get_color(row['Predicted_Status']),
            fill=True,
            fill_color=get_color(row['Predicted_Status']),
            fill_opacity=0.7,
            popup=row['Predicted_Status']
        ).add_to(vaccination_map)

    folium_static(vaccination_map)

# Upload CSV file section
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'latitude' in data.columns and 'longitude' in data.columns:
        predictions = preprocess_and_predict(data, model, scaler)
        if predictions is not None:
            visualize_predictions(data, predictions)
    else:
        st.error("The uploaded CSV must include 'latitude' and 'longitude' columns.")
