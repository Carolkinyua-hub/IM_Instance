import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler

# Function to get color based on vaccination status
def get_color(Vaccination_Status):
    if Vaccination_Status == 'Full_Defaulter':
        return 'red'
    elif Vaccination_Status == 'Partial_Defaulter':
        return 'orange'
    else:
        return 'green'


model = joblib.load('ridge_classifier_model.joblib')  # Update the path as needed
scaler = joblib.load('new_scalar_updated.joblib')  # Update the path as needed

# Streamlit app title
st.title('Vaccination Status Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df2 = pd.read_csv(uploaded_file)

    # Assuming the uploaded file is a CSV, adjust for other formats
    # Select columns for prediction
    features_for_prediction = df2[[ 'number_of_pentavalent_doses_received',
       'number_of_pneumococcal_doses_received',
       'number_of_rotavirus_doses_received',
       'number_of_measles_doses_received', 'number_of_polio_doses_received','latitude',
       'longitude']]

    # Scale features using the scaler object
    features_scaled = scaler.transform(features_for_prediction)

    # Make predictions
    y_pred = model.predict(features_scaled)

    # Add predictions to the DataFrame
    status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
    df2['Predicted_Status'] = [status_mapping[pred] for pred in y_pred]

    # Generate map
    mean_lat = df2['latitude'].mean()
    mean_long = df2['longitude'].mean()
    vaccination_map = folium.Map(location=[mean_lat, mean_long], zoom_start=6)

    for idx, row in df2.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=get_color(row['Predicted_Status']),
            fill=True,
            fill_color=get_color(row['Predicted_Status']),
            fill_opacity=0.7,
            popup=row['Predicted_Status']
        ).add_to(vaccination_map)

    # Display map in Streamlit
    folium_static(vaccination_map)
