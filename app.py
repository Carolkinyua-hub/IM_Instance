import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler  # Or your specific scaler

# Load your trained model (adjust the path as necessary)
model_path = 'ridge_classifier_model.joblib'
model = joblib.load(model_path)

# Load your pre-fitted scaler (adjust the path as necessary)
scaler_path = 'new_scalar_updated.joblib'
scaler = joblib.load(scaler_path)

# Streamlit app title
st.title('Vaccination Status Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df2 = pd.read_csv(uploaded_file)
    
    # Ensure correct column names
    df2.columns = ['number_of_polio_doses_received', 'number_of_pentavalent_doses_received', 
                   'number_of_pneumococcal_doses_received', 'number_of_rotavirus_doses_received', 
                   'latitude', 'longitude', 'number_of_measles_doses_received']  # Fix any typo in column names
    
    # Select columns for prediction and scale features using the scaler object
    feature_cols = ['number_of_polio_doses_received', 'number_of_pentavalent_doses_received', 
                    'number_of_pneumococcal_doses_received', 'number_of_rotavirus_doses_received',
                    'number_of_measles_doses_received']  # Exclude 'latitude' and 'longitude' for scaling
    
    X_validation_scaled = scaler.transform(df2[feature_cols])
    
    # Make predictions
    y_pred = model.predict(X_validation_scaled)
    
    # Add predictions to the DataFrame
    status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
    df2['Predicted_Status'] = [status_mapping[pred] for pred in y_pred]
    
    # Function to get color based on vaccination status
    def get_color(Vaccination_Status):
        if Vaccination_Status == 'Full_Defaulter':
            return 'red'
        elif Vaccination_Status == 'Partial_Defaulter':
            return 'orange'
        else:
            return 'green'
    
    # Generate and display map
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
    
    folium_static(vaccination_map)
