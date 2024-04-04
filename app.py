import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import folium
from streamlit_folium import folium_static

# Load the trained model and the pre-fitted scaler
@st.cache(allow_output_mutation=True)
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function to preprocess the uploaded data
def preprocess_data(data, scaler):
    feature_cols = [
        'number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received',
        'number_of_rotavirus_doses_received', 'number_of_measles_doses_received', 
        'number_of_polio_doses_received'
    ]
    
    if not set(feature_cols).issubset(data.columns):
        missing_cols = set(feature_cols) - set(data.columns)
        st.error(f"Missing columns in the uploaded data: {', '.join(missing_cols)}")
        return None, None

    # Scaling the features
    data_features = scaler.transform(data[feature_cols])

    return data_features, data[['latitude', 'longitude']]

# Function to visualize vaccination status on a map
def visualize_vaccination_status(data, y_pred):
    status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
    # Map predictions to status labels
    data['Predicted_Status'] = [status_mapping[pred] for pred in y_pred]
    
    mean_lat = data['latitude'].mean()
    mean_long = data['longitude'].mean()
    vaccination_map = folium.Map(location=[mean_lat, mean_long], zoom_start=6)
    
    for _, row in data.iterrows():
        color = 'red' if row['Predicted_Status'] == 'Full_Defaulter' else 'orange' if row['Predicted_Status'] == 'Partial_Defaulter' else 'green'
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=row['Predicted_Status']
        ).add_to(vaccination_map)
    
    folium_static(vaccination_map)

def main():
    st.title('Vaccination Status Prediction and Visualization')
    
    model, scaler = load_model_and_scaler('ridge_classifier_model.joblib', 'new_scalar_updated.joblib')
    
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            st.error("The uploaded CSV must include 'latitude' and 'longitude' columns.")
        else:
            st.write("Uploaded Data Preview (first 5 rows):")
            st.dataframe(data.head())

            data_features, data_location = preprocess_data(data, scaler)
            if data_features is not None:
                y_pred = model.predict(data_features)
                visualize_vaccination_status(pd.concat([data_location.reset_index(drop=True), 
                                                        pd.DataFrame(y_pred, columns=['Predicted_Status'])], axis=1), y_pred)

if __name__ == '__main__':
    main()
