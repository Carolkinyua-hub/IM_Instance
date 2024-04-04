import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load the trained model and the pre-fitted scaler
def load_model_and_scaler(model_path, scaler_path):
    try:
        classifier_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return classifier_model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

# Function to preprocess the uploaded data
def preprocess_data(data, scaler):
    # Assuming these are the features your model was trained on
    # Exclude 'latitude' and 'longitude' if they were not part of the training features
    feature_cols = ['number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received', 
                      'number_of_rotavirus_doses_received', 'number_of_measles_doses_received', 
                      'number_of_polio_doses_received']  # Update with your actual feature names
    
    if set(feature_cols).issubset(data.columns):
        data_features = data[feature_cols]
    else:
        missing_cols = set(feature_cols) - set(data.columns)
        st.error(f"Missing columns in the uploaded data: {', '.join(missing_cols)}")
        return None
    
    # Check for numeric data and NaN values
    if not all(pd.api.types.is_numeric_dtype(data_features[col]) for col in feature_cols):
        st.error("One or more selected columns are not numeric or contain NaN values.")
        return None
    
    # Scale the features using the loaded scaler
    data_scaled = scaler.transform(data_features)
    
    # Replace the original feature columns in the DataFrame with the scaled ones
    data[feature_cols] = data_scaled
    
    return data

# Function to visualize vaccination status on a map
def visualize_vaccination_status(X_validation, y_pred):
    X_validation['Predicted_Vaccination_Status'] = y_pred
    status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
    X_validation['Predicted_Status'] = X_validation['Predicted_Vaccination_Status'].map(status_mapping)
    
    def get_color(Vaccination_Status):
        return 'red' if Vaccination_Status == 'Full_Defaulter' else 'orange' if Vaccination_Status == 'Partial_Defaulter' else 'green'
    
    mean_lat = X_validation['latitude'].mean()
    mean_long = X_validation['longitude'].mean()
    vaccination_map = folium.Map(location=[mean_lat, mean_long], zoom_start=6)
    
    for idx, row in X_validation.iterrows():
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

# Main function to run the Streamlit app
def main():
    st.title('Vaccination Status Prediction and Visualization')
    
    # Load the model and scaler
    classifier_model, scaler = load_model_and_scaler('ridge_classifier_model.joblib', 'scaler.joblib')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset (first 5 rows):")
        st.write(df.head())
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.error("The uploaded CSV must include 'latitude' and 'longitude' columns.")
            return
        
        df_processed = preprocess_data(df, scaler)
        if df_processed is None:
            return
        
        # Assuming the model predicts a numeric status which needs mapping to a categorical status
        features_for_prediction = df_processed[['number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received', 
                      'number_of_rotavirus_doses_received', 'number_of_measles_doses_received', 
                      'number_of_polio_doses_received']]  # Use actual feature names
        y_pred = classifier_model.predict(features_for_prediction)
        
        visualize_vaccination_status(df_processed, y_pred)

if __name__ == '__main__':
    main()
