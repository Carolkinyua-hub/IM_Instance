import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load the trained model and the pre-fitted scaler
classifier_model = joblib.load('ridge_classifier_model.joblib')
scaler = joblib.load('scaler_updated.joblib')  # Assuming you have saved your scaler the same way as your model

# Function for data preprocessing
def preprocess_data(data):
    # Copy the data to avoid inplace modification
    data_processed = data.copy()
    
    # Drop any missing values
    data_processed.dropna(inplace=True)

    # Perform feature scaling on numerical columns
    numerical_cols = ['number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received', 
                      'number_of_rotavirus_doses_received', 'number_of_measles_doses_received', 
                      'number_of_polio_doses_received']
    data_processed[numerical_cols] = scaler.transform(data_processed[numerical_cols])
    
    return data_processed

def main():
    st.title('Vaccination Status Prediction and Visualization')

    # Upload validation dataset
    st.subheader('Upload Validation Dataset')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Preprocess data
        df_processed = preprocess_data(df)

        # Make predictions
        y_pred = classifier_model.predict(df_processed)
        df_processed['Predicted_Vaccination_Status'] = y_pred

        # Mapping predictions to status labels
        status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
        df_processed['Predicted_Status'] = [status_mapping[pred] for pred in y_pred]

        # Filter defaulters
        defaulters_df = df_processed[df_processed['Predicted_Status'].isin(['Full_Defaulter', 'Partial_Defaulter'])]

        # Visualization
        if not defaulters_df.empty:
            visualize_defaulters(defaulters_df)
        else:
            st.write("No defaulters found in the uploaded dataset.")

def visualize_defaulters(df):
    """Visualizes the defaulters on a map."""
    st.subheader('Predicted Defaulter Vaccination Status Visualization')
    mean_lat = df['latitude'].mean()
    mean_long = df['longitude'].mean()
    vaccination_map = folium.Map(location=[mean_lat, mean_long], zoom_start=6)

    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='red',  # Defaulters marked in red
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=row['Predicted_Status']
        ).add_to(vaccination_map)

    folium_static(vaccination_map)

if __name__ == '__main__':
    main()
