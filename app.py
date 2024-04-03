import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Function to load the model and scaler
def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load('ridge_classifier_model.joblib')
        scaler = joblib.load('scalar_updated.joblib')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

# Function for data preprocessing
def preprocess_data(data, scaler):
    # Define the expected columns
    expected_cols = {'number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received',
                     'number_of_rotavirus_doses_received', 'number_of_measles_doses_received',
                     'number_of_polio_doses_received', 'latitude', 'longitude'}
    # Check for missing columns
    if not expected_cols.issubset(data.columns):
        missing_cols = expected_cols - set(data.columns)
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
        return None

    # Drop any missing values in the expected columns
    data_processed = data.dropna(subset=expected_cols)
    
    # Select numerical columns for scaling
    numerical_cols = list(expected_cols - {'latitude', 'longitude'})
    data_processed[numerical_cols] = scaler.transform(data_processed[numerical_cols])
    
    return data_processed

def main():
    st.title('Vaccination Status Prediction and Visualization')

    # Load the trained model and scaler
    classifier_model, scaler = load_model_and_scaler('ridge_classifier_model.joblib', 'scalar_updated.joblib')

    # Upload the validation dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the uploaded dataset
        st.write("Uploaded Dataset (first 5 rows):")
        st.write(df.head())

        # Preprocess the data
        df_processed = preprocess_data(df, scaler)
        if df_processed is None:  # Stop further execution if there's an issue with the data
            return

        # Make predictions
        y_pred = classifier_model.predict(df_processed.drop(['latitude', 'longitude'], axis=1))
        df_processed['Predicted_Vaccination_Status'] = y_pred

        # Mapping predictions to status labels
        status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
        df_processed['Predicted_Status'] = df_processed['Predicted_Vaccination_Status'].map(status_mapping)

        st.write("Processed and Predicted Data:")
        st.write(df_processed)

        # Filter defaulters for visualization
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
