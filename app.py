import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load the trained model and the pre-fitted scaler
try:
    classifier_model = joblib.load('ridge_classifier_model.joblib')
    scaler = joblib.load('X_validation_scaled (1).joblib')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check the files and try again.")
    st.stop()


# Fit the loaded scaler to the training data
scaler.fit(X_train)

# Transform the validation data using the fitted scaler
X_validation_scaled = scaler.transform(X_validation)

# Function for data preprocessing
def preprocess_data(data):
    # Ensure the expected columns are present
    expected_cols = {'number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received',
                     'number_of_rotavirus_doses_received', 'number_of_measles_doses_received',
                     'number_of_polio_doses_received', 'latitude', 'longitude'}
    if not expected_cols.issubset(data.columns):
        missing_cols = expected_cols - set(data.columns)
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
        return None

    # Drop any missing values
    data_processed = data.dropna(subset=expected_cols)

    # Perform feature scaling on numerical columns
    numerical_cols = list(expected_cols - {'latitude', 'longitude'})
    data_processed[numerical_cols] = scaler.transform(data_processed[numerical_cols])
    
    return data_processed

def main():
    st.title('Vaccination Status Prediction and Visualization')

    # Upload validation dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the uploaded dataset
        st.write("Uploaded Dataset (first 5 rows):")
        st.write(df.head())

        # Preprocess data
        df_processed = preprocess_data(df)
        if df_processed is None:  # Stop further execution if there's an issue with the data
            return

        # Make predictions
        y_pred = classifier_model.predict(df_processed)
        df_processed['Predicted_Vaccination_Status'] = y_pred

        # Mapping predictions to status labels
        status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
        df_processed['Predicted_Status'] = df_processed['Predicted_Vaccination_Status'].map(status_mapping)

        st.write("Processed and Predicted Data:")
        st.write(df_processed)

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
