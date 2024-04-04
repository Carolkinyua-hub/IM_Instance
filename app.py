import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

def load_model_and_scaler(model_path, scaler_path):
    try:
        classifier_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return classifier_model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

def preprocess_data(data, scaler):
    # Replace 'your_features_here' with the actual features used during model training, excluding 'latitude' and 'longitude'
    feature_cols = ['your_features_here']
    
    # Check for missing columns in the uploaded data
    missing_cols = set(feature_cols) - set(data.columns)
    if missing_cols:
        st.error(f"Missing columns in the uploaded data: {', '.join(missing_cols)}")
        return None

    # Selecting the features from the uploaded data
    data_features = data[feature_cols]

    # Checking for numeric data and NaN values
    if not all(pd.api.types.is_numeric_dtype(data_features[col]) for col in data_features.columns):
        st.error("One or more selected columns are not numeric or contain NaN values.")
        return None

    # Scaling the features using the loaded scaler
    data_scaled = scaler.transform(data_features)

    # Replacing the original feature columns in the DataFrame with the scaled ones
    data[feature_cols] = data_scaled
    
    return data

def main():
    st.title('Vaccination Status Prediction and Visualization')
    
    # Update these paths to where your model and scaler are located
    model_path = 'ridge_classifier_model.joblib'
    scaler_path = 'new_scalar_updated.joblib'

    classifier_model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset (first 5 rows):")
        st.write(df.head())

        df_processed = preprocess_data(df, scaler)
        if df_processed is None:
            return

        # Assuming your model does not use 'latitude' and 'longitude' for prediction
        # Ensure 'your_features_here' matches the exact features list used during model training
        features_for_prediction = df_processed[['your_features_here']]
        
        y_pred = classifier_model.predict(features_for_prediction)
        
        df_processed['Predicted_Vaccination_Status'] = y_pred

        # Mapping predictions to status labels
        status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
        df_processed['Predicted_Status'] = df_processed['Predicted_Vaccination_Status'].map(status_mapping)

        st.write("Processed and Predicted Data:")
        st.write(df_processed)

        # Visualization
        defaulters_df = df_processed[df_processed['Predicted_Status'].isin(['Full_Defaulter', 'Partial_Defaulter'])]
        if not defaulters_df.empty:
            visualize_defaulters(defaulters_df)
        else:
            st.write("No defaulters found in the uploaded dataset.")

def visualize_defaulters(df):
    st.subheader('Predicted Defaulter Vaccination Status Visualization')
    mean_lat = df['latitude'].mean()
    mean_long = df['longitude'].mean()
    vaccination_map = folium.Map(location=[mean_lat, mean_long], zoom_start=6)

    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=row['Predicted_Status']
        ).add_to(vaccination_map)

    folium_static(vaccination_map)

if __name__ == '__main__':
    main()
