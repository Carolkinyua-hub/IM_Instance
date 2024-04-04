import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load the trained model and the pre-fitted scaler
def load_model_and_scaler(model_path, scaler_path):
    try:
        classifier_model = joblib.load('ridge_classifier_model.joblib')
        scaler = joblib.load('new_scalar_updated.joblib')
        if not hasattr(scaler, 'transform'):
            raise ValueError("Loaded scaler object does not have a transform method.")
        return classifier_model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Validation error: {e}")
        st.stop()
        return classifier_model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

# Function for data preprocessing
def preprocess_data(data, scaler):
    expected_cols = {'number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received',
                     'number_of_rotavirus_doses_received', 'number_of_measles_doses_received',
                     'number_of_polio_doses_received', 'latitude', 'longitude'}
    if not expected_cols.issubset(data.columns):
        missing_cols = expected_cols - set(data.columns)
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
        return None

    data_processed = data.dropna(subset=expected_cols)
    # Exclude 'latitude' and 'longitude' before scaling
    numerical_cols = [col for col in numerical_cols if col not in ['latitude', 'longitude']]
    scaled_data = scaler.transform(data[numerical_cols])
    
    return scaled_data

    # Ensure all selected columns are numeric
    if not all(pd.api.types.is_numeric_dtype(data_processed[col]) for col in numerical_cols):
        st.error("One or more selected columns are not numeric.")
        return None

    # Transform the numerical columns using the fitted scaler
    scaled_data = scaler.transform(data_processed[numerical_cols])
    data_processed.loc[:, numerical_cols] = scaled_data
    
    return data_processed

def main():
    st.title('Vaccination Status Prediction and Visualization')

    model_path = 'ridge_classifier_model.joblib'
    scaler_path = 'scalar_updated.joblib'
    classifier_model, scaler = load_model_and_scaler(model_path, scaler_path)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset (first 5 rows):")
        st.write(df.head())

        df_processed = preprocess_data(df, scaler)
        if df_processed is None:  # Stop further execution if there's an issue with the data
            return

        y_pred = classifier_model.predict(df_processed.drop(['latitude', 'longitude'], axis=1))
        df_processed['Predicted_Vaccination_Status'] = y_pred

        status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
        df_processed['Predicted_Status'] = df_processed['Predicted_Vaccination_Status'].map(status_mapping)

        st.write("Processed and Predicted Data:")
        st.write(df_processed)

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
