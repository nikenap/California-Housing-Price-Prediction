import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import requests
import joblib

# Load dataset and trained model
url = "https://github.com/nikenap/California-Housing-Price-Prediction/releases/download/v1.0.0/california_house_model.pkl"
response = requests.get(url)
with open("california_house_model.pkl", "wb") as f:
    f.write(response.content)

df_raw = pd.read_csv("data_california_house_cleaned.csv")
model = joblib.load('california_house_model.pkl')

# Set the title for the web app
st.write('''
## California Housing Price Prediction App
         
This app predicts the **California House Price** based on user input or a batch of data uploaded as a CSV file.

##### How to Use:
- **Single Prediction:** Manually input parameters using the options in the sidebar.
- **Batch Prediction:** Upload a CSV file containing the input data for multiple houses.

##### Output:
- The predicted house price(s) will be displayed below.
- If you upload a file, you can download the results with predictions as a CSV file.

For guidance on input file format, check the [Example CSV input file](https://github.com/nikenap/California-Housing-Price-Prediction/blob/main/data_california_house_example.csv).
''')
st.write('---')

# Sidebar for user input
st.sidebar.header('Specify Input Parameters')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Batch Prediction: Read and process uploaded CSV file
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(input_df.head())
    except Exception as e:
        st.error("Error reading the file. Please ensure it is a valid CSV format.")
else:
    # Single Prediction: Allow manual input
    def user_input_features():
        ocean_proximity = st.sidebar.selectbox(
            'Proximity to Ocean', 
            df_raw['ocean_proximity'].unique(),
            help="The proximity of the house to the ocean.")
        
        county = st.sidebar.selectbox(
            'County', 
            sorted(df_raw['county'].unique()),
            help="Select the county where the house is located. Counties in the dataset include major regions in California.")
        
        median_income = st.sidebar.slider(
            'Median Income (10k USD)', 
            float(df_raw['median_income'].min()),  
            float(df_raw['median_income'].max()),  
            float(df_raw['median_income'].mean()),
            help="The median income of households in the area, measured in tens of thousands of USD.") 
        
        housing_median_age = st.sidebar.slider(
            'House Age (years)', 
            int(df_raw['housing_median_age'].min()), 
            int(df_raw['housing_median_age'].max()), 
            int(df_raw['housing_median_age'].mean()),
            help="The median age of houses in the area, measured in years.")
        
        total_rooms = st.sidebar.slider(
            'Total Rooms', 
            int(df_raw['total_rooms'].min()), 
            int(df_raw['total_rooms'].max()), 
            int(df_raw['total_rooms'].mean()),
            help="The total number of rooms within a block. This includes bedrooms, living rooms, kitchens, etc.")
        
        pop_per_household = st.sidebar.slider(
            'Population per Household', 
            float(df_raw['pop_per_household'].min()), 
            float(df_raw['pop_per_household'].max()), 
            float(df_raw['pop_per_household'].mean()),
            help="The average number of people living in a single household within the area.")
        
        data = {'ocean_proximity': ocean_proximity,
                'county': county,
                'median_income': median_income,
                'housing_median_age': housing_median_age,
                'total_rooms': total_rooms,
                'pop_per_household': pop_per_household}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Display user input
st.subheader(' Specified Input Parameters')
st.write("User Input (Single Prediction)" if uploaded_file is None else "Uploaded File Data:")
st.write(input_df)
st.write('---')

# Apply model to make predictions
prediction = model.predict(input_df)

# Display predictions
st.subheader('Predicted House Prices')
if uploaded_file is not None:
    # Batch predictions: Display results for all rows
    input_df['Predicted Price'] = prediction
    st.write(input_df)
    # Option to download results
    csv = input_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="house_price_predictions.csv",
        mime="text/csv",
    )
else:
    # Single prediction: Show one result
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")
st.write('---')

# Load test dataset (replace with actual test dataset path)
test_data_path = "data_california_house_test.csv"  
test_data = pd.read_csv(test_data_path)

# Split into features and target
X_test = test_data.drop(columns=['median_house_value'])
y_test = test_data['median_house_value']

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Display Model Performance Metrics
st.write("## Model Performance Metrics")
st.write("---")
st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape:.2f}%")
st.write(f"**RMSE (Root Mean Squared Error):** ${rmse:,.2f}")
st.write(f"**R² (Coefficient of Determination):** {r2:.2f}")

# Toggle for Visualizations
if st.checkbox("Show Visualizations"):
    # Residual Plot
    st.write("### Residual Plot")
    fig, ax = plt.subplots()
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Predicted Values")
    st.pyplot(fig)

    # Prediction vs. Actual Plot
    st.write("### Prediction vs. Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Perfect Fit")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Prediction vs. Actual")
    ax.legend()
    st.pyplot(fig)