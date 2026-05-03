import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Neethu-Sathyan/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Wellness Tourism Prediction App")
st.write("Wellness Tourism Prediction App predicts whether the customer will purchase the wellness tourismpackage.")
st.write("Kindly enter the customer details to check whether they will purchase the wellness tourismpackage.")


# Collect user input based on tourism.csv
Age = st.number_input("Age (customer's age in years)", min_value=18.0, max_value=100.0, value=30.0)
TypeofContact = st.selectbox("Type of Contact (how the customer was reached)", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier (tier of the city the customer resides in)", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (length of the sales pitch in minutes)", min_value=0.0, value=15.0)
Occupation = st.selectbox("Occupation (customer's current occupation)", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender (customer's gender)", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (total number of people planning to travel)", min_value=1, value=1)
NumberOfFollowups = st.number_input("Number of Follow-ups (total number of follow-up calls/meetings)", min_value=0.0, value=3.0)
ProductPitched = st.selectbox("Product Pitched (type of travel package offered)", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.number_input("Preferred Property Star (preferred hotel rating)", min_value=1.0, max_value=5.0, value=3.0)
MaritalStatus = st.selectbox("Marital Status (customer's marital status)", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips (total trips taken by the customer previously)", min_value=0.0, value=1.0)
Passport = st.selectbox("Passport (does the customer possess a valid passport?)", [1,0],format_func=lambda x:'Yes' if x==1 else 'No')
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (customer satisfaction with the pitch)", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car (does the customer own a vehicle?)", [1,0],format_func=lambda x:'Yes' if x==1 else 'No')
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (total children in the travel group)", min_value=0.0, value=0.0)
Designation = st.selectbox("Designation (customer's professional rank)", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income (customer's monthly gross income)", min_value=0.0, value=25000.0)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,      
    'CityTier': CityTier,                
    'DurationOfPitch': DurationOfPitch,  
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,           
    'ProductPitched': ProductPitched,                 
    'PreferredPropertyStar': PreferredPropertyStar,   
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore, # Fixed
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting, # Fixed
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase wellness package" if prediction == 1 else "not purchase wellness package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
