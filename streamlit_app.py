import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Application Title
st.title("Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their profile and usage metrics.")

# Sidebar for user inputs
st.sidebar.header("Customer Details")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10.0, 150.0, 50.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    contract = st.sidebar.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display User Input
st.subheader("User Input Features")
st.write(df)

# Placeholder for Model Prediction
st.subheader("Prediction")
st.info("Model prediction logic will go here. Once you train a model (e.g., using scikit-learn), you can load it here to predict the churn probability.")

if st.button("Predict Churn"):
    # Mock prediction logic for demonstration
    probability = np.random.rand()
    prediction = "Churn" if probability > 0.5 else "No Churn"
    
    if prediction == "Churn":
        st.error(f"The customer is likely to churn! (Probability: {probability:.2f})")
    else:
        st.success(f"The customer is likely to stay. (Probability: {probability:.2f})")
