import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load models and preprocessing objects
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('encoders.pkl')
        expected_columns = joblib.load('expected_columns.pkl')
        return model, scaler, encoders, expected_columns
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure you run `train_model.py` first.")
        st.stop()

model, scaler, encoders, expected_columns = load_assets()

# Application Title
st.title("Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their profile and usage metrics.")

st.sidebar.header("Configure Input Details")
st.sidebar.info("Adjust the values on the right to see the churn prediction change.")

# Organize inputs nicely into columns for better UI
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ("Female", "Male"))
    senior_citizen = st.selectbox("Senior Citizen", ("No", "Yes"))
    partner = st.selectbox("Partner", ("No", "Yes"))
    dependents = st.selectbox("Dependents", ("No", "Yes"))
    
    st.subheader("Account Details")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))

with col2:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ("No", "Yes"))
    multiple_lines = st.selectbox("Multiple Lines", ("No phone service", "No", "Yes"))
    internet_service = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.selectbox("Online Security", ("No internet service", "No", "Yes"))
    online_backup = st.selectbox("Online Backup", ("No internet service", "No", "Yes"))
    device_protection = st.selectbox("Device Protection", ("No internet service", "No", "Yes"))

with col3:
    st.subheader("More Services & Billing")
    tech_support = st.selectbox("Tech Support", ("No internet service", "No", "Yes"))
    streaming_tv = st.selectbox("Streaming TV", ("No internet service", "No", "Yes"))
    streaming_movies = st.selectbox("Streaming Movies", ("No internet service", "No", "Yes"))
    paperless_billing = st.selectbox("Paperless Billing", ("No", "Yes"))
    payment_method = st.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

# Collect inputs into a dictionary matching the training feature names
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert to DataFrame
df_input = pd.DataFrame([input_data])

st.markdown("---")
st.subheader("User Input Features Summary")
st.dataframe(df_input)

if st.button("Predict Churn", type="primary"):
    # Preprocess the input
    
    # 1. Ensure columns are in the expected order
    df_processed = df_input[expected_columns].copy()
    
    # 2. Label Encoding using saved encoders
    for col in expected_columns:
        if col in encoders:
            try:
                df_processed[col] = encoders[col].transform(df_processed[col])
            except ValueError as e:
                # Handle unseen labels just in case
                st.error(f"Error encoding column {col}: {e}")
                st.stop()
    
    # 3. Scaling numerical features
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_processed[num_cols] = scaler.transform(df_processed[num_cols])
    
    # Predict using the loaded VotingClassifier model
    prediction = model.predict(df_processed)[0]
    
    # Check classes of the 'Churn' encoder to map prediction output back correctly
    churn_classes = encoders['Churn'].classes_
    # Typically ['No', 'Yes'], so 1 is 'Yes' (Churn), 0 is 'No'
    
    predicted_class = churn_classes[prediction]
    
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df_processed)[0]
        # Get probability of 'Yes'
        prob_yes = probability[1] if churn_classes[1] == 'Yes' else probability[0]
    else:
        prob_yes = None

    st.markdown("---")
    st.subheader("Prediction Result")
    
    if predicted_class == "Yes":
        if prob_yes is not None:
            st.error(f"🚨 **The customer is likely to CHURN!** (Probability: {prob_yes:.2%})")
        else:
            st.error("🚨 **The customer is likely to CHURN!**")
    else:
        if prob_yes is not None:
            st.success(f"✅ **The customer is likely to STAY.** (Probability of churning: {prob_yes:.2%})")
        else:
            st.success("✅ **The customer is likely to STAY.**")
