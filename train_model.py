import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    print("Loading data...")
    # Load dataset - make sure to place the WA_Fn-UseC_-Telco-Customer-Churn.csv in the correct location
    # Replace the path with your actual dataset path if needed
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    print("Preprocessing data...")
    # Preprocessing based on the notebook
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
    df.fillna(df["TotalCharges"].mean())
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    
    # Save the expected columns order
    expected_columns = df.drop(columns=['Churn']).columns.tolist()
    joblib.dump(expected_columns, 'expected_columns.pkl')

    # Label encode all categorical object columns and save encoders
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
    joblib.dump(encoders, 'encoders.pkl')

    print("Splitting data...")
    X = df.drop(columns=['Churn'])
    y = df['Churn'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

    print("Scaling features...")
    num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print("Training the final VotingClassifier model...")
    clf1 = GradientBoostingClassifier()
    clf2 = LogisticRegression()
    clf3 = AdaBoostClassifier()
    eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
    eclf1.fit(X_train, y_train)

    predictions = eclf1.predict(X_test)
    print("Final Accuracy Score:", accuracy_score(y_test, predictions))

    print("Saving the model and scaler...")
    # Save the model and the scaler to disk
    joblib.dump(eclf1, 'churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model 'churn_model.pkl' and 'scaler.pkl' have been saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
