import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def prepare_data(df):
    # Drop unnecessary columns and handle missing values
    df = df.drop(['AccidentID', 'DateTime'], axis=1)
    df = df.dropna()
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(df['SeverityLevel'])
    
    # Create dummy variables for categorical features
    X = pd.get_dummies(df.drop('SeverityLevel', axis=1))
    
    return X, y

def train_model():
    # Load data
    df = pd.read_csv('traffic_accidents_in_NY.csv')
    
    # Prepare features and target
    X, y = prepare_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with best parameters from optimization
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Save the model
    joblib.dump(model, 'rf_model.joblib')
    
    return model

if __name__ == "__main__":
    train_model()