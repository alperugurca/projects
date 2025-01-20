#!/usr/bin/env python
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import bentoml
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and feature columns
MODEL_PATH = 'models'

def load_model():
    """Load the trained model"""
    return bentoml.sklearn.load_model("sales_prediction_model:latest")

def prepare_features(data):
    """Prepare features for prediction"""
    # Add your feature preparation code here
    return data

def predict(data):
    """Make predictions using the loaded model"""
    model = load_model()
    prepared_data = prepare_features(data)
    return model.predict(prepared_data)

# Load model at startup
model, feature_columns = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions"""
    try:
        # Get input data
        data = request.get_json()
        
        # Validate input
        required_fields = ['date', 'store_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features
        df = prepare_features(data)
        
        # Select features in correct order
        X = df[feature_columns]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Return prediction
        return jsonify({
            'prediction': float(prediction),
            'store_id': data['store_id'],
            'date': data['date']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

def main():
    """Run the Flask app"""
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main() 