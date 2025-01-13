from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

model = joblib.load('rf_model.joblib')

CATEGORICAL_COLUMNS = {
    'County': ['Queens', 'Brooklyn', 'Manhattan', 'The Bronx', 'Staten Island'],
    'WeatherCondition': ['Rainy', 'Sunny', 'Snowy', 'Foggy'],
    'RoadCondition': ['Wet', 'Dry', 'Under Construction'],
    'VehicleType': ['Car', 'Truck', 'Bus', 'Motorcycle'],
    'Cause': ['Speeding', 'Drunk Driving', 'Distracted Driving', 'Poor Road Condition']
}

def prepare_input(data):
    df = pd.DataFrame([data])
    
    for col, values in CATEGORICAL_COLUMNS.items():
        dummies = pd.get_dummies(df[col], prefix=col)
        
        for val in values:
            dummy_col = f"{col}_{val}"
            if dummy_col not in dummies.columns:
                dummies[dummy_col] = 0
                
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
    
    df['Junction'] = df['Junction'].astype(bool)
    df['Traffic_Signal'] = df['Traffic_Signal'].astype(bool)
    df['Bump'] = df['Bump'].astype(bool)
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_encoded = prepare_input(data)
        
        prediction = model.predict(input_encoded)[0]
        
        severity_map = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
        result = severity_map[prediction]
        
        return jsonify({
            'status': 'success',
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 