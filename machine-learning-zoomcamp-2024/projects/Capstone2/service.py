from flask import Flask, request, jsonify
import bentoml
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model_runner = bentoml.sklearn.load_runner("sales_prediction_model:latest")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame and preprocess
        df = pd.DataFrame(data)
        
        # Make prediction
        prediction = model_runner.predict.run(df)
        
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 