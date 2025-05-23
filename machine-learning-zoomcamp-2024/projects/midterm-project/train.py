from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json  # Add this import

app = Flask(__name__)

# Load the model and feature names
model = pickle.load(open('rf_model.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder  # Set the custom encoder for the Flask app

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create a DataFrame with all possible genre columns (initialized to 0)
        genre_columns = [col for col in feature_names if col.startswith('Fav genre_')]
        input_data = pd.DataFrame(columns=feature_names, index=[0])
        input_data.fillna(0, inplace=True)
        
        # Fill in the numerical values
        input_data['Hours per day'] = data['Hours per day']
        input_data['Anxiety'] = data['Anxiety']
        input_data['Depression'] = data['Depression']
        input_data['Insomnia'] = data['Insomnia']
        input_data['OCD'] = data['OCD']
        
        # Set the genre column
        genre_col = f"Fav genre_{data['Fav genre']}"
        if genre_col in genre_columns:
            input_data[genre_col] = 1
            
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].tolist()
        
        return jsonify({
            'prediction': str(prediction),
            'probability': probability,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)