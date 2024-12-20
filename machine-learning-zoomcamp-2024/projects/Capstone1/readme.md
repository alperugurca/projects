# New York Traffic Accident Predictor

## Problem Description
Traffic accidents are a significant concern in New York, impacting public safety and causing economic losses. This project aims to predict the severity of traffic accidents using various features such as weather conditions, road conditions, and vehicle types. By accurately predicting accident severity, city planners and emergency services can better allocate resources and implement preventive measures.

---

## Dataset
- **Source**: [Kaggle - Traffic Accidents in NY 2023](https://www.kaggle.com/datasets/hfan83/traffic-accidents-in-ny-2023/data)
- **Features**:  
  - `AccidentID`: Unique identifier for each accident.  
  - `DateTime`: Date and time of the accident.  
  - `County`: The county where the accident occurred.  
  - `WeatherCondition`: Weather conditions at the time of the accident.  
  - `RoadCondition`: Condition of the road at the time of the accident.  
  - `VehicleType`: Type of vehicle(s) involved in the accident.  
  - `SeverityLevel`: The severity level of the accident (target variable).  
  - `NumberOfVehicles`: Number of vehicles involved in the accident.  
  - `NumberOfInjuries`: Number of injuries reported.  
  - `Cause`: Primary cause of the accident.  
  - `Junction`: Whether the accident occurred at a junction (True/False).  
  - `Traffic_Signal`: Whether a traffic signal was present (True/False).  
  - `Bump`: Whether a bump was involved (True/False).  

- **Target Variable**: `SeverityLevel`

---

## Project Structure
- `README.md`: Project documentation.  
- `notebook.ipynb`: Jupyter notebook with EDA and model development.  
- `train.py`: Script for training the final model.  
- `predict.py`: Script for serving predictions.     
- `Dockerfile`: Dockerfile for containerization.  
- `requirements.txt`: Project dependencies.  

---

## Environment Setup

### Local Development
1. Create a virtual environment:  
   `python -m venv venv`
2. Activate the virtual environment:  
   - On Windows: `venv\Scripts\activate`  
   - On Unix or MacOS: `source venv/bin/activate`
3. Install dependencies:  
   `pip install -r requirements.txt`

### Using Docker
1. Build the Docker image:  
   `docker build -t ny-traffic-predictor .`
2. Run the container:  
   `docker run -p 5000:5000 ny-traffic-predictor`

---

## Model Development

### Exploratory Data Analysis (EDA)
- Data cleaning and preparation: Handle missing values, inconsistent formats, and outliers.  
- Feature analysis and importance: Explore relationships between features and the target variable.  
- Target variable distribution: Analyze the distribution of `SeverityLevel` to handle class imbalance if needed.

### Model Training
1. Models Evaluated:
   - Random Forest  
   - XGBoost  
   - Neural Network  
2. Hyperparameter tuning: Optimized using Optuna.  
3. Final Model: Random Forest with the best parameters achieved optimal performance.

---

## Running the Project

### Training the Model
Run the following command to train the model:  
`python train.py`

### Making Predictions
Run the following command to serve predictions:  
`python predict.py`

---

### Prediction Endpoint
The prediction endpoint accepts POST requests with accident details in JSON format.  

### Example Request:
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{
  "County": "Queens", 
  "WeatherCondition": "Rainy", 
  "RoadCondition": "Dry", 
  "VehicleType": "Car", 
  "Cause": "Drunk Driving", 
  "Junction": false, 
  "Traffic_Signal": true, 
  "Bump": false
}' \
http://localhost:5000/predict
```


### Example Response:
```json
{
  "SeverityLevel": "High"
}
```

![Flask](flask.jpg)
