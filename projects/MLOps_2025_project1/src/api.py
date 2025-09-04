"""FastAPI application for serving predictions."""
import os
from typing import Dict, List

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Load environment variables
API_HOST = os.getenv("API_HOST", "127.0.0.1")  # Default to localhost
API_PORT = int(os.getenv("API_PORT", "8000"))

# Model and scaler paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning",
    version="1.0.0",
)


# Define input schema
class PredictionInput(BaseModel):
    """Input data for prediction."""

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# Define output schema
class PredictionOutput(BaseModel):
    """Output data for prediction."""

    prediction: int
    probability: float


# Initialize model and scaler as None
model = None
scaler = None


@app.on_event("startup")
def load_model() -> None:
    """Load the model and scaler on startup."""
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or scaler: {str(e)}")


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput) -> Dict[str, float]:
    """Make a prediction."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input data to feature array
        features = [
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal,
        ]

        # Scale features
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return {"prediction": int(prediction), "probability": float(probability)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server(API_HOST, API_PORT)
