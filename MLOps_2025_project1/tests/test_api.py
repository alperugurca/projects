"""Integration tests for the API."""
import json

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid input."""
    test_input = {
        "features": {
            "age": 65,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
    }

    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1


def test_predict_endpoint_missing_features():
    """Test prediction endpoint with missing features."""
    test_input = {
        "features": {
            "age": 65,
            "sex": 1
            # Missing other features
        }
    }

    response = client.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_values():
    """Test prediction endpoint with invalid feature values."""
    test_input = {
        "features": {
            "age": -1,  # Invalid age
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
    }

    response = client.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_json():
    """Test prediction endpoint with invalid JSON."""
    response = client.post(
        "/predict", data="invalid json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_wrong_method():
    """Test prediction endpoint with wrong HTTP method."""
    test_input = {
        "features": {
            "age": 65,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
    }

    response = client.get("/predict", json=test_input)
    assert response.status_code == 405  # Method not allowed
