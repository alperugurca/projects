import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.model import evaluate_model, preprocess_data, train_model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "age": np.random.normal(50, 10, n_samples),
        "sex": np.random.binomial(1, 0.5, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trestbps": np.random.normal(130, 20, n_samples),
        "chol": np.random.normal(200, 30, n_samples),
        "fbs": np.random.binomial(1, 0.2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.normal(150, 20, n_samples),
        "exang": np.random.binomial(1, 0.3, n_samples),
        "oldpeak": np.random.normal(1, 1, n_samples),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 4, n_samples),
        "thal": np.random.randint(0, 3, n_samples),
        "target": np.random.binomial(1, 0.4, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def model_and_scaler():
    """Create and return a trained model and scaler."""
    data = sample_data()
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def test_preprocess_data(sample_data):
    """Test data preprocessing function."""
    # Test with valid data
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    X_processed, y_processed = preprocess_data(X, y)

    assert isinstance(X_processed, np.ndarray)
    assert isinstance(y_processed, np.ndarray)
    assert X_processed.shape[1] == X.shape[1]
    assert y_processed.shape[0] == y.shape[0]

    # Test with missing values
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan

    with pytest.raises(ValueError):
        preprocess_data(X_with_nan, y)

    # Test with invalid feature names
    X_invalid = X.copy()
    X_invalid.columns = [f"feature_{i}" for i in range(X.shape[1])]

    with pytest.raises(ValueError):
        preprocess_data(X_invalid, y)


def test_train_model(sample_data):
    """Test model training function."""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    model, scaler = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    assert isinstance(scaler, StandardScaler)

    # Test prediction shape
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    assert predictions.shape[0] == X.shape[0]

    # Test probability predictions
    probabilities = model.predict_proba(X_scaled)
    assert probabilities.shape == (X.shape[0], 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_evaluate_model(model_and_scaler, sample_data):
    """Test model evaluation function."""
    model, scaler = model_and_scaler
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    metrics = evaluate_model(model, scaler, X, y)

    # Check metrics exist and are in valid ranges
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    for metric_name, value in metrics.items():
        assert isinstance(value, float)
        assert 0 <= value <= 1

    # Test with invalid input
    with pytest.raises(ValueError):
        evaluate_model(model, scaler, X.iloc[:, :-1], y)  # Missing feature


def test_model_prediction_values(model_and_scaler):
    """Test model predictions are valid."""
    model, scaler = model_and_scaler

    # Create a single test sample
    test_data = pd.DataFrame(
        {
            "age": [65],
            "sex": [1],
            "cp": [3],
            "trestbps": [145],
            "chol": [233],
            "fbs": [1],
            "restecg": [0],
            "thalach": [150],
            "exang": [0],
            "oldpeak": [2.3],
            "slope": [0],
            "ca": [0],
            "thal": [1],
        }
    )

    # Test prediction
    X_scaled = scaler.transform(test_data)
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
    assert probabilities.shape == (1, 2)
    assert np.isclose(np.sum(probabilities[0]), 1.0)
