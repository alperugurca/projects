import os
import shutil
import tempfile

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prefect import flow, task

from src.api import app
from src.monitoring.flow import monitoring_flow
from src.monitoring.metrics import ModelMonitor
from src.train import train_model_pipeline


@pytest.fixture(scope="module")
def test_data():
    """Create test dataset for integration testing."""
    np.random.seed(42)
    n_samples = 200

    # Generate synthetic data
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

    # Split into train and test
    df = pd.DataFrame(data)
    train_df = df.iloc[:150]
    test_df = df.iloc[150:]

    return train_df, test_df


@pytest.fixture(scope="module")
def test_environment():
    """Set up test environment with temporary directories."""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "models")
    mlflow_dir = os.path.join(temp_dir, "mlruns")
    os.makedirs(model_dir)
    os.makedirs(mlflow_dir)

    # Set up MLflow
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")

    yield {"temp_dir": temp_dir, "model_dir": model_dir, "mlflow_dir": mlflow_dir}

    # Cleanup
    shutil.rmtree(temp_dir)


def test_complete_pipeline(test_data, test_environment):
    """Test complete pipeline from training to monitoring."""
    train_df, test_df = test_data

    # Step 1: Train model and verify artifacts
    with mlflow.start_run() as run:
        model_info = train_model_pipeline(train_df)

        assert model_info["accuracy"] > 0.7
        assert os.path.exists(model_info["model_path"])
        assert os.path.exists(model_info["scaler_path"])

    # Step 2: Test model deployment and API
    client = TestClient(app)

    # Test health check
    health_response = client.get("/")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"

    # Test prediction endpoint
    test_sample = test_df.iloc[0].drop("target")
    prediction_input = {"features": test_sample.to_dict()}

    pred_response = client.post("/predict", json=prediction_input)
    assert pred_response.status_code == 200
    assert "prediction" in pred_response.json()
    assert "probability" in pred_response.json()

    # Step 3: Test monitoring system
    monitor = ModelMonitor()

    # Get predictions for test data
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    predictions = []
    for _, row in X_test.iterrows():
        response = client.post("/predict", json={"features": row.to_dict()})
        predictions.append(response.json()["prediction"])

    # Calculate monitoring metrics
    performance_metrics = monitor.calculate_performance_metrics(
        y_test, np.array(predictions)
    )
    drift_metrics = monitor.check_data_drift(X_test)
    prediction_drift = monitor.check_prediction_drift(predictions)

    # Verify monitoring results
    assert all(0 <= metric <= 1 for metric in performance_metrics.values())
    assert isinstance(drift_metrics, dict)
    assert isinstance(prediction_drift, dict)

    # Check for violations
    has_violations, violations = monitor.check_thresholds(
        performance_metrics, drift_metrics, prediction_drift
    )

    assert isinstance(has_violations, bool)
    assert isinstance(violations, list)


@pytest.mark.asyncio
async def test_concurrent_predictions(test_data, test_environment):
    """Test system behavior under concurrent load."""
    _, test_df = test_data
    client = TestClient(app)

    import asyncio

    import httpx

    async def make_prediction(sample):
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json={"features": sample.to_dict()})
            return response.status_code

    # Test concurrent predictions
    samples = [test_df.iloc[i].drop("target") for i in range(10)]
    tasks = [make_prediction(sample) for sample in samples]

    results = await asyncio.gather(*tasks)
    assert all(status == 200 for status in results)


def test_mlflow_integration(test_data, test_environment):
    """Test MLflow experiment tracking integration."""
    train_df, _ = test_data

    # Start MLflow run
    with mlflow.start_run() as run:
        # Train model
        model_info = train_model_pipeline(train_df)

        # Verify MLflow tracking
        run_id = run.info.run_id
        run_data = mlflow.get_run(run_id)

        assert "accuracy" in run_data.data.metrics
        assert "precision" in run_data.data.metrics
        assert "recall" in run_data.data.metrics
        assert "f1" in run_data.data.metrics

        # Verify model artifacts
        artifacts = mlflow.list_artifacts(run_id)
        assert any(artifact.path.endswith("model") for artifact in artifacts)
        assert any(artifact.path.endswith("scaler") for artifact in artifacts)


def test_prefect_workflow_integration(test_data, test_environment):
    """Test Prefect workflow orchestration."""
    train_df, test_df = test_data

    @task
    def prepare_data():
        return train_df, test_df

    @task
    def train_and_evaluate(data):
        train_data, test_data = data
        model_info = train_model_pipeline(train_data)
        return model_info

    @flow
    def test_flow():
        data = prepare_data()
        return train_and_evaluate(data)

    # Run the flow
    result = test_flow()

    assert "accuracy" in result
    assert "model_path" in result
    assert "scaler_path" in result


def test_monitoring_workflow_integration(test_data, test_environment):
    """Test monitoring workflow integration."""
    _, test_df = test_data

    # Initialize monitoring
    monitor = ModelMonitor()

    # Run monitoring flow
    @flow
    def test_monitoring_flow():
        # Get predictions
        client = TestClient(app)
        predictions = []

        for _, row in test_df.iloc[:10].iterrows():
            response = client.post(
                "/predict", json={"features": row.drop("target").to_dict()}
            )
            predictions.append(response.json()["prediction"])

        # Run monitoring checks
        metrics = monitor.calculate_performance_metrics(
            test_df.iloc[:10]["target"], np.array(predictions)
        )

        drift_metrics = monitor.check_data_drift(test_df.drop("target", axis=1))
        pred_drift = monitor.check_prediction_drift(predictions)

        return metrics, drift_metrics, pred_drift

    # Run the flow
    metrics, drift_metrics, pred_drift = test_monitoring_flow()

    assert isinstance(metrics, dict)
    assert isinstance(drift_metrics, dict)
    assert isinstance(pred_drift, dict)
