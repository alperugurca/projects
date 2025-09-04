"""Workflow definitions for the heart disease prediction project."""

import os
from datetime import timedelta

import mlflow
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash
from sklearn.model_selection import train_test_split

from src.data import load_data, preprocess_data
from src.monitoring import calculate_metrics, check_data_drift
from src.train import train_model


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def fetch_data(data_path: str) -> pd.DataFrame:
    """Fetch data from the specified path."""
    return load_data(data_path)


@task
def split_and_preprocess_data(data: pd.DataFrame):
    """Split and preprocess the data."""
    return preprocess_data(data)


@task
def train_and_register_model(X_train, X_test, y_train, y_test, scaler):
    """Train the model and register it in MLflow."""
    model, metrics = train_model()
    return model, scaler, metrics


@task
def evaluate_model_performance(model, X_test, y_test):
    """Calculate model performance metrics."""
    metrics = calculate_metrics(model, X_test, y_test)
    return metrics


@task
def check_model_drift(current_data: pd.DataFrame, reference_data: pd.DataFrame):
    """Check for data drift between current and reference data."""
    drift_report = check_data_drift(current_data, reference_data)
    return drift_report


@flow(name="heart-disease-prediction-training")
def training_workflow(
    data_path: str = "data/heart.csv",
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """Main training workflow."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("heart-disease-prediction")

    # Fetch and prepare data
    data = fetch_data(data_path)
    X_train, X_test, y_train, y_test, scaler = split_and_preprocess_data(data)

    # Train and evaluate model
    model, scaler, metrics = train_and_register_model(
        X_train, X_test, y_train, y_test, scaler
    )
    performance_metrics = evaluate_model_performance(model, X_test, y_test)

    # Check for data drift
    drift_metrics = check_model_drift(
        data, data
    )  # Using same data as reference for demo

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "performance": performance_metrics,
        "drift": drift_metrics,
    }


@flow(name="heart-disease-prediction-monitoring")
def monitoring_workflow(
    data_path: str = "data/heart.csv",
    model_path: str = "models/latest/model",
    reference_data_path: str = "data/reference.csv",
):
    """Model monitoring workflow."""
    # Load current data and model
    current_data = fetch_data(data_path)
    reference_data = fetch_data(reference_data_path)

    # Check for data drift
    drift_report = check_model_drift(current_data, reference_data)

    return {"drift_report": drift_report}


if __name__ == "__main__":
    # Run the training workflow
    training_workflow()
