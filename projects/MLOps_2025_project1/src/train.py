"""Model training module with MLflow tracking."""
import os
import shutil

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import load_data, preprocess_data


def create_model_signature(feature_names):
    """Create model signature for MLflow."""
    from mlflow.types.schema import ColSpec, Schema

    input_schema = Schema([ColSpec("double", name) for name in feature_names])
    output_schema = Schema([ColSpec("integer")])

    return mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)


def create_input_example(X_train):
    """Create input example for MLflow."""
    return X_train[0:1]


def ensure_model_dir():
    """Ensure model directory exists."""
    os.makedirs("models/latest", exist_ok=True)


def export_model(run_id: str):
    """Export model and scaler from MLflow to models/latest directory."""
    ensure_model_dir()

    # Download and copy model
    local_model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model",
    )
    shutil.copytree(local_model_path, "models/latest/model", dirs_exist_ok=True)

    # Download and copy scaler
    local_scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="scaler",
    )
    shutil.copytree(local_scaler_path, "models/latest/scaler", dirs_exist_ok=True)


def train_model():
    """Train the model and log metrics with MLflow."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("heart-disease-prediction")

    # Load and preprocess data
    data = load_data("data/heart.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Get feature names
    feature_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    # Train model
    with mlflow.start_run() as run:
        # Set parameters
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}

        # Create and train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Create model signature and input example
        signature = create_model_signature(feature_names)
        input_example = create_input_example(X_train)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model with signature and input example
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=input_example
        )

        # Save scaler with signature and input example
        scaler_signature = mlflow.models.ModelSignature(
            inputs=signature.inputs, outputs=signature.inputs
        )
        mlflow.sklearn.log_model(
            scaler, "scaler", signature=scaler_signature, input_example=input_example
        )

        # Export model and scaler to models/latest
        export_model(run.info.run_id)

        print("\nTraining Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

    return model, scaler, metrics


if __name__ == "__main__":
    train_model()
