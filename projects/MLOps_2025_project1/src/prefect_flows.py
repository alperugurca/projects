"""Prefect workflows for the heart disease prediction project."""

import mlflow
import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@task(name="load_data")
def load_data():
    """Load the heart disease dataset."""
    data = pd.read_csv("data/heart.csv")
    return data


@task(name="preprocess_data")
def preprocess_data(data: pd.DataFrame):
    """Preprocess the data."""
    # Split features and target
    X = data.drop("target", axis=1)
    y = data["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


@task(name="train_model")
def train_model(X_train, y_train):
    """Train the model and log to MLflow."""
    from sklearn.ensemble import RandomForestClassifier

    with mlflow.start_run() as run:
        # Set parameters
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(params)

        return model, run.info.run_id


@task(name="evaluate_model")
def evaluate_model(model, X_test, y_test, run_id):
    """Evaluate the model and log metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    with mlflow.start_run(run_id=run_id):
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        return metrics


@task(name="save_model")
def save_model(model, run_id):
    """Save the model to MLflow."""
    with mlflow.start_run(run_id=run_id):
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.save_model(model, "models/latest/model")


@flow(name="train_heart_disease_model")
def training_flow():
    """Main training workflow."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("heart-disease-prediction")

    # Load and preprocess data
    data = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Train model
    model, run_id = train_model(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, run_id)

    # Save model
    save_model(model, run_id)

    print("\nTraining Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")


@flow(name="monitor_model_performance")
def monitoring_flow():
    """Model monitoring workflow."""
    # Load latest data and predictions
    data = load_data()

    # Calculate drift metrics
    # TODO: Implement drift detection

    print("Model monitoring completed")


if __name__ == "__main__":
    training_flow()
