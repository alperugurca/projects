from datetime import timedelta
from typing import Dict, List, Tuple

import mlflow
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash

from .alerts import AlertSystem
from .metrics import ModelMonitor


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_production_data() -> pd.DataFrame:
    """Load recent production data."""
    # In production, this would load from your production database
    return pd.read_csv("data/production_data.csv")


@task
def evaluate_model(
    monitor: ModelMonitor, data: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float]]:
    """Evaluate model performance and check for drift."""
    # Get predictions
    model = mlflow.sklearn.load_model(f"models/{monitor.model_version}/model")
    X = data.drop("target", axis=1)
    y_true = data["target"]
    y_pred = model.predict(X)

    # Calculate metrics
    performance_metrics = monitor.calculate_performance_metrics(y_true, y_pred)
    drift_metrics = monitor.check_data_drift(X)
    prediction_drift = monitor.check_prediction_drift(y_pred.tolist())

    return performance_metrics, drift_metrics, prediction_drift


@task
def check_violations(
    monitor: ModelMonitor,
    performance_metrics: Dict[str, float],
    drift_metrics: Dict[str, Dict[str, float]],
    prediction_drift: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """Check for threshold violations."""
    return monitor.check_thresholds(
        performance_metrics, drift_metrics, prediction_drift
    )


@task
def handle_violations(
    has_violations: bool, violations: List[str], performance_metrics: Dict[str, float]
) -> None:
    """Handle any detected violations."""
    if not has_violations:
        return

    # Initialize alert system
    alert_system = AlertSystem()

    # Send alerts
    alert_system.trigger_alert(violations, performance_metrics)

    # If accuracy is severely degraded, trigger retraining
    if performance_metrics.get("accuracy", 1.0) < 0.75:
        from prefect.deployments import run_deployment

        run_deployment(
            name="model-training/training",
            flow_name="train_model",
            parameters={"force_retrain": True},
        )


@flow(name="model-monitoring")
def monitoring_flow():
    """Main monitoring flow."""
    # Initialize monitor
    monitor = ModelMonitor()

    # Load production data
    data = load_production_data()

    # Evaluate model and check for drift
    performance_metrics, drift_metrics, prediction_drift = evaluate_model(monitor, data)

    # Log metrics to MLflow
    monitor.log_metrics(performance_metrics, drift_metrics, prediction_drift)

    # Check for violations
    has_violations, violations = check_violations(
        monitor, performance_metrics, drift_metrics, prediction_drift
    )

    # Handle any violations
    handle_violations(has_violations, violations, performance_metrics)


if __name__ == "__main__":
    monitoring_flow()
