import numpy as np
import pandas as pd
import pytest

from src.monitoring.alerts import AlertSystem
from src.monitoring.metrics import ModelMonitor


@pytest.fixture
def sample_monitor():
    """Create a sample monitor instance."""
    return ModelMonitor(
        model_version="test",
        metrics_threshold={
            "accuracy": 0.85,
            "precision": 0.85,
            "recall": 0.85,
            "f1": 0.85,
        },
        drift_threshold=0.05,
    )


@pytest.fixture
def sample_production_data():
    """Create sample production data."""
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


def test_performance_metrics_calculation(sample_monitor):
    """Test performance metrics calculation."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = sample_monitor.calculate_performance_metrics(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert all(
        metric in metrics for metric in ["accuracy", "precision", "recall", "f1"]
    )
    assert all(0 <= value <= 1 for value in metrics.values())


def test_data_drift_detection(sample_monitor, sample_production_data):
    """Test data drift detection."""
    drift_metrics = sample_monitor.check_data_drift(sample_production_data)

    assert isinstance(drift_metrics, dict)
    assert (
        len(drift_metrics) == len(sample_production_data.columns) - 1
    )  # Excluding target

    for feature, metrics in drift_metrics.items():
        assert "statistic" in metrics
        assert "p_value" in metrics
        assert "has_drift" in metrics
        assert isinstance(metrics["has_drift"], bool)


def test_prediction_drift_detection(sample_monitor):
    """Test prediction drift detection."""
    # Test with sufficient data
    recent_predictions = [0, 1, 0, 1, 0] * 20  # 100 predictions
    drift_metrics = sample_monitor.check_prediction_drift(recent_predictions)

    assert isinstance(drift_metrics, dict)
    assert "drift" in drift_metrics
    assert "has_drift" in drift_metrics

    # Test with insufficient data
    short_predictions = [0, 1, 0]
    insufficient_metrics = sample_monitor.check_prediction_drift(short_predictions)
    assert insufficient_metrics["status"] == "insufficient_data"


def test_threshold_violations(sample_monitor):
    """Test threshold violation detection."""
    # Test with good metrics
    good_metrics = {"accuracy": 0.95, "precision": 0.90, "recall": 0.92, "f1": 0.91}

    has_violations, violations = sample_monitor.check_thresholds(
        good_metrics, {"feature1": {"has_drift": False}}, {"has_drift": False}
    )

    assert has_violations is False
    assert len(violations) == 0

    # Test with poor metrics
    poor_metrics = {"accuracy": 0.75, "precision": 0.70, "recall": 0.80, "f1": 0.75}

    has_violations, violations = sample_monitor.check_thresholds(
        poor_metrics, {"feature1": {"has_drift": True}}, {"has_drift": True}
    )

    assert has_violations is True
    assert len(violations) > 0


def test_alert_message_formatting():
    """Test alert message formatting."""
    alert_system = AlertSystem()

    violations = [
        "accuracy below threshold: 0.75 < 0.85",
        "Data drift detected in features: age, sex",
    ]

    metrics = {"accuracy": 0.75, "precision": 0.80, "recall": 0.85, "f1": 0.82}

    message = alert_system.format_alert_message(violations, metrics)

    assert "Model Monitoring Alert" in message
    assert "violations were detected" in message
    assert "Current Metrics" in message
    assert all(str(metric) in message for metric in metrics.values())


def test_alert_system_initialization():
    """Test alert system initialization with different configs."""
    # Test with no config
    alert_system = AlertSystem()
    assert alert_system.email_config is None
    assert alert_system.slack_webhook is None

    # Test with email config
    email_config = {
        "sender": "test@example.com",
        "recipient": "team@example.com",
        "smtp_server": "smtp.example.com",
        "username": "test",
        "password": "test",
    }
    alert_system = AlertSystem(email_config=email_config)
    assert alert_system.email_config == email_config

    # Test with slack webhook
    alert_system = AlertSystem(slack_webhook="https://hooks.slack.com/test")
    assert alert_system.slack_webhook is not None
