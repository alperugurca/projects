from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelMonitor:
    def __init__(
        self,
        model_version: str = "latest",
        metrics_threshold: Dict[str, float] = None,
        drift_threshold: float = 0.05,
    ):
        self.model_version = model_version
        self.metrics_threshold = metrics_threshold or {
            "accuracy": 0.85,
            "precision": 0.85,
            "recall": 0.85,
            "f1": 0.85,
        }
        self.drift_threshold = drift_threshold

        # Load reference data
        self.reference_data = self._load_reference_data()

    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference data from training set."""
        try:
            return pd.read_csv("data/heart.csv")
        except Exception as e:
            raise RuntimeError(f"Failed to load reference data: {str(e)}")

    def calculate_performance_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

    def check_data_drift(
        self, current_data: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Check for data drift using KS test."""
        if feature_columns is None:
            feature_columns = self.reference_data.columns.tolist()
            if "target" in feature_columns:
                feature_columns.remove("target")

        drift_metrics = {}
        for feature in feature_columns:
            ref_data = self.reference_data[feature]
            curr_data = current_data[feature]

            statistic, p_value = ks_2samp(ref_data, curr_data)
            drift_metrics[feature] = {
                "statistic": statistic,
                "p_value": p_value,
                "has_drift": p_value < self.drift_threshold,
            }

        return drift_metrics

    def check_prediction_drift(
        self, recent_predictions: List[int], window_size: int = 100
    ) -> Dict[str, float]:
        """Check for prediction distribution drift."""
        if len(recent_predictions) < window_size:
            return {"status": "insufficient_data"}

        # Calculate class distribution
        recent_dist = np.mean(recent_predictions[-window_size:])
        reference_dist = np.mean(self.reference_data["target"])

        drift = abs(recent_dist - reference_dist)
        return {
            "reference_positive_rate": reference_dist,
            "current_positive_rate": recent_dist,
            "drift": drift,
            "has_drift": drift > self.drift_threshold,
        }

    def log_metrics(
        self,
        performance_metrics: Dict[str, float],
        drift_metrics: Dict[str, Dict[str, float]],
        prediction_drift: Dict[str, float],
    ) -> None:
        """Log metrics to MLflow."""
        with mlflow.start_run(run_name="monitoring"):
            # Log performance metrics
            mlflow.log_metrics(performance_metrics)

            # Log drift metrics
            for feature, metrics in drift_metrics.items():
                mlflow.log_metrics(
                    {
                        f"drift_{feature}_statistic": metrics["statistic"],
                        f"drift_{feature}_p_value": metrics["p_value"],
                    }
                )

            # Log prediction drift
            if "drift" in prediction_drift:
                mlflow.log_metric("prediction_drift", prediction_drift["drift"])

    def check_thresholds(
        self,
        performance_metrics: Dict[str, float],
        drift_metrics: Dict[str, Dict[str, float]],
        prediction_drift: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """Check if any thresholds are violated."""
        violations = []

        # Check performance metrics
        for metric, value in performance_metrics.items():
            if value < self.metrics_threshold.get(metric, 0):
                violations.append(
                    f"{metric} below threshold: {value:.3f} < {self.metrics_threshold[metric]}"
                )

        # Check feature drift
        drifted_features = [
            feature
            for feature, metrics in drift_metrics.items()
            if metrics["has_drift"]
        ]
        if drifted_features:
            violations.append(
                f"Data drift detected in features: {', '.join(drifted_features)}"
            )

        # Check prediction drift
        if prediction_drift.get("has_drift", False):
            violations.append(
                f"Prediction drift detected: {prediction_drift['drift']:.3f} > {self.drift_threshold}"
            )

        return len(violations) > 0, violations
