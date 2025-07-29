"""Model monitoring module using Evidently AI."""
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.pipeline.column_mapping import ColumnMapping
from sklearn.metrics import accuracy_score, f1_score


def generate_data_drift_report(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, target_col: str = "target"
) -> dict:
    """Generate data drift report using Evidently AI."""
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.numerical_features = [
        col for col in reference_data.columns if col != target_col
    ]

    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(
        reference_data, current_data, column_mapping=column_mapping
    )

    report = data_drift_profile.json()

    # Extract key metrics
    metrics = {
        "data_drift_detected": report["data_drift"]["data"]["data_drift_detected"],
        "share_of_drifted_features": report["data_drift"]["data"][
            "share_of_drifted_features"
        ],
        "number_of_drifted_features": report["data_drift"]["data"][
            "number_of_drifted_features"
        ],
    }

    return metrics


def check_model_performance(
    y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.8
) -> dict:
    """Check model performance and trigger alerts if needed."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    alerts = []
    if metrics["accuracy"] < threshold:
        alerts.append(
            f"Model accuracy ({metrics['accuracy']:.2f}) below threshold ({threshold})"
        )
    if metrics["f1"] < threshold:
        alerts.append(
            f"Model F1 score ({metrics['f1']:.2f}) below threshold ({threshold})"
        )

    return {
        "metrics": metrics,
        "alerts": alerts,
        "status": "warning" if alerts else "ok",
    }
