"""Data processing module for heart disease prediction."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path: str) -> pd.DataFrame:
    """Load the heart disease dataset."""
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame):
    """Preprocess the data for training."""
    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
