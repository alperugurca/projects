"""Unit tests for data processing module."""
import numpy as np
import pandas as pd
import pytest

from src.data import preprocess_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "age": [45, 50, 55, 60],
            "sex": [1, 0, 1, 0],
            "cp": [1, 2, 3, 4],
            "target": [0, 1, 1, 0],
        }
    )


def test_preprocess_data(sample_data):
    """Test data preprocessing function."""
    X_train, X_test, y_train, y_test, scaler = preprocess_data(sample_data)

    # Check shapes
    assert X_train.shape[1] == 3  # Number of features
    assert X_test.shape[1] == 3
    assert len(y_train.shape) == 1
    assert len(y_test.shape) == 1

    # Check scaling
    assert np.abs(X_train.mean()) < 1e-10  # Scaled data should have mean close to 0
    assert np.abs(X_train.std() - 1) < 1e-10  # Scaled data should have std close to 1

    # Check data split
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)
