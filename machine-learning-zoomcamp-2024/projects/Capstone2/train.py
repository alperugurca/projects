#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from lightgbm import LGBMRegressor
from datetime import datetime
import pickle
import os

# Define data paths
DATA_PATH = '/kaggle/input/ml-zoomcamp-2024-competition'
MODEL_PATH = 'models'

def prepare_features(df, is_train=True):
    """Create features from the available data"""
    if is_train:
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Create date-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    if is_train:
        df['lag_1'] = df.groupby(['item_id', 'store_id'])['quantity'].shift(1)
        df['lag_7'] = df.groupby(['item_id', 'store_id'])['quantity'].shift(7)
        df['rolling_mean_7'] = df.groupby(['item_id', 'store_id'])['quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df['rolling_mean_30'] = df.groupby(['item_id', 'store_id'])['quantity'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean())
    else:
        df['lag_1'] = 0
        df['lag_7'] = 0
        df['rolling_mean_7'] = 0
        df['rolling_mean_30'] = 0
    
    df = df.fillna(0)
    return df

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train and tune multiple models"""
    models = {}
    results = {}
    
    # XGBoost
    print("\n=== Training XGBoost with GridSearch ===")
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    models['xgboost'] = xgb_grid.best_estimator_
    results['xgboost'] = {
        'best_params': xgb_grid.best_params_,
        'best_score': -xgb_grid.best_score_
    }
    
    # LightGBM
    print("\n=== Training LightGBM with GridSearch ===")
    lgb_params = {
        'n_estimators': [100, 200],
        'num_leaves': [31, 63],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    lgb_model = LGBMRegressor(random_state=42)
    lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    lgb_grid.fit(X_train, y_train)
    models['lightgbm'] = lgb_grid.best_estimator_
    results['lightgbm'] = {
        'best_params': lgb_grid.best_params_,
        'best_score': -lgb_grid.best_score_
    }
    
    # Evaluate models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name].update({
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2
        })
        
        print(f"\nModel: {name}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R2: {r2:.4f}")
        print(f"Best Parameters: {results[name]['best_params']}")
    
    return models, results

def save_model(model, model_name):
    """Save the trained model to disk"""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    model_file = os.path.join(MODEL_PATH, f'{model_name}.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}")

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(f'{DATA_PATH}/sales.csv')
    
    # Prepare features
    print("Preparing features...")
    train_df = prepare_features(train_df)
    
    # Define features
    feature_columns = [
        'year', 'month', 'day', 'day_of_week', 'is_weekend',
        'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30',
        'store_id'
    ]
    
    # Split data
    X = train_df[feature_columns]
    y = train_df['quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("Training models...")
    models, results = train_multiple_models(X_train, X_test, y_train, y_test)
    
    # Find best model
    best_model_name = min(results.items(), key=lambda x: x[1]['test_rmse'])[0]
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    
    # Save best model
    save_model(best_model, best_model_name)
    
    # Save feature columns for inference
    feature_cols_file = os.path.join(MODEL_PATH, 'feature_columns.pkl')
    with open(feature_cols_file, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"Feature columns saved to {feature_cols_file}")
    
    return best_model, feature_columns

if __name__ == "__main__":
    main() 