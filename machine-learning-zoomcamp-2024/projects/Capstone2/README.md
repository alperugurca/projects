# Sales Prediction ML Project

## Project Overview
This project is part of the ML Zoomcamp 2024 Competition, focusing on predicting future sales quantities for various items across different stores. The solution uses machine learning to forecast sales based on historical data and various temporal features.

## Setup and Installation

### Virtual Environment Setup
1. Create a virtual environment:
    python -m venv venv

2. Activate the virtual environment:
    - On Windows:
        .\venv\Scripts\activate
    - On Unix or MacOS:
        source venv/bin/activate

### Installing Dependencies
1. Install all required packages:
    pip install -r requirements.txt

2. Verify installation:
    python -c "import pandas, numpy, sklearn, xgboost, bentoml"

## Project Structure
- README.md
- requirements.txt
- train.py           # Training script
- predict.py         # Prediction service
- service.py         # BentoML service
- models/           # Saved models directory
- notebooks/        # Jupyter notebooks for EDA

## Problem Description
The challenge is to predict the quantity of items that will be sold in different stores. This is a regression problem where we need to forecast sales quantities based on:
- Historical sales data
- Store information
- Temporal patterns (daily, weekly, and monthly trends)

## Data Description
The dataset consists of two main files:
- sales.csv: Historical sales data containing:
  - date
  - store_id
  - item_id
  - quantity (target variable)
- test.csv: Test data for predictions

## Solution Approach
1. Feature Engineering:
   - Temporal features (year, month, day, day of week)
   - Weekend indicators
   - Lag features (1-day and 7-day)
   - Rolling mean features (7-day and 30-day windows)

2. Model:
   - XGBoost Regressor
   - Hyperparameters optimized for sales prediction

## Usage

### Training
1. Activate virtual environment (see Setup section)
2. Place data files in /kaggle/input/ml-zoomcamp-2024-competition
3. Run training:
    python train.py

### Making Predictions
1. Start the prediction service:
    python predict.py

2. Or use BentoML service:
    bentoml serve service:svc

### Docker Deployment

#### Prerequisites
- Docker installed on your system
- Model files in the models/ directory (run train.py first)

#### Building the Container
1. Build the Docker image:
    docker build -t sales-predictor

2. Run the container:
    docker run -p 5000:5000 sales-predictor

## Performance Metrics
The model is evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2) Score

## Dependencies
Main dependencies include:
- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- flask
- bentoml
- gunicorn

See requirements.txt for complete list and versions.

## Development
To set up for development:
1. Clone the repository
2. Create and activate virtual environment
3. Install development dependencies:
    pip install -r requirements-dev.txt

## Troubleshooting
Common issues and solutions:
1. If you see "Model not found" error:
   - Ensure you've run train.py first
   - Check models/ directory exists
2. For virtual environment issues:
   - Delete venv directory and recreate
   - Ensure Python version matches requirements

## License
MIT License

## Contact
For questions or issues, please open a GitHub issue.