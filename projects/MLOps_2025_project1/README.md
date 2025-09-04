# Heart Disease Prediction MLOps Project

An end-to-end MLOps project for heart disease prediction with comprehensive monitoring, deployment, and reproducibility.

## Prerequisites

- Python 3.9 or later (but less than 3.11)
- Docker Desktop
- Windows PowerShell or Git Bash
- 4GB+ RAM available
- Internet connection for downloading dependencies

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Run the setup script:
```powershell
.\setup.ps1
```

This will:
- Install Poetry (if not present)
- Create a virtual environment
- Install all dependencies
- Download the dataset
- Set up pre-commit hooks
- Create necessary directories
- Configure environment variables

3. Start all services:
```powershell
.\setup.ps1 start
```

4. Train and deploy the model:
```powershell
# Train the model
poetry run python src/train.py

# Deploy the model
poetry run python src/deploy.py

# Start monitoring
poetry run python src/monitoring/deploy.py
```

## Project Structure

```
├── data/                  # Dataset directory
├── models/               # Saved model artifacts
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── monitoring/     # Model monitoring
│   ├── train.py       # Training pipeline
│   ├── deploy.py      # Deployment script
│   └── api.py         # FastAPI service
├── tests/              # Test files
├── .env               # Environment variables
├── docker-compose.yml # Docker services
├── Dockerfile         # Model serving container
├── pyproject.toml    # Dependencies and config
└── setup.ps1         # Setup script
```

## Dependencies

All dependencies are managed through Poetry and are pinned to specific versions for reproducibility:

### Core Dependencies
- scikit-learn==1.3.0 - Machine learning
- pandas==2.0.3 - Data processing
- numpy==1.24.3 - Numerical computations
- mlflow==2.4.1 - Model tracking
- prefect==2.11.3 - Workflow orchestration
- fastapi==0.100.0 - API framework
- evidently==0.4.0 - Monitoring

### Development Dependencies
- pytest==7.4.0 - Testing
- black==23.3.0 - Code formatting
- pylint==2.17.4 - Linting
- pre-commit==3.3.3 - Git hooks

Full dependency list is in `pyproject.toml` and locked versions in `poetry.lock`.

## Available Services

After starting the services, you can access:

- MLflow UI: http://localhost:5000
  - View experiment tracking
  - Access model registry
  - Monitor metrics

- Prefect UI: http://localhost:4200
  - Monitor workflows
  - View task runs
  - Check schedules

- Model API: http://localhost:8000
  - /docs - API documentation
  - /predict - Make predictions
  - /health - Service health

- LocalStack: http://localhost:4566
  - Local AWS services
  - S3 storage
  - CloudWatch metrics

## Making Predictions

Test the deployed model:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 63,
      "sex": 1,
      "cp": 3,
      "trestbps": 145,
      "chol": 233,
      "fbs": 1,
      "restecg": 0,
      "thalach": 150,
      "exang": 0,
      "oldpeak": 2.3,
      "slope": 0,
      "ca": 0,
      "thal": 1
    }
  }'
```

## Development Workflow

1. Activate the environment:
```bash
poetry shell
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
```

4. Run linting:
```bash
pylint src tests
```

## Stopping Services

To stop all services:
```powershell
.\setup.ps1 stop
```

## Troubleshooting

1. Port conflicts:
   - Check if ports 5000, 4200, 8000, or 4566 are in use
   - Stop conflicting services or modify ports in .env

2. Docker issues:
   - Ensure Docker Desktop is running
   - Try restarting Docker Desktop
   - Check logs in logs/ directory

3. Database locks:
   - Stop all services: `.\setup.ps1 stop`
   - Delete mlflow.db
   - Restart services: `.\setup.ps1 start`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT

## Testing

### Unit Tests

The project includes comprehensive unit tests covering:

1. Model functionality:
   - Data preprocessing
   - Model training
   - Model evaluation
   - Prediction validation

2. API endpoints:
   - Health check
   - Prediction endpoint
   - Input validation
   - Error handling

3. Monitoring system:
   - Performance metrics calculation
   - Data drift detection
   - Prediction drift detection
   - Alert system functionality

### Integration Tests

The project includes end-to-end integration tests that verify the complete system workflow:

1. Complete Pipeline Test:
   - Model training → Deployment → Monitoring
   - Verifies all components work together
   - Tests data flow between components

2. Concurrent Load Testing:
   - Tests system behavior under parallel requests
   - Verifies API stability under load
   - Checks resource handling

3. MLflow Integration:
   - Experiment tracking
   - Model artifact management
   - Metric logging and retrieval

4. Prefect Workflow Integration:
   - Task orchestration
   - Data pipeline execution
   - Error handling and recovery

5. Monitoring System Integration:
   - Real-time metric calculation
   - Drift detection pipeline
   - Alert system integration

### Running Tests

Run all tests with coverage report:
```bash
poetry run pytest --cov=src --cov-report=term-missing
```

Run specific test suites:
```bash
# Unit tests
poetry run pytest tests/test_model.py
poetry run pytest tests/test_api.py
poetry run pytest tests/test_monitoring.py

# Integration tests
poetry run pytest tests/integration/test_end_to_end.py
```

Run tests with verbose output:
```bash
poetry run pytest -v
```

### Test Coverage

The tests cover:
- Input validation and error handling
- Edge cases and boundary conditions
- Performance metric calculations
- Data drift detection
- Alert system functionality
- API endpoint behavior
- Component integration
- System-wide workflows
- Concurrent operations
- Error recovery
