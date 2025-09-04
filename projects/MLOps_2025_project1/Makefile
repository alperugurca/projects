.PHONY: setup test lint format check clean build run-local run-docker down help

help:
	@echo "Available commands:"
	@echo "  setup       - Install dependencies and setup pre-commit hooks"
	@echo "  test        - Run all tests with pytest"
	@echo "  format      - Format code with black and isort"
	@echo "  lint        - Run code quality checks"
	@echo "  check       - Run all code quality checks"
	@echo "  clean       - Clean up temporary files"
	@echo "  build       - Build Docker image"
	@echo "  run-local   - Run services locally"
	@echo "  run-docker  - Run services in Docker"
	@echo "  down        - Stop Docker services"
	@echo "  train       - Train the model"
	@echo "  serve       - Run FastAPI server"

setup:
	poetry install
	poetry run pre-commit install

test:
	poetry run pytest

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run pylint src tests
	poetry run bandit -r src

check: format lint
	poetry run pre-commit run --all-files

train:
	poetry run python src/train.py

serve:
	poetry run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

build:
	docker build -t heart-disease-prediction:latest .

run-local:
	poetry run python -m src.monitoring.monitor &
	poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts &
	poetry run prefect server start &
	poetry run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

run-docker:
	docker-compose up -d

down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
