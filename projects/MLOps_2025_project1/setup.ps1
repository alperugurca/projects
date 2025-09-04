# Setup script for Heart Disease Prediction MLOps Project

# Function to check if a command exists
function Test-Command {
    param($Command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $Command) { return $true }
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Function to download the dataset
function Get-HeartDataset {
    $dataDir = "data"
    if (-not (Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir
    }

    if (-not (Test-Path "$dataDir/heart.csv")) {
        Write-Host "Downloading heart disease dataset..."
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ronitgavaskar/Heart_Disease/master/heart.csv" -OutFile "$dataDir/heart.csv"
    }
}

# Check Python installation
if (-not (Test-Command python)) {
    Write-Host "Python is not installed. Please install Python 3.9 or later from https://www.python.org/downloads/"
    exit 1
}

# Check Poetry installation
if (-not (Test-Command poetry)) {
    Write-Host "Installing Poetry..."
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
}

# Check Docker installation
if (-not (Test-Command docker)) {
    Write-Host "Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
}

# Create virtual environment and install dependencies
Write-Host "Installing project dependencies..."
poetry install

# Download dataset
Get-HeartDataset

# Initialize pre-commit hooks
Write-Host "Setting up pre-commit hooks..."
poetry run pre-commit install

# Create necessary directories
$dirs = @(
    "models",
    "models/latest",
    "mlruns",
    "mlartifacts",
    "monitoring/dashboards",
    "logs"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir
    }
}

# Set up environment variables
if (-not (Test-Path .env)) {
    @"
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
PREFECT_API_URL=http://127.0.0.1:4200/api
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_DEFAULT_REGION=us-east-1
LOCALSTACK_ENDPOINT=http://localhost:4566
"@ | Out-File -FilePath .env -Encoding utf8
}

# Function to start services
function Start-Services {
    # Start MLflow server
    Start-Process -NoNewWindow poetry -ArgumentList "run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts" -RedirectStandardOutput "logs/mlflow.log"

    # Start Prefect server
    Start-Process -NoNewWindow poetry -ArgumentList "run prefect server start" -RedirectStandardOutput "logs/prefect.log"

    # Start LocalStack
    docker-compose up -d
}

# Function to stop services
function Stop-Services {
    # Stop MLflow
    Get-Process | Where-Object { $_.Name -like "*mlflow*" } | Stop-Process -Force

    # Stop Prefect
    Get-Process | Where-Object { $_.Name -like "*prefect*" } | Stop-Process -Force

    # Stop LocalStack
    docker-compose down
}

# Parse command line arguments
param(
    [Parameter(Position=0)]
    [string]$Command
)

switch ($Command) {
    "start" {
        Start-Services
        Write-Host "All services started. Check logs/ directory for output."
    }
    "stop" {
        Stop-Services
        Write-Host "All services stopped."
    }
    default {
        Write-Host "Setup completed successfully!"
        Write-Host @"

Next steps:
1. Start all services:
   .\setup.ps1 start

2. Train the model:
   poetry run python src/train.py

3. Deploy the model:
   poetry run python src/deploy.py

4. Start monitoring:
   poetry run python src/monitoring/deploy.py

Services will be available at:
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200
- Model API: http://localhost:8000
- LocalStack: http://localhost:4566

To stop all services:
.\setup.ps1 stop
"@
    }
}
