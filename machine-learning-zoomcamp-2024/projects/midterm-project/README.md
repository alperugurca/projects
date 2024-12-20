# Therapy Music Recommendation System

## Description
A machine learning system that recommends music based on mental health indicators. The system analyzes factors like anxiety, depression, insomnia, OCD levels, and music preferences to provide personalized recommendations.

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)

### Local Setup
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the API:
    ```bash
    python predict.py
    ```

### Docker Setup
1. Build and run:
```bash
docker build -t music-recommender .
docker run -p 5000:5000 music-recommender
```

## Testing the API
Run the test script:
```bash
python test_api.py
```

Example request:
```json
{
    "Hours per day": 3.0,
    "Anxiety": 7.0,
    "Depression": 5.0,
    "Insomnia": 6.0,
    "OCD": 4.0,
    "Fav genre": "Rock"
}
```

## Project Structure
- `notebook.ipynb`: Model development and analysis
- `predict.py`: API implementation
- `train.py`: Model training
- `test_api.py`: API testing
- `requirements.txt`: Dependencies
- `Dockerfile`: Docker configuration

## Model Performance
- Accuracy: ~74.59%
- API Endpoint: http://localhost:5000/predict