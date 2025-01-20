# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files and application code
COPY models/ ./models/
COPY predict.py .
COPY service.py .

# Make port 5000 available
EXPOSE 5000

# Set environment variables
ENV MODEL_PATH=models/model.joblib
ENV PYTHONUNBUFFERED=1

# Run the service
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "service:app"] 