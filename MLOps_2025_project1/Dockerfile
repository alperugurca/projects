# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the necessary files
COPY models/latest/model /app/models/latest/model
COPY models/latest/scaler /app/models/latest/scaler
COPY src/api.py /app/src/api.py
COPY pyproject.toml poetry.lock ./

# Install Poetry and dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --without dev --no-interaction --no-ansi

# Set environment variables
ENV MODEL_PATH=/app/models/latest/model
ENV SCALER_PATH=/app/models/latest/scaler
ENV MODEL_VERSION=latest

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["python", "src/api.py"]
