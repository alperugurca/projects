FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY predict.py .
COPY rf_model.pkl .
COPY feature_names.pkl .

EXPOSE 5000

CMD ["python", "predict.py"] 