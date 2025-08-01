# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY app.py .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn pandas mlflow scikit-learn

EXPOSE 8000

# Run with explicit logging
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
