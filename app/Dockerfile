# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY app.py .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install \
        fastapi \
        uvicorn \
        pandas \
        scikit-learn \
        mlflow \
        shap \
        prometheus_client \
        prometheus-fastapi-instrumentator \
        matplotlib \
        seaborn \
        pydantic


EXPOSE 8000

# Run with explicit logging
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]

