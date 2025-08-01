import os
import sys
import logging
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === Configure Logging ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)

if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(stream_handler)

# === Set and Log MLflow Tracking URI ===
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
logger.info(f"🚀 Using MLflow tracking URI: {tracking_uri}")

# === Load Registered Model ===
model_name = "CaliforniaPriceModelBest"
try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")
    logger.info(f"✅ Loaded model '{model_name}' from MLflow registry.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# === FastAPI App ===
app = FastAPI()

# === Input Schema ===
class HousingFeatures(BaseModel):
    median_income: float
    housing_median_age: float
    avg_rooms: float
    avg_bedrooms: float
    population: float
    avg_occupancy: float
    latitude: float
    longitude: float

# === Prediction Endpoint ===
@app.post("/predict")
def predict(features: HousingFeatures):
    logger.info("📥 Received prediction request.")
    try:
        data = pd.DataFrame([features.dict()])
        prediction = model.predict(data)
        logger.info(f"✅ Prediction successful: {prediction[0]}")
        return {"predicted_house_value": prediction[0]}
    except Exception as e:
        logger.exception("❌ Prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))
