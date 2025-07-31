import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tempfile
from pydantic import BaseModel, confloat, ValidationError
from typing import List

# === Define schema with Pydantic ===
class HousingRecord(BaseModel):
    median_income: confloat(ge=0)
    housing_median_age: confloat(ge=0)
    avg_rooms: confloat(gt=0)
    avg_bedrooms: confloat(ge=0)
    population: confloat(ge=0)
    avg_occupancy: confloat(gt=0)
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)
    median_house_value: confloat(ge=0)

# === Load and clean data ===
data_path = "data/raw/california.csv"
print(f"📥 Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Rename columns
print("🧹 Renaming columns...")
df.rename(columns={
    'MedInc': 'median_income',
    'HouseAge': 'housing_median_age',
    'AveRooms': 'avg_rooms',
    'AveBedrms': 'avg_bedrooms',
    'Population': 'population',
    'AveOccup': 'avg_occupancy',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'MedHouseVal': 'median_house_value'
}, inplace=True)

print("🔍 Columns after renaming:", list(df.columns))

# === Pydantic data validation ===
print("🔎 Validating data using Pydantic...")
errors = []
for idx, row in df.iterrows():
    try:
        HousingRecord(**row.to_dict())
    except ValidationError as e:
        errors.append((idx, e))

if errors:
    print(f"❌ Validation failed for {len(errors)} rows.")
    for idx, err in errors[:5]:  # Show only first 5 errors
        print(f"Row {idx}: {err}")
    raise ValueError("Dataset validation failed.")
else:
    print("✅ Data validation passed.")

# === Prepare features and target ===
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
print(f"📊 Dataset shape: Features={X.shape}, Target={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✂️ Data split into train and test sets")

# === Model setup ===
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

# === Start MLflow Experiment ===
experiment_name = "housing-price-prediction-5"
print(f"🚀 Starting MLflow experiment: {experiment_name}")
mlflow.set_experiment(experiment_name)

# === Training and logging ===
for name, model in models.items():
    print(f"\n🔧 Training model: {name}")
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"📉 MSE: {mse:.4f}, R²: {r2:.4f}")

        # Log parameters, metrics, model
        mlflow.log_param("model_type", name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Register model version
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "CaliforniaPriceModel")

        # Plot and log residuals
        with tempfile.TemporaryDirectory() as tmp_dir:
            residuals = y_test - preds
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=preds, y=residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f"{name} - Residuals")
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            path = os.path.join(tmp_dir, "residuals.png")
            plt.savefig(path)
            mlflow.log_artifact(path, artifact_path="plots")
            plt.close()

            # Actual vs predicted
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=preds)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.title(f"{name} - Actual vs Predicted")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            path = os.path.join(tmp_dir, "actual_vs_predicted.png")
            plt.savefig(path)
            mlflow.log_artifact(path, artifact_path="plots")
            plt.close()

        print(f"📁 Model logged and registered under version control.")

print("\n✅ All models trained, validated, and tracked successfully!")
