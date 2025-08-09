import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import mlflow
import uuid
from datetime import datetime
import mlflow.sklearn
import tempfile
from pydantic import BaseModel, confloat, ValidationError
from typing import List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from prometheus_client import start_http_server, Summary
import time

# Start Prometheus HTTP server
start_http_server(8001)
print("Prometheus metrics server started on port 8001")
TRAIN_DURATION = Summary('train_duration_seconds', 'Time spent training model')

# 1. Define Schema with Pydantic
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

# 2. Load Dataset
data_path = "data/raw/california.csv"
print(f"\nStep 1: Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# 3. Rename Columns to Match Schema
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

# 4. Validate Data Using Pydantic
for idx, row in df.iterrows():
    try:
        HousingRecord(**row.to_dict())
    except ValidationError:
        continue

# 5. Split Features and Target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define Models and Hyperparameters
models = {
    "LinearRegression": (LinearRegression(), {}),
    "DecisionTree": (DecisionTreeRegressor(random_state=42), {
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5]
    })
}

# 7. Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_experiment_name = f"housing-price-prediction-{timestamp}"
mlflow.set_experiment(unique_experiment_name)
mlflow.autolog()

model_metrics = []

#  8. Train, Evaluate and Log Models 
@TRAIN_DURATION.time()
def train_and_log():
    for name, (model, param_grid) in models.items():
        with mlflow.start_run(run_name=name) as run:
            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=3)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                mlflow.log_params(grid.best_params_)
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            signature = infer_signature(X_test, preds)
            mlflow.sklearn.log_model(model, "model", signature=signature)

            model_metrics.append({"model": name, "mse": mse, "r2": r2})

            # SHAP Plot
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    if name == "DecisionTree":
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)
                    else:
                        explainer = shap.LinearExplainer(model, X_train)
                        shap_values = explainer.shap_values(X_test)

                    shap.summary_plot(shap_values, X_test, show=False)
                    shap_path = os.path.join(tmp, f"shap_summary_{name}.png")
                    plt.savefig(shap_path, bbox_inches='tight')
                    mlflow.log_artifact(shap_path, artifact_path="plots")
                    plt.close()
                except Exception as e:
                    print(f"SHAP plot failed: {e}")

                # Residual Plot
                residuals = y_test - preds
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=preds, y=residuals)
                plt.axhline(0, color='red', linestyle='--')
                plt.title(f"{name} - Residuals")
                plt.xlabel("Predicted")
                plt.ylabel("Residuals")
                residual_path = os.path.join(tmp, f"residuals_{name}.png")
                plt.savefig(residual_path)
                mlflow.log_artifact(residual_path, artifact_path="plots")
                plt.close()

                #  Actual vs Predicted 
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=preds)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.title(f"{name} - Actual vs Predicted")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                avp_path = os.path.join(tmp, f"actual_vs_predicted_{name}.png")
                plt.savefig(avp_path)
                mlflow.log_artifact(avp_path, artifact_path="plots")
                plt.close()

train_and_log()

#  9. Comparison Plot
metrics_df = pd.DataFrame(model_metrics)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=metrics_df, x="model", y="r2", ax=axes[0], palette="crest")
axes[0].set_title("R² Score Comparison")
axes[0].set_ylim(0, 1)

sns.barplot(data=metrics_df, x="model", y="mse", ax=axes[1], palette="flare")
axes[1].set_title("Mean Squared Error Comparison")

plt.tight_layout()
with tempfile.TemporaryDirectory() as tmp:
    plot_path = os.path.join(tmp, "model_comparison.png")
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path, artifact_path="plots")
    plt.close()

print("All models trained and visualized. Prometheus running at :8001/metrics")
time.sleep(240)
