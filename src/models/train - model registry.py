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
from mlflow.tracking import MlflowClient

# === Load data ===
data_path = "data/raw/california.csv"
print(f" Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Rename columns to standard names
print(" Renaming dataset columns...")
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

print(" Columns after renaming:", list(df.columns))

# === Split features and target ===
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
print(f" Dataset shape: Features={X.shape}, Target={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Data split into train and test sets")

# === Models to train ===
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

# === Start MLflow Experiment ===
experiment_name = "housing-price-prediction-4"
model_registry_name = "CaliforniaPriceModel"
print(f" Setting MLflow experiment: {experiment_name}")
mlflow.set_experiment(experiment_name)
client = MlflowClient()

# === Train each model ===
for name, model in models.items():
    print(f"\n Training model: {name}")
    with mlflow.start_run(run_name=name) as run:
        print(" Fitting the model...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f" MSE: {mse:.4f}, R²: {r2:.4f}")

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print(" Logging model to MLflow...")
        mlflow.sklearn.log_model(model, artifact_path="model")

        # === Register the model version ===
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f" Registering model to: {model_registry_name}")
        mv = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
        print(f" Registered version: {mv.version}")

        # === Create and log plots ===
        print("Generating plots...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            residuals = y_test - preds
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=preds, y=residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.title(f"{name} - Residual Plot")
            residuals_path = os.path.join(tmp_dir, "residuals.png")
            plt.savefig(residuals_path)
            mlflow.log_artifact(residuals_path, artifact_path="plots")
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=preds)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{name} - Actual vs Predicted")
            preds_path = os.path.join(tmp_dir, "actual_vs_predicted.png")
            plt.savefig(preds_path)
            mlflow.log_artifact(preds_path, artifact_path="plots")
            plt.close()

        print(f" {name} run completed and registered to MLflow Model Registry.")

print("\n✅ All models trained, logged, and versioned successfully!")
