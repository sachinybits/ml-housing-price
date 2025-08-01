import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import mlflow
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

# === 1. Define Schema with Pydantic ===
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

# === 2. Load Dataset ===
data_path = "data/raw/california.csv"
print(f"\n📥 Step 1: Loading dataset from: {data_path}")
try:
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print(f"❌ ERROR: Dataset file not found at {data_path}")
    raise

# === 3. Rename Columns to Match Schema ===
print("\n🧹 Step 2: Renaming columns to match Pydantic schema...")
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
print(f"🔍 Columns renamed: {list(df.columns)}")

# === 4. Validate Data Using Pydantic ===
print("\n🔎 Step 3: Validating dataset using Pydantic...")
errors = []
for idx, row in df.iterrows():
    try:
        HousingRecord(**row.to_dict())
    except ValidationError as e:
        print(f"❌ Validation error at row {idx}: {e}")
        errors.append((idx, e))

if errors:
    print(f"🚫 Found {len(errors)} invalid rows during validation. Aborting.")
    raise ValueError("Dataset validation failed.")
else:
    print("✅ All rows validated successfully.")

# === 5. Split Features and Target ===
print("\n📊 Step 4: Preparing features and target...")
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
print(f"✅ Feature shape: {X.shape}, Target shape: {y.shape}")

print("✂️ Splitting into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📦 Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# === 6. Define Models and Hyperparameters ===
print("\n🧠 Step 5: Initializing models with hyperparameter tuning...")
models = {
    "LinearRegression": (LinearRegression(), {}),
    "DecisionTree": (DecisionTreeRegressor(random_state=42), {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10]
    })
}

# === 7. Set up MLflow ===
experiment_name = "housing-price-prediction-verbose12"

# Set tracking URI if using a remote or custom MLflow server
mlflow.set_tracking_uri("http://localhost:5000")  # or use your server URL

print(f"\n🚀 Step 6: Starting MLflow experiment '{experiment_name}'")
mlflow.set_experiment(experiment_name)

#print(f"\n🚀 Step 6: Starting MLflow experiment '{experiment_name}'")
#mlflow.set_experiment(experiment_name)

# === Step 7.1: Store metrics for model comparison later ===
model_metrics = []

# === 8. Train, Evaluate and Log Models ===
for name, (model, param_grid) in models.items():
    print(f"\n🔧 Step 7: Training model: {name}")
    with mlflow.start_run(run_name=name) as run:
        if param_grid:
            print("🔍 Running GridSearchCV...")
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"✅ Best Params: {grid_search.best_params_}")
            mlflow.log_params(grid_search.best_params_)
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        print("🔍 Predicting on test set...")
        preds = best_model.predict(X_test)

        print("📏 Evaluating performance...")
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"📉 {name} | MSE = {mse:.4f} | R² = {r2:.4f}")

        model_metrics.append({
            "model": name,
            "mse": mse,
            "r2": r2,
            "run_id": run.info.run_id
        })

        print("📝 Logging metrics to MLflow...")
        mlflow.log_param("model_type", name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print("🔍 Inferring model signature and logging model...")
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, preds)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        # === SHAP Explainability ===
        print("🔍 Generating SHAP explainability plots...")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if name == "DecisionTree":
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_test)
                elif name == "LinearRegression":
                    X_train_df = pd.DataFrame(X_train, columns=X.columns)
                    X_test_df = pd.DataFrame(X_test, columns=X.columns)
                    explainer = shap.LinearExplainer(best_model, X_train_df)
                    shap_values = explainer.shap_values(X_test_df)
                    X_test = X_test_df
                else:
                    print(f"⚠️ SHAP not supported for model: {name}")
                    shap_values = None

                if shap_values is not None:
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test, show=False)
                    shap_path = os.path.join(tmp_dir, f"shap_summary_{name}.png")
                    plt.savefig(shap_path, bbox_inches='tight')
                    mlflow.log_artifact(shap_path, artifact_path="plots")
                    plt.close()
                    print("✅ SHAP plot logged to MLflow.")
        except Exception as e:
            print(f"⚠️ Failed to generate SHAP plot: {e}")

        print("📊 Creating diagnostic plots...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            residuals = y_test - preds
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=preds, y=residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f"{name} - Residuals")
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            residual_path = os.path.join(tmp_dir, f"residuals_{name}.png")
            plt.savefig(residual_path)
            mlflow.log_artifact(residual_path, artifact_path="plots")
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=preds)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.title(f"{name} - Actual vs Predicted")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            avp_path = os.path.join(tmp_dir, f"actual_vs_predicted_{name}.png")
            plt.savefig(avp_path)
            mlflow.log_artifact(avp_path, artifact_path="plots")
            plt.close()

print("\n✅ Step 9: All models trained, validated, and logged successfully!")

# === Step 10: Model Comparison Plot ===
print("\n📊 Step 10: Generating model performance comparison plot...")
metrics_df = pd.DataFrame(model_metrics)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=metrics_df, x="model", y="r2", ax=axes[0], palette="Blues_d")
axes[0].set_title("R² Score Comparison")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("R² Score")

sns.barplot(data=metrics_df, x="model", y="mse", ax=axes[1], palette="Oranges_d")
axes[1].set_title("Mean Squared Error Comparison")
axes[1].set_ylabel("MSE")

plt.tight_layout()
with tempfile.TemporaryDirectory() as tmp_dir:
    plot_path = os.path.join(tmp_dir, "model_comparison.png")
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(plot_path, artifact_path="plots")
    plt.close()

print("✅ Model comparison plot logged to MLflow: model_comparison.png")

# === Step 11: Register the Best Model ===
print("\n🏆 Step 11: Selecting and registering the best model...")
best_entry = max(model_metrics, key=lambda x: x["r2"])
best_model_name = best_entry["model"]
best_run_id = best_entry["run_id"]
print(f"✅ Best model based on R²: {best_model_name}")

model_uri = f"runs:/{best_run_id}/model"
final_model_name = "CaliforniaPriceModelBest"
#mlflow.register_model(model_uri, final_model_name)
client = MlflowClient()
model_names = [m.name for m in client.search_registered_models()]

if final_model_name not in model_names:
    mlflow.register_model(model_uri, final_model_name)
    print(f"✅ Registered new model: {final_model_name}")
else:
    client.create_model_version(name=final_model_name, source=model_uri, run_id=best_run_id)
    print(f"🔄 Added new version to existing model: {final_model_name}")

print(f"✅ Best model registered as: {final_model_name}")
