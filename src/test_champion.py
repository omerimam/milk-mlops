import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels

from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------------------------------
# 1. MLflow Configuration
# -------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/MilkSARIMA@champion"

DATA_PATH = "data/monthly_milk_production.csv"


# -------------------------------------------------
# 2. Load Champion Model from MLflow Registry
# -------------------------------------------------
print("\nLoading model from MLflow Registry...")
print("Model URI:", MODEL_URI)

model = mlflow.statsmodels.load_model(MODEL_URI)

print("\nChampion model loaded successfully")
print("Model type:", type(model))


# -------------------------------------------------
# 3. Load Data
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")


# -------------------------------------------------
# 4. Get same Test Data
# -------------------------------------------------
test = df["Production"].iloc[-12:]


# -------------------------------------------------
# 5. Forecast using Champion Model
# -------------------------------------------------
forecast = model.forecast(
    steps=len(test)
)


# -------------------------------------------------
# 6. Evaluation
# -------------------------------------------------
mae = mean_absolute_error(
    test,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        forecast
    )
)


print("\nChampion Model Evaluation")
print("-------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


# -------------------------------------------------
# 7. Show Actual vs Forecast
# -------------------------------------------------
comparison = pd.DataFrame(
    {
        "Actual": test.values,
        "Forecast": forecast.values
    },
    index=test.index
)

print("\nActual vs Forecast")
print("------------------")
print(comparison)