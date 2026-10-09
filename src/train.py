import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.statsmodels

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------------------------------
# MLflow Configuration
# -------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Milk_SARIMA_Experiment")


# -------------------------------------------------
# 1. Paths
# -------------------------------------------------
DATA_PATH = "data/monthly_milk_production.csv"
MODEL_PATH = "models/sarima_model.pkl"
REPORT_PATH = "reports/forecast.png"


# -------------------------------------------------
# 2. Load data
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

print("\nData:")
print(df.head())

print("\nRows:", len(df))
print("Start:", df.index.min())
print("End:", df.index.max())


# -------------------------------------------------
# 3. Train / Test split
# Last 12 months = test
# -------------------------------------------------
train = df["Production"].iloc[:-12]
test = df["Production"].iloc[-12:]

print("\nTrain rows:", len(train))
print("Test rows:", len(test))


# -------------------------------------------------
# 4. SARIMA Parameters
# -------------------------------------------------
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 12)


# -------------------------------------------------
# 5. Start MLflow Run
# -------------------------------------------------
with mlflow.start_run(run_name="SARIMA_111_111_12") as run:

    print("\nMLflow Run started")
    print("Run ID:", run.info.run_id)


    # -------------------------------------------------
    # 6. Train SARIMA model
    # -------------------------------------------------
    model = SARIMAX(
        train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)


    # -------------------------------------------------
    # 7. Forecast
    # -------------------------------------------------
    forecast = result.forecast(
        steps=len(test)
    )


    # -------------------------------------------------
    # 8. Evaluation
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

    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")


    # -------------------------------------------------
    # 9. Log Parameters to MLflow
    # -------------------------------------------------
    mlflow.log_param(
        "order",
        str(ORDER)
    )

    mlflow.log_param(
        "seasonal_order",
        str(SEASONAL_ORDER)
    )

    mlflow.log_param(
        "train_rows",
        len(train)
    )

    mlflow.log_param(
        "test_rows",
        len(test)
    )


    # -------------------------------------------------
    # 10. Log Metrics to MLflow
    # -------------------------------------------------
    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "RMSE",
        rmse
    )


    # -------------------------------------------------
    # 11. Save model locally
    # -------------------------------------------------
    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        result,
        MODEL_PATH
    )

    print(
        f"\nModel saved locally to: {MODEL_PATH}"
    )


    # -------------------------------------------------
    # 12. Save forecast chart
    # -------------------------------------------------
    os.makedirs(
        "reports",
        exist_ok=True
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        train.index,
        train,
        label="Train"
    )

    plt.plot(
        test.index,
        test,
        label="Actual"
    )

    plt.plot(
        test.index,
        forecast,
        label="Forecast"
    )

    plt.xlabel("Date")
    plt.ylabel("Milk Production")
    plt.title("Milk Production - SARIMA Baseline")

    plt.legend()
    plt.tight_layout()

    plt.savefig(REPORT_PATH)
    plt.close()

    print(
        f"Forecast chart saved to: {REPORT_PATH}"
    )


    # -------------------------------------------------
    # 13. Log normal artifacts to MLflow
    # -------------------------------------------------
    mlflow.log_artifact(
        MODEL_PATH,
        artifact_path="model"
    )

    mlflow.log_artifact(
        REPORT_PATH,
        artifact_path="reports"
    )

    print("\nArtifacts logged to MLflow")


    # -------------------------------------------------
    # 14. Log + Register MLflow Model
    # -------------------------------------------------
    model_info = mlflow.statsmodels.log_model(
        statsmodels_model=result,
        name="sarima_model",
        registered_model_name="MilkSARIMA"
    )

    print("\nMLflow Model registered successfully")
    print("Model URI:", model_info.model_uri)


# -------------------------------------------------
# 15. Finished
# -------------------------------------------------
print("\nMLflow Run completed successfully")