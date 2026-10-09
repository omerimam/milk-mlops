import time
import os


from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from fastapi.responses import Response
from contextlib import asynccontextmanager

import mlflow
import mlflow.statsmodels

from fastapi import FastAPI, Query, HTTPException


# -------------------------------------------------
# 1. MLflow Configuration
# -------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/MilkSARIMA@champion"
)
# -------------------------------------------------
# Prometheus Metrics
# -------------------------------------------------
FORECAST_REQUESTS = Counter(
    "milk_forecast_requests_total",
    "Total number of milk forecast requests"
)

FORECAST_ERRORS = Counter(
    "milk_forecast_errors_total",
    "Total number of milk forecast errors"
)

FORECAST_LATENCY = Histogram(
    "milk_forecast_latency_seconds",
    "Time spent generating milk forecasts"
)

# -------------------------------------------------
# 2. Lifespan
# Load Champion Model when FastAPI starts
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\nLoading champion model...")
    print("Model URI:", MODEL_URI)

    app.state.model = mlflow.statsmodels.load_model(
        MODEL_URI
    )

    print("Champion model loaded successfully")

    yield

    print("API shutting down")


# -------------------------------------------------
# 3. Create FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="Milk Forecast API",
    version="1.0",
    lifespan=lifespan
)


# -------------------------------------------------
# 4. Home Endpoint
# -------------------------------------------------
@app.get("/")
def home():

    return {
        "message": "Milk Forecast API is running"
    }


# -------------------------------------------------
# 5. Health Endpoint
# -------------------------------------------------
@app.get("/health")
def health():

    model_loaded = hasattr(
        app.state,
        "model"
    )

    return {
        "status": "ready" if model_loaded else "not_ready",
        "model_loaded": model_loaded,
        "model_uri": MODEL_URI
    }


# -------------------------------------------------
# 6. Model Information Endpoint
# -------------------------------------------------
@app.get("/model")
def model_info():

    model_loaded = hasattr(
        app.state,
        "model"
    )

    return {
        "model_name": "MilkSARIMA",
        "alias": "champion",
        "model_uri": MODEL_URI,
        "loaded": model_loaded
    }


# -------------------------------------------------
# 7. Forecast Endpoint
# -------------------------------------------------
@app.get("/forecast")
def forecast(
    months: int = Query(
        default=12,
        ge=1,
        le=36
    )
):

    start_time = time.perf_counter()

    FORECAST_REQUESTS.inc()

    try:

        if not hasattr(app.state, "model"):

            raise RuntimeError(
                "Model is not loaded"
            )

        model = app.state.model

        predictions = model.forecast(
            steps=months
        )

        forecast_values = [
            float(value)
            for value in predictions
        ]

        return {
            "model": "MilkSARIMA",
            "alias": "champion",
            "months": months,
            "forecast": forecast_values
        }

    except Exception as e:

        FORECAST_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        FORECAST_LATENCY.observe(
            elapsed_time
        )
# -------------------------------------------------
# Prometheus Metrics Endpoint
# -------------------------------------------------
@app.get("/metrics", include_in_schema=False)
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )