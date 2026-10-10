import os
import pandas as pd

from scipy.stats import ks_2samp


# -------------------------------------------------
# Evidently imports
# Support newer and older Evidently versions
# -------------------------------------------------
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    NEW_EVIDENTLY = True

except ImportError:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    NEW_EVIDENTLY = False


# -------------------------------------------------
# 1. Paths
# -------------------------------------------------
REFERENCE_DATA_PATH = "monitoring-data/reference.csv"
CURRENT_DATA_PATH = "monitoring-data/current.csv"

REPORT_PATH = "reports/evidently_drift_report.html"
DRIFT_SUMMARY_PATH = "reports/drift_summary.csv"


# -------------------------------------------------
# 2. Features
# -------------------------------------------------
FEATURES = [
    "Production",
    "lag_1",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_12",
]


# -------------------------------------------------
# 3. Feature Engineering Function
# -------------------------------------------------
def prepare_data(path, dataset_name):

    print(f"\nLoading {dataset_name} data...")
    print("Path:", path)

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{dataset_name} data is empty."
        )

    required_columns = [
        "date",
        "Production",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -----------------------------
    # Time-Series Features
    # -----------------------------
    df["lag_1"] = (
        df["Production"]
        .shift(1)
    )

    df["lag_12"] = (
        df["Production"]
        .shift(12)
    )

    df["rolling_mean_3"] = (
        df["Production"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["rolling_mean_12"] = (
        df["Production"]
        .shift(1)
        .rolling(window=12)
        .mean()
    )

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"{dataset_name} does not contain enough rows "
            "after feature engineering."
        )

    print("Rows after feature engineering:", len(df))
    print(
        "Period:",
        df["date"].min(),
        "->",
        df["date"].max(),
    )

    return df


# -------------------------------------------------
# 4. Load Reference and Current Data
# -------------------------------------------------
reference = prepare_data(
    REFERENCE_DATA_PATH,
    "REFERENCE"
)

current = prepare_data(
    CURRENT_DATA_PATH,
    "CURRENT"
)


# -------------------------------------------------
# 5. Select Monitoring Features
# -------------------------------------------------
reference_data = (
    reference[FEATURES]
    .copy()
)

current_data = (
    current[FEATURES]
    .copy()
)


print("\nMonitoring Features")
print("-------------------")

for feature in FEATURES:
    print("-", feature)


# -------------------------------------------------
# 6. Drift Magnitude Analysis
# -------------------------------------------------
print("\nDrift Magnitude Analysis")
print("------------------------")

drift_results = []


for feature in FEATURES:

    ks_result = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    reference_mean = (
        reference_data[feature]
        .mean()
    )

    current_mean = (
        current_data[feature]
        .mean()
    )

    if reference_mean != 0:

        mean_change = (
            (
                current_mean
                - reference_mean
            )
            / reference_mean
        ) * 100

    else:

        mean_change = 0


    drift_detected = (
        ks_result.pvalue
        < 0.05
    )


    print(f"\nFeature: {feature}")

    print(
        f"Reference Mean : "
        f"{reference_mean:.2f}"
    )

    print(
        f"Current Mean   : "
        f"{current_mean:.2f}"
    )

    print(
        f"Mean Change %  : "
        f"{mean_change:.2f}%"
    )

    print(
        f"KS Statistic   : "
        f"{ks_result.statistic:.4f}"
    )

    print(
        f"P-value        : "
        f"{ks_result.pvalue:.6f}"
    )

    print(
        f"Drift Detected : "
        f"{drift_detected}"
    )


    drift_results.append(
        {
            "feature": feature,
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "mean_change_percent": mean_change,
            "ks_statistic": ks_result.statistic,
            "p_value": ks_result.pvalue,
            "drift_detected": drift_detected,
        }
    )


# -------------------------------------------------
# 7. Create Reports Directory
# -------------------------------------------------
os.makedirs(
    "reports",
    exist_ok=True,
)


# -------------------------------------------------
# 8. Save Drift Summary
# -------------------------------------------------
drift_summary_df = pd.DataFrame(
    drift_results
)

drift_summary_df.to_csv(
    DRIFT_SUMMARY_PATH,
    index=False
)

print("\nDrift Summary saved to:")
print(DRIFT_SUMMARY_PATH)


# -------------------------------------------------
# 9. Evidently Data Drift Report
# -------------------------------------------------
print(
    "\nRunning Evidently "
    "Data Drift analysis..."
)


if NEW_EVIDENTLY:

    report = Report(
        [
            DataDriftPreset()
        ]
    )

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

else:

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
    )


# -------------------------------------------------
# 10. Save HTML Report
# -------------------------------------------------
if NEW_EVIDENTLY:

    result.save_html(
        REPORT_PATH
    )

else:

    report.save_html(
        REPORT_PATH
    )


print("\nEvidently HTML Report saved to:")
print(REPORT_PATH)


# -------------------------------------------------
# 11. Final Summary
# -------------------------------------------------
drifted_features = (
    drift_summary_df[
        drift_summary_df[
            "drift_detected"
        ] == True
    ]
)


print("\nFinal Drift Summary")
print("-------------------")

print(
    "Total Features:",
    len(FEATURES)
)

print(
    "Drifted Features:",
    len(drifted_features)
)

print(
    "Drift Share:",
    f"{len(drifted_features) / len(FEATURES):.2%}"
)

print(
    "\nEvidently monitoring "
    "completed successfully"
)