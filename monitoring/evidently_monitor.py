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
DATA_PATH = "data/monthly_milk_production.csv"

REPORT_PATH = "reports/evidently_drift_report.html"

MONITORING_DATA_PATH = "reports/evidently_monitoring_data.csv"


# -------------------------------------------------
# 2. Load Data
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


print("\nOriginal Data")
print("----------------")

print("Rows:", len(df))
print("Start:", df["date"].min())
print("End:", df["date"].max())


# -------------------------------------------------
# 3. Time-Series Feature Engineering
# -------------------------------------------------

# Production value one month ago
df["lag_1"] = df["Production"].shift(1)


# Production value 12 months ago
df["lag_12"] = df["Production"].shift(12)


# Mean of the previous 3 months
df["rolling_mean_3"] = (
    df["Production"]
    .shift(1)
    .rolling(window=3)
    .mean()
)


# Mean of the previous 12 months
df["rolling_mean_12"] = (
    df["Production"]
    .shift(1)
    .rolling(window=12)
    .mean()
)


# Remove rows with missing values caused by lag/rolling calculations
df = df.dropna().reset_index(drop=True)


print("\nData After Feature Engineering")
print("--------------------------------")

print("Rows:", len(df))

print(
    df[
        [
            "date",
            "Production",
            "lag_1",
            "lag_12",
            "rolling_mean_3",
            "rolling_mean_12",
        ]
    ].head()
)


# -------------------------------------------------
# 4. Split Reference and Current Data
# -------------------------------------------------

split_point = int(len(df) * 0.80)

reference = df.iloc[:split_point].copy()

current = df.iloc[split_point:].copy()


print("\nReference / Current")
print("---------------------")

print("Reference rows:", len(reference))
print("Current rows:", len(current))


print("\nReference period:")
print(
    reference["date"].min(),
    "->",
    reference["date"].max(),
)


print("\nCurrent period:")
print(
    current["date"].min(),
    "->",
    current["date"].max(),
)


# -------------------------------------------------
# 5. Features Monitored
# -------------------------------------------------

FEATURES = [
    "Production",
    "lag_1",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_12",
]


reference_data = reference[FEATURES].copy()

current_data = current[FEATURES].copy()


print("\nMonitoring Features")
print("-------------------")

for feature in FEATURES:
    print("-", feature)


# -------------------------------------------------
# 6. Manual Drift Magnitude Analysis
# -------------------------------------------------

print("\nDrift Magnitude Analysis")
print("------------------------")

drift_results = []


for feature in FEATURES:

    ks_result = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    reference_mean = reference_data[feature].mean()

    current_mean = current_data[feature].mean()


    mean_change = (
        (current_mean - reference_mean)
        / reference_mean
    ) * 100


    drift_detected = ks_result.pvalue < 0.05


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
# 7. Save Drift Summary CSV
# -------------------------------------------------

os.makedirs(
    "reports",
    exist_ok=True,
)


drift_summary_df = pd.DataFrame(
    drift_results
)


DRIFT_SUMMARY_PATH = (
    "reports/drift_summary.csv"
)


drift_summary_df.to_csv(
    DRIFT_SUMMARY_PATH,
    index=False,
)


print("\nDrift Summary saved to:")
print(DRIFT_SUMMARY_PATH)


# -------------------------------------------------
# 8. Create Evidently Data Drift Report
# -------------------------------------------------

print("\nRunning Evidently Data Drift analysis...")


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
# 9. Save Evidently HTML Report
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
# 10. Save Monitoring Dataset
# -------------------------------------------------

df.to_csv(
    MONITORING_DATA_PATH,
    index=False,
)


print("\nMonitoring Data saved to:")
print(MONITORING_DATA_PATH)


# -------------------------------------------------
# 11. Final Summary
# -------------------------------------------------

drifted_features = drift_summary_df[
    drift_summary_df["drift_detected"] == True
]


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


print("\nEvidently monitoring completed successfully")