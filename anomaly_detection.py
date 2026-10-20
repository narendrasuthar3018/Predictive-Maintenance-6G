# ============================================================
# PREDICTIVE MAINTENANCE & ANOMALY DETECTION
# 6G-INTEGRATED SMART MANUFACTURING SYSTEM
# ============================================================
# File: anomaly_detection.py
# Purpose:
#   - Detect abnormal machine behavior
#   - Detect sensor/network/quality anomalies
#   - Calculate anomaly scores
#   - Classify anomaly severity
#   - Identify anomaly domain
#   - Generate machine-level anomaly summary
#   - Save trained Isolation Forest model
# ============================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "engineered_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "isolation_forest.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "anomaly_features.json"
)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "anomaly_detection_results.csv"
)

MACHINE_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "machine_anomaly_summary.csv"
)


# ============================================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

CONTAMINATION = 0.05


# ============================================================
# 4. ANOMALY DETECTION FEATURES
# ============================================================
# Important:
# We intentionally DO NOT use:
#   - Predictive_Maintenance_Score
#   - Efficiency_Status
#   - preliminary_risk_score
#   - preliminary_risk_category
#   - Machine_ID
#   - Date / Timestamp / datetime
#
# This prevents circular anomaly detection and keeps the
# anomaly model based on actual machine/network/quality behavior.
# ============================================================

ANOMALY_FEATURES = [

    # --------------------------------------------------------
    # MACHINE SENSOR FEATURES
    # --------------------------------------------------------

    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",

    # --------------------------------------------------------
    # NETWORK / 6G FEATURES
    # --------------------------------------------------------

    "Network_Latency_ms",
    "Packet_Loss_%",

    # --------------------------------------------------------
    # QUALITY FEATURES
    # --------------------------------------------------------

    "Quality_Control_Defect_Rate_%",
    "Error_Rate_%",

    # --------------------------------------------------------
    # PRODUCTION
    # --------------------------------------------------------

    "Production_Speed_units_per_hr",

    # --------------------------------------------------------
    # ENGINEERED SENSOR RISK
    # --------------------------------------------------------

    "thermal_risk",
    "mechanical_risk",
    "power_risk",
    "error_risk",
    "quality_risk",

    # --------------------------------------------------------
    # NETWORK RISK
    # --------------------------------------------------------

    "network_risk",
    "network_health",

    # --------------------------------------------------------
    # OPERATIONAL RISK
    # --------------------------------------------------------

    "production_load",
    "error_per_100_units",
    "defect_per_100_units",

    # --------------------------------------------------------
    # COMPOSITE HEALTH
    # --------------------------------------------------------

    "sensor_risk",
    "machine_health_score",

    # --------------------------------------------------------
    # HISTORICAL CHANGE FEATURES
    # --------------------------------------------------------

    "temperature_change_rate",
    "vibration_change_rate",
    "power_change_rate",
    "latency_change_rate",
    "packet_loss_change_rate",
    "error_change_rate",
    "defect_change_rate",
    "production_speed_change_rate",

    # --------------------------------------------------------
    # INSTABILITY FEATURES
    # --------------------------------------------------------

    "temperature_instability",
    "vibration_instability",
    "error_instability",
    "network_latency_instability",
    "packet_loss_instability",

    # --------------------------------------------------------
    # STRESS COUNTS
    # --------------------------------------------------------

    "sensor_stress_count",
    "network_issue_count",
    "quality_issue_count",
]


# ============================================================
# 5. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 70)
    print("LOADING ENGINEERED DATASET")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nEngineered dataset not found:\n{INPUT_FILE}\n\n"
            "Please run feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset loaded successfully.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    return df


# ============================================================
# 6. VALIDATE FEATURES
# ============================================================

def validate_features(df):

    print("\n" + "=" * 70)
    print("VALIDATING ANOMALY FEATURES")
    print("=" * 70)

    available_features = []

    missing_features = []

    for feature in ANOMALY_FEATURES:

        if feature in df.columns:

            available_features.append(feature)

        else:

            missing_features.append(feature)

    print(f"Required features : {len(ANOMALY_FEATURES)}")
    print(f"Available features: {len(available_features)}")

    if missing_features:

        print("\nWARNING: Missing features detected:")

        for feature in missing_features:

            print(f"  - {feature}")

    if len(available_features) < 10:

        raise ValueError(
            "Not enough valid anomaly detection features."
        )

    return available_features


# ============================================================
# 7. PREPARE FEATURE MATRIX
# ============================================================

def prepare_features(df, features):

    print("\n" + "=" * 70)
    print("PREPARING FEATURE MATRIX")
    print("=" * 70)

    X = df[features].copy()

    # Convert everything to numeric
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Median imputation
    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(X)

    # Convert back to DataFrame
    X_imputed = pd.DataFrame(
        X_imputed,
        columns=features,
        index=df.index
    )

    print(
        f"Feature matrix shape: "
        f"{X_imputed.shape[0]:,} × {X_imputed.shape[1]}"
    )

    print(
        f"Remaining missing values: "
        f"{X_imputed.isna().sum().sum()}"
    )

    return X_imputed, imputer


# ============================================================
# 8. TRAIN ISOLATION FOREST
# ============================================================

def train_anomaly_model(X):

    print("\n" + "=" * 70)
    print("TRAINING ISOLATION FOREST")
    print("=" * 70)

    print(f"Estimators    : {N_ESTIMATORS}")
    print(f"Contamination : {CONTAMINATION}")
    print(f"Random state  : {RANDOM_STATE}")

    model = IsolationForest(

        n_estimators=N_ESTIMATORS,

        contamination=CONTAMINATION,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        bootstrap=False,

    )

    model.fit(X)

    print("\nIsolation Forest training completed successfully.")

    return model


# ============================================================
# 9. GENERATE ANOMALY SCORES
# ============================================================

def generate_anomaly_scores(model, X):

    print("\n" + "=" * 70)
    print("GENERATING ANOMALY SCORES")
    print("=" * 70)

    # Isolation Forest:
    # Higher score = more normal
    # Therefore we invert it so:
    # Higher score = more anomalous

    raw_score = -model.score_samples(X)

    prediction = model.predict(X)

    # -1 = anomaly
    # +1 = normal

    is_anomaly = prediction == -1

    # --------------------------------------------------------
    # Min-Max anomaly score
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    anomaly_score = scaler.fit_transform(
        raw_score.reshape(-1, 1)
    ).flatten()

    # --------------------------------------------------------
    # Percentile score
    # --------------------------------------------------------

    rank_series = pd.Series(raw_score)

    anomaly_percentile = (
        rank_series.rank(
            method="average",
            pct=True
        )
        .values
        * 100
    )

    print(
        f"Anomalies detected: "
        f"{is_anomaly.sum():,}"
    )

    print(
        f"Normal records   : "
        f"{(~is_anomaly).sum():,}"
    )

    print(
        f"Anomaly rate     : "
        f"{is_anomaly.mean() * 100:.2f}%"
    )

    return (
        raw_score,
        anomaly_score,
        anomaly_percentile,
        is_anomaly
    )


# ============================================================
# 10. ANOMALY SEVERITY
# ============================================================

def calculate_severity(
    is_anomaly,
    anomaly_percentile
):

    severity = np.full(
        len(is_anomaly),
        "Normal",
        dtype=object
    )

    # Medium anomaly
    medium_mask = (
        is_anomaly
        & (anomaly_percentile < 99)
    )

    severity[medium_mask] = "Medium"

    # High anomaly
    high_mask = (
        is_anomaly
        & (anomaly_percentile >= 99)
        & (anomaly_percentile < 99.5)
    )

    severity[high_mask] = "High"

    # Critical anomaly
    critical_mask = (
        is_anomaly
        & (anomaly_percentile >= 99.5)
    )

    severity[critical_mask] = "Critical"

    return severity


# ============================================================
# 11. DETERMINE ANOMALY DOMAIN
# ============================================================

def determine_anomaly_domain(row):

    domains = []

    # --------------------------------------------------------
    # Sensor conditions
    # --------------------------------------------------------

    sensor_problem = False

    sensor_columns = [

        "high_temperature_flag",
        "high_vibration_flag",
        "high_power_flag",
        "sensor_stress_count",

    ]

    for column in sensor_columns:

        if column in row.index:

            try:

                if float(row[column]) > 0:

                    sensor_problem = True

            except:

                pass

    if sensor_problem:

        domains.append("Sensor")


    # --------------------------------------------------------
    # Network / 6G conditions
    # --------------------------------------------------------

    network_problem = False

    network_columns = [

        "high_latency_flag",
        "high_packet_loss_flag",
        "network_issue_count",

    ]

    for column in network_columns:

        if column in row.index:

            try:

                if float(row[column]) > 0:

                    network_problem = True

            except:

                pass

    if network_problem:

        domains.append("Network/6G")


    # --------------------------------------------------------
    # Quality conditions
    # --------------------------------------------------------

    quality_problem = False

    quality_columns = [

        "high_error_flag",
        "high_defect_flag",
        "quality_issue_count",

    ]

    for column in quality_columns:

        if column in row.index:

            try:

                if float(row[column]) > 0:

                    quality_problem = True

            except:

                pass

    if quality_problem:

        domains.append("Quality")


    # --------------------------------------------------------
    # Operational conditions
    # --------------------------------------------------------

    operational_problem = False

    if "production_load" in row.index:

        try:

            if float(row["production_load"]) >= 0.80:

                operational_problem = True

        except:

            pass

    if operational_problem:

        domains.append("Operational")


    # --------------------------------------------------------
    # Final domain
    # --------------------------------------------------------

    if len(domains) == 0:

        return "Unclassified"

    if len(domains) >= 2:

        return "Multi-Factor"

    return domains[0]


# ============================================================
# 12. EXPLAIN ANOMALY
# ============================================================

def generate_anomaly_explanation(row):

    factors = []

    # Temperature
    if "Temperature_C" in row.index:

        if row["Temperature_C"] >= 80:

            factors.append("High Temperature")

    # Vibration
    if "Vibration_Hz" in row.index:

        if row["Vibration_Hz"] >= 4:

            factors.append("High Vibration")

    # Power
    if "Power_Consumption_kW" in row.index:

        if row["Power_Consumption_kW"] >= 8:

            factors.append("High Power Consumption")

    # Latency
    if "Network_Latency_ms" in row.index:

        if row["Network_Latency_ms"] >= 40:

            factors.append("High Network Latency")

    # Packet loss
    if "Packet_Loss_%" in row.index:

        if row["Packet_Loss_%"] >= 4:

            factors.append("High Packet Loss")

    # Error
    if "Error_Rate_%" in row.index:

        if row["Error_Rate_%"] >= 12:

            factors.append("High Error Rate")

    # Defect
    if "Quality_Control_Defect_Rate_%" in row.index:

        if row["Quality_Control_Defect_Rate_%"] >= 8:

            factors.append("High Defect Rate")

    # Production
    if "Production_Speed_units_per_hr" in row.index:

        if row["Production_Speed_units_per_hr"] <= 100:

            factors.append("Low Production Speed")

    if len(factors) == 0:

        return "Unusual multivariate pattern"

    return " | ".join(factors)


# ============================================================
# 13. CREATE RESULTS
# ============================================================

def create_results(
    df,
    raw_score,
    anomaly_score,
    anomaly_percentile,
    is_anomaly
):

    print("\n" + "=" * 70)
    print("CREATING ANOMALY RESULTS")
    print("=" * 70)

    results = df.copy()

    results["anomaly_raw_score"] = raw_score

    results["anomaly_score"] = anomaly_score

    results["anomaly_percentile"] = anomaly_percentile

    results["is_anomaly"] = is_anomaly.astype(int)

    results["anomaly_label"] = np.where(
        is_anomaly,
        "Anomaly",
        "Normal"
    )

    results["anomaly_severity"] = calculate_severity(
        is_anomaly,
        anomaly_percentile
    )

    # --------------------------------------------------------
    # Domain
    # --------------------------------------------------------

    results["anomaly_domain"] = "Normal"

    anomaly_indices = results.index[
        results["is_anomaly"] == 1
    ]

    for index in anomaly_indices:

        results.loc[
            index,
            "anomaly_domain"
        ] = determine_anomaly_domain(
            results.loc[index]
        )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    results["anomaly_explanation"] = "No significant anomaly"

    for index in anomaly_indices:

        results.loc[
            index,
            "anomaly_explanation"
        ] = generate_anomaly_explanation(
            results.loc[index]
        )

    return results


# ============================================================
# 14. MACHINE LEVEL SUMMARY
# ============================================================

def create_machine_summary(results):

    print("\n" + "=" * 70)
    print("CREATING MACHINE-LEVEL SUMMARY")
    print("=" * 70)

    if "Machine_ID" not in results.columns:

        print(
            "Machine_ID column not found. "
            "Skipping machine summary."
        )

        return None

    summary = (
        results
        .groupby("Machine_ID")
        .agg(

            total_records=(
                "Machine_ID",
                "size"
            ),

            anomaly_count=(
                "is_anomaly",
                "sum"
            ),

            average_anomaly_score=(
                "anomaly_score",
                "mean"
            ),

            maximum_anomaly_score=(
                "anomaly_score",
                "max"
            ),

            critical_count=(
                "anomaly_severity",
                lambda x:
                (x == "Critical").sum()
            ),

            high_count=(
                "anomaly_severity",
                lambda x:
                (x == "High").sum()
            ),

            medium_count=(
                "anomaly_severity",
                lambda x:
                (x == "Medium").sum()
            ),

        )
        .reset_index()
    )

    summary["anomaly_rate_%"] = (
        summary["anomaly_count"]
        / summary["total_records"]
        * 100
    )

    summary["machine_anomaly_status"] = np.select(

        [

            summary["critical_count"] >= 5,

            summary["high_count"] >= 5,

            summary["anomaly_rate_%"] >= 5,

        ],

        [

            "Critical",

            "High",

            "Watch",

        ],

        default="Normal"

    )

    summary = summary.sort_values(
        by=[
            "critical_count",
            "high_count",
            "anomaly_rate_%",
            "average_anomaly_score"
        ],
        ascending=False
    )

    return summary


# ============================================================
# 15. SAVE MODEL
# ============================================================

def save_model(
    model,
    imputer,
    features
):

    print("\n" + "=" * 70)
    print("SAVING ANOMALY MODEL")
    print("=" * 70)

    model_package = {

        "model": model,

        "imputer": imputer,

        "features": features,

        "random_state": RANDOM_STATE,

        "contamination": CONTAMINATION,

    }

    joblib.dump(
        model_package,
        MODEL_FILE
    )

    with open(
        FEATURE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            features,
            file,
            indent=4
        )

    print(
        f"Model saved:\n{MODEL_FILE}"
    )

    print(
        f"Feature configuration saved:\n"
        f"{FEATURE_FILE}"
    )


# ============================================================
# 16. SAVE RESULTS
# ============================================================

def save_results(
    results,
    machine_summary
):

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    print(
        f"Anomaly results saved:\n"
        f"{RESULT_FILE}"
    )

    if machine_summary is not None:

        machine_summary.to_csv(
            MACHINE_SUMMARY_FILE,
            index=False
        )

        print(
            f"Machine summary saved:\n"
            f"{MACHINE_SUMMARY_FILE}"
        )


# ============================================================
# 17. PRINT FINAL REPORT
# ============================================================

def print_final_report(
    results,
    machine_summary
):

    print("\n")
    print("=" * 70)
    print("ANOMALY DETECTION FINAL REPORT")
    print("=" * 70)

    total_records = len(results)

    anomaly_count = int(
        results["is_anomaly"].sum()
    )

    normal_count = (
        total_records
        - anomaly_count
    )

    anomaly_rate = (
        anomaly_count
        / total_records
        * 100
    )

    print(
        f"\nTotal Records      : "
        f"{total_records:,}"
    )

    print(
        f"Normal Records     : "
        f"{normal_count:,}"
    )

    print(
        f"Anomalies          : "
        f"{anomaly_count:,}"
    )

    print(
        f"Anomaly Rate       : "
        f"{anomaly_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    print("\nAnomaly Severity:")

    severity_counts = (
        results[
            results["is_anomaly"] == 1
        ]["anomaly_severity"]
        .value_counts()
    )

    if len(severity_counts) == 0:

        print("No anomalies detected.")

    else:

        for severity in [
            "Medium",
            "High",
            "Critical"
        ]:

            count = int(
                severity_counts.get(
                    severity,
                    0
                )
            )

            print(
                f"  {severity:<10}: "
                f"{count:,}"
            )

    # --------------------------------------------------------
    # Domains
    # --------------------------------------------------------

    print("\nAnomaly Domains:")

    domain_counts = (
        results[
            results["is_anomaly"] == 1
        ]["anomaly_domain"]
        .value_counts()
    )

    for domain, count in domain_counts.items():

        print(
            f"  {domain:<15}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Top machines
    # --------------------------------------------------------

    if machine_summary is not None:

        print("\nTop 10 At-Risk Machines:")

        top_machines = (
            machine_summary
            .head(10)
        )

        for _, row in top_machines.iterrows():

            print(
                f"  Machine {int(row['Machine_ID']):02d} "
                f"| Anomalies: "
                f"{int(row['anomaly_count']):4d} "
                f"| Rate: "
                f"{row['anomaly_rate_%']:.2f}% "
                f"| Status: "
                f"{row['machine_anomaly_status']}"
            )

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# 18. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE & ANOMALY DETECTION")
    print("6G-INTEGRATED SMART MANUFACTURING SYSTEM")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        features = validate_features(df)

        # ----------------------------------------------------
        # Prepare
        # ----------------------------------------------------

        X, imputer = prepare_features(
            df,
            features
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_anomaly_model(
            X
        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        (
            raw_score,
            anomaly_score,
            anomaly_percentile,
            is_anomaly
        ) = generate_anomaly_scores(
            model,
            X
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        results = create_results(

            df,

            raw_score,

            anomaly_score,

            anomaly_percentile,

            is_anomaly

        )

        # ----------------------------------------------------
        # Machine summary
        # ----------------------------------------------------

        machine_summary = create_machine_summary(
            results
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        save_model(

            model,

            imputer,

            features

        )

        # ----------------------------------------------------
        # Save outputs
        # ----------------------------------------------------

        save_results(

            results,

            machine_summary

        )

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        print_final_report(

            results,

            machine_summary

        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("ANOMALY DETECTION FAILED")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        raise


# ============================================================
# 19. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()