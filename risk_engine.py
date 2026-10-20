# ============================================================
# PREDICTIVE MAINTENANCE & ANOMALY DETECTION
# 6G-INTEGRATED SMART MANUFACTURING SYSTEM
# ============================================================
# File: risk_engine.py
# Purpose:
#   - Combine sensor, network, quality and anomaly signals
#   - Calculate final maintenance risk score
#   - Classify machine/record risk
#   - Identify dominant risk factors
#   - Generate machine-level risk summary
#   - Prepare dashboard-ready risk data
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "anomaly_detection_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

RISK_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "risk_engine_results.csv"
)

MACHINE_RISK_FILE = os.path.join(
    OUTPUT_DIR,
    "machine_risk_summary.csv"
)

RISK_FACTOR_FILE = os.path.join(
    OUTPUT_DIR,
    "risk_factor_summary.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Weights are intentionally transparent.
#
# Sensor condition:
#       30%
#
# Network / 6G condition:
#       20%
#
# Quality condition:
#       15%
#
# Historical machine behavior:
#       15%
#
# Anomaly detection:
#       20%
#
# Total:
#       100%
# ============================================================

SENSOR_WEIGHT = 0.30
NETWORK_WEIGHT = 0.20
QUALITY_WEIGHT = 0.15
HISTORICAL_WEIGHT = 0.15
ANOMALY_WEIGHT = 0.20


# ============================================================
# 3. REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING ANOMALY DETECTION RESULTS")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nRequired input file not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Please run anomaly_detection.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    return df


# ============================================================
# 5. NUMERIC SAFETY
# ============================================================

def numeric_column(
    df,
    column,
    default=0.0
):

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(default)


# ============================================================
# 6. NORMALIZE 0-1
# ============================================================

def normalize_series(
    series,
    lower=0.0,
    upper=1.0
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    if upper == lower:

        return pd.Series(
            0.0,
            index=values.index
        )

    normalized = (
        (values - lower)
        / (upper - lower)
    )

    return normalized.clip(
        0.0,
        1.0
    )


# ============================================================
# 7. CALCULATE SENSOR RISK
# ============================================================

def calculate_sensor_risk(df):

    print("\nCalculating sensor risk...")

    # --------------------------------------------------------
    # Prefer already engineered sensor risk.
    # --------------------------------------------------------

    if "sensor_risk" in df.columns:

        sensor_risk = normalize_series(
            numeric_column(
                df,
                "sensor_risk"
            )
        )

    else:

        # ----------------------------------------------------
        # Fallback calculation
        # ----------------------------------------------------

        temperature = normalize_series(
            numeric_column(
                df,
                "Temperature_C"
            ),
            30,
            90
        )

        vibration = normalize_series(
            numeric_column(
                df,
                "Vibration_Hz"
            ),
            0.1,
            5.0
        )

        power = normalize_series(
            numeric_column(
                df,
                "Power_Consumption_kW"
            ),
            1.5,
            10.0
        )

        sensor_risk = (
            0.40 * temperature
            + 0.30 * vibration
            + 0.30 * power
        ).clip(
            0,
            1
        )

    return sensor_risk


# ============================================================
# 8. CALCULATE NETWORK / 6G RISK
# ============================================================

def calculate_network_risk(df):

    print("Calculating network / 6G risk...")

    if "network_risk" in df.columns:

        network_risk = normalize_series(
            numeric_column(
                df,
                "network_risk"
            )
        )

    else:

        latency = normalize_series(
            numeric_column(
                df,
                "Network_Latency_ms"
            ),
            1,
            50
        )

        packet_loss = normalize_series(
            numeric_column(
                df,
                "Packet_Loss_%"
            ),
            0,
            5
        )

        network_risk = (
            0.60 * latency
            + 0.40 * packet_loss
        ).clip(
            0,
            1
        )

    return network_risk


# ============================================================
# 9. CALCULATE QUALITY RISK
# ============================================================

def calculate_quality_risk(df):

    print("Calculating quality risk...")

    if "quality_risk" in df.columns:

        quality_risk = normalize_series(
            numeric_column(
                df,
                "quality_risk"
            )
        )

    else:

        defect = normalize_series(
            numeric_column(
                df,
                "Quality_Control_Defect_Rate_%"
            ),
            0,
            10
        )

        error = normalize_series(
            numeric_column(
                df,
                "Error_Rate_%"
            ),
            0,
            15
        )

        quality_risk = (
            0.50 * defect
            + 0.50 * error
        ).clip(
            0,
            1
        )

    return quality_risk


# ============================================================
# 10. CALCULATE HISTORICAL RISK
# ============================================================

def calculate_historical_risk(df):

    print("Calculating historical risk...")

    if "historical_sensor_risk" in df.columns:

        historical_risk = normalize_series(
            numeric_column(
                df,
                "historical_sensor_risk"
            )
        )

    else:

        change_columns = [

            "temperature_change_rate",
            "vibration_change_rate",
            "power_change_rate",
            "latency_change_rate",
            "packet_loss_change_rate",
            "error_change_rate",
            "defect_change_rate",

        ]

        available = [
            column
            for column in change_columns
            if column in df.columns
        ]

        if not available:

            return pd.Series(
                0.0,
                index=df.index
            )

        change_matrix = pd.DataFrame(
            index=df.index
        )

        for column in available:

            values = numeric_column(
                df,
                column
            )

            change_matrix[column] = (
                values.abs()
            )

        # Row-wise average of changes
        historical_risk = (
            change_matrix
            .mean(axis=1)
        )

        # Normalize using robust percentile boundaries
        low = historical_risk.quantile(
            0.05
        )

        high = historical_risk.quantile(
            0.95
        )

        if high <= low:

            historical_risk = pd.Series(
                0.0,
                index=df.index
            )

        else:

            historical_risk = normalize_series(
                historical_risk,
                low,
                high
            )

    return historical_risk


# ============================================================
# 11. CALCULATE ANOMALY RISK
# ============================================================

def calculate_anomaly_risk(df):

    print("Calculating anomaly risk...")

    if "anomaly_score" in df.columns:

        anomaly_risk = normalize_series(
            numeric_column(
                df,
                "anomaly_score"
            )
        )

    elif "is_anomaly" in df.columns:

        anomaly_risk = numeric_column(
            df,
            "is_anomaly"
        ).clip(
            0,
            1
        )

    else:

        anomaly_risk = pd.Series(
            0.0,
            index=df.index
        )

    return anomaly_risk


# ============================================================
# 12. CALCULATE FINAL RISK SCORE
# ============================================================

def calculate_final_risk_score(
    sensor_risk,
    network_risk,
    quality_risk,
    historical_risk,
    anomaly_risk
):

    print("\nCalculating final maintenance risk...")

    # --------------------------------------------------------
    # Weighted risk model
    # --------------------------------------------------------

    final_risk = (

        SENSOR_WEIGHT
        * sensor_risk

        +

        NETWORK_WEIGHT
        * network_risk

        +

        QUALITY_WEIGHT
        * quality_risk

        +

        HISTORICAL_WEIGHT
        * historical_risk

        +

        ANOMALY_WEIGHT
        * anomaly_risk

    )

    final_risk = final_risk.clip(
        0,
        1
    )

    return final_risk


# ============================================================
# 13. RISK SCORE 0-100
# ============================================================

def create_risk_score(final_risk):

    return (
        final_risk
        * 100
    ).round(2)


# ============================================================
# 14. RISK CATEGORY
# ============================================================

def create_risk_category(risk_score):

    category = np.select(

        [

            risk_score < 25,

            risk_score < 50,

            risk_score < 75,

            risk_score >= 75,

        ],

        [

            "Low",

            "Medium",

            "High",

            "Critical",

        ],

        default="Medium"

    )

    return category


# ============================================================
# 15. MAINTENANCE PRIORITY
# ============================================================

def create_priority(
    risk_score
):

    priority = np.select(

        [

            risk_score >= 85,

            risk_score >= 70,

            risk_score >= 50,

            risk_score >= 25,

        ],

        [

            "Immediate",

            "High",

            "Planned",

            "Monitor",

        ],

        default="Routine"

    )

    return priority


# ============================================================
# 16. DOMINANT RISK FACTOR
# ============================================================

def determine_dominant_factor(
    sensor,
    network,
    quality,
    historical,
    anomaly
):

    values = {

        "Sensor": sensor,

        "Network/6G": network,

        "Quality": quality,

        "Historical": historical,

        "Anomaly": anomaly,

    }

    return max(
        values,
        key=values.get
    )


# ============================================================
# 17. GENERATE RISK EXPLANATION
# ============================================================

def generate_risk_explanation(
    row
):

    factors = []

    risk_score = float(
        row.get(
            "maintenance_risk_score",
            0
        )
    )

    # --------------------------------------------------------
    # Sensor
    # --------------------------------------------------------

    if row.get(
        "sensor_risk",
        0
    ) >= 0.70:

        factors.append(
            "Elevated sensor stress"
        )

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    if row.get(
        "network_risk",
        0
    ) >= 0.70:

        factors.append(
            "Network/6G degradation"
        )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    if row.get(
        "quality_risk",
        0
    ) >= 0.70:

        factors.append(
            "High quality/error risk"
        )

    # --------------------------------------------------------
    # Historical
    # --------------------------------------------------------

    if row.get(
        "historical_risk",
        0
    ) >= 0.70:

        factors.append(
            "Abnormal historical change"
        )

    # --------------------------------------------------------
    # Anomaly
    # --------------------------------------------------------

    if row.get(
        "anomaly_risk",
        0
    ) >= 0.70:

        factors.append(
            "Strong anomaly signal"
        )

    # --------------------------------------------------------
    # Final explanation
    # --------------------------------------------------------

    if not factors:

        if risk_score < 25:

            return (
                "Machine operating within "
                "low-risk conditions"
            )

        if risk_score < 50:

            return (
                "Moderate operational risk "
                "requires monitoring"
            )

        return (
            "Risk elevated; detailed "
            "inspection recommended"
        )

    return " | ".join(
        factors
    )


# ============================================================
# 18. BUILD RISK DATASET
# ============================================================

def build_risk_dataset(
    df
):

    print("\n" + "=" * 70)
    print("BUILDING RISK ENGINE")
    print("=" * 70)

    results = df.copy()

    # --------------------------------------------------------
    # Individual risk dimensions
    # --------------------------------------------------------

    sensor_risk = calculate_sensor_risk(
        results
    )

    network_risk = calculate_network_risk(
        results
    )

    quality_risk = calculate_quality_risk(
        results
    )

    historical_risk = calculate_historical_risk(
        results
    )

    anomaly_risk = calculate_anomaly_risk(
        results
    )

    # --------------------------------------------------------
    # Store components
    # --------------------------------------------------------

    results["sensor_risk_engine"] = (
        sensor_risk.round(4)
    )

    results["network_risk_engine"] = (
        network_risk.round(4)
    )

    results["quality_risk_engine"] = (
        quality_risk.round(4)
    )

    results["historical_risk"] = (
        historical_risk.round(4)
    )

    results["anomaly_risk"] = (
        anomaly_risk.round(4)
    )

    # --------------------------------------------------------
    # Final risk
    # --------------------------------------------------------

    final_risk = calculate_final_risk_score(

        sensor_risk,

        network_risk,

        quality_risk,

        historical_risk,

        anomaly_risk

    )

    results["maintenance_risk"] = (
        final_risk.round(4)
    )

    results["maintenance_risk_score"] = (
        create_risk_score(
            final_risk
        )
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    results["maintenance_risk_category"] = (
        create_risk_category(
            results[
                "maintenance_risk_score"
            ]
        )
    )

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    results["maintenance_priority"] = (
        create_priority(
            results[
                "maintenance_risk_score"
            ]
        )
    )

    # --------------------------------------------------------
    # Dominant factor
    # --------------------------------------------------------

    results["dominant_risk_factor"] = [

        determine_dominant_factor(

            sensor_risk.iloc[index],

            network_risk.iloc[index],

            quality_risk.iloc[index],

            historical_risk.iloc[index],

            anomaly_risk.iloc[index]

        )

        for index in range(
            len(results)
        )

    ]

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation_rows = []

    for index in range(
        len(results)
    ):

        row = {

            "maintenance_risk_score":
                results.iloc[index][
                    "maintenance_risk_score"
                ],

            "sensor_risk":
                sensor_risk.iloc[index],

            "network_risk":
                network_risk.iloc[index],

            "quality_risk":
                quality_risk.iloc[index],

            "historical_risk":
                historical_risk.iloc[index],

            "anomaly_risk":
                anomaly_risk.iloc[index],

        }

        explanation_rows.append(
            generate_risk_explanation(
                row
            )
        )

    results[
        "maintenance_risk_explanation"
    ] = explanation_rows

    return results


# ============================================================
# 19. MACHINE LEVEL RISK SUMMARY
# ============================================================

def create_machine_risk_summary(
    results
):

    print("\n" + "=" * 70)
    print("CREATING MACHINE RISK SUMMARY")
    print("=" * 70)

    if "Machine_ID" not in results.columns:

        print(
            "Machine_ID not available."
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

            average_risk_score=(
                "maintenance_risk_score",
                "mean"
            ),

            maximum_risk_score=(
                "maintenance_risk_score",
                "max"
            ),

            critical_records=(
                "maintenance_risk_category",
                lambda x:
                (x == "Critical").sum()
            ),

            high_records=(
                "maintenance_risk_category",
                lambda x:
                (x == "High").sum()
            ),

            medium_records=(
                "maintenance_risk_category",
                lambda x:
                (x == "Medium").sum()
            ),

            low_records=(
                "maintenance_risk_category",
                lambda x:
                (x == "Low").sum()
            ),

            anomaly_records=(
                "is_anomaly",
                "sum"
            )
            if "is_anomaly" in results.columns
            else (
                "maintenance_risk_score",
                lambda x: 0
            ),

        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Risk rate
    # --------------------------------------------------------

    summary["high_or_critical_rate_%"] = (

        (
            summary["critical_records"]
            +
            summary["high_records"]
        )

        /

        summary["total_records"]

        * 100

    )

    # --------------------------------------------------------
    # Machine status
    # --------------------------------------------------------

    summary["machine_risk_status"] = np.select(

        [

            summary["maximum_risk_score"] >= 85,

            summary["average_risk_score"] >= 70,

            summary["average_risk_score"] >= 50,

            summary["average_risk_score"] >= 25,

        ],

        [

            "Critical",

            "High",

            "Medium",

            "Low",

        ],

        default="Low"

    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    summary = summary.sort_values(

        by=[

            "machine_risk_status",

            "maximum_risk_score",

            "average_risk_score",

        ],

        ascending=[

            True,

            False,

            False,

        ]

    )

    return summary


# ============================================================
# 20. RISK FACTOR SUMMARY
# ============================================================

def create_risk_factor_summary(
    results
):

    print("\n" + "=" * 70)
    print("CREATING RISK FACTOR SUMMARY")
    print("=" * 70)

    factor_columns = {

        "Sensor": "sensor_risk_engine",

        "Network/6G": "network_risk_engine",

        "Quality": "quality_risk_engine",

        "Historical": "historical_risk",

        "Anomaly": "anomaly_risk",

    }

    rows = []

    for factor, column in factor_columns.items():

        if column not in results.columns:

            continue

        values = numeric_column(
            results,
            column
        )

        rows.append({

            "risk_factor": factor,

            "average_risk": round(
                values.mean(),
                4
            ),

            "maximum_risk": round(
                values.max(),
                4
            ),

            "high_risk_records": int(
                (values >= 0.70).sum()
            ),

            "high_risk_percentage": round(
                (values >= 0.70).mean()
                * 100,
                2
            ),

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 21. SAVE OUTPUTS
# ============================================================

def save_outputs(
    results,
    machine_summary,
    factor_summary
):

    print("\n" + "=" * 70)
    print("SAVING RISK ENGINE OUTPUTS")
    print("=" * 70)

    results.to_csv(
        RISK_RESULTS_FILE,
        index=False
    )

    print(
        f"\nRisk results:\n"
        f"{RISK_RESULTS_FILE}"
    )

    if machine_summary is not None:

        machine_summary.to_csv(
            MACHINE_RISK_FILE,
            index=False
        )

        print(
            f"\nMachine risk summary:\n"
            f"{MACHINE_RISK_FILE}"
        )

    if factor_summary is not None:

        factor_summary.to_csv(
            RISK_FACTOR_FILE,
            index=False
        )

        print(
            f"\nRisk factor summary:\n"
            f"{RISK_FACTOR_FILE}"
        )


# ============================================================
# 22. FINAL REPORT
# ============================================================

def print_final_report(
    results,
    machine_summary,
    factor_summary
):

    print("\n")
    print("=" * 70)
    print("RISK ENGINE FINAL REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    total = len(results)

    average_score = results[
        "maintenance_risk_score"
    ].mean()

    maximum_score = results[
        "maintenance_risk_score"
    ].max()

    minimum_score = results[
        "maintenance_risk_score"
    ].min()

    print(
        f"\nTotal Records       : "
        f"{total:,}"
    )

    print(
        f"Average Risk Score  : "
        f"{average_score:.2f}/100"
    )

    print(
        f"Maximum Risk Score  : "
        f"{maximum_score:.2f}/100"
    )

    print(
        f"Minimum Risk Score  : "
        f"{minimum_score:.2f}/100"
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print("\nRisk Distribution:")

    distribution = (
        results[
            "maintenance_risk_category"
        ]
        .value_counts()
    )

    for category in [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]:

        count = int(
            distribution.get(
                category,
                0
            )
        )

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"  {category:<10}: "
            f"{count:>7,} "
            f"({percentage:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    print("\nMaintenance Priority:")

    priority_counts = (
        results[
            "maintenance_priority"
        ]
        .value_counts()
    )

    for priority, count in (
        priority_counts.items()
    ):

        print(
            f"  {priority:<12}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Dominant factors
    # --------------------------------------------------------

    print("\nDominant Risk Factors:")

    factor_counts = (
        results[
            "dominant_risk_factor"
        ]
        .value_counts()
    )

    for factor, count in (
        factor_counts.items()
    ):

        print(
            f"  {factor:<15}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Top machines
    # --------------------------------------------------------

    if machine_summary is not None:

        print(
            "\nTop 10 Machines by Maximum Risk:"
        )

        top = (
            machine_summary
            .sort_values(
                "maximum_risk_score",
                ascending=False
            )
            .head(10)
        )

        for _, row in top.iterrows():

            print(

                f"  Machine "
                f"{int(row['Machine_ID']):02d} "
                f"| Avg: "
                f"{row['average_risk_score']:.2f} "
                f"| Max: "
                f"{row['maximum_risk_score']:.2f} "
                f"| Status: "
                f"{row['machine_risk_status']}"

            )

    # --------------------------------------------------------
    # Factor statistics
    # --------------------------------------------------------

    if factor_summary is not None:

        print(
            "\nRisk Factor Statistics:"
        )

        for _, row in (
            factor_summary.iterrows()
        ):

            print(

                f"  {row['risk_factor']:<15} "
                f"| Avg: "
                f"{row['average_risk']:.3f} "
                f"| High-risk: "
                f"{row['high_risk_percentage']:.2f}%"

            )

    print("\n" + "=" * 70)
    print(
        "RISK ENGINE COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)


# ============================================================
# 23. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "PREDICTIVE MAINTENANCE RISK ENGINE"
    )
    print(
        "6G-INTEGRATED SMART MANUFACTURING SYSTEM"
    )
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        df = load_data()

        # ----------------------------------------------------
        # Build risk engine
        # ----------------------------------------------------

        results = build_risk_dataset(
            df
        )

        # ----------------------------------------------------
        # Machine summary
        # ----------------------------------------------------

        machine_summary = (
            create_machine_risk_summary(
                results
            )
        )

        # ----------------------------------------------------
        # Factor summary
        # ----------------------------------------------------

        factor_summary = (
            create_risk_factor_summary(
                results
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_outputs(

            results,

            machine_summary,

            factor_summary

        )

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        print_final_report(

            results,

            machine_summary,

            factor_summary

        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("RISK ENGINE FAILED")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        raise


# ============================================================
# 24. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()