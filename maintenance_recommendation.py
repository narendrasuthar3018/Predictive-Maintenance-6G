# ============================================================
# PREDICTIVE MAINTENANCE RECOMMENDATION ENGINE
# 6G-INTEGRATED SMART MANUFACTURING SYSTEM
#
# Version : 3.0.0
# Author  : Narendra Suthar
#
# Input:
#   outputs/risk_engine_results.csv
#
# Outputs:
#   outputs/maintenance_recommendation_results.csv
#   outputs/machine_maintenance_summary.csv
#   outputs/maintenance_priority_queue.csv
#   outputs/fleet_maintenance_summary.csv
# ============================================================

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

RISK_FILE = (
    OUTPUT_DIR /
    "risk_engine_results.csv"
)

RECOMMENDATION_FILE = (
    OUTPUT_DIR /
    "maintenance_recommendation_results.csv"
)

MACHINE_SUMMARY_FILE = (
    OUTPUT_DIR /
    "machine_maintenance_summary.csv"
)

PRIORITY_QUEUE_FILE = (
    OUTPUT_DIR /
    "maintenance_priority_queue.csv"
)

FLEET_SUMMARY_FILE = (
    OUTPUT_DIR /
    "fleet_maintenance_summary.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

PRIORITY_ORDER = {
    "P1 - Emergency": 1,
    "P2 - High": 2,
    "P3 - Planned": 3,
    "P4 - Routine": 4
}


RISK_ORDER = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1
}


# ============================================================
# SAFE NUMERIC HELPER
# ============================================================

def numeric_series(
    df,
    column,
    default=0.0
):
    """
    Safely return a numeric Series.

    Handles:
    - missing columns
    - duplicate columns
    - strings
    - NaN
    """

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    value = df[column]

    # Duplicate column names can return a DataFrame.
    if isinstance(value, pd.DataFrame):

        value = value.iloc[:, 0]

    return pd.to_numeric(
        value,
        errors="coerce"
    ).fillna(default)


# ============================================================
# SAFE TEXT HELPER
# ============================================================

def text_series(
    df,
    column,
    default="Unknown"
):
    """
    Safely return a text Series.
    """

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype="object"
        )

    value = df[column]

    if isinstance(value, pd.DataFrame):

        value = value.iloc[:, 0]

    value = value.astype(str)

    value = value.replace(
        {
            "nan": default,
            "None": default,
            "NaN": default
        }
    )

    return value


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(
    df,
    candidates
):
    """
    Find the first available column from candidate names.
    """

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(
            candidate
        ).strip().lower()

        if key in normalized:

            return normalized[key]

    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_risk_results():

    print("\n" + "=" * 75)
    print("LOADING RISK ENGINE RESULTS")
    print("=" * 75)

    if not RISK_FILE.exists():

        raise FileNotFoundError(
            f"Risk engine output not found:\n{RISK_FILE}"
        )

    df = pd.read_csv(
        RISK_FILE,
        low_memory=False
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Remove duplicate column names
    # --------------------------------------------------------

    duplicated = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicated:

        print(
            f"Duplicate columns detected : "
            f"{len(duplicated)}"
        )

        df = df.loc[
            :,
            ~df.columns.duplicated()
        ].copy()

        print(
            "Duplicate columns removed."
        )

    return df


# ============================================================
# STANDARDIZE COLUMNS
# ============================================================

def standardize_columns(df):

    print("\n" + "=" * 75)
    print("STANDARDIZING INPUT COLUMNS")
    print("=" * 75)

    df = df.copy()

    # --------------------------------------------------------
    # Machine ID
    # --------------------------------------------------------

    machine_column = find_column(
        df,
        [
            "Machine_ID",
            "Machine ID",
            "machine_id",
            "Machine"
        ]
    )

    if machine_column is None:

        df["Machine_ID"] = "Unknown"

    elif machine_column != "Machine_ID":

        df["Machine_ID"] = df[
            machine_column
        ]

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_column = find_column(
        df,
        [
            "maintenance_risk_score",
            "final_maintenance_risk",
            "maintenance_risk",
            "risk_score",
            "Risk_Score"
        ]
    )

    if risk_column is None:

        # Try preliminary risk score.
        risk_column = find_column(
            df,
            [
                "preliminary_risk_score",
                "risk"
            ]
        )

    if risk_column is None:

        df["maintenance_risk_score"] = 0.0

    else:

        df["maintenance_risk_score"] = (
            numeric_series(
                df,
                risk_column
            )
        )

    # --------------------------------------------------------
    # Normalize risk score if it appears to be 0-1
    # --------------------------------------------------------

    maximum_risk = (
        df["maintenance_risk_score"]
        .max()
    )

    if maximum_risk <= 1.0:

        df["maintenance_risk_score"] = (
            df["maintenance_risk_score"]
            * 100
        )

    df["maintenance_risk_score"] = (
        df["maintenance_risk_score"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    category_column = find_column(
        df,
        [
            "maintenance_risk_category",
            "risk_category",
            "Risk_Category"
        ]
    )

    if category_column is not None:

        df["maintenance_risk_category"] = (
            text_series(
                df,
                category_column
            )
        )

    else:

        df["maintenance_risk_category"] = (
            pd.cut(
                df["maintenance_risk_score"],
                bins=[
                    -np.inf,
                    25,
                    50,
                    75,
                    np.inf
                ],
                labels=[
                    "Low",
                    "Medium",
                    "High",
                    "Critical"
                ]
            )
            .astype(str)
        )

    # --------------------------------------------------------
    # Anomaly score
    # --------------------------------------------------------

    anomaly_column = find_column(
        df,
        [
            "anomaly_score",
            "Anomaly_Score",
            "isolation_forest_score"
        ]
    )

    if anomaly_column is not None:

        df["anomaly_score"] = (
            numeric_series(
                df,
                anomaly_column
            )
        )

    else:

        df["anomaly_score"] = 0.0

    # --------------------------------------------------------
    # Anomaly flag
    # --------------------------------------------------------

    anomaly_flag_column = find_column(
        df,
        [
            "is_anomaly",
            "anomaly_flag",
            "Anomaly"
        ]
    )

    if anomaly_flag_column is not None:

        anomaly_values = text_series(
            df,
            anomaly_flag_column,
            "0"
        ).str.lower()

        df["is_anomaly"] = (
            anomaly_values
            .isin(
                [
                    "1",
                    "true",
                    "yes",
                    "anomaly",
                    "anomalous"
                ]
            )
            .astype(int)
        )

    else:

        # Use anomaly severity if available.
        severity_column = find_column(
            df,
            [
                "anomaly_severity",
                "anomaly_level"
            ]
        )

        if severity_column is not None:

            severity = text_series(
                df,
                severity_column
            ).str.lower()

            df["is_anomaly"] = (
                severity
                .isin(
                    [
                        "high",
                        "critical",
                        "severe",
                        "anomaly"
                    ]
                )
                .astype(int)
            )

        else:

            # Conservative threshold.
            df["is_anomaly"] = (
                df["anomaly_score"] >= 0.70
            ).astype(int)

    # --------------------------------------------------------
    # Sensor risk
    # --------------------------------------------------------

    sensor_column = find_column(
        df,
        [
            "sensor_risk",
            "sensor_risk_score",
            "sensor_health_risk"
        ]
    )

    if sensor_column is not None:

        df["sensor_risk"] = (
            numeric_series(
                df,
                sensor_column
            )
        )

    else:

        df["sensor_risk"] = 0.0

    # --------------------------------------------------------
    # Network risk
    # --------------------------------------------------------

    network_column = find_column(
        df,
        [
            "network_risk",
            "network_6g_risk",
            "network_risk_score",
            "Network_Risk"
        ]
    )

    if network_column is not None:

        df["network_risk"] = (
            numeric_series(
                df,
                network_column
            )
        )

    else:

        df["network_risk"] = 0.0

    # --------------------------------------------------------
    # Quality risk
    # --------------------------------------------------------

    quality_column = find_column(
        df,
        [
            "quality_risk",
            "quality_risk_score",
            "Quality_Risk"
        ]
    )

    if quality_column is not None:

        df["quality_risk"] = (
            numeric_series(
                df,
                quality_column
            )
        )

    else:

        df["quality_risk"] = 0.0

    # --------------------------------------------------------
    # Historical risk
    # --------------------------------------------------------

    historical_column = find_column(
        df,
        [
            "historical_risk",
            "historical_risk_score",
            "history_risk"
        ]
    )

    if historical_column is not None:

        df["historical_risk"] = (
            numeric_series(
                df,
                historical_column
            )
        )

    else:

        df["historical_risk"] = 0.0

    return df


# ============================================================
# DETECT SENSOR CONDITIONS
# ============================================================

def detect_sensor_conditions(df):

    temperature = numeric_series(
        df,
        "Temperature_C"
    )

    vibration = numeric_series(
        df,
        "Vibration_Hz"
    )

    power = numeric_series(
        df,
        "Power_Consumption_kW"
    )

    error_rate = numeric_series(
        df,
        "Error_Rate_%"
    )

    conditions = []

    for index in df.index:

        issues = []

        temp = temperature.loc[index]
        vib = vibration.loc[index]
        pwr = power.loc[index]
        err = error_rate.loc[index]

        if temp > 80:

            issues.append(
                "High temperature"
            )

        elif temp > 70:

            issues.append(
                "Elevated temperature"
            )

        if vib > 8:

            issues.append(
                "Excessive vibration"
            )

        elif vib > 6:

            issues.append(
                "Elevated vibration"
            )

        if pwr > 90:

            issues.append(
                "High power consumption"
            )

        if err > 5:

            issues.append(
                "High equipment error rate"
            )

        conditions.append(
            "; ".join(issues)
            if issues
            else "Normal sensor condition"
        )

    return pd.Series(
        conditions,
        index=df.index
    )


# ============================================================
# DETECT NETWORK CONDITIONS
# ============================================================

def detect_network_conditions(df):

    latency = numeric_series(
        df,
        "Network_Latency_ms"
    )

    packet_loss = numeric_series(
        df,
        "Packet_Loss_%"
    )

    conditions = []

    for index in df.index:

        issues = []

        lat = latency.loc[index]
        loss = packet_loss.loc[index]

        if lat > 80:

            issues.append(
                "Very high network latency"
            )

        elif lat > 50:

            issues.append(
                "Elevated network latency"
            )

        if loss > 5:

            issues.append(
                "High packet loss"
            )

        elif loss > 2:

            issues.append(
                "Elevated packet loss"
            )

        conditions.append(
            "; ".join(issues)
            if issues
            else "Normal network condition"
        )

    return pd.Series(
        conditions,
        index=df.index
    )


# ============================================================
# DETECT QUALITY CONDITIONS
# ============================================================

def detect_quality_conditions(df):

    defect = numeric_series(
        df,
        "Quality_Control_Defect_Rate_%"
    )

    production_error = numeric_series(
        df,
        "Error_Rate_%"
    )

    conditions = []

    for index in df.index:

        issues = []

        defect_value = defect.loc[index]
        error_value = production_error.loc[index]

        if defect_value > 5:

            issues.append(
                "High defect rate"
            )

        elif defect_value > 3:

            issues.append(
                "Elevated defect rate"
            )

        if error_value > 5:

            issues.append(
                "High production error rate"
            )

        conditions.append(
            "; ".join(issues)
            if issues
            else "Normal quality condition"
        )

    return pd.Series(
        conditions,
        index=df.index
    )


# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

def calculate_root_cause(row):

    factors = {
        "Sensor / Mechanical": float(
            row.get(
                "sensor_risk",
                0
            )
        ),

        "Network / 6G": float(
            row.get(
                "network_risk",
                0
            )
        ),

        "Quality / Production": float(
            row.get(
                "quality_risk",
                0
            )
        ),

        "Historical Trend": float(
            row.get(
                "historical_risk",
                0
            )
        )
    }

    # --------------------------------------------------------
    # Anomaly is a valid risk factor ONLY when the anomaly
    # detector explicitly classifies the record as anomalous.
    # --------------------------------------------------------

    is_anomaly = int(
        row.get(
            "is_anomaly",
            0
        )
    )

    anomaly_score = float(
        row.get(
            "anomaly_score",
            0
        )
    )

    if is_anomaly == 1:
        factors["Anomaly"] = anomaly_score
    else:
        factors["Anomaly"] = 0.0

    # --------------------------------------------------------
    # Select the strongest actual risk contributor
    # --------------------------------------------------------

    cause = max(
        factors,
        key=factors.get
    )

    return cause

# ============================================================
# MAINTENANCE PRIORITY
# ============================================================

def calculate_priority(
    risk,
    anomaly,
    category
):

    risk = float(risk)
    anomaly = float(anomaly)

    category = str(
        category
    ).strip()

    if (
        category == "Critical"
        or risk >= 75
    ):

        return "P1 - Emergency"

    if (
        category == "High"
        or risk >= 60
        or anomaly >= 0.80
    ):

        return "P2 - High"

    if (
        category == "Medium"
        or risk >= 35
        or anomaly >= 0.60
    ):

        return "P3 - Planned"

    return "P4 - Routine"


# ============================================================
# URGENCY
# ============================================================

def calculate_urgency(
    priority
):

    mapping = {
        "P1 - Emergency": "Immediate",
        "P2 - High": "Within 24 Hours",
        "P3 - Planned": "Within 7 Days",
        "P4 - Routine": "Routine Monitoring"
    }

    return mapping.get(
        priority,
        "Review Required"
    )


# ============================================================
# INSPECTION TYPE
# ============================================================

def calculate_inspection_type(
    cause,
    sensor_condition,
    network_condition,
    quality_condition
):

    cause = str(
        cause
    )

    if cause == "Sensor / Mechanical":

        return "Mechanical and Sensor Inspection"

    if cause == "Network / 6G":

        return "Network and Edge Connectivity Inspection"

    if cause == "Quality / Production":

        return "Quality and Production Line Inspection"

    if cause == "Historical Trend":

        return "Trend and Preventive Maintenance Inspection"

    if cause == "Anomaly":

        return "Comprehensive Anomaly Investigation"

    combined = " ".join(
        [
            str(sensor_condition),
            str(network_condition),
            str(quality_condition)
        ]
    ).lower()

    if (
        "temperature" in combined
        or "vibration" in combined
    ):

        return "Sensor and Mechanical Inspection"

    if (
        "latency" in combined
        or "packet loss" in combined
    ):

        return "Network Inspection"

    return "General Preventive Inspection"


# ============================================================
# MAINTENANCE ACTION
# ============================================================

def calculate_action(
    cause,
    priority
):

    if priority == "P1 - Emergency":

        return (
            "Stop or isolate affected operation if safe; "
            "perform immediate technical inspection."
        )

    if priority == "P2 - High":

        return (
            "Schedule urgent maintenance and inspect "
            "the dominant risk factor."
        )

    if priority == "P3 - Planned":

        return (
            "Schedule preventive maintenance and monitor "
            "the dominant condition."
        )

    if cause == "Network / 6G":

        return (
            "Continue monitoring network health, latency "
            "and packet loss."
        )

    return (
        "Continue routine monitoring and preventive "
        "maintenance."
    )


# ============================================================
# DOWNTIME ESTIMATION
# ============================================================

def estimate_downtime(
    risk,
    priority
):

    risk = float(risk)

    if priority == "P1 - Emergency":

        return round(
            4.0 + (risk - 75) * 0.08,
            2
        )

    if priority == "P2 - High":

        return round(
            2.0 + (risk - 60) * 0.05,
            2
        )

    if priority == "P3 - Planned":

        return round(
            1.0 + (risk - 35) * 0.025,
            2
        )

    return 0.5


# ============================================================
# MAINTENANCE COST ESTIMATION
# ============================================================

def estimate_cost(
    priority,
    downtime
):

    base_cost = {
        "P1 - Emergency": 15000,
        "P2 - High": 9000,
        "P3 - Planned": 5000,
        "P4 - Routine": 1500
    }

    base = base_cost.get(
        priority,
        1500
    )

    hourly_cost = 1000

    return round(
        base
        +
        downtime * hourly_cost,
        2
    )


# ============================================================
# PRODUCTION IMPACT
# ============================================================

def estimate_production_impact(
    df
):

    speed = numeric_series(
        df,
        "Production_Speed_units_per_hr"
    )

    return speed


# ============================================================
# BUSINESS IMPACT
# ============================================================

def calculate_business_impact(
    downtime,
    production_speed
):

    downtime = float(
        downtime
    )

    production_speed = float(
        production_speed
    )

    return round(
        downtime * production_speed,
        2
    )


# ============================================================
# MONITORING FREQUENCY
# ============================================================

def monitoring_frequency(
    priority
):

    mapping = {
        "P1 - Emergency": "Continuous",
        "P2 - High": "Every 15 Minutes",
        "P3 - Planned": "Hourly",
        "P4 - Routine": "Daily"
    }

    return mapping.get(
        priority,
        "Daily"
    )


# ============================================================
# BUILD RECOMMENDATIONS
# ============================================================

def build_recommendations(df):

    print("\n" + "=" * 75)
    print("BUILDING MAINTENANCE RECOMMENDATIONS")
    print("=" * 75)

    df = df.copy()

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    print(
        "Analyzing sensor conditions..."
    )

    df["sensor_condition"] = (
        detect_sensor_conditions(df)
    )

    print(
        "Analyzing network / 6G conditions..."
    )

    df["network_condition"] = (
        detect_network_conditions(df)
    )

    print(
        "Analyzing quality conditions..."
    )

    df["quality_condition"] = (
        detect_quality_conditions(df)
    )

    # --------------------------------------------------------
    # Root cause
    # --------------------------------------------------------

    print(
        "Calculating root causes..."
    )

    df["dominant_risk_factor"] = (
        df.apply(
            calculate_root_cause,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    print(
        "Calculating maintenance priorities..."
    )

    df["maintenance_priority"] = [
        calculate_priority(
            risk,
            anomaly,
            category
        )
        for risk, anomaly, category
        in zip(
            df["maintenance_risk_score"],
            df["anomaly_score"],
            df["maintenance_risk_category"]
        )
    ]

    # --------------------------------------------------------
    # Urgency
    # --------------------------------------------------------

    df["maintenance_urgency"] = (
        df["maintenance_priority"]
        .apply(
            calculate_urgency
        )
    )

    # --------------------------------------------------------
    # Inspection
    # --------------------------------------------------------

    print(
        "Creating inspection recommendations..."
    )

    df["recommended_inspection"] = [
        calculate_inspection_type(
            cause,
            sensor,
            network,
            quality
        )
        for cause, sensor, network, quality
        in zip(
            df["dominant_risk_factor"],
            df["sensor_condition"],
            df["network_condition"],
            df["quality_condition"]
        )
    ]

    # --------------------------------------------------------
    # Maintenance action
    # --------------------------------------------------------

    df["recommended_action"] = [
        calculate_action(
            cause,
            priority
        )
        for cause, priority
        in zip(
            df["dominant_risk_factor"],
            df["maintenance_priority"]
        )
    ]

    # --------------------------------------------------------
    # Downtime
    # --------------------------------------------------------

    print(
        "Estimating downtime..."
    )

    df["estimated_downtime_hours"] = [
        estimate_downtime(
            risk,
            priority
        )
        for risk, priority
        in zip(
            df["maintenance_risk_score"],
            df["maintenance_priority"]
        )
    ]

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    print(
        "Estimating maintenance cost..."
    )

    df["estimated_maintenance_cost"] = [
        estimate_cost(
            priority,
            downtime
        )
        for priority, downtime
        in zip(
            df["maintenance_priority"],
            df["estimated_downtime_hours"]
        )
    ]

    # --------------------------------------------------------
    # Production impact
    # --------------------------------------------------------

    production_speed = (
        estimate_production_impact(df)
    )

    df["estimated_production_impact_units"] = [
        calculate_business_impact(
            downtime,
            speed
        )
        for downtime, speed
        in zip(
            df["estimated_downtime_hours"],
            production_speed
        )
    ]

    # --------------------------------------------------------
    # Monitoring
    # --------------------------------------------------------

    df["monitoring_frequency"] = (
        df["maintenance_priority"]
        .apply(
            monitoring_frequency
        )
    )

    # --------------------------------------------------------
    # Health score
    # --------------------------------------------------------

    df["machine_health_score"] = (
        100
        -
        df["maintenance_risk_score"]
    ).clip(
        0,
        100
    ).round(2)

    # --------------------------------------------------------
    # Risk rank
    # --------------------------------------------------------

    df["risk_rank"] = (
        df["maintenance_risk_category"]
        .map(
            RISK_ORDER
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Priority rank
    # --------------------------------------------------------

    df["priority_rank"] = (
        df["maintenance_priority"]
        .map(
            PRIORITY_ORDER
        )
        .fillna(99)
        .astype(int)
    )

    # --------------------------------------------------------
    # Overall queue score
    # --------------------------------------------------------

    df["maintenance_priority_score"] = (
        (
            df["maintenance_risk_score"]
            * 0.70
        )
        +
        (
            df["anomaly_score"]
            * 100
            * 0.30
        )
    ).clip(
        0,
        100
    ).round(2)

    # --------------------------------------------------------
    # Maintenance status
    # --------------------------------------------------------

    df["maintenance_status"] = np.select(
        [
            df["maintenance_priority"]
            == "P1 - Emergency",

            df["maintenance_priority"]
            == "P2 - High",

            df["maintenance_priority"]
            == "P3 - Planned"
        ],
        [
            "Emergency Attention",
            "High Attention",
            "Planned Maintenance"
        ],
        default="Routine Monitoring"
    )

    return df


# ============================================================
# CREATE MACHINE SUMMARY
# ============================================================

def create_machine_summary(
    df
):

    print("\n" + "=" * 75)
    print("CREATING MACHINE MAINTENANCE SUMMARY")
    print("=" * 75)

    data = df.copy()

    # --------------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------------

    data = data.loc[
        :,
        ~data.columns.duplicated()
    ].copy()

    # --------------------------------------------------------
    # Ensure Machine_ID
    # --------------------------------------------------------

    if "Machine_ID" not in data.columns:

        data["Machine_ID"] = "Unknown"

    # --------------------------------------------------------
    # Helper flags
    # --------------------------------------------------------

    data["_critical"] = (
        data["maintenance_risk_category"]
        .astype(str)
        .eq("Critical")
        .astype(int)
    )

    data["_high"] = (
        data["maintenance_risk_category"]
        .astype(str)
        .eq("High")
        .astype(int)
    )

    data["_p1"] = (
        data["maintenance_priority"]
        .astype(str)
        .eq("P1 - Emergency")
        .astype(int)
    )

    data["_p2"] = (
        data["maintenance_priority"]
        .astype(str)
        .eq("P2 - High")
        .astype(int)
    )

    data["_p3"] = (
        data["maintenance_priority"]
        .astype(str)
        .eq("P3 - Planned")
        .astype(int)
    )

    data["_anomaly"] = (
        data["is_anomaly"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Group aggregation
    # --------------------------------------------------------

    summary = (
        data
        .groupby(
            "Machine_ID",
            dropna=False
        )
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

            minimum_risk_score=(
                "maintenance_risk_score",
                "min"
            ),

            risk_std=(
                "maintenance_risk_score",
                "std"
            ),

            average_anomaly_score=(
                "anomaly_score",
                "mean"
            ),

            anomaly_records=(
                "_anomaly",
                "sum"
            ),

            critical_records=(
                "_critical",
                "sum"
            ),

            high_records=(
                "_high",
                "sum"
            ),

            emergency_actions=(
                "_p1",
                "sum"
            ),

            high_priority_actions=(
                "_p2",
                "sum"
            ),

            planned_actions=(
                "_p3",
                "sum"
            ),

            total_downtime_hours=(
                "estimated_downtime_hours",
                "sum"
            ),

            total_maintenance_cost=(
                "estimated_maintenance_cost",
                "sum"
            ),

            total_production_impact_units=(
                "estimated_production_impact_units",
                "sum"
            ),

            average_health_score=(
                "machine_health_score",
                "mean"
            ),

            average_priority_score=(
                "maintenance_priority_score",
                "mean"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Fill NaN
    # --------------------------------------------------------

    summary["risk_std"] = (
        summary["risk_std"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    summary["critical_rate_pct"] = (
        summary["critical_records"]
        /
        summary["total_records"]
        *
        100
    ).round(2)

    summary["high_risk_rate_pct"] = (
        (
            summary["critical_records"]
            +
            summary["high_records"]
        )
        /
        summary["total_records"]
        *
        100
    ).round(2)

    summary["anomaly_rate_pct"] = (
        summary["anomaly_records"]
        /
        summary["total_records"]
        *
        100
    ).round(2)

    # --------------------------------------------------------
    # Machine status
    # --------------------------------------------------------

    summary["machine_status"] = np.select(
        [
            summary["emergency_actions"] > 0,

            summary["high_priority_actions"] > 0,

            summary["average_risk_score"] >= 35
        ],
        [
            "Emergency Attention",

            "High Attention",

            "Planned Maintenance"
        ],
        default="Routine Monitoring"
    )

    # --------------------------------------------------------
    # Machine priority score
    # --------------------------------------------------------

    summary["machine_priority_score"] = (
        (
            summary["average_risk_score"]
            * 0.50
        )
        +
        (
            summary["maximum_risk_score"]
            * 0.30
        )
        +
        (
            summary["average_anomaly_score"]
            * 100
            * 0.20
        )
    ).clip(
        0,
        100
    ).round(2)

    # --------------------------------------------------------
    # Maintenance workload
    # --------------------------------------------------------

    summary["maintenance_workload_score"] = (
        summary["emergency_actions"] * 4
        +
        summary["high_priority_actions"] * 3
        +
        summary["planned_actions"] * 2
        +
        summary["high_records"]
    ).round(2)

    # --------------------------------------------------------
    # Fleet ranking
    # --------------------------------------------------------

    summary = summary.sort_values(
        by=[
            "machine_priority_score",
            "maximum_risk_score"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    summary["fleet_priority_rank"] = (
        np.arange(
            len(summary)
        )
        + 1
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    print(
        f"Machines analyzed : "
        f"{len(summary):,}"
    )

    if len(summary) > 0:

        print(
            "Highest priority machine : "
            f"{summary.iloc[0]['Machine_ID']}"
        )

        print(
            "Highest priority score : "
            f"{summary.iloc[0]['machine_priority_score']:.2f}"
        )

    return summary


# ============================================================
# CREATE PRIORITY QUEUE
# ============================================================

def create_priority_queue(
    df
):

    print("\n" + "=" * 75)
    print("CREATING MAINTENANCE PRIORITY QUEUE")
    print("=" * 75)

    queue = df.copy()

    # --------------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------------

    queue = queue.loc[
        :,
        ~queue.columns.duplicated()
    ].copy()

    # --------------------------------------------------------
    # Ensure priority rank
    # --------------------------------------------------------

    queue["priority_rank"] = (
        queue["maintenance_priority"]
        .astype(str)
        .apply(
            lambda value:
            PRIORITY_ORDER.get(
                value,
                99
            )
        )
    )

    # --------------------------------------------------------
    # Ensure score numeric
    # --------------------------------------------------------

    queue["maintenance_priority_score"] = (
        pd.to_numeric(
            queue[
                "maintenance_priority_score"
            ],
            errors="coerce"
        )
        .fillna(0)
    )

    queue["maintenance_risk_score"] = (
        pd.to_numeric(
            queue[
                "maintenance_risk_score"
            ],
            errors="coerce"
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    queue = queue.sort_values(
        by=[
            "priority_rank",
            "maintenance_priority_score",
            "maintenance_risk_score"
        ],
        ascending=[
            True,
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Queue rank
    # --------------------------------------------------------

    queue["queue_rank"] = (
        np.arange(
            len(queue)
        )
        + 1
    )

    # --------------------------------------------------------
    # Priority category
    # --------------------------------------------------------

    queue["priority_category"] = np.select(
        [
            queue["priority_rank"] == 1,
            queue["priority_rank"] == 2,
            queue["priority_rank"] == 3,
            queue["priority_rank"] == 4
        ],
        [
            "Emergency",
            "High",
            "Planned",
            "Routine"
        ],
        default="Review"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    queue.to_csv(
        PRIORITY_QUEUE_FILE,
        index=False
    )

    print(
        f"Saved : {PRIORITY_QUEUE_FILE}"
    )

    print(
        f"Queue records : {len(queue):,}"
    )

    return queue


# ============================================================
# CREATE FLEET SUMMARY
# ============================================================

def create_fleet_summary(
    df,
    machine_summary
):

    print("\n" + "=" * 75)
    print("CREATING FLEET SUMMARY")
    print("=" * 75)

    total_records = len(df)

    total_machines = (
        df["Machine_ID"]
        .nunique()
    )

    average_risk = (
        df["maintenance_risk_score"]
        .mean()
    )

    maximum_risk = (
        df["maintenance_risk_score"]
        .max()
    )

    average_health = (
        df["machine_health_score"]
        .mean()
    )

    anomaly_count = (
        df["is_anomaly"]
        .sum()
    )

    critical_count = (
        df["maintenance_risk_category"]
        .astype(str)
        .eq("Critical")
        .sum()
    )

    high_count = (
        df["maintenance_risk_category"]
        .astype(str)
        .eq("High")
        .sum()
    )

    p1_count = (
        df["maintenance_priority"]
        .astype(str)
        .eq("P1 - Emergency")
        .sum()
    )

    p2_count = (
        df["maintenance_priority"]
        .astype(str)
        .eq("P2 - High")
        .sum()
    )

    p3_count = (
        df["maintenance_priority"]
        .astype(str)
        .eq("P3 - Planned")
        .sum()
    )

    p4_count = (
        df["maintenance_priority"]
        .astype(str)
        .eq("P4 - Routine")
        .sum()
    )

    total_downtime = (
        df["estimated_downtime_hours"]
        .sum()
    )

    total_cost = (
        df["estimated_maintenance_cost"]
        .sum()
    )

    total_production_impact = (
        df["estimated_production_impact_units"]
        .sum()
    )

    # --------------------------------------------------------
    # Dominant risk factor
    # --------------------------------------------------------

    factor_counts = {}

    if "dominant_risk_factor" in df.columns:

        factor_counts = (
            df["dominant_risk_factor"]
            .value_counts()
            .to_dict()
        )

    dominant_factor = (
        max(
            factor_counts,
            key=factor_counts.get
        )
        if factor_counts
        else "Unknown"
    )

    # --------------------------------------------------------
    # Fleet status
    # --------------------------------------------------------

    if (
        critical_count > 0
        or p1_count > 0
    ):

        fleet_status = (
            "Immediate Maintenance Attention"
        )

    elif (
        high_count > 0
        or p2_count > 0
    ):

        fleet_status = (
            "Elevated Maintenance Attention"
        )

    elif p3_count > 0:

        fleet_status = (
            "Planned Maintenance Required"
        )

    else:

        fleet_status = (
            "Routine Monitoring"
        )

    # --------------------------------------------------------
    # Summary dataframe
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "total_records": total_records,

                "total_machines": total_machines,

                "average_maintenance_risk":
                    round(
                        average_risk,
                        2
                    ),

                "maximum_maintenance_risk":
                    round(
                        maximum_risk,
                        2
                    ),

                "average_machine_health":
                    round(
                        average_health,
                        2
                    ),

                "anomaly_records":
                    int(anomaly_count),

                "anomaly_rate_pct":
                    round(
                        anomaly_count
                        /
                        max(
                            total_records,
                            1
                        )
                        *
                        100,
                        2
                    ),

                "critical_records":
                    int(critical_count),

                "high_records":
                    int(high_count),

                "P1_emergency_records":
                    int(p1_count),

                "P2_high_records":
                    int(p2_count),

                "P3_planned_records":
                    int(p3_count),

                "P4_routine_records":
                    int(p4_count),

                "estimated_total_downtime_hours":
                    round(
                        total_downtime,
                        2
                    ),

                "estimated_total_maintenance_cost":
                    round(
                        total_cost,
                        2
                    ),

                "estimated_total_production_impact_units":
                    round(
                        total_production_impact,
                        2
                    ),

                "dominant_risk_factor":
                    dominant_factor,

                "fleet_status":
                    fleet_status
            }
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary.to_csv(
        FLEET_SUMMARY_FILE,
        index=False
    )

    print(
        f"Saved : {FLEET_SUMMARY_FILE}"
    )

    return summary


# ============================================================
# PRINT FINAL REPORT
# ============================================================

def print_final_report(
    df,
    machine_summary,
    priority_queue,
    fleet_summary
):

    print("\n")
    print("=" * 80)
    print("PREDICTIVE MAINTENANCE FINAL REPORT")
    print("6G-INTEGRATED SMART MANUFACTURING SYSTEM")
    print("=" * 80)

    print(
        f"\nTotal records       : "
        f"{len(df):,}"
    )

    print(
        f"Total machines      : "
        f"{df['Machine_ID'].nunique():,}"
    )

    print(
        f"Average risk        : "
        f"{df['maintenance_risk_score'].mean():.2f}/100"
    )

    print(
        f"Maximum risk        : "
        f"{df['maintenance_risk_score'].max():.2f}/100"
    )

    print(
        f"Average health      : "
        f"{df['machine_health_score'].mean():.2f}/100"
    )

    print("\nRisk Distribution:")

    risk_distribution = (
        df[
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
            risk_distribution.get(
                category,
                0
            )
        )

        percentage = (
            count
            /
            max(
                len(df),
                1
            )
            *
            100
        )

        print(
            f"  {category:<10}: "
            f"{count:>8,} "
            f"({percentage:>6.2f}%)"
        )

    print("\nMaintenance Priority:")

    for priority in [
        "P1 - Emergency",
        "P2 - High",
        "P3 - Planned",
        "P4 - Routine"
    ]:

        count = int(
            (
                df[
                    "maintenance_priority"
                ]
                == priority
            ).sum()
        )

        print(
            f"  {priority:<20}: "
            f"{count:>8,}"
        )

    print("\nDominant Risk Factors:")

    factor_counts = (
        df[
            "dominant_risk_factor"
        ]
        .value_counts()
    )

    for factor, count in (
        factor_counts.head(5).items()
    ):

        print(
            f"  {str(factor):<30}: "
            f"{int(count):>8,}"
        )

    print("\nBusiness Impact Estimates:")

    print(
        f"  Downtime hours : "
        f"{df['estimated_downtime_hours'].sum():,.2f}"
    )

    print(
        f"  Maintenance cost : "
        f"{df['estimated_maintenance_cost'].sum():,.2f}"
    )

    print(
        f"  Production impact : "
        f"{df['estimated_production_impact_units'].sum():,.2f} units"
    )

    print("\nTop Priority Machines:")

    top_machines = (
        machine_summary
        .head(10)
    )

    for _, row in top_machines.iterrows():

        print(
            f"  {str(row['Machine_ID']):<15} | "
            f"Avg Risk: "
            f"{row['average_risk_score']:.2f} | "
            f"Max Risk: "
            f"{row['maximum_risk_score']:.2f} | "
            f"Priority: "
            f"{row['machine_priority_score']:.2f}"
        )

    print("\nOutput Files:")

    print(
        f"  1. {RECOMMENDATION_FILE}"
    )

    print(
        f"  2. {MACHINE_SUMMARY_FILE}"
    )

    print(
        f"  3. {PRIORITY_QUEUE_FILE}"
    )

    print(
        f"  4. {FLEET_SUMMARY_FILE}"
    )

    print("\n" + "=" * 80)
    print("MAINTENANCE RECOMMENDATION ENGINE COMPLETED")
    print("=" * 80)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDUSTRY-LEVEL MAINTENANCE RECOMMENDATION ENGINE")
    print("6G-INTEGRATED SMART MANUFACTURING SYSTEM")
    print("Version 3.0.0")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # Create output directory
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        df = load_risk_results()

        # ----------------------------------------------------
        # Standardize
        # ----------------------------------------------------

        df = standardize_columns(
            df
        )

        # ----------------------------------------------------
        # Build recommendations
        # ----------------------------------------------------

        df = build_recommendations(
            df
        )

        # ----------------------------------------------------
        # Save row-level recommendations
        # ----------------------------------------------------

        print("\n" + "=" * 75)
        print("SAVING RECOMMENDATION RESULTS")
        print("=" * 75)

        df.to_csv(
            RECOMMENDATION_FILE,
            index=False
        )

        print(
            f"Saved : {RECOMMENDATION_FILE}"
        )

        # ----------------------------------------------------
        # Machine summary
        # ----------------------------------------------------

        machine_summary = (
            create_machine_summary(
                df
            )
        )

        machine_summary.to_csv(
            MACHINE_SUMMARY_FILE,
            index=False
        )

        print(
            f"Saved : {MACHINE_SUMMARY_FILE}"
        )

        # ----------------------------------------------------
        # Priority queue
        # ----------------------------------------------------

        priority_queue = (
            create_priority_queue(
                df
            )
        )

        # ----------------------------------------------------
        # Fleet summary
        # ----------------------------------------------------

        fleet_summary = (
            create_fleet_summary(
                df,
                machine_summary
            )
        )

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        print_final_report(
            df,
            machine_summary,
            priority_queue,
            fleet_summary
        )

        return 0

    except Exception as error:

        print("\n")
        print("=" * 80)
        print("MAINTENANCE RECOMMENDATION ENGINE FAILED")
        print("=" * 80)

        print(
            f"\nError: {error}"
        )

        print("\nTraceback:")

        traceback.print_exc()

        return 1


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )

