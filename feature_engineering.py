# ============================================================
# PREDICTIVE MAINTENANCE & ANOMALY DETECTION
# 6G-INTEGRATED SMART MANUFACTURING SYSTEM
# ============================================================
# File: feature_engineering.py
# Purpose:
#   - Clean and transform raw manufacturing data
#   - Create temporal features
#   - Create machine-wise historical features
#   - Create sensor, network, quality and operational features
#   - Prevent data leakage
#   - Generate engineered_dataset.csv
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Thales_Group_Manufacturing.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "engineered_dataset.csv"


# ============================================================
# CONFIGURATION
# ============================================================

MACHINE_COLUMN = "Machine_ID"
DATE_COLUMN = "Date"
TIME_COLUMN = "Timestamp"

# These columns are descriptive / target-like columns.
# They should NOT directly become predictive features for
# the core anomaly/risk engine.
EXCLUDED_FROM_CORE_FEATURES = [
    "Predictive_Maintenance_Score",
    "Efficiency_Status",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(df, columns):
    """
    Convert selected columns to numeric safely.
    Invalid values become NaN.
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def minmax_fixed(series, minimum, maximum):
    """
    Fixed-range normalization.

    Using fixed physical/data ranges instead of dataset-wide
    min/max prevents unnecessary dependence on the complete
    dataset.
    """
    denominator = maximum - minimum

    if denominator == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    result = (series - minimum) / denominator

    return result.clip(0, 1)


def add_cyclical_feature(df, column, period, prefix):
    """
    Create sine/cosine representation of a cyclical variable.
    """
    df[f"{prefix}_sin"] = np.sin(
        2 * np.pi * df[column] / period
    )

    df[f"{prefix}_cos"] = np.cos(
        2 * np.pi * df[column] / period
    )

    return df


def add_previous_feature(df, column):
    """
    Create previous-observation value for every machine.

    shift(1) is extremely important because it prevents the
    current observation from being used as its own history.
    """
    previous_column = f"{column}_previous"

    df[previous_column] = (
        df.groupby(MACHINE_COLUMN, sort=False)[column]
        .shift(1)
    )

    return df


def add_delta_feature(df, column):
    """
    Difference between current and previous machine observation.
    """
    delta_column = f"{column}_delta"

    previous = (
        df.groupby(MACHINE_COLUMN, sort=False)[column]
        .shift(1)
    )

    df[delta_column] = df[column] - previous

    return df


def add_rolling_features(df, column, windows=(5, 15)):
    """
    Create machine-wise rolling historical statistics.

    Important:
    The current row is excluded using shift(1), which prevents
    information leakage.
    """
    grouped = df.groupby(MACHINE_COLUMN, sort=False)[column]

    shifted = grouped.shift(1)

    for window in windows:

        df[f"{column}_rolling_mean_{window}"] = (
            shifted.groupby(df[MACHINE_COLUMN])
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=2
                ).mean()
            )
        )

        df[f"{column}_rolling_std_{window}"] = (
            shifted.groupby(df[MACHINE_COLUMN])
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=2
                ).std()
            )
        )

    return df


# ============================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================

def feature_engineering(input_df):
    """
    Main feature engineering pipeline.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Raw manufacturing dataset.

    Returns
    -------
    pandas.DataFrame
        Engineered dataset.
    """

    # --------------------------------------------------------
    # COPY DATA
    # --------------------------------------------------------

    df = input_df.copy()

    print("\n" + "=" * 70)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 70)

    print(f"Initial rows    : {len(df):,}")
    print(f"Initial columns : {len(df.columns)}")


    # --------------------------------------------------------
    # STANDARDIZE COLUMN NAMES
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
    )


    # --------------------------------------------------------
    # REQUIRED COLUMN VALIDATION
    # --------------------------------------------------------

    required_columns = [
        "Date",
        "Timestamp",
        "Machine_ID",
        "Operation_Mode",
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr",
        "Predictive_Maintenance_Score",
        "Error_Rate_%",
        "Efficiency_Status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )


    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    numeric_columns = [
        "Machine_ID",
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr",
        "Predictive_Maintenance_Score",
        "Error_Rate_%",
    ]

    df = safe_numeric(
        df,
        numeric_columns
    )


    # --------------------------------------------------------
    # DATE + TIMESTAMP PROCESSING
    # --------------------------------------------------------

        # --------------------------------------------------------
    # DATE + TIMESTAMP PROCESSING
    # --------------------------------------------------------

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
        dayfirst=True
    )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN].astype(str).str.strip(),
        format="%H:%M:%S",
        errors="coerce"
    )

    # Create unified datetime.
    time_delta = pd.to_timedelta(
        df[TIME_COLUMN].dt.hour * 3600
        + df[TIME_COLUMN].dt.minute * 60
        + df[TIME_COLUMN].dt.second,
        unit="s"
    )

    df["datetime"] = (
        df[DATE_COLUMN].dt.normalize()
        + time_delta
    )


    # --------------------------------------------------------
    # SORT BY MACHINE + TIME
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            MACHINE_COLUMN,
            "datetime"
        ],
        kind="mergesort"
    ).reset_index(drop=True)
    # --------------------------------------------------------
    # SORT BY MACHINE + TIME
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            MACHINE_COLUMN,
            "datetime"
        ],
        kind="mergesort"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day

    df["day_of_year"] = (
        df["datetime"].dt.dayofyear
    )

    df["day_of_week"] = (
        df["datetime"].dt.dayofweek
    )

    df["hour"] = (
        df["datetime"].dt.hour
    )

    df["minute"] = (
        df["datetime"].dt.minute
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_business_hour"] = (
        (df["hour"] >= 8)
        & (df["hour"] < 18)
    ).astype(int)


    # --------------------------------------------------------
    # CYCLICAL TIME FEATURES
    # --------------------------------------------------------

    df = add_cyclical_feature(
        df,
        "hour",
        24,
        "hour"
    )

    df = add_cyclical_feature(
        df,
        "minute",
        60,
        "minute"
    )

    df = add_cyclical_feature(
        df,
        "day_of_week",
        7,
        "day_of_week"
    )

    df = add_cyclical_feature(
        df,
        "day_of_year",
        365,
        "day_of_year"
    )


    # --------------------------------------------------------
    # OPERATION MODE FEATURES
    # --------------------------------------------------------

    df["is_active"] = (
        df["Operation_Mode"]
        .astype(str)
        .str.lower()
        .eq("active")
        .astype(int)
    )

    df["is_idle"] = (
        df["Operation_Mode"]
        .astype(str)
        .str.lower()
        .eq("idle")
        .astype(int)
    )

    df["is_maintenance_mode"] = (
        df["Operation_Mode"]
        .astype(str)
        .str.lower()
        .eq("maintenance")
        .astype(int)
    )


    # --------------------------------------------------------
    # NETWORK / 6G INTELLIGENCE FEATURES
    # --------------------------------------------------------

    # Latency normalization:
    # expected dataset range = 1 to 50 ms
    df["latency_normalized"] = minmax_fixed(
        df["Network_Latency_ms"],
        1,
        50
    )

    # Packet loss normalization:
    # expected dataset range = 0 to 5 %
    df["packet_loss_normalized"] = minmax_fixed(
        df["Packet_Loss_%"],
        0,
        5
    )

    # Network risk combines latency and packet loss.
    df["network_risk"] = (
        0.55 * df["latency_normalized"]
        + 0.45 * df["packet_loss_normalized"]
    )

    # Network health is inverse of network risk.
    df["network_health"] = (
        1 - df["network_risk"]
    ).clip(0, 1)

    # High-latency indicator.
    df["high_latency_flag"] = (
        df["Network_Latency_ms"] >= 40
    ).astype(int)

    # High packet-loss indicator.
    df["high_packet_loss_flag"] = (
        df["Packet_Loss_%"] >= 4
    ).astype(int)


    # --------------------------------------------------------
    # THERMAL / SENSOR FEATURES
    # --------------------------------------------------------

    df["temperature_normalized"] = minmax_fixed(
        df["Temperature_C"],
        30,
        90
    )

    df["vibration_normalized"] = minmax_fixed(
        df["Vibration_Hz"],
        0.1,
        5
    )

    df["power_normalized"] = minmax_fixed(
        df["Power_Consumption_kW"],
        1.5,
        10
    )

    df["error_normalized"] = minmax_fixed(
        df["Error_Rate_%"],
        0,
        15
    )

    df["defect_normalized"] = minmax_fixed(
        df["Quality_Control_Defect_Rate_%"],
        0,
        10
    )


    # --------------------------------------------------------
    # SENSOR RISK FEATURES
    # --------------------------------------------------------

    df["thermal_risk"] = (
        df["temperature_normalized"]
    )

    df["mechanical_risk"] = (
        df["vibration_normalized"]
    )

    df["power_risk"] = (
        df["power_normalized"]
    )

    df["error_risk"] = (
        df["error_normalized"]
    )

    df["quality_risk"] = (
        0.55 * df["defect_normalized"]
        + 0.45 * df["error_normalized"]
    )


    # --------------------------------------------------------
    # OPERATIONAL FEATURES
    # --------------------------------------------------------

    df["production_speed_normalized"] = minmax_fixed(
        df["Production_Speed_units_per_hr"],
        50,
        500
    )

    # Higher speed is not automatically dangerous.
    # This feature simply describes production load.
    df["production_load"] = (
        df["production_speed_normalized"]
    )

    # Error per production unit.
    df["error_per_100_units"] = (
        df["Error_Rate_%"]
        / df["Production_Speed_units_per_hr"].replace(0, np.nan)
        * 100
    )

    df["defect_per_100_units"] = (
        df["Quality_Control_Defect_Rate_%"]
        / df["Production_Speed_units_per_hr"].replace(0, np.nan)
        * 100
    )


    # --------------------------------------------------------
    # COMPOSITE MACHINE HEALTH INDICATORS
    # --------------------------------------------------------

    df["sensor_risk"] = (
        0.30 * df["thermal_risk"]
        + 0.30 * df["mechanical_risk"]
        + 0.20 * df["power_risk"]
        + 0.20 * df["error_risk"]
    ).clip(0, 1)

    df["machine_health_score"] = (
        1 - df["sensor_risk"]
    ).clip(0, 1)


    # --------------------------------------------------------
    # EXTREME CONDITION FLAGS
    # --------------------------------------------------------

    df["high_temperature_flag"] = (
        df["Temperature_C"] >= 80
    ).astype(int)

    df["high_vibration_flag"] = (
        df["Vibration_Hz"] >= 4
    ).astype(int)

    df["high_power_flag"] = (
        df["Power_Consumption_kW"] >= 8
    ).astype(int)

    df["high_error_flag"] = (
        df["Error_Rate_%"] >= 12
    ).astype(int)

    df["high_defect_flag"] = (
        df["Quality_Control_Defect_Rate_%"] >= 8
    ).astype(int)


    # --------------------------------------------------------
    # COMBINED CONDITION FLAGS
    # --------------------------------------------------------

    df["sensor_stress_count"] = (
        df["high_temperature_flag"]
        + df["high_vibration_flag"]
        + df["high_power_flag"]
        + df["high_error_flag"]
    )

    df["network_issue_count"] = (
        df["high_latency_flag"]
        + df["high_packet_loss_flag"]
    )

    df["quality_issue_count"] = (
        df["high_defect_flag"]
        + df["high_error_flag"]
    )


    # --------------------------------------------------------
    # MACHINE HISTORY FEATURES
    # --------------------------------------------------------

    historical_columns = [
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr",
        "Error_Rate_%",
    ]

    print("\nCreating machine historical features...")

    for column in historical_columns:

        # Previous observation
        df = add_previous_feature(
            df,
            column
        )

        # Change from previous observation
        df = add_delta_feature(
            df,
            column
        )


    # --------------------------------------------------------
    # ROLLING MACHINE HISTORY
    # --------------------------------------------------------

    rolling_columns = [
        "Temperature_C",
        "Vibration_Hz",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Error_Rate_%",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr",
    ]

    print("Creating rolling historical statistics...")

    for column in rolling_columns:

        df = add_rolling_features(
            df,
            column,
            windows=(5, 15)
        )


    # --------------------------------------------------------
    # TREND / CHANGE FEATURES
    # --------------------------------------------------------

    df["temperature_change_rate"] = (
        df["Temperature_C_delta"]
        / df["Temperature_C_previous"].abs().replace(0, np.nan)
    )

    df["vibration_change_rate"] = (
        df["Vibration_Hz_delta"]
        / df["Vibration_Hz_previous"].abs().replace(0, np.nan)
    )

    df["error_change_rate"] = (
        df["Error_Rate_%_delta"]
        / df["Error_Rate_%_previous"].abs().replace(0, np.nan)
    )

    df["latency_change_rate"] = (
        df["Network_Latency_ms_delta"]
        / df["Network_Latency_ms_previous"].abs().replace(0, np.nan)
    )

    df["packet_loss_change_rate"] = (
        df["Packet_Loss_%_delta"]
        / df["Packet_Loss_%_previous"].abs().replace(0, np.nan)
    )


    # --------------------------------------------------------
    # MACHINE HISTORICAL RISK
    # --------------------------------------------------------

    df["historical_sensor_risk"] = (
        0.35 * df[
            "Vibration_Hz_rolling_mean_15"
        ].pipe(
            lambda x: minmax_fixed(x, 0.1, 5)
        )
        + 0.30 * df[
            "Temperature_C_rolling_mean_15"
        ].pipe(
            lambda x: minmax_fixed(x, 30, 90)
        )
        + 0.35 * df[
            "Error_Rate_%_rolling_mean_15"
        ].pipe(
            lambda x: minmax_fixed(x, 0, 15)
        )
    ).clip(0, 1)


    # --------------------------------------------------------
    # STABILITY INDICATORS
    # --------------------------------------------------------

    df["temperature_instability"] = (
        df["Temperature_C_rolling_std_15"]
    )

    df["vibration_instability"] = (
        df["Vibration_Hz_rolling_std_15"]
    )

    df["error_instability"] = (
        df["Error_Rate_%_rolling_std_15"]
    )

    df["network_latency_instability"] = (
        df["Network_Latency_ms_rolling_std_15"]
    )

    df["packet_loss_instability"] = (
        df["Packet_Loss_%_rolling_std_15"]
    )


    # --------------------------------------------------------
    # DATA QUALITY FLAGS
    # --------------------------------------------------------

    df["missing_sensor_count"] = (
        df[
            [
                "Temperature_C",
                "Vibration_Hz",
                "Power_Consumption_kW",
                "Network_Latency_ms",
                "Packet_Loss_%",
                "Quality_Control_Defect_Rate_%",
                "Production_Speed_units_per_hr",
                "Error_Rate_%",
            ]
        ]
        .isna()
        .sum(axis=1)
    )

    df["data_quality_flag"] = (
        df["missing_sensor_count"] == 0
    ).astype(int)


    # --------------------------------------------------------
    # FINAL NUMERIC CLEANING
    # --------------------------------------------------------

    # Replace infinite values generated by division.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # MISSING VALUE IMPUTATION
    # --------------------------------------------------------

    # Historical features naturally contain NaN values for the
    # first observation of each machine.
    #
    # We use machine-level forward fill first, then global
    # median fallback.
    #
    # This does NOT use future observations.

    numeric_feature_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    history_columns = [
        column
        for column in numeric_feature_columns
        if (
            "previous" in column
            or "rolling" in column
            or "delta" in column
            or "change_rate" in column
            or "instability" in column
            or column == "historical_sensor_risk"
        )
    ]

    for column in history_columns:

        df[column] = (
            df.groupby(
                MACHINE_COLUMN,
                sort=False
            )[column]
            .ffill()
        )

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        df[column] = (
            df[column]
            .fillna(median_value)
        )


    # --------------------------------------------------------
    # RE-CALCULATE MACHINE HEALTH AFTER CLEANING
    # --------------------------------------------------------

    df["machine_health_score"] = (
        1 - df["sensor_risk"]
    ).clip(0, 1)


    # --------------------------------------------------------
    # ENGINEERED RISK INDICATOR
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # This is NOT ground-truth machine failure.
    #
    # It is an engineered intelligence feature that combines:
    # sensor risk + network risk + quality risk + history.
    #
    # The final project will later build a more complete
    # transparent risk engine around these components.

    df["preliminary_risk_score"] = (
        0.35 * df["sensor_risk"]
        + 0.25 * df["network_risk"]
        + 0.20 * df["quality_risk"]
        + 0.20 * df["historical_sensor_risk"]
    ).clip(0, 1)


    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    df["preliminary_risk_category"] = pd.cut(
        df["preliminary_risk_score"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            0.75,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ],
        include_lowest=True
    )


    # --------------------------------------------------------
    # RESTORE CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    df = df.sort_values(
        by="datetime",
        kind="mergesort"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # ROUND FLOAT VALUES
    # --------------------------------------------------------

    float_columns = df.select_dtypes(
        include=["float64", "float32"]
    ).columns

    df[float_columns] = df[float_columns].round(6)


    # --------------------------------------------------------
    # OUTPUT VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print(f"Final rows       : {len(df):,}")
    print(f"Final columns    : {len(df.columns):,}")

    print(
        f"New features     : "
        f"{len(df.columns) - len(input_df.columns):,}"
    )

    print(
        f"Remaining missing values: "
        f"{df.isna().sum().sum():,}"
    )

    print("\nRisk distribution:")

    print(
        df["preliminary_risk_category"]
        .value_counts()
        .sort_index()
    )

    print("\nNetwork health:")
    print(
        df["network_health"]
        .describe()
        .round(4)
    )

    print("\nMachine health:")
    print(
        df["machine_health_score"]
        .describe()
        .round(4)
    )

    return df


# ============================================================
# SAVE ENGINEERED DATASET
# ============================================================

def run_feature_engineering():
    """
    Load raw CSV, perform feature engineering and save output.
    """

    print("\n")
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Please verify that the CSV exists inside the data folder."
        )

    print(f"\nInput file : {DATA_PATH}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Loaded dataset: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    engineered_df = feature_engineering(
        df
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    engineered_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT SAVED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nOutput file:")
    print(OUTPUT_PATH)

    print(
        f"\nDataset shape: "
        f"{engineered_df.shape[0]:,} rows × "
        f"{engineered_df.shape[1]:,} columns"
    )

    print("\nFirst 5 rows:")
    print(
        engineered_df.head()
        .to_string()
    )

    print("\nFeature engineering pipeline finished successfully.")

    return engineered_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_feature_engineering()

    except Exception as error:

        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING FAILED")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        raise