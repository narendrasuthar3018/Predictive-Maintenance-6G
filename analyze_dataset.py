import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PREDICTIVE MAINTENANCE
# DATASET RELATIONSHIP & TARGET ANALYSIS
# ============================================================

DATA_PATH = Path("data") / "Thales_Group_Manufacturing.csv"

print("=" * 75)
print("PREDICTIVE MAINTENANCE - ADVANCED DATASET ANALYSIS")
print("=" * 75)

# ============================================================
# 1. LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    print("\nERROR: Dataset not found!")
    print(f"Expected: {DATA_PATH}")
    raise SystemExit

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

# ============================================================
# 2. TARGET / IMPORTANT COLUMN DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 75)
print("PREDICTIVE MAINTENANCE SCORE")
print("=" * 75)

score = df["Predictive_Maintenance_Score"]

print(f"Mean   : {score.mean():.4f}")
print(f"Median : {score.median():.4f}")
print(f"Std    : {score.std():.4f}")
print(f"Min    : {score.min():.4f}")
print(f"Max    : {score.max():.4f}")

# Risk buckets for analysis only
df["Risk_Band"] = pd.cut(
    score,
    bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
    labels=["Critical", "High", "Medium", "Low"]
)

print("\nRisk Band Distribution:")
print(
    df["Risk_Band"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nRisk Band Percentage:")
print(
    (df["Risk_Band"].value_counts(normalize=True)
     .sort_index() * 100)
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)

# ============================================================
# 3. EFFICIENCY STATUS
# ============================================================

print("\n" + "=" * 75)
print("EFFICIENCY STATUS")
print("=" * 75)

status_counts = df["Efficiency_Status"].value_counts()

print(status_counts.to_string())

print("\nPercentage:")
print(
    (df["Efficiency_Status"].value_counts(normalize=True) * 100)
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)

# ============================================================
# 4. OPERATION MODE
# ============================================================

print("\n" + "=" * 75)
print("OPERATION MODE")
print("=" * 75)

mode_counts = df["Operation_Mode"].value_counts()

print(mode_counts.to_string())

# ============================================================
# 5. MACHINE-WISE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("MACHINE-WISE ANALYSIS")
print("=" * 75)

machine_summary = (
    df.groupby("Machine_ID")
    .agg(
        Records=("Machine_ID", "size"),
        Avg_Temperature=("Temperature_C", "mean"),
        Avg_Vibration=("Vibration_Hz", "mean"),
        Avg_Power=("Power_Consumption_kW", "mean"),
        Avg_Latency=("Network_Latency_ms", "mean"),
        Avg_Packet_Loss=("Packet_Loss_%", "mean"),
        Avg_Defect_Rate=("Quality_Control_Defect_Rate_%", "mean"),
        Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
        Avg_Maintenance_Score=("Predictive_Maintenance_Score", "mean"),
        Avg_Error_Rate=("Error_Rate_%", "mean")
    )
)

print("\nMachine Summary:")
print(machine_summary.round(3).to_string())

# ============================================================
# 6. CORRELATION WITH MAINTENANCE SCORE
# ============================================================

print("\n" + "=" * 75)
print("CORRELATION WITH PREDICTIVE MAINTENANCE SCORE")
print("=" * 75)

numeric_columns = df.select_dtypes(include=np.number).columns

correlations = (
    df[numeric_columns]
    .corr()["Predictive_Maintenance_Score"]
    .drop("Predictive_Maintenance_Score")
    .sort_values(ascending=False)
)

print(correlations.round(4).to_string())

# ============================================================
# 7. CORRELATION WITH ERROR RATE
# ============================================================

print("\n" + "=" * 75)
print("CORRELATION WITH ERROR RATE")
print("=" * 75)

error_correlations = (
    df[numeric_columns]
    .corr()["Error_Rate_%"]
    .drop("Error_Rate_%")
    .sort_values(ascending=False)
)

print(error_correlations.round(4).to_string())

# ============================================================
# 8. CORRELATION WITH DEFECT RATE
# ============================================================

print("\n" + "=" * 75)
print("CORRELATION WITH QUALITY DEFECT RATE")
print("=" * 75)

defect_correlations = (
    df[numeric_columns]
    .corr()["Quality_Control_Defect_Rate_%"]
    .drop("Quality_Control_Defect_Rate_%")
    .sort_values(ascending=False)
)

print(defect_correlations.round(4).to_string())

# ============================================================
# 9. EFFICIENCY STATUS vs NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 75)
print("EFFICIENCY STATUS vs MACHINE FEATURES")
print("=" * 75)

feature_columns = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]

status_analysis = (
    df.groupby("Efficiency_Status")[feature_columns]
    .mean()
    .round(3)
)

print(status_analysis.to_string())

# ============================================================
# 10. OPERATION MODE vs MAINTENANCE SCORE
# ============================================================

print("\n" + "=" * 75)
print("OPERATION MODE vs MAINTENANCE SCORE")
print("=" * 75)

mode_analysis = (
    df.groupby("Operation_Mode")[feature_columns]
    .mean()
    .round(3)
)

print(mode_analysis.to_string())

# ============================================================
# 11. RISK BAND vs MACHINE FEATURES
# ============================================================

print("\n" + "=" * 75)
print("RISK BAND vs MACHINE FEATURES")
print("=" * 75)

risk_analysis = (
    df.groupby("Risk_Band", observed=True)[feature_columns]
    .mean()
    .round(3)
)

print(risk_analysis.to_string())

# ============================================================
# 12. HIGH-RISK MACHINE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("HIGH-RISK MACHINE ANALYSIS")
print("=" * 75)

high_risk = df[df["Risk_Band"].isin(["Critical", "High"])]

print(f"High/Critical Risk Records: {len(high_risk):,}")

if len(high_risk) > 0:

    high_risk_machine = (
        high_risk.groupby("Machine_ID")
        .agg(
            High_Risk_Records=("Machine_ID", "size"),
            Avg_Temperature=("Temperature_C", "mean"),
            Avg_Vibration=("Vibration_Hz", "mean"),
            Avg_Power=("Power_Consumption_kW", "mean"),
            Avg_Latency=("Network_Latency_ms", "mean"),
            Avg_Packet_Loss=("Packet_Loss_%", "mean"),
            Avg_Defect_Rate=("Quality_Control_Defect_Rate_%", "mean"),
            Avg_Error_Rate=("Error_Rate_%", "mean"),
            Avg_Maintenance_Score=(
                "Predictive_Maintenance_Score",
                "mean"
            )
        )
        .sort_values(
            "High_Risk_Records",
            ascending=False
        )
    )

    print(
        high_risk_machine
        .head(15)
        .round(3)
        .to_string()
    )

# ============================================================
# 13. NETWORK HEALTH ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("6G / NETWORK HEALTH ANALYSIS")
print("=" * 75)

network_columns = [
    "Network_Latency_ms",
    "Packet_Loss_%"
]

print("\nNetwork Statistics:")

print(
    df[network_columns]
    .describe()
    .round(3)
    .to_string()
)

# ============================================================
# 14. EXTREME SENSOR CONDITIONS
# ============================================================

print("\n" + "=" * 75)
print("EXTREME SENSOR CONDITIONS")
print("=" * 75)

conditions = {
    "Temperature >= 80 C": df["Temperature_C"] >= 80,
    "Vibration >= 4 Hz": df["Vibration_Hz"] >= 4,
    "Power >= 8 kW": df["Power_Consumption_kW"] >= 8,
    "Latency >= 40 ms": df["Network_Latency_ms"] >= 40,
    "Packet Loss >= 4%": df["Packet_Loss_%"] >= 4,
    "Defect Rate >= 8%": df["Quality_Control_Defect_Rate_%"] >= 8,
    "Error Rate >= 12%": df["Error_Rate_%"] >= 12
}

for name, condition in conditions.items():

    count = condition.sum()
    percentage = (count / len(df)) * 100

    print(
        f"{name:<30} "
        f"{count:>8,} records "
        f"({percentage:>6.2f}%)"
    )

# ============================================================
# 15. DATASET TIME RANGE
# ============================================================

print("\n" + "=" * 75)
print("TIME RANGE")
print("=" * 75)

date_series = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

print(f"Start Date : {date_series.min()}")
print(f"End Date   : {date_series.max()}")

# ============================================================
# 16. FINAL
# ============================================================

print("\n" + "=" * 75)
print("ADVANCED ANALYSIS COMPLETED")
print("=" * 75)