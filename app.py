import streamlit as st
import pandas as pd
import io
import plotly.express as px
import numpy as np
from pathlib import Path


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Predictive Maintenance & Anomaly Detection System"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Narendra Suthar"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="ðŸ­",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(filename):
    file_path = OUTPUT_DIR / filename

    if file_path.exists():
        try:
            return pd.read_csv(file_path)
        except Exception as exc:
            st.error(f"Error loading {filename}: {type(exc).__name__}: {exc}")
            return None

    # Use compressed maintenance results when the full CSV is unavailable.
    if filename == "maintenance_recommendation_results.csv":
        zip_path = OUTPUT_DIR / "maintenance_recommendation_results.zip"

        if zip_path.exists():
            try:
                with zip_path.open("rb") as zip_file:
                    return pd.read_csv(zip_file, compression="zip")
            except Exception as exc:
                st.error(f"Error loading maintenance ZIP: {type(exc).__name__}: {exc}")
                return None

    st.warning(f"CSV file not found: {filename}")
    return None

maintenance_df = load_csv(
    "maintenance_recommendation_results.csv"
)

machine_df = load_csv(
    "machine_maintenance_summary.csv"
)

priority_df = load_csv(
    "maintenance_priority_queue.csv"
)

fleet_df = load_csv(
    "fleet_maintenance_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'ðŸ­ Predictive Maintenance & Anomaly Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">'
    f'6G-Integrated Smart Manufacturing Intelligence | '
    f'v{APP_VERSION} | {APP_AUTHOR}'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# DATA AVAILABILITY CHECK
# ============================================================

if maintenance_df is None:

    st.error(
        "Maintenance recommendation dataset is not available."
    )

    st.info(
        "Please run maintenance_recommendation.py first."
    )

    st.stop()
# ============================================================
# DASHBOARD DATA COMPATIBILITY LAYER
# ============================================================
# The maintenance engine uses production-oriented column names.
# The dashboard internally uses shorter analytical aliases.
# Create aliases once here so downstream sections remain compatible.

if maintenance_df is not None and not maintenance_df.empty:

    if (
        "risk_score" not in maintenance_df.columns
        and "maintenance_risk_score" in maintenance_df.columns
    ):
        maintenance_df["risk_score"] = pd.to_numeric(
            maintenance_df["maintenance_risk_score"],
            errors="coerce"
        )

    if (
        "risk_level" not in maintenance_df.columns
        and "maintenance_risk_category" in maintenance_df.columns
    ):
        maintenance_df["risk_level"] = (
            maintenance_df["maintenance_risk_category"]
            .astype(str)
            .str.strip()
        )

    if (
        "root_cause" not in maintenance_df.columns
        and "dominant_risk_factor" in maintenance_df.columns
    ):
        maintenance_df["root_cause"] = (
            maintenance_df["dominant_risk_factor"]
            .astype(str)
            .str.strip()
        )



# ============================================================
# BASIC SYSTEM INFORMATION
# ============================================================

total_records = len(maintenance_df)

if "Machine_ID" in maintenance_df.columns:
    total_machines = maintenance_df["Machine_ID"].nunique()
else:
    total_machines = 0


if "maintenance_risk_score" in maintenance_df.columns:
    avg_risk = maintenance_df[
        "maintenance_risk_score"
    ].mean()
else:
    avg_risk = 0


if "maintenance_risk_category" in maintenance_df.columns:

    critical_count = (
        maintenance_df[
            "maintenance_risk_category"
        ]
        .astype(str)
        .str.lower()
        .eq("critical")
        .sum()
    )

    high_count = (
        maintenance_df[
            "maintenance_risk_category"
        ]
        .astype(str)
        .str.lower()
        .eq("high")
        .sum()
    )

else:

    critical_count = 0
    high_count = 0


# ============================================================
# EXECUTIVE KPIs
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Executive Overview'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{total_records:,}"
    )

with col2:
    st.metric(
        "Machines",
        f"{total_machines:,}"
    )

with col3:
    st.metric(
        "Average Risk",
        f"{avg_risk:.2f}/100"
    )

with col4:
    st.metric(
        "Critical Risk",
        f"{critical_count:,}"
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'System Status'
    '</div>',
    unsafe_allow_html=True
)

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.success("âœ… Dataset Loaded")

with status_col2:
    if machine_df is not None:
        st.success("âœ… Machine Analytics")
    else:
        st.warning("âš ï¸ Machine Analytics")

with status_col3:
    if priority_df is not None:
        st.success("âœ… Priority Queue")
    else:
        st.warning("âš ï¸ Priority Queue")

with status_col4:
    if fleet_df is not None:
        st.success("âœ… Fleet Analytics")
    else:
        st.warning("âš ï¸ Fleet Analytics")


# ============================================================
# DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Maintenance Intelligence Dataset'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    f"Loaded **{total_records:,} records** "
    f"from the maintenance intelligence engine."
)

st.dataframe(
    maintenance_df.head(10),
    width="stretch"
)

# ============================================================
# RISK & FLEET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Risk & Fleet Intelligence'
    '</div>',
    unsafe_allow_html=True
)

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)


# ------------------------------------------------------------
# Average Machine Health
# ------------------------------------------------------------

if "machine_health_score" in maintenance_df.columns:

    average_health = pd.to_numeric(
        maintenance_df["machine_health_score"],
        errors="coerce"
    ).mean()

else:

    average_health = 0


# ------------------------------------------------------------
# High Risk Records
# ------------------------------------------------------------

if "maintenance_risk_category" in maintenance_df.columns:

    risk_categories = (
        maintenance_df["maintenance_risk_category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    high_risk_count = (
        risk_categories.isin(
            ["high", "critical"]
        )
        .sum()
    )

else:

    high_risk_count = 0


# ------------------------------------------------------------
# Anomaly Records
# ------------------------------------------------------------

if "is_anomaly" in maintenance_df.columns:

    anomaly_count = (
        pd.to_numeric(
            maintenance_df["is_anomaly"],
            errors="coerce"
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

else:

    anomaly_count = 0


# ------------------------------------------------------------
# Average Network Health
# ------------------------------------------------------------

if "network_health" in maintenance_df.columns:

    average_network_health = (
        pd.to_numeric(
            maintenance_df["network_health"],
            errors="coerce"
        )
        .mean()
        * 100
    )

else:

    average_network_health = 0


with risk_col1:

    st.metric(
        "Average Machine Health",
        f"{average_health:.2f}/100"
    )


with risk_col2:

    st.metric(
        "High + Critical Risk",
        f"{high_risk_count:,}"
    )


with risk_col3:

    st.metric(
        "Detected Anomalies",
        f"{anomaly_count:,}"
    )


with risk_col4:

    st.metric(
        "Network / 6G Health",
        f"{average_network_health:.2f}%"
    )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.markdown(
    "### Risk Distribution"
)


if "maintenance_risk_category" in maintenance_df.columns:

    risk_distribution = (
        maintenance_df[
            "maintenance_risk_category"
        ]
        .astype(str)
        .str.strip()
        .str.title()
        .value_counts()
        .rename_axis("Risk Category")
        .reset_index(name="Records")
    )

    st.bar_chart(
        risk_distribution.set_index(
            "Risk Category"
        )
    )


# ============================================================
# DOMINANT RISK FACTORS
# ============================================================

st.markdown(
    "### Dominant Risk Factors"
)


if "dominant_risk_factor" in maintenance_df.columns:

    factor_distribution = (
        maintenance_df[
            "dominant_risk_factor"
        ]
        .astype(str)
        .value_counts()
        .rename_axis("Risk Factor")
        .reset_index(name="Records")
    )

    st.bar_chart(
        factor_distribution.set_index(
            "Risk Factor"
        )
    )

# ============================================================
# MAINTENANCE PRIORITY DISTRIBUTION
# ============================================================
# ============================================================
# MAINTENANCE PRIORITY DISTRIBUTION
# ============================================================

st.markdown("### Maintenance Priority")

priority_counts = (
    maintenance_df["maintenance_priority"]
    .astype(str)
    .str.strip()
    .value_counts()
)

p1_count = int(
    priority_counts.get("P1 - Emergency", 0)
)

p2_count = int(
    priority_counts.get("P2 - High", 0)
)

p3_count = int(
    priority_counts.get("P3 - Planned", 0)
)

p4_count = int(
    priority_counts.get("P4 - Routine", 0)
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "P1 - Emergency",
        f"{p1_count:,}",
        help="Immediate maintenance action required."
    )

with col2:
    st.metric(
        "P2 - High",
        f"{p2_count:,}",
        help="High-priority maintenance should be scheduled soon."
    )

with col3:
    st.metric(
        "P3 - Planned",
        f"{p3_count:,}",
        help="Planned preventive maintenance."
    )

with col4:
    st.metric(
        "P4 - Routine",
        f"{p4_count:,}",
        help="Routine monitoring and maintenance."
    )
# ============================================================
## ============================================================
# TOP PRIORITY MACHINES
# ============================================================
# ============================================================
# TOP PRIORITY MACHINES
# ============================================================

st.markdown("### Top Priority Machines")

if machine_df is not None and not machine_df.empty:

    top_machines = machine_df.copy()

    # --------------------------------------------------------
    # Convert important numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "average_risk_score",
        "maximum_risk_score",
        "average_health_score",
        "machine_priority_score",
        "fleet_priority_rank"
    ]

    for column in numeric_columns:

        if column in top_machines.columns:

            top_machines[column] = pd.to_numeric(
                top_machines[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Sort by machine priority score
    # --------------------------------------------------------

    if "machine_priority_score" in top_machines.columns:

        top_machines = (
            top_machines
            .sort_values(
                by="machine_priority_score",
                ascending=False,
                na_position="last"
            )
            .head(10)
        )

    # --------------------------------------------------------
    # Select available columns
    # --------------------------------------------------------

    display_columns = [
        "Machine_ID",
        "average_risk_score",
        "maximum_risk_score",
        "average_health_score",
        "machine_priority_score",
        "fleet_priority_rank",
        "machine_status"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in top_machines.columns
    ]

    if available_columns:

        top_machines = top_machines[
            available_columns
        ].copy()

        # ----------------------------------------------------
        # Rename columns for dashboard
        # ----------------------------------------------------

        top_machines = top_machines.rename(
            columns={
                "Machine_ID": "Machine ID",
                "average_risk_score": "Average Risk",
                "maximum_risk_score": "Maximum Risk",
                "average_health_score": "Average Health",
                "machine_priority_score": "Priority Score",
                "fleet_priority_rank": "Fleet Rank",
                "machine_status": "Machine Status"
            }
        )

        # ----------------------------------------------------
        # Round numerical values
        # ----------------------------------------------------

        for column in [
            "Average Risk",
            "Maximum Risk",
            "Average Health",
            "Priority Score"
        ]:

            if column in top_machines.columns:

                top_machines[column] = (
                    top_machines[column]
                    .round(2)
                )

        # ----------------------------------------------------
        # Format Machine ID / Rank
        # ----------------------------------------------------

        if "Machine ID" in top_machines.columns:

            top_machines["Machine ID"] = (
                pd.to_numeric(
                    top_machines["Machine ID"],
                    errors="coerce"
                )
                .astype("Int64")
            )

        if "Fleet Rank" in top_machines.columns:

            top_machines["Fleet Rank"] = (
                pd.to_numeric(
                    top_machines["Fleet Rank"],
                    errors="coerce"
                )
                .astype("Int64")
            )

        # ----------------------------------------------------
        # Display table
        # ----------------------------------------------------

        st.dataframe(
            top_machines,
            width="stretch",
            hide_index=True
        )

    else:

        st.warning(
            "Required machine priority columns are not available."
        )

else:

    st.warning(
        "Machine maintenance summary is not available."
    )

    st.warning(
        "Machine maintenance summary is not available."
    )

# ============================================================
# MACHINE HEALTH MONITORING
# ============================================================

st.markdown("### Machine Health Monitoring")

if (
    machine_df is not None
    and not machine_df.empty
    and "average_health_score" in machine_df.columns
):

    health_df = machine_df.copy()

    # --------------------------------------------------------
    # Convert health score to numeric
    # --------------------------------------------------------

    health_df["average_health_score"] = pd.to_numeric(
        health_df["average_health_score"],
        errors="coerce"
    )

    health_df = health_df.dropna(
        subset=["average_health_score"]
    )

    # --------------------------------------------------------
    # Health statistics
    # --------------------------------------------------------

    average_health = health_df[
        "average_health_score"
    ].mean()

    minimum_health = health_df[
        "average_health_score"
    ].min()

    maximum_health = health_df[
        "average_health_score"
    ].max()

    healthy_machines = int(
        (
            health_df["average_health_score"] >= 70
        ).sum()
    )

    attention_machines = int(
        (
            (
                health_df["average_health_score"] >= 40
            )
            &
            (
                health_df["average_health_score"] < 70
            )
        ).sum()
    )

    critical_machines = int(
        (
            health_df["average_health_score"] < 40
        ).sum()
    )

    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    health_col1, health_col2, health_col3 = st.columns(3)

    with health_col1:

        st.metric(
            "Fleet Health",
            f"{average_health:.2f}/100"
        )

    with health_col2:

        st.metric(
            "Healthy Machines",
            f"{healthy_machines:,}"
        )

    with health_col3:

        st.metric(
            "Machines Requiring Attention",
            f"{attention_machines + critical_machines:,}"
        )

    # --------------------------------------------------------
    # Health distribution
    # --------------------------------------------------------

    health_distribution = pd.DataFrame(
        {
            "Health Status": [
                "Healthy",
                "Attention",
                "Critical"
            ],
            "Machines": [
                healthy_machines,
                attention_machines,
                critical_machines
            ]
        }
    )

    st.markdown("#### Machine Health Distribution")

    st.bar_chart(
        health_distribution.set_index(
            "Health Status"
        )
    )

    # --------------------------------------------------------
    # Lowest health machines
    # --------------------------------------------------------

    st.markdown(
        "#### Machines Requiring Attention"
    )

    low_health_columns = [
        "Machine_ID",
        "average_health_score",
        "average_risk_score",
        "machine_priority_score",
        "machine_status"
    ]

    available_health_columns = [
        column
        for column in low_health_columns
        if column in health_df.columns
    ]

    if available_health_columns:

        low_health_df = (
            health_df[
                available_health_columns
            ]
            .sort_values(
                "average_health_score",
                ascending=True
            )
            .head(10)
            .copy()
        )

        low_health_df = low_health_df.rename(
            columns={
                "Machine_ID": "Machine ID",
                "average_health_score": "Average Health",
                "average_risk_score": "Average Risk",
                "machine_priority_score": "Priority Score",
                "machine_status": "Machine Status"
            }
        )

        for column in [
            "Average Health",
            "Average Risk",
            "Priority Score"
        ]:

            if column in low_health_df.columns:

                low_health_df[column] = (
                    pd.to_numeric(
                        low_health_df[column],
                        errors="coerce"
                    )
                    .round(2)
                )

        st.dataframe(
            low_health_df,
            width="stretch",
            hide_index=True
        )

else:

    st.warning(
        "Machine health data is not available."
    )

# ============================================================
# ANOMALY INTELLIGENCE
# ============================================================

st.markdown("### Anomaly Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
    and "is_anomaly" in maintenance_df.columns
):

    anomaly_df = maintenance_df.copy()

    # --------------------------------------------------------
    # Normalize anomaly fields
    # --------------------------------------------------------

    anomaly_df["is_anomaly"] = pd.to_numeric(
        anomaly_df["is_anomaly"],
        errors="coerce"
    ).fillna(0).astype(int)

    if "anomaly_score" in anomaly_df.columns:

        anomaly_df["anomaly_score"] = pd.to_numeric(
            anomaly_df["anomaly_score"],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------------
    # Core anomaly metrics
    # --------------------------------------------------------

    total_records = len(anomaly_df)

    anomaly_records = int(
        anomaly_df["is_anomaly"].sum()
    )

    normal_records = (
        total_records - anomaly_records
    )

    anomaly_rate = (
        anomaly_records / total_records * 100
        if total_records > 0
        else 0
    )

    if "anomaly_severity" in anomaly_df.columns:

        critical_anomalies = int(
            (
                (
                    anomaly_df["anomaly_severity"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                == "critical"
            ).sum()
        )

        high_anomalies = int(
            (
                (
                    anomaly_df["anomaly_severity"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                == "high"
            ).sum()
        )

        medium_anomalies = int(
            (
                (
                    anomaly_df["anomaly_severity"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                == "medium"
            ).sum()
        )

    else:

        critical_anomalies = 0
        high_anomalies = 0
        medium_anomalies = 0

    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    anomaly_col1, anomaly_col2, anomaly_col3, anomaly_col4 = (
        st.columns(4)
    )

    with anomaly_col1:

        st.metric(
            "Detected Anomalies",
            f"{anomaly_records:,}"
        )

    with anomaly_col2:

        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.2f}%"
        )

    with anomaly_col3:

        st.metric(
            "Critical Anomalies",
            f"{critical_anomalies:,}"
        )

    with anomaly_col4:

        st.metric(
            "High Anomalies",
            f"{high_anomalies:,}"
        )

    # --------------------------------------------------------
    # Anomaly severity distribution
    # --------------------------------------------------------

    if "anomaly_severity" in anomaly_df.columns:

        severity_distribution = (
            anomaly_df[
                anomaly_df["is_anomaly"] == 1
            ]["anomaly_severity"]
            .astype(str)
            .str.strip()
            .value_counts()
            .rename_axis("Severity")
            .reset_index(
                name="Records"
            )
        )

        st.markdown(
            "#### Anomaly Severity Distribution"
        )

        if not severity_distribution.empty:

            st.bar_chart(
                severity_distribution.set_index(
                    "Severity"
                )
            )

        else:

            st.info(
                "No anomalous records available."
            )

    # --------------------------------------------------------
    # Anomaly domain distribution
    # --------------------------------------------------------

    if "anomaly_domain" in anomaly_df.columns:

        domain_distribution = (
            anomaly_df[
                anomaly_df["is_anomaly"] == 1
            ]["anomaly_domain"]
            .astype(str)
            .str.strip()
            .value_counts()
            .rename_axis("Anomaly Domain")
            .reset_index(
                name="Records"
            )
        )

        st.markdown(
            "#### Anomaly Domain Distribution"
        )

        if not domain_distribution.empty:

            st.bar_chart(
                domain_distribution.set_index(
                    "Anomaly Domain"
                )
            )

        else:

            st.info(
                "No anomaly domain information available."
            )

    # --------------------------------------------------------
    # Top anomalous records
    # --------------------------------------------------------

    st.markdown(
        "#### Top Anomalous Records"
    )

    anomalous_records = anomaly_df[
        anomaly_df["is_anomaly"] == 1
    ].copy()

    if not anomalous_records.empty:

        if "anomaly_score" in anomalous_records.columns:

            anomalous_records = (
                anomalous_records
                .sort_values(
                    "anomaly_score",
                    ascending=False
                )
                .head(15)
            )

        anomaly_display_columns = [
            "Machine_ID",
            "Date",
            "Timestamp",
            "anomaly_score",
            "anomaly_severity",
            "anomaly_domain",
            "anomaly_explanation"
        ]

        available_anomaly_columns = [
            column
            for column in anomaly_display_columns
            if column in anomalous_records.columns
        ]

        anomaly_table = anomalous_records[
            available_anomaly_columns
        ].copy()

        anomaly_table = anomaly_table.rename(
            columns={
                "Machine_ID": "Machine ID",
                "Date": "Date",
                "Timestamp": "Timestamp",
                "anomaly_score": "Anomaly Score",
                "anomaly_severity": "Severity",
                "anomaly_domain": "Domain",
                "anomaly_explanation": "Explanation"
            }
        )

        if "Anomaly Score" in anomaly_table.columns:

            anomaly_table["Anomaly Score"] = (
                pd.to_numeric(
                    anomaly_table["Anomaly Score"],
                    errors="coerce"
                )
                .round(4)
            )

        st.dataframe(
            anomaly_table,
            width="stretch",
            hide_index=True
        )

    else:

        st.success(
            "No anomalous records detected."
        )

else:

    st.warning(
        "Anomaly detection results are not available."
    )
# ============================================================
# ANOMALY TRENDS & MACHINE ANALYSIS
# ============================================================

st.markdown("### Anomaly Trends & Machine Analysis")

if (
    maintenance_df is not None
    and not maintenance_df.empty
    and "is_anomaly" in maintenance_df.columns
):

    trend_df = maintenance_df.copy()

    # --------------------------------------------------------
    # Normalize anomaly flag
    # --------------------------------------------------------

    trend_df["is_anomaly"] = pd.to_numeric(
        trend_df["is_anomaly"],
        errors="coerce"
    ).fillna(0).astype(int)

    # --------------------------------------------------------
    # Prepare date column
    # --------------------------------------------------------

    if "Date" in trend_df.columns:

        trend_df["Date"] = pd.to_datetime(
            trend_df["Date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Daily anomaly trend
    # --------------------------------------------------------

    if "Date" in trend_df.columns:

        daily_anomaly = (
            trend_df
            .dropna(subset=["Date"])
            .groupby("Date")
            .agg(
                Total_Records=("is_anomaly", "size"),
                Anomalies=("is_anomaly", "sum")
            )
            .reset_index()
        )

        daily_anomaly["Anomaly Rate"] = (
            daily_anomaly["Anomalies"]
            / daily_anomaly["Total_Records"]
            * 100
        )

        st.markdown(
            "#### Daily Anomaly Trend"
        )

        if not daily_anomaly.empty:

            st.line_chart(
                daily_anomaly.set_index(
                    "Date"
                )[["Anomalies", "Anomaly Rate"]]
            )

    # --------------------------------------------------------
    # Machine-wise anomaly analysis
    # --------------------------------------------------------

    if "Machine_ID" in trend_df.columns:

        machine_anomaly = (
            trend_df
            .groupby("Machine_ID")
            .agg(
                Total_Records=("is_anomaly", "size"),
                Anomalies=("is_anomaly", "sum")
            )
            .reset_index()
        )

        machine_anomaly["Anomaly Rate"] = (
            machine_anomaly["Anomalies"]
            / machine_anomaly["Total_Records"]
            * 100
        )

        machine_anomaly = (
            machine_anomaly
            .sort_values(
                "Anomaly Rate",
                ascending=False
            )
            .head(10)
        )

        st.markdown(
            "#### Highest Anomaly-Rate Machines"
        )

        display_machine_anomaly = (
            machine_anomaly
            .copy()
        )

        display_machine_anomaly[
            "Anomaly Rate"
        ] = (
            display_machine_anomaly[
                "Anomaly Rate"
            ]
            .round(2)
        )

        st.dataframe(
            display_machine_anomaly,
            width="stretch",
            hide_index=True
        )

    # --------------------------------------------------------
    # Average anomaly score
    # --------------------------------------------------------

    if "anomaly_score" in trend_df.columns:

        trend_df["anomaly_score"] = pd.to_numeric(
            trend_df["anomaly_score"],
            errors="coerce"
        )

        average_anomaly_score = (
            trend_df["anomaly_score"]
            .mean()
        )

        anomalous_score = (
            trend_df.loc[
                trend_df["is_anomaly"] == 1,
                "anomaly_score"
            ]
            .mean()
        )

        normal_score = (
            trend_df.loc[
                trend_df["is_anomaly"] == 0,
                "anomaly_score"
            ]
            .mean()
        )

        score_col1, score_col2, score_col3 = (
            st.columns(3)
        )

        with score_col1:

            st.metric(
                "Average Anomaly Score",
                f"{average_anomaly_score:.4f}"
            )

        with score_col2:

            st.metric(
                "Anomalous Record Score",
                f"{anomalous_score:.4f}"
            )

        with score_col3:

            st.metric(
                "Normal Record Score",
                f"{normal_score:.4f}"
            )

    # --------------------------------------------------------
    # Normal vs anomaly comparison
    # --------------------------------------------------------

    comparison_data = pd.DataFrame(
        {
            "Record Type": [
                "Normal",
                "Anomaly"
            ],
            "Records": [
                int(
                    (
                        trend_df["is_anomaly"] == 0
                    ).sum()
                ),
                int(
                    (
                        trend_df["is_anomaly"] == 1
                    ).sum()
                )
            ]
        }
    )

    st.markdown(
        "#### Normal vs Anomalous Records"
    )

    st.bar_chart(
        comparison_data.set_index(
            "Record Type"
        )
    )
# ============================================================
# SENSOR INTELLIGENCE
# ============================================================

st.markdown("### Sensor Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    sensor_df = maintenance_df.copy()

    sensor_columns = [
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Production_Speed_units_per_hr"
    ]

    available_sensor_columns = [
        column
        for column in sensor_columns
        if column in sensor_df.columns
    ]

    if available_sensor_columns:

        # ----------------------------------------------------
        # Convert sensor columns to numeric
        # ----------------------------------------------------

        for column in available_sensor_columns:

            sensor_df[column] = pd.to_numeric(
                sensor_df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Sensor KPI calculations
        # ----------------------------------------------------

        sensor_stats = {}

        for column in available_sensor_columns:

            series = sensor_df[column].dropna()

            if not series.empty:

                sensor_stats[column] = {
                    "mean": series.mean(),
                    "minimum": series.min(),
                    "maximum": series.max(),
                    "std": series.std()
                }

        # ----------------------------------------------------
        # Sensor overview table
        # ----------------------------------------------------

        overview_rows = []

        sensor_labels = {
            "Temperature_C": "Temperature",
            "Vibration_Hz": "Vibration",
            "Power_Consumption_kW": "Power Consumption",
            "Production_Speed_units_per_hr": "Production Speed"
        }

        for column in available_sensor_columns:

            stats = sensor_stats.get(column)

            if stats is not None:

                overview_rows.append(
                    {
                        "Sensor": sensor_labels.get(
                            column,
                            column
                        ),
                        "Average": round(
                            stats["mean"],
                            2
                        ),
                        "Minimum": round(
                            stats["minimum"],
                            2
                        ),
                        "Maximum": round(
                            stats["maximum"],
                            2
                        ),
                        "Std Deviation": round(
                            stats["std"],
                            2
                        )
                    }
                )

        sensor_overview = pd.DataFrame(
            overview_rows
        )

        st.markdown(
            "#### Sensor Overview"
        )

        if not sensor_overview.empty:

            st.dataframe(
                sensor_overview,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # Sensor trend selector
        # ----------------------------------------------------

        st.markdown(
            "#### Sensor Trend Analysis"
        )

        sensor_selector_options = {
            "Temperature": "Temperature_C",
            "Vibration": "Vibration_Hz",
            "Power Consumption":
                "Power_Consumption_kW",
            "Production Speed":
                "Production_Speed_units_per_hr"
        }

        available_selector_options = {
            label: column
            for label, column
            in sensor_selector_options.items()
            if column in sensor_df.columns
        }

        if available_selector_options:

            selected_sensor_label = st.selectbox(
                "Select Sensor",
                list(
                    available_selector_options.keys()
                )
            )

            selected_sensor = (
                available_selector_options[
                    selected_sensor_label
                ]
            )

            # ----------------------------------------------
            # Daily aggregation
            # ----------------------------------------------

            if "Date" in sensor_df.columns:

                sensor_df["Date"] = pd.to_datetime(
                    sensor_df["Date"],
                    errors="coerce"
                )

                daily_sensor = (
                    sensor_df
                    .dropna(subset=["Date"])
                    .groupby("Date")[
                        selected_sensor
                    ]
                    .mean()
                    .reset_index()
                )

                if not daily_sensor.empty:

                    st.line_chart(
                        daily_sensor.set_index(
                            "Date"
                        )
                    )

            # ----------------------------------------------
            # Distribution
            # ----------------------------------------------

            sensor_distribution = (
                sensor_df[
                    selected_sensor
                ]
                .dropna()
                .describe()
            )

            st.markdown(
                f"#### {selected_sensor_label} Statistics"
            )

            statistics_table = pd.DataFrame(
                {
                    "Metric": [
                        "Mean",
                        "Minimum",
                        "Maximum",
                        "Standard Deviation"
                    ],
                    "Value": [
                        round(
                            sensor_distribution.get(
                                "mean",
                                0
                            ),
                            2
                        ),
                        round(
                            sensor_distribution.get(
                                "min",
                                0
                            ),
                            2
                        ),
                        round(
                            sensor_distribution.get(
                                "max",
                                0
                            ),
                            2
                        ),
                        round(
                            sensor_distribution.get(
                                "std",
                                0
                            ),
                            2
                        )
                    ]
                }
            )

            st.dataframe(
                statistics_table,
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "No supported sensor columns were found."
        )

else:

    st.warning(
        "Sensor data is not available."
    )
# ============================================================
# SENSOR INTELLIGENCE - PART 2
# SENSOR ANOMALY & CONDITION ANALYSIS
# ============================================================

st.markdown("### Sensor Condition & Anomaly Analysis")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    condition_df = maintenance_df.copy()

    sensor_condition_columns = [
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Production_Speed_units_per_hr"
    ]

    available_condition_columns = [
        column
        for column in sensor_condition_columns
        if column in condition_df.columns
    ]

    if available_condition_columns:

        for column in available_condition_columns:

            condition_df[column] = pd.to_numeric(
                condition_df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # SENSOR CONDITION SCORE
        # ----------------------------------------------------

        condition_scores = []

        for column in available_condition_columns:

            series = condition_df[column]

            if series.notna().sum() < 10:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - (
                1.5 * iqr
            )

            upper_bound = q3 + (
                1.5 * iqr
            )

            abnormal_mask = (
                (series < lower_bound)
                |
                (series > upper_bound)
            )

            abnormal_count = int(
                abnormal_mask.sum()
            )

            total_count = int(
                series.notna().sum()
            )

            abnormal_rate = (
                abnormal_count
                / total_count
                * 100
                if total_count > 0
                else 0
            )

            condition_scores.append(
                {
                    "Sensor": column,
                    "Abnormal Records":
                        abnormal_count,
                    "Total Records":
                        total_count,
                    "Abnormal Rate (%)":
                        round(
                            abnormal_rate,
                            2
                        ),
                    "Lower Bound":
                        round(
                            lower_bound,
                            2
                        ),
                    "Upper Bound":
                        round(
                            upper_bound,
                            2
                        )
                }
            )

        sensor_condition_summary = pd.DataFrame(
            condition_scores
        )

        if not sensor_condition_summary.empty:

            sensor_condition_summary[
                "Sensor"
            ] = (
                sensor_condition_summary[
                    "Sensor"
                ]
                .replace(
                    {
                        "Temperature_C":
                            "Temperature",
                        "Vibration_Hz":
                            "Vibration",
                        "Power_Consumption_kW":
                            "Power Consumption",
                        "Production_Speed_units_per_hr":
                            "Production Speed"
                    }
                )
            )

            st.markdown(
                "#### Sensor Abnormality Summary"
            )

            st.dataframe(
                sensor_condition_summary,
                width="stretch",
                hide_index=True
            )

            st.markdown(
                "#### Sensor Abnormality Rate"
            )

            abnormal_chart = (
                sensor_condition_summary[
                    [
                        "Sensor",
                        "Abnormal Rate (%)"
                    ]
                ]
                .set_index("Sensor")
            )

            if not abnormal_chart.empty:
                abnormal_chart = abnormal_chart.copy()

                abnormal_chart["Abnormal Rate (%)"] = pd.to_numeric(
                    abnormal_chart["Abnormal Rate (%)"],
                    errors="coerce"
                )

                abnormal_chart = abnormal_chart.dropna(
                    subset=["Abnormal Rate (%)"]
                )

                if not abnormal_chart.empty:
                    fig_sensor_abnormality = px.bar(
                        abnormal_chart.reset_index(),
                        x="Sensor",
                        y="Abnormal Rate (%)",
                        title="Sensor Abnormality Rate",
                        labels={
                            "Sensor": "Sensor",
                            "Abnormal Rate (%)": "Abnormal Rate (%)"
                        },
                        text="Abnormal Rate (%)"
                    )

                    fig_sensor_abnormality.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside"
                    )

                    fig_sensor_abnormality.update_layout(
                        height=420,
                        margin=dict(
                            l=20,
                            r=20,
                            t=60,
                            b=20
                        ),
                        yaxis=dict(
                            title="Abnormal Rate (%)",
                            rangemode="tozero"
                        ),
                        xaxis=dict(
                            title="Sensor"
                        )
                    )

                    st.plotly_chart(
                        fig_sensor_abnormality,
                        width="stretch",
                        key="sensor_abnormality_rate_chart"
                    )
                else:
                    st.info(
                        "No valid sensor abnormality values are available."
                    )

        # ----------------------------------------------------
        # MACHINE-WISE SENSOR HEALTH
        # ----------------------------------------------------

        if "Machine_ID" in condition_df.columns:

            machine_sensor_health = (
                condition_df
                .groupby("Machine_ID")
                [
                    available_condition_columns
                ]
                .mean()
                .reset_index()
            )

            normalized_sensor_values = []

            for column in available_condition_columns:

                column_min = (
                    condition_df[column]
                    .min()
                )

                column_max = (
                    condition_df[column]
                    .max()
                )

                denominator = (
                    column_max
                    - column_min
                )

                if (
                    pd.isna(denominator)
                    or denominator == 0
                ):

                    normalized_values = (
                        pd.Series(
                            100.0,
                            index=machine_sensor_health.index
                        )
                    )

                else:

                    normalized_values = (
                        (
                            machine_sensor_health[column]
                            - column_min
                        )
                        / denominator
                        * 100
                    )

                normalized_sensor_values.append(
                    normalized_values
                )

            if normalized_sensor_values:

                sensor_matrix = pd.concat(
                    normalized_sensor_values,
                    axis=1
                )

                sensor_matrix.columns = (
                    available_condition_columns
                )

                machine_sensor_health[
                    "Sensor Profile Score"
                ] = (
                    sensor_matrix.mean(
                        axis=1
                    )
                )

                machine_sensor_health[
                    "Sensor Profile Score"
                ] = (
                    machine_sensor_health[
                        "Sensor Profile Score"
                    ]
                    .round(2)
                )

                machine_sensor_health = (
                    machine_sensor_health
                    .sort_values(
                        "Sensor Profile Score",
                        ascending=True
                    )
                )

                st.markdown(
                    "#### Machine Sensor Profile"
                )

                machine_sensor_display = (
                    machine_sensor_health
                    [
                        [
                            "Machine_ID",
                            "Sensor Profile Score"
                        ]
                    ]
                    .head(15)
                    .copy()
                )

                machine_sensor_display = (
                    machine_sensor_display
                    .rename(
                        columns={
                            "Machine_ID":
                                "Machine ID",
                            "Sensor Profile Score":
                                "Sensor Profile Score"
                        }
                    )
                )

                st.dataframe(
                    machine_sensor_display,
                    width="stretch",
                    hide_index=True
                )

        # ----------------------------------------------------
        # SENSOR CORRELATION
        # ----------------------------------------------------

        if len(
            available_condition_columns
        ) >= 2:

            correlation_matrix = (
                condition_df[
                    available_condition_columns
                ]
                .corr()
            )

            correlation_matrix = (
                correlation_matrix
                .round(3)
            )

            correlation_matrix = (
                correlation_matrix
                .rename(
                    columns={
                        "Temperature_C":
                            "Temperature",
                        "Vibration_Hz":
                            "Vibration",
                        "Power_Consumption_kW":
                            "Power Consumption",
                        "Production_Speed_units_per_hr":
                            "Production Speed"
                    }
                )
                .rename(
                    index={
                        "Temperature_C":
                            "Temperature",
                        "Vibration_Hz":
                            "Vibration",
                        "Power_Consumption_kW":
                            "Power Consumption",
                        "Production_Speed_units_per_hr":
                            "Production Speed"
                    }
                )
            )

            st.markdown(
                "#### Sensor Correlation Matrix"
            )

            st.dataframe(
                correlation_matrix,
                width="stretch"
            )

        # ----------------------------------------------------
        # HIGH-RISK SENSOR RECORDS
        # ----------------------------------------------------

        st.markdown(
            "#### Highest-Risk Sensor Records"
        )

        risk_sensor_columns = [
            "Machine_ID",
            "Date",
            "Timestamp",
            "Temperature_C",
            "Vibration_Hz",
            "Power_Consumption_kW",
            "Production_Speed_units_per_hr",
            "sensor_risk",
            "risk_score"
        ]

        available_risk_sensor_columns = [
            column
            for column in risk_sensor_columns
            if column in condition_df.columns
        ]

        if available_risk_sensor_columns:

            high_risk_sensor_df = (
                condition_df[
                    available_risk_sensor_columns
                ]
                .copy()
            )

            if "sensor_risk" in high_risk_sensor_df.columns:

                high_risk_sensor_df[
                    "sensor_risk"
                ] = pd.to_numeric(
                    high_risk_sensor_df[
                        "sensor_risk"
                    ],
                    errors="coerce"
                )

                high_risk_sensor_df = (
                    high_risk_sensor_df
                    .sort_values(
                        "sensor_risk",
                        ascending=False
                    )
                    .head(15)
                )

            elif "risk_score" in high_risk_sensor_df.columns:

                high_risk_sensor_df[
                    "risk_score"
                ] = pd.to_numeric(
                    high_risk_sensor_df[
                        "risk_score"
                    ],
                    errors="coerce"
                )

                high_risk_sensor_df = (
                    high_risk_sensor_df
                    .sort_values(
                        "risk_score",
                        ascending=False
                    )
                    .head(15)
                )

            high_risk_sensor_df = (
                high_risk_sensor_df
                .rename(
                    columns={
                        "Machine_ID":
                            "Machine ID",
                        "Temperature_C":
                            "Temperature",
                        "Vibration_Hz":
                            "Vibration",
                        "Power_Consumption_kW":
                            "Power Consumption",
                        "Production_Speed_units_per_hr":
                            "Production Speed",
                        "sensor_risk":
                            "Sensor Risk",
                        "risk_score":
                            "Overall Risk"
                    }
                )
            )

            for column in [
                "Temperature",
                "Vibration",
                "Power Consumption",
                "Production Speed",
                "Sensor Risk",
                "Overall Risk"
            ]:

                if column in high_risk_sensor_df.columns:

                    high_risk_sensor_df[
                        column
                    ] = pd.to_numeric(
                        high_risk_sensor_df[
                            column
                        ],
                        errors="coerce"
                    ).round(2)

            st.dataframe(
                high_risk_sensor_df,
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "No supported sensor intelligence "
            "columns were found."
        )

else:

    st.warning(
        "Sensor intelligence data is not available."
    )
# ============================================================
# NETWORK / 6G INTELLIGENCE - PART 1
# NETWORK HEALTH & CONNECTIVITY ANALYSIS
# ============================================================

st.markdown("### Network / 6G Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    network_df = maintenance_df.copy()

    network_columns = [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "network_health",
        "network_risk"
    ]

    available_network_columns = [
        column
        for column in network_columns
        if column in network_df.columns
    ]

    if available_network_columns:

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        for column in available_network_columns:

            network_df[column] = pd.to_numeric(
                network_df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # NETWORK KPI CALCULATIONS
        # ----------------------------------------------------

        latency_mean = 0
        latency_min = 0
        latency_max = 0

        packet_loss_mean = 0
        packet_loss_max = 0

        network_health_mean = 0
        network_risk_mean = 0

        if "Network_Latency_ms" in network_df.columns:

            latency_series = (
                network_df[
                    "Network_Latency_ms"
                ]
                .dropna()
            )

            if not latency_series.empty:

                latency_mean = (
                    latency_series.mean()
                )

                latency_min = (
                    latency_series.min()
                )

                latency_max = (
                    latency_series.max()
                )

        if "Packet_Loss_%" in network_df.columns:

            packet_loss_series = (
                network_df[
                    "Packet_Loss_%"
                ]
                .dropna()
            )

            if not packet_loss_series.empty:

                packet_loss_mean = (
                    packet_loss_series.mean()
                )

                packet_loss_max = (
                    packet_loss_series.max()
                )

        if "network_health" in network_df.columns:

            network_health_series = (
                network_df[
                    "network_health"
                ]
                .dropna()
            )

            if not network_health_series.empty:

                network_health_mean = (
                    network_health_series.mean()
                )

                # Handle both 0-1 and 0-100 scales.
                if network_health_mean <= 1:

                    network_health_display = (
                        network_health_mean * 100
                    )

                else:

                    network_health_display = (
                        network_health_mean
                    )

            else:

                network_health_display = 0

        else:

            network_health_display = 0

        if "network_risk" in network_df.columns:

            network_risk_series = (
                network_df[
                    "network_risk"
                ]
                .dropna()
            )

            if not network_risk_series.empty:

                network_risk_mean = (
                    network_risk_series.mean()
                )

                # Handle both 0-1 and 0-100 scales.
                if network_risk_mean <= 1:

                    network_risk_display = (
                        network_risk_mean * 100
                    )

                else:

                    network_risk_display = (
                        network_risk_mean
                    )

            else:

                network_risk_display = 0

        else:

            network_risk_display = 0

        # ----------------------------------------------------
        # NETWORK KPI CARDS
        # ----------------------------------------------------

        network_col1, network_col2, network_col3, network_col4 = (
            st.columns(4)
        )

        with network_col1:

            st.metric(
                "Avg Network Latency",
                f"{latency_mean:.2f} ms"
            )

        with network_col2:

            st.metric(
                "Avg Packet Loss",
                f"{packet_loss_mean:.3f}%"
            )

        with network_col3:

            st.metric(
                "Network Health",
                f"{network_health_display:.2f}/100"
            )

        with network_col4:

            st.metric(
                "Network Risk",
                f"{network_risk_display:.2f}/100"
            )

        # ----------------------------------------------------
        # NETWORK CONDITION SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "#### Network Condition Summary"
        )

        condition_rows = []

        if "Network_Latency_ms" in network_df.columns:

            condition_rows.append(
                {
                    "Metric": "Network Latency",
                    "Average": round(
                        latency_mean,
                        3
                    ),
                    "Minimum": round(
                        latency_min,
                        3
                    ),
                    "Maximum": round(
                        latency_max,
                        3
                    ),
                    "Unit": "ms"
                }
            )

        if "Packet_Loss_%" in network_df.columns:

            condition_rows.append(
                {
                    "Metric": "Packet Loss",
                    "Average": round(
                        packet_loss_mean,
                        4
                    ),
                    "Minimum": round(
                        packet_loss_series.min(),
                        4
                    ),
                    "Maximum": round(
                        packet_loss_max,
                        4
                    ),
                    "Unit": "%"
                }
            )

        if "network_health" in network_df.columns:

            condition_rows.append(
                {
                    "Metric": "Network Health",
                    "Average": round(
                        network_health_display,
                        2
                    ),
                    "Minimum": round(
                        (
                            network_df[
                                "network_health"
                            ].min()
                            * 100
                            if network_df[
                                "network_health"
                            ].mean() <= 1
                            else network_df[
                                "network_health"
                            ].min()
                        ),
                        2
                    ),
                    "Maximum": round(
                        (
                            network_df[
                                "network_health"
                            ].max()
                            * 100
                            if network_df[
                                "network_health"
                            ].mean() <= 1
                            else network_df[
                                "network_health"
                            ].max()
                        ),
                        2
                    ),
                    "Unit": "Score"
                }
            )

        if "network_risk" in network_df.columns:

            condition_rows.append(
                {
                    "Metric": "Network Risk",
                    "Average": round(
                        network_risk_display,
                        2
                    ),
                    "Minimum": round(
                        (
                            network_df[
                                "network_risk"
                            ].min()
                            * 100
                            if network_df[
                                "network_risk"
                            ].mean() <= 1
                            else network_df[
                                "network_risk"
                            ].min()
                        ),
                        2
                    ),
                    "Maximum": round(
                        (
                            network_df[
                                "network_risk"
                            ].max()
                            * 100
                            if network_df[
                                "network_risk"
                            ].mean() <= 1
                            else network_df[
                                "network_risk"
                            ].max()
                        ),
                        2
                    ),
                    "Unit": "Score"
                }
            )

        network_condition_summary = pd.DataFrame(
            condition_rows
        )

        if not network_condition_summary.empty:

            st.dataframe(
                network_condition_summary,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # NETWORK HEALTH DISTRIBUTION
        # ----------------------------------------------------

        if "network_health" in network_df.columns:

            health_values = (
                network_df[
                    "network_health"
                ]
                .dropna()
                .copy()
            )

            if not health_values.empty:

                if health_values.mean() <= 1:

                    health_values = (
                        health_values * 100
                    )

                network_health_bins = pd.cut(
                    health_values,
                    bins=[
                        -float("inf"),
                        25,
                        50,
                        75,
                        float("inf")
                    ],
                    labels=[
                        "Critical",
                        "Poor",
                        "Good",
                        "Excellent"
                    ]
                )
                network_health_bins = network_health_bins.astype(str)

                health_distribution = (
                    network_health_bins
                    .value_counts(
                        sort=False
                    )
                    .rename_axis(
                        "Network Health"
                    )
                    .reset_index(
                        name="Records"
                    )
                )

                st.markdown(
                    "#### Network Health Distribution"
                )

                st.bar_chart(
                    health_distribution.set_index(
                        "Network Health"
                    )
                )

        # ----------------------------------------------------
        # NETWORK RISK DISTRIBUTION
        # ----------------------------------------------------

        if "network_risk" in network_df.columns:

            risk_values = (
                network_df[
                    "network_risk"
                ]
                .dropna()
                .copy()
            )

            if not risk_values.empty:

                if risk_values.mean() <= 1:

                    risk_values = (
                        risk_values * 100
                    )

                network_risk_bins = pd.cut(
                    risk_values,
                    bins=[
                        -float("inf"),
                        25,
                        50,
                        75,
                        float("inf")
                    ],
                    labels=[
                        "Low",
                        "Moderate",
                        "High",
                        "Critical"
                    ]
                )
                network_risk_bins = network_risk_bins.astype(str)

                risk_distribution = (
                    network_risk_bins
                    .value_counts(
                        sort=False
                    )
                    .rename_axis(
                        "Network Risk"
                    )
                    .reset_index(
                        name="Records"
                    )
                )

                st.markdown(
                    "#### Network Risk Distribution"
                )

                st.bar_chart(
                    risk_distribution.set_index(
                        "Network Risk"
                    )
                )

    else:

        st.info(
            "No network or 6G telemetry columns "
            "were found."
        )

else:

    st.warning(
        "Network intelligence data is not available."
    )
# ============================================================
# NETWORK / 6G INTELLIGENCE - PART 2
# MACHINE, TREND & CONNECTIVITY ANALYSIS
# ============================================================

st.markdown("### Network Connectivity & Machine Analysis")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    connectivity_df = maintenance_df.copy()

    connectivity_columns = [
        "Machine_ID",
        "Date",
        "Timestamp",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "network_health",
        "network_risk"
    ]

    available_connectivity_columns = [
        column
        for column in connectivity_columns
        if column in connectivity_df.columns
    ]

    if available_connectivity_columns:

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        for column in [
            "Network_Latency_ms",
            "Packet_Loss_%",
            "network_health",
            "network_risk"
        ]:

            if column in connectivity_df.columns:

                connectivity_df[column] = pd.to_numeric(
                    connectivity_df[column],
                    errors="coerce"
                )

        # ----------------------------------------------------
        # MACHINE-WISE NETWORK PERFORMANCE
        # ----------------------------------------------------

        if "Machine_ID" in connectivity_df.columns:

            machine_group_columns = []

            for column in [
                "Network_Latency_ms",
                "Packet_Loss_%",
                "network_health",
                "network_risk"
            ]:

                if column in connectivity_df.columns:
                    machine_group_columns.append(
                        column
                    )

            if machine_group_columns:

                machine_network = (
                    connectivity_df
                    .groupby("Machine_ID")
                    [
                        machine_group_columns
                    ]
                    .mean()
                    .reset_index()
                )

                # Convert normalized scores to 0-100.
                if "network_health" in machine_network.columns:

                    if (
                        machine_network[
                            "network_health"
                        ].mean()
                        <= 1
                    ):

                        machine_network[
                            "network_health"
                        ] = (
                            machine_network[
                                "network_health"
                            ] * 100
                        )

                if "network_risk" in machine_network.columns:

                    if (
                        machine_network[
                            "network_risk"
                        ].mean()
                        <= 1
                    ):

                        machine_network[
                            "network_risk"
                        ] = (
                            machine_network[
                                "network_risk"
                            ] * 100
                        )

                st.markdown(
                    "#### Machine-wise Network Performance"
                )

                display_machine_network = (
                    machine_network
                    .copy()
                )

                display_machine_network = (
                    display_machine_network
                    .sort_values(
                        "network_risk",
                        ascending=False
                    )
                    .head(15)
                )

                display_machine_network = (
                    display_machine_network
                    .rename(
                        columns={
                            "Machine_ID":
                                "Machine ID",
                            "Network_Latency_ms":
                                "Avg Latency (ms)",
                            "Packet_Loss_%":
                                "Avg Packet Loss (%)",
                            "network_health":
                                "Network Health",
                            "network_risk":
                                "Network Risk"
                        }
                    )
                )

                for column in [
                    "Avg Latency (ms)",
                    "Avg Packet Loss (%)",
                    "Network Health",
                    "Network Risk"
                ]:

                    if column in display_machine_network.columns:

                        display_machine_network[
                            column
                        ] = pd.to_numeric(
                            display_machine_network[
                                column
                            ],
                            errors="coerce"
                        ).round(3)

                st.dataframe(
                    display_machine_network,
                    width="stretch",
                    hide_index=True
                )

        # ----------------------------------------------------
        # LATENCY TREND
        # ----------------------------------------------------

        if (
            "Date" in connectivity_df.columns
            and "Network_Latency_ms"
            in connectivity_df.columns
        ):

            connectivity_df["Date"] = pd.to_datetime(
                connectivity_df["Date"],
                errors="coerce"
            )

            daily_latency = (
                connectivity_df
                .dropna(
                    subset=["Date"]
                )
                .groupby("Date")
                [
                    "Network_Latency_ms"
                ]
                .agg(
                    [
                        "mean",
                        "min",
                        "max"
                    ]
                )
                .reset_index()
            )

            daily_latency = daily_latency.rename(
                columns={
                    "mean":
                        "Average Latency",
                    "min":
                        "Minimum Latency",
                    "max":
                        "Maximum Latency"
                }
            )

            st.markdown(
                "#### Network Latency Trend"
            )

            if not daily_latency.empty:

                st.line_chart(
                    daily_latency.set_index(
                        "Date"
                    )
                )

        # ----------------------------------------------------
        # PACKET LOSS TREND
        # ----------------------------------------------------

        if (
            "Date" in connectivity_df.columns
            and "Packet_Loss_%"
            in connectivity_df.columns
        ):

            daily_packet_loss = (
                connectivity_df
                .dropna(
                    subset=["Date"]
                )
                .groupby("Date")
                [
                    "Packet_Loss_%"
                ]
                .agg(
                    [
                        "mean",
                        "min",
                        "max"
                    ]
                )
                .reset_index()
            )

            daily_packet_loss = (
                daily_packet_loss
                .rename(
                    columns={
                        "mean":
                            "Average Packet Loss",
                        "min":
                            "Minimum Packet Loss",
                        "max":
                            "Maximum Packet Loss"
                    }
                )
            )

            st.markdown(
                "#### Packet Loss Trend"
            )

            if not daily_packet_loss.empty:

                st.line_chart(
                    daily_packet_loss.set_index(
                        "Date"
                    )
                )

        # ----------------------------------------------------
        # NETWORK HEALTH TREND
        # ----------------------------------------------------

        if (
            "Date" in connectivity_df.columns
            and "network_health"
            in connectivity_df.columns
        ):

            daily_network_health = (
                connectivity_df
                .dropna(
                    subset=["Date"]
                )
                .groupby("Date")
                [
                    "network_health"
                ]
                .mean()
                .reset_index()
            )

            if (
                not daily_network_health.empty
                and daily_network_health[
                    "network_health"
                ].mean() <= 1
            ):

                daily_network_health[
                    "network_health"
                ] = (
                    daily_network_health[
                        "network_health"
                    ] * 100
                )

            daily_network_health = (
                daily_network_health
                .rename(
                    columns={
                        "network_health":
                            "Network Health"
                    }
                )
            )

            st.markdown(
                "#### Network Health Trend"
            )

            if not daily_network_health.empty:

                st.line_chart(
                    daily_network_health.set_index(
                        "Date"
                    )
                )

        # ----------------------------------------------------
        # NETWORK RISK TREND
        # ----------------------------------------------------

        if (
            "Date" in connectivity_df.columns
            and "network_risk"
            in connectivity_df.columns
        ):

            daily_network_risk = (
                connectivity_df
                .dropna(
                    subset=["Date"]
                )
                .groupby("Date")
                [
                    "network_risk"
                ]
                .mean()
                .reset_index()
            )

            if (
                not daily_network_risk.empty
                and daily_network_risk[
                    "network_risk"
                ].mean() <= 1
            ):

                daily_network_risk[
                    "network_risk"
                ] = (
                    daily_network_risk[
                        "network_risk"
                    ] * 100
                )

            daily_network_risk = (
                daily_network_risk
                .rename(
                    columns={
                        "network_risk":
                            "Network Risk"
                    }
                )
            )

            st.markdown(
                "#### Network Risk Trend"
            )

            if not daily_network_risk.empty:

                st.line_chart(
                    daily_network_risk.set_index(
                        "Date"
                    )
                )

        # ----------------------------------------------------
        # NETWORK STRESS RECORDS
        # ----------------------------------------------------

        st.markdown(
            "#### Highest Network Stress Records"
        )

        stress_columns = [
            "Machine_ID",
            "Date",
            "Timestamp",
            "Network_Latency_ms",
            "Packet_Loss_%",
            "network_health",
            "network_risk"
        ]

        available_stress_columns = [
            column
            for column in stress_columns
            if column in connectivity_df.columns
        ]

        if available_stress_columns:

            stress_df = (
                connectivity_df[
                    available_stress_columns
                ]
                .copy()
            )

            if "network_risk" in stress_df.columns:

                stress_df = (
                    stress_df
                    .sort_values(
                        "network_risk",
                        ascending=False
                    )
                    .head(15)
                )

            elif "Network_Latency_ms" in stress_df.columns:

                stress_df = (
                    stress_df
                    .sort_values(
                        "Network_Latency_ms",
                        ascending=False
                    )
                    .head(15)
                )

            stress_df = (
                stress_df
                .rename(
                    columns={
                        "Machine_ID":
                            "Machine ID",
                        "Network_Latency_ms":
                            "Latency (ms)",
                        "Packet_Loss_%":
                            "Packet Loss (%)",
                        "network_health":
                            "Network Health",
                        "network_risk":
                            "Network Risk"
                    }
                )
            )

            for column in [
                "Latency (ms)",
                "Packet Loss (%)",
                "Network Health",
                "Network Risk"
            ]:

                if column in stress_df.columns:

                    stress_df[column] = (
                        pd.to_numeric(
                            stress_df[column],
                            errors="coerce"
                        )
                        .round(3)
                    )

            st.dataframe(
                stress_df,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # NETWORK ALERT SUMMARY
        # ----------------------------------------------------

        alert_counts = {
            "High Latency": 0,
            "High Packet Loss": 0,
            "High Network Risk": 0
        }

        if "Network_Latency_ms" in connectivity_df.columns:

            latency_threshold = (
                connectivity_df[
                    "Network_Latency_ms"
                ].quantile(0.95)
            )

            alert_counts[
                "High Latency"
            ] = int(
                (
                    connectivity_df[
                        "Network_Latency_ms"
                    ]
                    >= latency_threshold
                ).sum()
            )

        if "Packet_Loss_%" in connectivity_df.columns:

            packet_threshold = (
                connectivity_df[
                    "Packet_Loss_%"
                ].quantile(0.95)
            )

            alert_counts[
                "High Packet Loss"
            ] = int(
                (
                    connectivity_df[
                        "Packet_Loss_%"
                    ]
                    >= packet_threshold
                ).sum()
            )

        if "network_risk" in connectivity_df.columns:

            risk_values = connectivity_df[
                "network_risk"
            ]

            if risk_values.mean() <= 1:

                risk_values = (
                    risk_values * 100
                )

            risk_threshold = (
                risk_values.quantile(0.95)
            )

            alert_counts[
                "High Network Risk"
            ] = int(
                (
                    risk_values
                    >= risk_threshold
                ).sum()
            )

        network_alert_df = pd.DataFrame(
            {
                "Network Alert": list(
                    alert_counts.keys()
                ),
                "Records": list(
                    alert_counts.values()
                )
            }
        )

        st.markdown(
            "#### Network Alert Summary"
        )

        st.bar_chart(
            network_alert_df.set_index(
                "Network Alert"
            )
        )

    else:

        st.info(
            "Required network connectivity "
            "columns are not available."
        )

else:

    st.warning(
        "Network connectivity data is not available."
    )    
# ============================================================
# QUALITY & PRODUCTION INTELLIGENCE - PART 1
# QUALITY, ERROR & PRODUCTION PERFORMANCE
# ============================================================

st.markdown("### Quality & Production Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    quality_df = maintenance_df.copy()

    quality_columns = [
        "Quality_Control_Defect_Rate_%",
        "Error_Rate_%",
        "Production_Speed_units_per_hr",
        "Efficiency_Status",
        "quality_risk"
    ]

    available_quality_columns = [
        column
        for column in quality_columns
        if column in quality_df.columns
    ]

    if available_quality_columns:

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        for column in [
            "Quality_Control_Defect_Rate_%",
            "Error_Rate_%",
            "Production_Speed_units_per_hr",
            "quality_risk"
        ]:

            if column in quality_df.columns:

                quality_df[column] = pd.to_numeric(
                    quality_df[column],
                    errors="coerce"
                )

        # ----------------------------------------------------
        # QUALITY & PRODUCTION KPIs
        # ----------------------------------------------------

        defect_mean = 0
        defect_max = 0

        error_mean = 0
        error_max = 0

        production_mean = 0
        production_min = 0
        production_max = 0

        quality_risk_mean = 0

        if (
            "Quality_Control_Defect_Rate_%"
            in quality_df.columns
        ):

            defect_series = (
                quality_df[
                    "Quality_Control_Defect_Rate_%"
                ]
                .dropna()
            )

            if not defect_series.empty:

                defect_mean = (
                    defect_series.mean()
                )

                defect_max = (
                    defect_series.max()
                )

        if "Error_Rate_%" in quality_df.columns:

            error_series = (
                quality_df[
                    "Error_Rate_%"
                ]
                .dropna()
            )

            if not error_series.empty:

                error_mean = (
                    error_series.mean()
                )

                error_max = (
                    error_series.max()
                )

        if (
            "Production_Speed_units_per_hr"
            in quality_df.columns
        ):

            production_series = (
                quality_df[
                    "Production_Speed_units_per_hr"
                ]
                .dropna()
            )

            if not production_series.empty:

                production_mean = (
                    production_series.mean()
                )

                production_min = (
                    production_series.min()
                )

                production_max = (
                    production_series.max()
                )

        if "quality_risk" in quality_df.columns:

            quality_risk_series = (
                quality_df[
                    "quality_risk"
                ]
                .dropna()
            )

            if not quality_risk_series.empty:

                quality_risk_mean = (
                    quality_risk_series.mean()
                )

                if quality_risk_mean <= 1:

                    quality_risk_display = (
                        quality_risk_mean * 100
                    )

                else:

                    quality_risk_display = (
                        quality_risk_mean
                    )

            else:

                quality_risk_display = 0

        else:

            quality_risk_display = 0

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        quality_col1, quality_col2, quality_col3, quality_col4 = (
            st.columns(4)
        )

        with quality_col1:

            st.metric(
                "Avg Defect Rate",
                f"{defect_mean:.3f}%"
            )

        with quality_col2:

            st.metric(
                "Avg Error Rate",
                f"{error_mean:.3f}%"
            )

        with quality_col3:

            st.metric(
                "Avg Production Speed",
                f"{production_mean:.2f} units/hr"
            )

        with quality_col4:

            st.metric(
                "Quality Risk",
                f"{quality_risk_display:.2f}/100"
            )

        # ----------------------------------------------------
        # QUALITY PERFORMANCE SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "#### Quality Performance Summary"
        )

        quality_summary_rows = []

        if (
            "Quality_Control_Defect_Rate_%"
            in quality_df.columns
        ):

            quality_summary_rows.append(
                {
                    "Metric": "Defect Rate",
                    "Average": round(
                        defect_mean,
                        3
                    ),
                    "Maximum": round(
                        defect_max,
                        3
                    ),
                    "Unit": "%"
                }
            )

        if "Error_Rate_%" in quality_df.columns:

            quality_summary_rows.append(
                {
                    "Metric": "Error Rate",
                    "Average": round(
                        error_mean,
                        3
                    ),
                    "Maximum": round(
                        error_max,
                        3
                    ),
                    "Unit": "%"
                }
            )

        if (
            "Production_Speed_units_per_hr"
            in quality_df.columns
        ):

            quality_summary_rows.append(
                {
                    "Metric": "Production Speed",
                    "Average": round(
                        production_mean,
                        2
                    ),
                    "Minimum": round(
                        production_min,
                        2
                    ),
                    "Maximum": round(
                        production_max,
                        2
                    ),
                    "Unit": "units/hr"
                }
            )

        if "quality_risk" in quality_df.columns:

            quality_summary_rows.append(
                {
                    "Metric": "Quality Risk",
                    "Average": round(
                        quality_risk_display,
                        2
                    ),
                    "Unit": "Score / 100"
                }
            )

        quality_summary = pd.DataFrame(
            quality_summary_rows
        )

        if not quality_summary.empty:

            st.dataframe(
                quality_summary,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # EFFICIENCY STATUS DISTRIBUTION
        # ----------------------------------------------------

        if "Efficiency_Status" in quality_df.columns:

            efficiency_distribution = (
                quality_df[
                    "Efficiency_Status"
                ]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "nan": "Unknown"
                    }
                )
                .value_counts()
                .rename_axis(
                    "Efficiency Status"
                )
                .reset_index(
                    name="Records"
                )
            )

            st.markdown(
                "#### Efficiency Status Distribution"
            )

            if not efficiency_distribution.empty:

                st.bar_chart(
                    efficiency_distribution.set_index(
                        "Efficiency Status"
                    )
                )

        # ----------------------------------------------------
        # PRODUCTION SPEED DISTRIBUTION
        # ----------------------------------------------------

        if (
            "Production_Speed_units_per_hr"
            in quality_df.columns
        ):

            production_values = (
                quality_df[
                    "Production_Speed_units_per_hr"
                ]
                .dropna()
            )

            if not production_values.empty:

                production_bins = pd.cut(
                    production_values,
                    bins=5
                )

                production_distribution = (
                    production_bins
                    .value_counts(
                        sort=False
                    )
                    .rename_axis(
                        "Production Speed Range"
                    )
                    .reset_index(
                        name="Records"
                    )
                )

                st.markdown(
                    "#### Production Speed Distribution"
                )

                production_distribution["Production Speed Range"] = (
                    production_distribution["Production Speed Range"].astype(str)
                )

                st.bar_chart(
                    production_distribution.set_index(
                        "Production Speed Range"
                    )["Records"]
                )

        # ----------------------------------------------------
        # QUALITY RISK DISTRIBUTION
        # ----------------------------------------------------

        if "quality_risk" in quality_df.columns:

            quality_risk_values = (
                quality_df[
                    "quality_risk"
                ]
                .dropna()
                .copy()
            )

            if not quality_risk_values.empty:

                if (
                    quality_risk_values.mean()
                    <= 1
                ):

                    quality_risk_values = (
                        quality_risk_values * 100
                    )

                quality_risk_bins = pd.cut(
                    quality_risk_values,
                    bins=[
                        -float("inf"),
                        25,
                        50,
                        75,
                        float("inf")
                    ],
                    labels=[
                        "Low",
                        "Moderate",
                        "High",
                        "Critical"
                    ]
                )
                quality_risk_bins = quality_risk_bins.astype(str)

                quality_risk_distribution = (
                    quality_risk_bins
                    .value_counts(
                        sort=False
                    )
                    .rename_axis(
                        "Quality Risk"
                    )
                    .reset_index(
                        name="Records"
                    )
                )

                st.markdown(
                    "#### Quality Risk Distribution"
                )

                st.bar_chart(
                    quality_risk_distribution.set_index(
                        "Quality Risk"
                    )
                )

    else:

        st.info(
            "No quality or production columns "
            "were found."
        )

else:

    st.warning(
        "Quality and production data is not available."
    )
# ============================================================
# QUALITY & PRODUCTION INTELLIGENCE - PART 2
# MACHINE-WISE QUALITY & PRODUCTION ANALYSIS
# ============================================================

st.markdown("### Machine Quality & Production Analysis")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    machine_quality_df = maintenance_df.copy()

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in [
        "Quality_Control_Defect_Rate_%",
        "Error_Rate_%",
        "Production_Speed_units_per_hr",
        "quality_risk"
    ]:

        if column in machine_quality_df.columns:

            machine_quality_df[column] = pd.to_numeric(
                machine_quality_df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # MACHINE-WISE QUALITY PERFORMANCE
    # --------------------------------------------------------

    if "Machine_ID" in machine_quality_df.columns:

        machine_quality_metrics = []

        group_metrics = [
            "Quality_Control_Defect_Rate_%",
            "Error_Rate_%",
            "Production_Speed_units_per_hr",
            "quality_risk"
        ]

        available_group_metrics = [
            column
            for column in group_metrics
            if column in machine_quality_df.columns
        ]

        if available_group_metrics:

            machine_quality_summary = (
                machine_quality_df
                .groupby("Machine_ID")
                [
                    available_group_metrics
                ]
                .mean()
                .reset_index()
            )

            if "quality_risk" in machine_quality_summary.columns:

                if (
                    machine_quality_summary[
                        "quality_risk"
                    ].mean()
                    <= 1
                ):

                    machine_quality_summary[
                        "quality_risk"
                    ] = (
                        machine_quality_summary[
                            "quality_risk"
                        ] * 100
                    )

            st.markdown(
                "#### Machine-wise Quality Performance"
            )

            display_quality_machine = (
                machine_quality_summary
                .copy()
            )

            if "quality_risk" in display_quality_machine.columns:

                display_quality_machine = (
                    display_quality_machine
                    .sort_values(
                        "quality_risk",
                        ascending=False
                    )
                )

            display_quality_machine = (
                display_quality_machine
                .head(15)
                .rename(
                    columns={
                        "Machine_ID":
                            "Machine ID",
                        "Quality_Control_Defect_Rate_%":
                            "Avg Defect Rate (%)",
                        "Error_Rate_%":
                            "Avg Error Rate (%)",
                        "Production_Speed_units_per_hr":
                            "Avg Production Speed",
                        "quality_risk":
                            "Quality Risk"
                    }
                )
            )

            for column in [
                "Avg Defect Rate (%)",
                "Avg Error Rate (%)",
                "Avg Production Speed",
                "Quality Risk"
            ]:

                if column in display_quality_machine.columns:

                    display_quality_machine[
                        column
                    ] = pd.to_numeric(
                        display_quality_machine[
                            column
                        ],
                        errors="coerce"
                    ).round(3)

            st.dataframe(
                display_quality_machine,
                width="stretch",
                hide_index=True
            )

    # --------------------------------------------------------
    # DAILY QUALITY TREND
    # --------------------------------------------------------

    if (
        "Date" in machine_quality_df.columns
        and "Quality_Control_Defect_Rate_%"
        in machine_quality_df.columns
    ):

        machine_quality_df["Date"] = pd.to_datetime(
            machine_quality_df["Date"],
            errors="coerce"
        )

        daily_defect = (
    machine_quality_df[
        [
            "Date",
            "Quality_Control_Defect_Rate_%"
        ]
    ]
    .dropna(
        subset=["Date"]
    )
    .groupby("Date")
    [
        "Quality_Control_Defect_Rate_%"
    ]
            .agg(
                [
                    "mean",
                    "min",
                    "max"
                ]
            )
            .reset_index()
        )

        daily_defect = (
            daily_defect
            .rename(
                columns={
                    "mean":
                        "Average Defect Rate",
                    "min":
                        "Minimum Defect Rate",
                    "max":
                        "Maximum Defect Rate"
                }
            )
        )

        st.markdown(
            "#### Daily Defect Rate Trend"
        )

        if not daily_defect.empty:

            st.line_chart(
                daily_defect.set_index(
                    "Date"
                )
            )

    # --------------------------------------------------------
    # DAILY ERROR RATE TREND
    # --------------------------------------------------------

    if (
        "Date" in machine_quality_df.columns
        and "Error_Rate_%"
        in machine_quality_df.columns
    ):

        daily_error = (
            machine_quality_df
            .dropna(
                subset=["Date"]
            )
            .groupby("Date")
            [
                "Error_Rate_%"
            ]
            .agg(
                [
                    "mean",
                    "min",
                    "max"
                ]
            )
            .reset_index()
        )

        daily_error = (
            daily_error
            .rename(
                columns={
                    "mean":
                        "Average Error Rate",
                    "min":
                        "Minimum Error Rate",
                    "max":
                        "Maximum Error Rate"
                }
            )
        )

        st.markdown(
            "#### Daily Error Rate Trend"
        )

        if not daily_error.empty:

            st.line_chart(
                daily_error.set_index(
                    "Date"
                )
            )

    # --------------------------------------------------------
    # DAILY PRODUCTION SPEED TREND
    # --------------------------------------------------------

    if (
        "Date" in machine_quality_df.columns
        and "Production_Speed_units_per_hr"
        in machine_quality_df.columns
    ):

        daily_production = (
            machine_quality_df
            .dropna(
                subset=["Date"]
            )
            .groupby("Date")
            [
                "Production_Speed_units_per_hr"
            ]
            .agg(
                [
                    "mean",
                    "min",
                    "max"
                ]
            )
            .reset_index()
        )

        daily_production = (
            daily_production
            .rename(
                columns={
                    "mean":
                        "Average Production Speed",
                    "min":
                        "Minimum Production Speed",
                    "max":
                        "Maximum Production Speed"
                }
            )
        )

        st.markdown(
            "#### Daily Production Speed Trend"
        )

        if not daily_production.empty:

            st.line_chart(
                daily_production.set_index(
                    "Date"
                )
            )

    # --------------------------------------------------------
    # DAILY QUALITY RISK TREND
    # --------------------------------------------------------

    if (
        "Date" in machine_quality_df.columns
        and "quality_risk"
        in machine_quality_df.columns
    ):

        daily_quality_risk = (
            machine_quality_df
            .dropna(
                subset=["Date"]
            )
            .groupby("Date")
            [
                "quality_risk"
            ]
            .mean()
            .reset_index()
        )

        if (
            not daily_quality_risk.empty
            and daily_quality_risk[
                "quality_risk"
            ].mean()
            <= 1
        ):

            daily_quality_risk[
                "quality_risk"
            ] = (
                daily_quality_risk[
                    "quality_risk"
                ] * 100
            )

        daily_quality_risk = (
            daily_quality_risk
            .rename(
                columns={
                    "quality_risk":
                        "Quality Risk"
                }
            )
        )

        st.markdown(
            "#### Daily Quality Risk Trend"
        )

        if not daily_quality_risk.empty:

            st.line_chart(
                daily_quality_risk.set_index(
                    "Date"
                )
            )

    # --------------------------------------------------------
    # EFFICIENCY STATUS BY MACHINE
    # --------------------------------------------------------

    if (
        "Machine_ID" in machine_quality_df.columns
        and "Efficiency_Status"
        in machine_quality_df.columns
    ):

        machine_efficiency = (
            machine_quality_df
            .groupby(
                [
                    "Machine_ID",
                    "Efficiency_Status"
                ]
            )
            .size()
            .reset_index(
                name="Records"
            )
        )

        st.markdown(
            "#### Machine Efficiency Status"
        )

        efficiency_machine_options = sorted(
            machine_efficiency[
                "Machine_ID"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if efficiency_machine_options:

            selected_machine = st.selectbox(
                "Select Machine",
                efficiency_machine_options,
                key="quality_machine_selector"
            )

            selected_machine_efficiency = (
                machine_efficiency[
                    machine_efficiency[
                        "Machine_ID"
                    ]
                    == selected_machine
                ]
                .copy()
            )

            selected_machine_efficiency = (
                selected_machine_efficiency
                .set_index(
                    "Efficiency_Status"
                )
            )

            st.bar_chart(
                selected_machine_efficiency[
                    ["Records"]
                ]
            )

    # --------------------------------------------------------
    # QUALITY ALERT RECORDS
    # --------------------------------------------------------

    st.markdown(
        "#### Highest Quality-Risk Records"
    )

    quality_alert_columns = [
        "Machine_ID",
        "Date",
        "Timestamp",
        "Quality_Control_Defect_Rate_%",
        "Error_Rate_%",
        "Production_Speed_units_per_hr",
        "Efficiency_Status",
        "quality_risk"
    ]

    available_quality_alert_columns = [
        column
        for column in quality_alert_columns
        if column in machine_quality_df.columns
    ]

    if available_quality_alert_columns:

        quality_alert_df = (
            machine_quality_df[
                available_quality_alert_columns
            ]
            .copy()
        )

        if "quality_risk" in quality_alert_df.columns:

            quality_alert_df = (
                quality_alert_df
                .sort_values(
                    "quality_risk",
                    ascending=False
                )
                .head(15)
            )

        quality_alert_df = (
            quality_alert_df
            .rename(
                columns={
                    "Machine_ID":
                        "Machine ID",
                    "Quality_Control_Defect_Rate_%":
                        "Defect Rate (%)",
                    "Error_Rate_%":
                        "Error Rate (%)",
                    "Production_Speed_units_per_hr":
                        "Production Speed",
                    "Efficiency_Status":
                        "Efficiency Status",
                    "quality_risk":
                        "Quality Risk"
                }
            )
        )

        for column in [
            "Defect Rate (%)",
            "Error Rate (%)",
            "Production Speed",
            "Quality Risk"
        ]:

            if column in quality_alert_df.columns:

                quality_alert_df[
                    column
                ] = pd.to_numeric(
                    quality_alert_df[
                        column
                    ],
                    errors="coerce"
                ).round(3)

        st.dataframe(
            quality_alert_df,
            width="stretch",
            hide_index=True
        )

else:

    st.warning(
        "Machine quality analysis data "
        "is not available."
    )
# ============================================================
# ROOT CAUSE ANALYSIS - PART 1
# RISK DRIVER & PRIMARY CAUSE ANALYSIS
# ============================================================

st.markdown("### Root Cause Analysis")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    rca_df = maintenance_df.copy()

    # --------------------------------------------------------
    # RISK DRIVER COLUMNS
    # --------------------------------------------------------

    rca_columns = [
        "sensor_risk",
        "network_risk",
        "quality_risk",
        "historical_risk",
        "anomaly_score",
        "risk_score"
    ]

    available_rca_columns = [
        column
        for column in rca_columns
        if column in rca_df.columns
    ]

    if available_rca_columns:

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        for column in available_rca_columns:

            rca_df[column] = pd.to_numeric(
                rca_df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # NORMALIZE RISK DRIVERS TO 0-100
        # ----------------------------------------------------

        driver_columns = [
            "sensor_risk",
            "network_risk",
            "quality_risk",
            "historical_risk",
            "anomaly_score"
        ]

        for column in driver_columns:

            if column in rca_df.columns:

                column_mean = (
                    rca_df[column]
                    .mean()
                )

                if (
                    not pd.isna(column_mean)
                    and column_mean <= 1
                ):

                    rca_df[column] = (
                        rca_df[column] * 100
                    )

        # ----------------------------------------------------
        # ANOMALY SCORE SHOULD ONLY MATTER FOR ANOMALOUS DATA
        # ----------------------------------------------------

        if (
            "anomaly_score" in rca_df.columns
            and "is_anomaly" in rca_df.columns
        ):

            anomaly_flag = pd.to_numeric(
                rca_df["is_anomaly"],
                errors="coerce"
            ).fillna(0)

            rca_df.loc[
                anomaly_flag == 0,
                "anomaly_score"
            ] = 0

        # ----------------------------------------------------
        # DRIVER LABELS
        # ----------------------------------------------------

        driver_labels = {
            "sensor_risk":
                "Sensor / Mechanical",
            "network_risk":
                "Network / 6G",
            "quality_risk":
                "Quality / Production",
            "historical_risk":
                "Historical Trend",
            "anomaly_score":
                "Anomaly"
        }

        # ----------------------------------------------------
        # PRIMARY ROOT CAUSE
        # ----------------------------------------------------

        available_driver_columns = [
            column
            for column in driver_labels
            if column in rca_df.columns
        ]

        if available_driver_columns:

            rca_df["primary_root_cause"] = (
                rca_df[
                    available_driver_columns
                ]
                .idxmax(axis=1)
                .map(driver_labels)
            )

            # ------------------------------------------------
            # PRIMARY CAUSE SCORE
            # ------------------------------------------------

            rca_df["primary_cause_score"] = (
                rca_df[
                    available_driver_columns
                ]
                .max(axis=1)
                .round(2)
            )

            # ------------------------------------------------
            # SECONDARY ROOT CAUSE
            # ------------------------------------------------

            def get_secondary_cause(row):

                values = {}

                for column in available_driver_columns:

                    value = row.get(
                        column,
                        0
                    )

                    if pd.isna(value):
                        value = 0

                    values[
                        driver_labels[column]
                    ] = float(value)

                sorted_values = sorted(
                    values.items(),
                    key=lambda item: item[1],
                    reverse=True
                )

                if len(sorted_values) >= 2:

                    return sorted_values[1][0]

                return "None"

            rca_df["secondary_root_cause"] = (
                rca_df.apply(
                    get_secondary_cause,
                    axis=1
                )
            )

            # ------------------------------------------------
            # ROOT CAUSE CONFIDENCE
            # ------------------------------------------------

            total_driver_score = (
                rca_df[
                    available_driver_columns
                ]
                .sum(axis=1)
            )

            rca_df["root_cause_confidence"] = (
                (
                    rca_df[
                        "primary_cause_score"
                    ]
                    /
                    total_driver_score.replace(
                        0,
                        np.nan
                    )
                )
                * 100
            )

            rca_df[
                "root_cause_confidence"
            ] = (
                rca_df[
                    "root_cause_confidence"
                ]
                .fillna(0)
                .clip(
                    0,
                    100
                )
                .round(2)
            )

            # ------------------------------------------------
            # ROOT CAUSE DISTRIBUTION
            # ------------------------------------------------

            root_cause_distribution = (
                rca_df[
                    "primary_root_cause"
                ]
                .astype(str)
                .str.strip()
                .value_counts()
                .rename_axis(
                    "Primary Root Cause"
                )
                .reset_index(
                    name="Records"
                )
            )

            st.markdown(
                "#### Primary Root Cause Distribution"
            )

            if not root_cause_distribution.empty:

                st.bar_chart(
                    root_cause_distribution.set_index(
                        "Primary Root Cause"
                    )
                )

            # ------------------------------------------------
            # ROOT CAUSE KPI CARDS
            # ------------------------------------------------

            total_rca_records = len(
                rca_df
            )

            strongest_cause = "N/A"
            strongest_cause_count = 0

            if not root_cause_distribution.empty:

                strongest_cause = (
                    root_cause_distribution.iloc[
                        0
                    ]["Primary Root Cause"]
                )

                strongest_cause_count = int(
                    root_cause_distribution.iloc[
                        0
                    ]["Records"]
                )

            average_confidence = (
                rca_df[
                    "root_cause_confidence"
                ]
                .mean()
            )

            average_primary_score = (
                rca_df[
                    "primary_cause_score"
                ]
                .mean()
            )

            rca_col1, rca_col2, rca_col3, rca_col4 = (
                st.columns(4)
            )

            with rca_col1:

                st.metric(
                    "RCA Records",
                    f"{total_rca_records:,}"
                )

            with rca_col2:

                st.metric(
                    "Dominant Root Cause",
                    strongest_cause
                )

            with rca_col3:

                st.metric(
                    "Dominant Cause Records",
                    f"{strongest_cause_count:,}"
                )

            with rca_col4:

                st.metric(
                    "Avg RCA Confidence",
                    f"{average_confidence:.2f}%"
                )

            # ------------------------------------------------
            # ROOT CAUSE DRIVER SUMMARY
            # ------------------------------------------------

            driver_summary_rows = []

            for column in available_driver_columns:

                driver_series = (
                    rca_df[column]
                    .dropna()
                )

                if not driver_series.empty:

                    driver_summary_rows.append(
                        {
                            "Risk Driver":
                                driver_labels[column],
                            "Average Score":
                                round(
                                    driver_series.mean(),
                                    2
                                ),
                            "Maximum Score":
                                round(
                                    driver_series.max(),
                                    2
                                ),
                            "High Driver Records":
                                int(
                                    (
                                        driver_series
                                        >= 75
                                    ).sum()
                                )
                        }
                    )

            driver_summary = pd.DataFrame(
                driver_summary_rows
            )

            st.markdown(
                "#### Risk Driver Summary"
            )

            if not driver_summary.empty:

                st.dataframe(
                    driver_summary,
                    width="stretch",
                    hide_index=True
                )

            # ------------------------------------------------
            # TOP ROOT CAUSE RECORDS
            # ------------------------------------------------

            st.markdown(
                "#### Top Root Cause Records"
            )

            top_rca_columns = [
                "Machine_ID",
                "Date",
                "Timestamp",
                "risk_score",
                "primary_root_cause",
                "secondary_root_cause",
                "primary_cause_score",
                "root_cause_confidence"
            ]

            available_top_rca_columns = [
                column
                for column in top_rca_columns
                if column in rca_df.columns
            ]

            if available_top_rca_columns:

                top_rca = (
                    rca_df[
                        available_top_rca_columns
                    ]
                    .copy()
                )

                sort_column = None

                if (
                    "primary_cause_score"
                    in top_rca.columns
                ):

                    sort_column = (
                        "primary_cause_score"
                    )

                elif (
                    "risk_score"
                    in top_rca.columns
                ):

                    sort_column = "risk_score"

                if sort_column is not None:

                    top_rca = (
                        top_rca
                        .sort_values(
                            sort_column,
                            ascending=False
                        )
                        .head(20)
                    )

                top_rca = (
                    top_rca
                    .rename(
                        columns={
                            "Machine_ID":
                                "Machine ID",
                            "risk_score":
                                "Overall Risk",
                            "primary_root_cause":
                                "Primary Root Cause",
                            "secondary_root_cause":
                                "Secondary Root Cause",
                            "primary_cause_score":
                                "Cause Score",
                            "root_cause_confidence":
                                "RCA Confidence"
                        }
                    )
                )

                for column in [
                    "Overall Risk",
                    "Cause Score",
                    "RCA Confidence"
                ]:

                    if column in top_rca.columns:

                        top_rca[column] = (
                            pd.to_numeric(
                                top_rca[column],
                                errors="coerce"
                            )
                            .round(2)
                        )

                st.dataframe(
                    top_rca,
                    width="stretch",
                    hide_index=True
                )

        else:

            st.info(
                "No risk driver columns are available "
                "for root cause analysis."
            )

    else:

        st.info(
            "Risk driver data is not available."
        )

else:

    st.warning(
        "Root cause analysis data is not available."
    )
# ============================================================
# ROOT CAUSE ANALYSIS - PART 2
# MACHINE-WISE CAUSE & DRIVER CONTRIBUTION
# ============================================================

st.markdown("### Machine Root Cause Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    machine_rca_df = maintenance_df.copy()

    # --------------------------------------------------------
    # REQUIRED DRIVER COLUMNS
    # --------------------------------------------------------

    machine_rca_drivers = [
        "sensor_risk",
        "network_risk",
        "quality_risk",
        "historical_risk",
        "anomaly_score"
    ]

    available_machine_rca_drivers = [
        column
        for column in machine_rca_drivers
        if column in machine_rca_df.columns
    ]

    if (
        "Machine_ID" in machine_rca_df.columns
        and available_machine_rca_drivers
    ):

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        for column in available_machine_rca_drivers:

            machine_rca_df[column] = pd.to_numeric(
                machine_rca_df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # NORMALIZE DRIVER SCORES
        # ----------------------------------------------------

        for column in available_machine_rca_drivers:

            column_mean = (
                machine_rca_df[column]
                .mean()
            )

            if (
                not pd.isna(column_mean)
                and column_mean <= 1
            ):

                machine_rca_df[column] = (
                    machine_rca_df[column] * 100
                )

        # ----------------------------------------------------
        # ANOMALY FILTER
        # ----------------------------------------------------

        if "is_anomaly" in machine_rca_df.columns:

            anomaly_flag = pd.to_numeric(
                machine_rca_df["is_anomaly"],
                errors="coerce"
            ).fillna(0)

            if "anomaly_score" in machine_rca_df.columns:

                machine_rca_df.loc[
                    anomaly_flag == 0,
                    "anomaly_score"
                ] = 0

        # ----------------------------------------------------
        # MACHINE-WISE AVERAGE DRIVERS
        # ----------------------------------------------------

        machine_driver_summary = (
            machine_rca_df
            .groupby("Machine_ID")
            [
                available_machine_rca_drivers
            ]
            .mean()
            .reset_index()
        )

        # ----------------------------------------------------
        # PRIMARY MACHINE ROOT CAUSE
        # ----------------------------------------------------

        driver_name_map = {
            "sensor_risk":
                "Sensor / Mechanical",
            "network_risk":
                "Network / 6G",
            "quality_risk":
                "Quality / Production",
            "historical_risk":
                "Historical Trend",
            "anomaly_score":
                "Anomaly"
        }

        machine_driver_summary[
            "Primary Root Cause"
        ] = (
            machine_driver_summary[
                available_machine_rca_drivers
            ]
            .idxmax(axis=1)
            .map(driver_name_map)
        )

        machine_driver_summary[
            "Primary Cause Score"
        ] = (
            machine_driver_summary[
                available_machine_rca_drivers
            ]
            .max(axis=1)
            .round(2)
        )

        # ----------------------------------------------------
        # MACHINE RCA CONFIDENCE
        # ----------------------------------------------------

        machine_driver_total = (
            machine_driver_summary[
                available_machine_rca_drivers
            ]
            .sum(axis=1)
        )

        machine_driver_summary[
            "RCA Confidence"
        ] = (
            machine_driver_summary[
                "Primary Cause Score"
            ]
            /
            machine_driver_total.replace(
                0,
                np.nan
            )
            * 100
        )

        machine_driver_summary[
            "RCA Confidence"
        ] = (
            machine_driver_summary[
                "RCA Confidence"
            ]
            .fillna(0)
            .clip(
                0,
                100
            )
            .round(2)
        )

        # ----------------------------------------------------
        # MACHINE RCA TABLE
        # ----------------------------------------------------

        st.markdown(
            "#### Machine-wise Primary Root Cause"
        )

        machine_rca_display = (
            machine_driver_summary
            .copy()
            .sort_values(
                "Primary Cause Score",
                ascending=False
            )
            .head(20)
        )

        machine_rca_display = (
            machine_rca_display
            .rename(
                columns={
                    "Machine_ID":
                        "Machine ID"
                }
            )
        )

        st.dataframe(
            machine_rca_display[
                [
                    "Machine ID",
                    "Primary Root Cause",
                    "Primary Cause Score",
                    "RCA Confidence"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # MACHINE ROOT CAUSE DISTRIBUTION
        # ----------------------------------------------------

        machine_cause_distribution = (
            machine_driver_summary[
                "Primary Root Cause"
            ]
            .value_counts()
            .rename_axis(
                "Root Cause"
            )
            .reset_index(
                name="Machines"
            )
        )

        st.markdown(
            "#### Root Cause Across Machine Fleet"
        )

        if not machine_cause_distribution.empty:

            st.bar_chart(
                machine_cause_distribution.set_index(
                    "Root Cause"
                )
            )

        # ----------------------------------------------------
        # DRIVER CONTRIBUTION
        # ----------------------------------------------------

        contribution_rows = []

        for column in available_machine_rca_drivers:

            score_series = (
                machine_driver_summary[column]
                .dropna()
            )

            if not score_series.empty:

                contribution_rows.append(
                    {
                        "Risk Driver":
                            driver_name_map.get(
                                column,
                                column
                            ),
                        "Fleet Average":
                            round(
                                score_series.mean(),
                                2
                            ),
                        "Fleet Maximum":
                            round(
                                score_series.max(),
                                2
                            ),
                        "Machines >= 50":
                            int(
                                (
                                    score_series
                                    >= 50
                                ).sum()
                            ),
                        "Machines >= 75":
                            int(
                                (
                                    score_series
                                    >= 75
                                ).sum()
                            )
                    }
                )

        contribution_df = pd.DataFrame(
            contribution_rows
        )

        st.markdown(
            "#### Fleet Risk Driver Contribution"
        )

        if not contribution_df.empty:

            st.dataframe(
                contribution_df,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # SELECT MACHINE FOR RCA DRILL-DOWN
        # ----------------------------------------------------

        st.markdown(
            "#### Machine RCA Drill-down"
        )

        machine_options = sorted(
            machine_driver_summary[
                "Machine_ID"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if machine_options:

            selected_rca_machine = st.selectbox(
                "Select Machine for Root Cause Analysis",
                machine_options,
                key="rca_machine_selector"
            )

            selected_machine_rca = (
                machine_driver_summary[
                    machine_driver_summary[
                        "Machine_ID"
                    ]
                    == selected_rca_machine
                ]
                .copy()
            )

            if not selected_machine_rca.empty:

                selected_row = (
                    selected_machine_rca
                    .iloc[0]
                )

                selected_primary_cause = (
                    selected_row[
                        "Primary Root Cause"
                    ]
                )

                selected_cause_score = (
                    selected_row[
                        "Primary Cause Score"
                    ]
                )

                selected_confidence = (
                    selected_row[
                        "RCA Confidence"
                    ]
                )

                drill_col1, drill_col2, drill_col3 = (
                    st.columns(3)
                )

                with drill_col1:

                    st.metric(
                        "Primary Root Cause",
                        str(
                            selected_primary_cause
                        )
                    )

                with drill_col2:

                    st.metric(
                        "Cause Score",
                        f"{selected_cause_score:.2f}/100"
                    )

                with drill_col3:

                    st.metric(
                        "RCA Confidence",
                        f"{selected_confidence:.2f}%"
                    )

                # --------------------------------------------
                # DRIVER SCORES FOR SELECTED MACHINE
                # --------------------------------------------

                selected_driver_rows = []

                for column in available_machine_rca_drivers:

                    selected_driver_rows.append(
                        {
                            "Risk Driver":
                                driver_name_map.get(
                                    column,
                                    column
                                ),
                            "Score":
                                round(
                                    float(
                                        selected_row[
                                            column
                                        ]
                                    ),
                                    2
                                )
                        }
                    )

                selected_driver_df = pd.DataFrame(
                    selected_driver_rows
                )

                st.markdown(
                    "##### Machine Risk Driver Profile"
                )

                st.bar_chart(
                    selected_driver_df.set_index(
                        "Risk Driver"
                    )
                )

                # --------------------------------------------
                # AUTOMATIC RCA EXPLANATION
                # --------------------------------------------

                explanation_map = {
                    "Sensor / Mechanical":
                        (
                            "The machine's dominant risk "
                            "is associated with sensor or "
                            "mechanical operating conditions. "
                            "Inspect vibration, temperature, "
                            "power consumption and related "
                            "equipment behaviour."
                        ),

                    "Network / 6G":
                        (
                            "The dominant risk is associated "
                            "with communication performance. "
                            "Review network latency, packet "
                            "loss and connectivity quality "
                            "between machine systems."
                        ),

                    "Quality / Production":
                        (
                            "The dominant risk is associated "
                            "with production or quality "
                            "conditions. Review defect rate, "
                            "error rate and production "
                            "performance."
                        ),

                    "Historical Trend":
                        (
                            "The dominant risk is associated "
                            "with historical operating "
                            "behaviour. Review recent trends "
                            "and repeated degradation patterns."
                        ),

                    "Anomaly":
                        (
                            "The dominant risk is associated "
                            "with an abnormal operating "
                            "pattern detected by the anomaly "
                            "detection engine. Investigate "
                            "the corresponding sensor, "
                            "network and production signals."
                        )
                }

                selected_explanation = (
                    explanation_map.get(
                        selected_primary_cause,
                        "No automated explanation available."
                    )
                )

                st.info(
                    selected_explanation
                )

    else:

        st.info(
            "Machine-level root cause analysis "
            "requires Machine_ID and risk driver data."
        )

else:

    st.warning(
        "Machine root cause data is not available."
    )
# ============================================================
# FORECASTING & TREND INTELLIGENCE - PART 1
# OPERATIONAL RISK TREND FORECASTING
# ============================================================

st.markdown("### Forecasting & Trend Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    forecast_df = maintenance_df.copy()

    # --------------------------------------------------------
    # DATE PREPARATION
    # --------------------------------------------------------

    if "Date" in forecast_df.columns:

        forecast_df["Date"] = pd.to_datetime(
            forecast_df["Date"],
            errors="coerce"
        )

        forecast_df = forecast_df.dropna(
            subset=["Date"]
        )

    # --------------------------------------------------------
    # FORECASTING SIGNALS
    # --------------------------------------------------------

    forecast_signals = [
        "risk_score",
        "sensor_risk",
        "network_risk",
        "quality_risk",
        "anomaly_score"
    ]

    available_forecast_signals = [
        column
        for column in forecast_signals
        if column in forecast_df.columns
    ]

    if (
        "Date" in forecast_df.columns
        and available_forecast_signals
    ):

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        for column in available_forecast_signals:

            forecast_df[column] = pd.to_numeric(
                forecast_df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # DAILY AGGREGATION
        # ----------------------------------------------------

        daily_forecast = (
            forecast_df
            .groupby("Date")
            [
                available_forecast_signals
            ]
            .mean()
            .reset_index()
        )

        # ----------------------------------------------------
        # NORMALIZE RISK SIGNALS
        # ----------------------------------------------------

        for column in available_forecast_signals:

            signal_mean = (
                daily_forecast[column]
                .mean()
            )

            if (
                not pd.isna(signal_mean)
                and signal_mean <= 1
            ):

                daily_forecast[column] = (
                    daily_forecast[column] * 100
                )

        # ----------------------------------------------------
        # TREND DIRECTION FUNCTION
        # ----------------------------------------------------

        def calculate_trend_direction(series):

            clean_series = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
                .dropna()
            )

            if len(clean_series) < 3:

                return "Insufficient Data"

            recent_window = (
                clean_series
                .tail(
                    min(
                        7,
                        len(clean_series)
                    )
                )
            )

            earlier_window = (
                clean_series
                .head(
                    min(
                        7,
                        len(clean_series)
                    )
                )
            )

            recent_mean = (
                recent_window.mean()
            )

            earlier_mean = (
                earlier_window.mean()
            )

            difference = (
                recent_mean
                - earlier_mean
            )

            threshold = max(
                abs(earlier_mean) * 0.03,
                0.5
            )

            if difference > threshold:

                return "Increasing"

            if difference < -threshold:

                return "Decreasing"

            return "Stable"

        # ----------------------------------------------------
        # TREND SUMMARY
        # ----------------------------------------------------

        trend_rows = []

        signal_labels = {
            "risk_score":
                "Overall Risk",
            "sensor_risk":
                "Sensor Risk",
            "network_risk":
                "Network / 6G Risk",
            "quality_risk":
                "Quality Risk",
            "anomaly_score":
                "Anomaly Signal"
        }

        for column in available_forecast_signals:

            series = (
                daily_forecast[column]
                .dropna()
            )

            if series.empty:
                continue

            trend_rows.append(
                {
                    "Signal":
                        signal_labels.get(
                            column,
                            column
                        ),
                    "Current":
                        round(
                            float(
                                series.iloc[-1]
                            ),
                            2
                        ),
                    "Average":
                        round(
                            float(
                                series.mean()
                            ),
                            2
                        ),
                    "Minimum":
                        round(
                            float(
                                series.min()
                            ),
                            2
                        ),
                    "Maximum":
                        round(
                            float(
                                series.max()
                            ),
                            2
                        ),
                    "Trend":
                        calculate_trend_direction(
                            series
                        )
                }
            )

        trend_summary = pd.DataFrame(
            trend_rows
        )

        st.markdown(
            "#### Operational Trend Summary"
        )

        if not trend_summary.empty:

            st.dataframe(
                trend_summary,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # OVERALL RISK TREND
        # ----------------------------------------------------

        if "risk_score" in daily_forecast.columns:

            st.markdown(
                "#### Overall Risk Trend"
            )

            risk_trend = (
                daily_forecast[
                    [
                        "Date",
                        "risk_score"
                    ]
                ]
                .dropna()
                .copy()
            )

            risk_trend = (
                risk_trend
                .rename(
                    columns={
                        "risk_score":
                            "Overall Risk"
                    }
                )
                .set_index("Date")
            )

            if not risk_trend.empty:

                st.line_chart(
                    risk_trend
                )

        # ----------------------------------------------------
        # SENSOR RISK TREND
        # ----------------------------------------------------

        if "sensor_risk" in daily_forecast.columns:

            st.markdown(
                "#### Sensor Risk Trend"
            )

            sensor_trend = (
                daily_forecast[
                    [
                        "Date",
                        "sensor_risk"
                    ]
                ]
                .dropna()
                .copy()
            )

            sensor_trend = (
                sensor_trend
                .rename(
                    columns={
                        "sensor_risk":
                            "Sensor Risk"
                    }
                )
                .set_index("Date")
            )

            if not sensor_trend.empty:

                st.line_chart(
                    sensor_trend
                )

        # ----------------------------------------------------
        # NETWORK RISK TREND
        # ----------------------------------------------------

        if "network_risk" in daily_forecast.columns:

            st.markdown(
                "#### Network / 6G Risk Trend"
            )

            network_trend = (
                daily_forecast[
                    [
                        "Date",
                        "network_risk"
                    ]
                ]
                .dropna()
                .copy()
            )

            network_trend = (
                network_trend
                .rename(
                    columns={
                        "network_risk":
                            "Network / 6G Risk"
                    }
                )
                .set_index("Date")
            )

            if not network_trend.empty:

                st.line_chart(
                    network_trend
                )

        # ----------------------------------------------------
        # QUALITY RISK TREND
        # ----------------------------------------------------

        if "quality_risk" in daily_forecast.columns:

            st.markdown(
                "#### Quality Risk Trend"
            )

            quality_trend = (
                daily_forecast[
                    [
                        "Date",
                        "quality_risk"
                    ]
                ]
                .dropna()
                .copy()
            )

            quality_trend = (
                quality_trend
                .rename(
                    columns={
                        "quality_risk":
                            "Quality Risk"
                    }
                )
                .set_index("Date")
            )

            if not quality_trend.empty:

                st.line_chart(
                    quality_trend
                )

    else:

        st.info(
            "Date and forecasting signals are "
            "required for trend intelligence."
        )

else:

    st.warning(
        "Forecasting data is not available."
    )
# ============================================================
# FORECASTING & TREND INTELLIGENCE - PART 2
# SHORT-HORIZON RISK PROJECTION
# ============================================================

st.markdown("### Short-Horizon Risk Projection")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    projection_df = maintenance_df.copy()

    # --------------------------------------------------------
    # DATE PREPARATION
    # --------------------------------------------------------

    if "Date" in projection_df.columns:

        projection_df["Date"] = pd.to_datetime(
            projection_df["Date"],
            errors="coerce"
        )

        projection_df = projection_df.dropna(
            subset=["Date"]
        )

    if (
        "Date" in projection_df.columns
        and "risk_score" in projection_df.columns
    ):

        projection_df["risk_score"] = pd.to_numeric(
            projection_df["risk_score"],
            errors="coerce"
        )

        daily_risk_projection = (
            projection_df
            .dropna(
                subset=[
                    "Date",
                    "risk_score"
                ]
            )
            .groupby("Date")[
                "risk_score"
            ]
            .mean()
            .reset_index()
        )

        # ----------------------------------------------------
        # NORMALIZE RISK SCORE
        # ----------------------------------------------------

        if (
            not daily_risk_projection.empty
            and daily_risk_projection[
                "risk_score"
            ].mean() <= 1
        ):

            daily_risk_projection[
                "risk_score"
            ] = (
                daily_risk_projection[
                    "risk_score"
                ] * 100
            )

        # ----------------------------------------------------
        # LINEAR TREND FORECAST
        # ----------------------------------------------------

        if len(
            daily_risk_projection
        ) >= 7:

            historical_values = (
                daily_risk_projection[
                    "risk_score"
                ]
                .to_numpy()
            )

            historical_dates = (
                daily_risk_projection[
                    "Date"
                ]
                .to_numpy()
            )

            x_values = np.arange(
                len(
                    historical_values
                )
            )

            # Use recent observations for
            # short-horizon trend estimation.
            window_size = min(
                14,
                len(
                    historical_values
                )
            )

            recent_values = (
                historical_values[
                    -window_size:
                ]
            )

            recent_x = np.arange(
                window_size
            )

            if window_size >= 2:

                slope, intercept = np.polyfit(
                    recent_x,
                    recent_values,
                    1
                )

                forecast_horizon = 7

                future_x = np.arange(
                    window_size,
                    window_size
                    + forecast_horizon
                )

                forecast_values = (
                    slope * future_x
                    + intercept
                )

                forecast_values = np.clip(
                    forecast_values,
                    0,
                    100
                )

                last_date = (
                    daily_risk_projection[
                        "Date"
                    ]
                    .iloc[-1]
                )

                future_dates = pd.date_range(
                    start=(
                        last_date
                        + pd.Timedelta(
                            days=1
                        )
                    ),
                    periods=forecast_horizon,
                    freq="D"
                )

                forecast_output = pd.DataFrame(
                    {
                        "Date": future_dates,
                        "Projected Risk":
                            np.round(
                                forecast_values,
                                2
                            ),
                        "Forecast Type":
                            "Trend Projection"
                    }
                )

                # ------------------------------------------------
                # CURRENT VS PROJECTED
                # ------------------------------------------------

                current_risk = float(
                    historical_values[-1]
                )

                projected_risk = float(
                    forecast_values[-1]
                )

                projected_change = (
                    projected_risk
                    - current_risk
                )

                if projected_change > 2:

                    forecast_direction = (
                        "Increasing"
                    )

                elif projected_change < -2:

                    forecast_direction = (
                        "Decreasing"
                    )

                else:

                    forecast_direction = (
                        "Stable"
                    )

                # ------------------------------------------------
                # FORECAST KPIs
                # ------------------------------------------------

                forecast_col1, forecast_col2, forecast_col3, forecast_col4 = (
                    st.columns(4)
                )

                with forecast_col1:

                    st.metric(
                        "Current Risk",
                        f"{current_risk:.2f}/100"
                    )

                with forecast_col2:

                    st.metric(
                        "7-Day Projected Risk",
                        f"{projected_risk:.2f}/100"
                    )

                with forecast_col3:

                    st.metric(
                        "Projected Change",
                        f"{projected_change:+.2f}"
                    )

                with forecast_col4:

                    st.metric(
                        "Forecast Direction",
                        forecast_direction
                    )

                # ------------------------------------------------
                # FORECAST TABLE
                # ------------------------------------------------

                st.markdown(
                    "#### Seven-Day Risk Projection"
                )

                st.dataframe(
                    forecast_output,
                    width="stretch",
                    hide_index=True
                )

                # ------------------------------------------------
                # HISTORICAL + FORECAST DATA
                # ------------------------------------------------

                historical_chart = (
                    daily_risk_projection[
                        [
                            "Date",
                            "risk_score"
                        ]
                    ]
                    .copy()
                )

                historical_chart = (
                    historical_chart
                    .rename(
                        columns={
                            "risk_score":
                                "Historical Risk"
                        }
                    )
                )

                forecast_chart = (
                    forecast_output[
                        [
                            "Date",
                            "Projected Risk"
                        ]
                    ]
                    .copy()
                )

                historical_chart = (
                    historical_chart
                    .set_index("Date")
                )

                forecast_chart = (
                    forecast_chart
                    .set_index("Date")
                )

                combined_forecast_chart = (
                    historical_chart
                    .join(
                        forecast_chart,
                        how="outer"
                    )
                )

                st.markdown(
                    "#### Historical vs Projected Risk"
                )

                st.line_chart(
                    combined_forecast_chart
                )

                # ------------------------------------------------
                # FORECAST STATUS
                # ------------------------------------------------

                if forecast_direction == "Increasing":

                    st.warning(
                        "Projected risk is increasing "
                        "over the short horizon. "
                        "Review active maintenance "
                        "priorities and dominant risk drivers."
                    )

                elif forecast_direction == "Decreasing":

                    st.success(
                        "Projected risk is decreasing "
                        "over the short horizon."
                    )

                else:

                    st.info(
                        "Projected risk remains relatively "
                        "stable over the short horizon."
                    )

                # ------------------------------------------------
                # FORECAST LIMITATION NOTE
                # ------------------------------------------------

                st.caption(
                    "Forecast is a statistical trend projection "
                    "based on historical risk scores. It does "
                    "not represent a guaranteed machine-failure "
                    "prediction or Remaining Useful Life estimate."
                )

            else:

                st.info(
                    "Not enough historical observations "
                    "are available for trend projection."
                )

        else:

            st.info(
                "At least seven daily observations are "
                "required for short-horizon projection."
            )

    else:

        st.info(
            "Date and risk_score columns are required "
            "for risk projection."
        )

else:

    st.warning(
        "Risk projection data is not available."
    )
# ============================================================
# MACHINE DRILL-DOWN INTELLIGENCE - PART 1
# MACHINE HEALTH & RISK PROFILE
# ============================================================

st.markdown("### Machine Drill-down Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    drilldown_df = maintenance_df.copy()

    if "Machine_ID" in drilldown_df.columns:

        machine_ids = sorted(
            drilldown_df[
                "Machine_ID"
            ]
            .dropna()
            .unique()
            .tolist(),
            key=lambda value: str(value)
        )

        if machine_ids:

            selected_machine = st.selectbox(
                "Select Machine",
                machine_ids,
                key="machine_drilldown_selector"
            )

            selected_machine_df = (
                drilldown_df[
                    drilldown_df[
                        "Machine_ID"
                    ]
                    == selected_machine
                ]
                .copy()
            )

            if not selected_machine_df.empty:

                # ------------------------------------------------
                # HELPER FUNCTION
                # ------------------------------------------------

                def machine_metric_value(
                    dataframe,
                    column,
                    default=0.0
                ):

                    if column not in dataframe.columns:

                        return default

                    values = pd.to_numeric(
                        dataframe[column],
                        errors="coerce"
                    ).dropna()

                    if values.empty:

                        return default

                    return float(
                        values.mean()
                    )

                # ------------------------------------------------
                # MACHINE METRICS
                # ------------------------------------------------

                machine_risk = machine_metric_value(
                    selected_machine_df,
                    "risk_score"
                )

                if machine_risk <= 1:

                    machine_risk *= 100

                machine_health = machine_metric_value(
                    selected_machine_df,
                    "health_score"
                )

                if machine_health <= 1:

                    machine_health *= 100

                machine_anomaly_count = 0

                if "is_anomaly" in selected_machine_df.columns:

                    machine_anomaly_count = int(
                        pd.to_numeric(
                            selected_machine_df[
                                "is_anomaly"
                            ],
                            errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    )

                machine_records = len(
                    selected_machine_df
                )

                anomaly_rate = (
                    machine_anomaly_count
                    / machine_records
                    * 100
                    if machine_records > 0
                    else 0
                )

                # ------------------------------------------------
                # MACHINE STATUS
                # ------------------------------------------------

                machine_status = "Healthy"

                if machine_risk >= 75:

                    machine_status = "Critical"

                elif machine_risk >= 50:

                    machine_status = "High Risk"

                elif machine_risk >= 30:

                    machine_status = "Medium Risk"

                # ------------------------------------------------
                # KPI CARDS
                # ------------------------------------------------

                machine_col1, machine_col2, machine_col3, machine_col4 = (
                    st.columns(4)
                )

                with machine_col1:

                    st.metric(
                        "Machine Risk",
                        f"{machine_risk:.2f}/100"
                    )

                with machine_col2:

                    st.metric(
                        "Machine Health",
                        f"{machine_health:.2f}/100"
                    )

                with machine_col3:

                    st.metric(
                        "Anomaly Rate",
                        f"{anomaly_rate:.2f}%"
                    )

                with machine_col4:

                    st.metric(
                        "Machine Status",
                        machine_status
                    )

                # ------------------------------------------------
                # MACHINE PROFILE
                # ------------------------------------------------

                st.markdown(
                    f"#### Machine Profile â€” {selected_machine}"
                )

                profile_rows = [
                    {
                        "Metric":
                            "Machine ID",
                        "Value":
                            str(selected_machine)
                    },
                    {
                        "Metric":
                            "Records Analyzed",
                        "Value":
                            f"{machine_records:,}"
                    },
                    {
                        "Metric":
                            "Average Risk Score",
                        "Value":
                            f"{machine_risk:.2f}/100"
                    },
                    {
                        "Metric":
                            "Average Health Score",
                        "Value":
                            f"{machine_health:.2f}/100"
                    },
                    {
                        "Metric":
                            "Detected Anomalies",
                        "Value":
                            f"{machine_anomaly_count:,}"
                    },
                    {
                        "Metric":
                            "Anomaly Rate",
                        "Value":
                            f"{anomaly_rate:.2f}%"
                    },
                    {
                        "Metric":
                            "Operational Status",
                        "Value":
                            machine_status
                    }
                ]

                profile_df = pd.DataFrame(
                    profile_rows
                )

                st.dataframe(
                    profile_df,
                    width="stretch",
                    hide_index=True
                )

                # ------------------------------------------------
                # SENSOR PROFILE
                # ------------------------------------------------

                st.markdown(
                    "#### Sensor Performance"
                )

                sensor_profile_columns = {
                    "Temperature_C":
                        "Temperature (Â°C)",
                    "Vibration_Hz":
                        "Vibration (Hz)",
                    "Power_Consumption_kW":
                        "Power Consumption (kW)",
                    "Production_Speed_units_per_hr":
                        "Production Speed (units/hr)"
                }

                sensor_profile_rows = []

                for column, label in (
                    sensor_profile_columns.items()
                ):

                    if column in selected_machine_df.columns:

                        values = pd.to_numeric(
                            selected_machine_df[
                                column
                            ],
                            errors="coerce"
                        ).dropna()

                        if not values.empty:

                            sensor_profile_rows.append(
                                {
                                    "Sensor":
                                        label,
                                    "Average":
                                        round(
                                            float(
                                                values.mean()
                                            ),
                                            2
                                        ),
                                    "Minimum":
                                        round(
                                            float(
                                                values.min()
                                            ),
                                            2
                                        ),
                                    "Maximum":
                                        round(
                                            float(
                                                values.max()
                                            ),
                                            2
                                        ),
                                    "Standard Deviation":
                                        round(
                                            float(
                                                values.std()
                                            ),
                                            2
                                        )
                                }
                            )

                sensor_profile_df = pd.DataFrame(
                    sensor_profile_rows
                )

                if not sensor_profile_df.empty:

                    st.dataframe(
                        sensor_profile_df,
                        width="stretch",
                        hide_index=True
                    )

                # ------------------------------------------------
                # NETWORK PROFILE
                # ------------------------------------------------

                st.markdown(
                    "#### Network / 6G Performance"
                )

                network_profile_rows = []

                network_profile_columns = {
                    "Network_Latency_ms":
                        "Network Latency (ms)",
                    "Packet_Loss_%":
                        "Packet Loss (%)",
                    "network_health":
                        "Network Health",
                    "network_risk":
                        "Network Risk"
                }

                for column, label in (
                    network_profile_columns.items()
                ):

                    if column in selected_machine_df.columns:

                        values = pd.to_numeric(
                            selected_machine_df[
                                column
                            ],
                            errors="coerce"
                        ).dropna()

                        if not values.empty:

                            value = float(
                                values.mean()
                            )

                            if column in [
                                "network_health",
                                "network_risk"
                            ] and value <= 1:

                                value *= 100

                            network_profile_rows.append(
                                {
                                    "Metric":
                                        label,
                                    "Average":
                                        round(
                                            value,
                                            2
                                        )
                                }
                            )

                network_profile_df = pd.DataFrame(
                    network_profile_rows
                )

                if not network_profile_df.empty:

                    st.dataframe(
                        network_profile_df,
                        width="stretch",
                        hide_index=True
                    )

                # ------------------------------------------------
                # QUALITY PROFILE
                # ------------------------------------------------

                st.markdown(
                    "#### Quality & Production Profile"
                )

                quality_profile_rows = []

                quality_profile_columns = {
                    "Quality_Control_Defect_Rate_%":
                        "Defect Rate (%)",
                    "Error_Rate_%":
                        "Error Rate (%)",
                    "Production_Speed_units_per_hr":
                        "Production Speed",
                    "quality_risk":
                        "Quality Risk"
                }

                for column, label in (
                    quality_profile_columns.items()
                ):

                    if column in selected_machine_df.columns:

                        values = pd.to_numeric(
                            selected_machine_df[
                                column
                            ],
                            errors="coerce"
                        ).dropna()

                        if not values.empty:

                            value = float(
                                values.mean()
                            )

                            if (
                                column == "quality_risk"
                                and value <= 1
                            ):

                                value *= 100

                            quality_profile_rows.append(
                                {
                                    "Metric":
                                        label,
                                    "Average":
                                        round(
                                            value,
                                            2
                                        )
                                }
                            )

                quality_profile_df = pd.DataFrame(
                    quality_profile_rows
                )

                if not quality_profile_df.empty:

                    st.dataframe(
                        quality_profile_df,
                        width="stretch",
                        hide_index=True
                    )

    else:

        st.info(
            "Machine_ID is required for "
            "machine drill-down analysis."
        )

else:

    st.warning(
        "Machine data is not available."
    )
  # ============================================================
# MACHINE DRILL-DOWN INTELLIGENCE - PART 2
# MACHINE TRENDS, ANOMALIES & OPERATIONAL ANALYSIS
# ============================================================

st.markdown("### Machine Historical Intelligence")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    historical_machine_df = maintenance_df.copy()

    # --------------------------------------------------------
    # MACHINE SELECTION
    # --------------------------------------------------------

    if "Machine_ID" in historical_machine_df.columns:

        machine_list = sorted(
            historical_machine_df[
                "Machine_ID"
            ]
            .dropna()
            .unique()
            .tolist(),
            key=lambda value: str(value)
        )

        if machine_list:

            historical_machine_id = st.selectbox(
                "Select Machine for Historical Analysis",
                machine_list,
                key="machine_historical_selector"
            )

            machine_history = (
                historical_machine_df[
                    historical_machine_df[
                        "Machine_ID"
                    ]
                    == historical_machine_id
                ]
                .copy()
            )

            # ------------------------------------------------
            # DATE PREPARATION
            # ------------------------------------------------

            if "Date" in machine_history.columns:

                machine_history["Date"] = pd.to_datetime(
                    machine_history["Date"],
                    errors="coerce"
                )

            # ------------------------------------------------
            # TIMESTAMP PREPARATION
            # ------------------------------------------------

            if "Timestamp" in machine_history.columns:

                machine_history["Timestamp"] = pd.to_datetime(
                    machine_history["Timestamp"],
                    errors="coerce"
                )

            # =================================================
            # RISK TREND
            # =================================================

            st.markdown(
                f"#### Risk Trend â€” Machine {historical_machine_id}"
            )

            if (
                "Date" in machine_history.columns
                and "risk_score" in machine_history.columns
            ):

                machine_risk_trend = (
                    machine_history[
                        [
                            "Date",
                            "risk_score"
                        ]
                    ]
                    .dropna()
                    .groupby("Date")[
                        "risk_score"
                    ]
                    .mean()
                    .reset_index()
                )

                if not machine_risk_trend.empty:

                    machine_risk_trend[
                        "risk_score"
                    ] = pd.to_numeric(
                        machine_risk_trend[
                            "risk_score"
                        ],
                        errors="coerce"
                    )

                    if (
                        machine_risk_trend[
                            "risk_score"
                        ].mean() <= 1
                    ):

                        machine_risk_trend[
                            "risk_score"
                        ] *= 100

                    machine_risk_trend = (
                        machine_risk_trend
                        .dropna()
                        .set_index("Date")
                    )

                    st.line_chart(
                        machine_risk_trend[
                            ["risk_score"]
                        ]
                    )

            # =================================================
            # SENSOR TRENDS
            # =================================================

            st.markdown(
                "#### Sensor Trend Analysis"
            )

            sensor_trend_options = {
                "Temperature_C":
                    "Temperature (Â°C)",
                "Vibration_Hz":
                    "Vibration (Hz)",
                "Power_Consumption_kW":
                    "Power Consumption (kW)",
                "Production_Speed_units_per_hr":
                    "Production Speed (units/hr)"
            }

            available_sensor_trends = [
                column
                for column in sensor_trend_options
                if column in machine_history.columns
            ]

            if available_sensor_trends:

                selected_sensor = st.selectbox(
                    "Select Sensor",
                    available_sensor_trends,
                    format_func=lambda column:
                        sensor_trend_options[column],
                    key="machine_sensor_trend_selector"
                )

                if "Date" in machine_history.columns:

                    sensor_trend = (
                        machine_history[
                            [
                                "Date",
                                selected_sensor
                            ]
                        ]
                        .dropna()
                        .groupby("Date")[
                            selected_sensor
                        ]
                        .mean()
                    )

                    if not sensor_trend.empty:

                        st.line_chart(
                            sensor_trend
                        )

            else:

                st.info(
                    "Sensor trend data is not available."
                )

            # =================================================
            # NETWORK / 6G TREND
            # =================================================

            st.markdown(
                "#### Network / 6G Trend"
            )

            network_trend_columns = []

            if (
                "network_health"
                in machine_history.columns
            ):

                network_trend_columns.append(
                    "network_health"
                )

            if (
                "network_risk"
                in machine_history.columns
            ):

                network_trend_columns.append(
                    "network_risk"
                )

            if network_trend_columns:

                network_trend_data = (
                    machine_history[
                        [
                            "Date"
                        ]
                        + network_trend_columns
                    ]
                    .dropna(subset=["Date"])
                    .groupby("Date")[
                        network_trend_columns
                    ]
                    .mean()
                )

                for column in network_trend_columns:

                    network_trend_data[
                        column
                    ] = pd.to_numeric(
                        network_trend_data[
                            column
                        ],
                        errors="coerce"
                    )

                    if (
                        network_trend_data[
                            column
                        ].mean() <= 1
                    ):

                        network_trend_data[
                            column
                        ] *= 100

                if not network_trend_data.empty:

                    st.line_chart(
                        network_trend_data
                    )

            else:

                st.info(
                    "Network trend data is not available."
                )

            # =================================================
            # ANOMALY TIMELINE
            # =================================================

            st.markdown(
                "#### Machine Anomaly Timeline"
            )

            if (
                "Date" in machine_history.columns
                and "is_anomaly" in machine_history.columns
            ):

                anomaly_timeline = (
                    machine_history[
                        [
                            "Date",
                            "is_anomaly"
                        ]
                    ]
                    .dropna()
                    .groupby("Date")[
                        "is_anomaly"
                    ]
                    .sum()
                    .reset_index()
                )

                anomaly_timeline[
                    "Detected Anomalies"
                ] = (
                    anomaly_timeline[
                        "is_anomaly"
                    ]
                    .astype(int)
                )

                anomaly_timeline = (
                    anomaly_timeline[
                        [
                            "Date",
                            "Detected Anomalies"
                        ]
                    ]
                    .set_index("Date")
                )

                if not anomaly_timeline.empty:

                    st.bar_chart(
                        anomaly_timeline
                    )

            else:

                st.info(
                    "Anomaly timeline data is not available."
                )

            # =================================================
            # OPERATION MODE ANALYSIS
            # =================================================

            st.markdown(
                "#### Operation Mode Analysis"
            )

            if "Operation_Mode" in machine_history.columns:

                operation_mode_counts = (
                    machine_history[
                        "Operation_Mode"
                    ]
                    .astype(str)
                    .str.strip()
                    .value_counts()
                    .reset_index()
                )

                operation_mode_counts.columns = [
                    "Operation Mode",
                    "Records"
                ]

                st.bar_chart(
                    operation_mode_counts
                    .set_index(
                        "Operation Mode"
                    )
                )

            else:

                st.info(
                    "Operation mode data is not available."
                )

            # =================================================
            # EFFICIENCY STATUS
            # =================================================

            st.markdown(
                "#### Efficiency Status"
            )

            if "Efficiency_Status" in machine_history.columns:

                efficiency_counts = (
                    machine_history[
                        "Efficiency_Status"
                    ]
                    .astype(str)
                    .str.strip()
                    .value_counts()
                    .reset_index()
                )

                efficiency_counts.columns = [
                    "Efficiency Status",
                    "Records"
                ]

                st.bar_chart(
                    efficiency_counts
                    .set_index(
                        "Efficiency Status"
                    )
                )

            else:

                st.info(
                    "Efficiency status data is not available."
                )

            # =================================================
            # MACHINE RISK DRIVER PROFILE
            # =================================================

            st.markdown(
                "#### Machine Risk Driver Profile"
            )

            driver_columns = {
                "sensor_risk":
                    "Sensor / Mechanical",
                "network_risk":
                    "Network / 6G",
                "quality_risk":
                    "Quality / Production",
                "historical_risk":
                    "Historical Trend",
                "anomaly_score":
                    "Anomaly"
            }

            driver_rows = []

            for column, label in driver_columns.items():

                if column in machine_history.columns:

                    values = pd.to_numeric(
                        machine_history[
                            column
                        ],
                        errors="coerce"
                    ).dropna()

                    if not values.empty:

                        value = float(
                            values.mean()
                        )

                        if (
                            column == "anomaly_score"
                            and "is_anomaly"
                            in machine_history.columns
                        ):

                            anomaly_mask = (
                                pd.to_numeric(
                                    machine_history[
                                        "is_anomaly"
                                    ],
                                    errors="coerce"
                                )
                                .fillna(0)
                                == 1
                            )

                            anomaly_values = values[
                                anomaly_mask.loc[
                                    values.index
                                ]
                            ]

                            if anomaly_values.empty:

                                value = 0.0

                            else:

                                value = float(
                                    anomaly_values.mean()
                                )

                        if value <= 1:

                            value *= 100

                        driver_rows.append(
                            {
                                "Risk Driver":
                                    label,
                                "Average Score":
                                    round(
                                        value,
                                        2
                                    )
                            }
                        )

            driver_profile_df = pd.DataFrame(
                driver_rows
            )

            if not driver_profile_df.empty:

                st.bar_chart(
                    driver_profile_df
                    .set_index(
                        "Risk Driver"
                    )
                )

                st.dataframe(
                    driver_profile_df,
                    width="stretch",
                    hide_index=True
                )

            # =================================================
            # MACHINE RECORD DETAILS
            # =================================================

            st.markdown(
                "#### Machine Record Details"
            )

            display_columns = [
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
                "risk_score",
                "is_anomaly",
                "anomaly_severity",
                "anomaly_domain",
                "maintenance_priority"
            ]

            available_display_columns = [
                column
                for column in display_columns
                if column in machine_history.columns
            ]

            if available_display_columns:

                detail_table = (
                    machine_history[
                        available_display_columns
                    ]
                    .sort_values(
                        by="Date"
                        if "Date"
                        in available_display_columns
                        else available_display_columns[0],
                        ascending=False
                    )
                    .head(100)
                )

                st.dataframe(
                    detail_table,
                    width="stretch",
                    hide_index=True
                )

        else:

            st.info(
                "No machines are available "
                "for drill-down analysis."
            )

    else:

        st.info(
            "Machine_ID is required for "
            "machine historical analysis."
        )

else:

    st.warning(
        "Machine historical data is not available."
    )
# ============================================================
# MAINTENANCE RECOMMENDATIONS & ACTION QUEUE - PART 1
# ============================================================

st.markdown("### Maintenance Recommendations & Action Queue")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    recommendation_df = maintenance_df.copy()

    # --------------------------------------------------------
    # HELPER FUNCTIONS
    # --------------------------------------------------------

    def recommendation_numeric(
        dataframe,
        column,
        default=0.0
    ):

        if column not in dataframe.columns:

            return default

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        ).dropna()

        if values.empty:

            return default

        return float(values.mean())

    # --------------------------------------------------------
    # PRIORITY COUNTS
    # --------------------------------------------------------

    if "maintenance_priority" in recommendation_df.columns:

        recommendation_priority = (
            recommendation_df[
                "maintenance_priority"
            ]
            .astype(str)
            .str.strip()
        )

        p1_actions = int(
            (
                recommendation_priority
                == "P1 - Emergency"
            ).sum()
        )

        p2_actions = int(
            (
                recommendation_priority
                == "P2 - High"
            ).sum()
        )

        p3_actions = int(
            (
                recommendation_priority
                == "P3 - Planned"
            ).sum()
        )

        p4_actions = int(
            (
                recommendation_priority
                == "P4 - Routine"
            ).sum()
        )

    else:

        p1_actions = 0
        p2_actions = 0
        p3_actions = 0
        p4_actions = 0

    # --------------------------------------------------------
    # ACTION QUEUE KPI
    # --------------------------------------------------------

    action_col1, action_col2, action_col3, action_col4 = (
        st.columns(4)
    )

    with action_col1:

        st.metric(
            "Emergency Actions",
            f"{p1_actions:,}"
        )

    with action_col2:

        st.metric(
            "High Priority Actions",
            f"{p2_actions:,}"
        )

    with action_col3:

        st.metric(
            "Planned Actions",
            f"{p3_actions:,}"
        )

    with action_col4:

        st.metric(
            "Routine Actions",
            f"{p4_actions:,}"
        )

    # --------------------------------------------------------
    # MACHINE-LEVEL ACTION QUEUE
    # --------------------------------------------------------

    st.markdown(
        "#### Machine-Level Maintenance Queue"
    )

    if "Machine_ID" in recommendation_df.columns:

        machine_action_rows = []

        grouped_machines = (
            recommendation_df
            .groupby("Machine_ID")
        )

        for machine_id, machine_group in grouped_machines:

            avg_risk = recommendation_numeric(
                machine_group,
                "risk_score"
            )

            max_risk = 0.0

            if "risk_score" in machine_group.columns:

                risk_values = pd.to_numeric(
                    machine_group[
                        "risk_score"
                    ],
                    errors="coerce"
                ).dropna()

                if not risk_values.empty:

                    max_risk = float(
                        risk_values.max()
                    )

            if avg_risk <= 1:

                avg_risk *= 100

            if max_risk <= 1:

                max_risk *= 100

            anomaly_count = 0

            if "is_anomaly" in machine_group.columns:

                anomaly_count = int(
                    pd.to_numeric(
                        machine_group[
                            "is_anomaly"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            priority = "P4 - Routine"

            if (
                max_risk >= 75
                or anomaly_count >= 10
            ):

                priority = "P1 - Emergency"

            elif (
                avg_risk >= 50
                or max_risk >= 60
                or anomaly_count >= 5
            ):

                priority = "P2 - High"

            elif avg_risk >= 30:

                priority = "P3 - Planned"

            # ------------------------------------------------
            # RECOMMENDED ACTION
            # ------------------------------------------------

            action = (
                "Routine monitoring and "
                "scheduled preventive inspection."
            )

            if priority == "P1 - Emergency":

                action = (
                    "Immediate engineering inspection, "
                    "isolate critical condition if required, "
                    "and perform corrective maintenance."
                )

            elif priority == "P2 - High":

                action = (
                    "Schedule priority inspection and "
                    "perform targeted preventive maintenance."
                )

            elif priority == "P3 - Planned":

                action = (
                    "Include machine in the next "
                    "planned maintenance cycle."
                )

            # ------------------------------------------------
            # DOMINANT DRIVER
            # ------------------------------------------------

            driver_values = {}

            driver_columns = {
                "sensor_risk":
                    "Sensor / Mechanical",
                "network_risk":
                    "Network / 6G",
                "quality_risk":
                    "Quality / Production",
                "historical_risk":
                    "Historical Trend"
            }

            for column, label in driver_columns.items():

                if column in machine_group.columns:

                    value = recommendation_numeric(
                        machine_group,
                        column
                    )

                    if value <= 1:

                        value *= 100

                    driver_values[label] = value

            if driver_values:

                dominant_driver = max(
                    driver_values,
                    key=driver_values.get
                )

                dominant_score = driver_values[
                    dominant_driver
                ]

            else:

                dominant_driver = "Not Available"
                dominant_score = 0.0

            # ------------------------------------------------
            # MACHINE HEALTH
            # ------------------------------------------------

            machine_health = (
                100 - avg_risk
            )

            machine_health = max(
                0,
                min(
                    100,
                    machine_health
                )
            )

            machine_action_rows.append(
                {
                    "Machine_ID":
                        machine_id,
                    "Average Risk":
                        round(
                            avg_risk,
                            2
                        ),
                    "Maximum Risk":
                        round(
                            max_risk,
                            2
                        ),
                    "Machine Health":
                        round(
                            machine_health,
                            2
                        ),
                    "Anomalies":
                        anomaly_count,
                    "Priority":
                        priority,
                    "Dominant Risk Driver":
                        dominant_driver,
                    "Driver Score":
                        round(
                            dominant_score,
                            2
                        ),
                    "Recommended Action":
                        action
                }
            )

        action_queue_df = pd.DataFrame(
            machine_action_rows
        )

        if not action_queue_df.empty:

            # ------------------------------------------------
            # PRIORITY ORDER
            # ------------------------------------------------

            priority_order = {
                "P1 - Emergency": 1,
                "P2 - High": 2,
                "P3 - Planned": 3,
                "P4 - Routine": 4
            }

            action_queue_df[
                "_priority_order"
            ] = (
                action_queue_df[
                    "Priority"
                ]
                .map(priority_order)
                .fillna(99)
            )

            action_queue_df = (
                action_queue_df
                .sort_values(
                    by=[
                        "_priority_order",
                        "Average Risk"
                    ],
                    ascending=[
                        True,
                        False
                    ]
                )
                .drop(
                    columns=[
                        "_priority_order"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            # ------------------------------------------------
            # TOP ACTION QUEUE
            # ------------------------------------------------

            st.dataframe(
                action_queue_df.head(20),
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # PRIORITY DISTRIBUTION
            # ------------------------------------------------

            st.markdown(
                "#### Machine Maintenance Priority Distribution"
            )

            machine_priority_distribution = (
                action_queue_df[
                    "Priority"
                ]
                .value_counts()
                .reindex(
                    [
                        "P1 - Emergency",
                        "P2 - High",
                        "P3 - Planned",
                        "P4 - Routine"
                    ],
                    fill_value=0
                )
            )

            st.bar_chart(
                machine_priority_distribution
            )

            # ------------------------------------------------
            # TOP MACHINES BY RISK
            # ------------------------------------------------

            st.markdown(
                "#### Highest-Risk Machines"
            )

            top_machine_columns = [
                "Machine_ID",
                "Average Risk",
                "Maximum Risk",
                "Machine Health",
                "Anomalies",
                "Priority",
                "Dominant Risk Driver"
            ]

            available_top_columns = [
                column
                for column in top_machine_columns
                if column in action_queue_df.columns
            ]

            st.dataframe(
                action_queue_df[
                    available_top_columns
                ].head(10),
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "Machine_ID is required for "
            "maintenance action queue."
        )

else:

    st.warning(
        "Maintenance recommendation data "
        "is not available."
    )
# ============================================================
# MAINTENANCE RECOMMENDATIONS & ACTION QUEUE - PART 2
# DETAILED INTERVENTION PLANNING
# ============================================================

st.markdown("### Detailed Maintenance Intervention Planning")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    intervention_df = maintenance_df.copy()

    # --------------------------------------------------------
    # NUMERIC HELPER
    # --------------------------------------------------------

    def intervention_numeric(
        dataframe,
        column,
        default=0.0
    ):

        if column not in dataframe.columns:

            return default

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        ).dropna()

        if values.empty:

            return default

        return float(
            values.mean()
        )

    # --------------------------------------------------------
    # BUILD DETAILED INTERVENTION QUEUE
    # --------------------------------------------------------

    if "Machine_ID" in intervention_df.columns:

        intervention_rows = []

        for machine_id, machine_group in (
            intervention_df.groupby("Machine_ID")
        ):

            avg_risk = intervention_numeric(
                machine_group,
                "risk_score"
            )

            max_risk = 0.0

            if "risk_score" in machine_group.columns:

                risk_values = pd.to_numeric(
                    machine_group[
                        "risk_score"
                    ],
                    errors="coerce"
                ).dropna()

                if not risk_values.empty:

                    max_risk = float(
                        risk_values.max()
                    )

            if avg_risk <= 1:

                avg_risk *= 100

            if max_risk <= 1:

                max_risk *= 100

            # ------------------------------------------------
            # ANOMALIES
            # ------------------------------------------------

            anomaly_count = 0

            if "is_anomaly" in machine_group.columns:

                anomaly_count = int(
                    pd.to_numeric(
                        machine_group[
                            "is_anomaly"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            # ------------------------------------------------
            # RISK DRIVERS
            # ------------------------------------------------

            driver_values = {}

            driver_columns = {
                "sensor_risk":
                    "Sensor / Mechanical",
                "network_risk":
                    "Network / 6G",
                "quality_risk":
                    "Quality / Production",
                "historical_risk":
                    "Historical Trend"
            }

            for column, label in driver_columns.items():

                if column in machine_group.columns:

                    value = intervention_numeric(
                        machine_group,
                        column
                    )

                    if value <= 1:

                        value *= 100

                    driver_values[label] = value

            if driver_values:

                dominant_driver = max(
                    driver_values,
                    key=driver_values.get
                )

                dominant_score = float(
                    driver_values[
                        dominant_driver
                    ]
                )

            else:

                dominant_driver = "General Risk"
                dominant_score = avg_risk

            # ------------------------------------------------
            # PRIORITY
            # ------------------------------------------------

            if (
                max_risk >= 75
                or anomaly_count >= 10
            ):

                priority = "P1 - Emergency"
                urgency = "Immediate"
                intervention_window = (
                    "Now / Within 24 Hours"
                )

            elif (
                avg_risk >= 50
                or max_risk >= 60
                or anomaly_count >= 5
            ):

                priority = "P2 - High"
                urgency = "High"
                intervention_window = (
                    "Within 24-72 Hours"
                )

            elif avg_risk >= 30:

                priority = "P3 - Planned"
                urgency = "Moderate"
                intervention_window = (
                    "Within 7 Days"
                )

            else:

                priority = "P4 - Routine"
                urgency = "Low"
                intervention_window = (
                    "Next Scheduled Cycle"
                )

            # ------------------------------------------------
            # INTERVENTION TYPE
            # ------------------------------------------------

            if dominant_driver == "Sensor / Mechanical":

                intervention_type = (
                    "Mechanical & Sensor Inspection"
                )

                recommended_action = (
                    "Inspect vibration, temperature, "
                    "power behavior and mechanical "
                    "components. Verify sensor calibration."
                )

                inspection_scope = (
                    "Temperature, vibration, power "
                    "consumption and mechanical condition"
                )

            elif dominant_driver == "Network / 6G":

                intervention_type = (
                    "Network / 6G Reliability Inspection"
                )

                recommended_action = (
                    "Inspect network latency, packet loss "
                    "and communication reliability. "
                    "Validate connectivity and network "
                    "configuration."
                )

                inspection_scope = (
                    "Latency, packet loss, network health "
                    "and communication stability"
                )

            elif dominant_driver == "Quality / Production":

                intervention_type = (
                    "Quality & Production Inspection"
                )

                recommended_action = (
                    "Review production speed, defect rate "
                    "and error rate. Inspect process "
                    "parameters and quality controls."
                )

                inspection_scope = (
                    "Defect rate, error rate, production "
                    "speed and process quality"
                )

            else:

                intervention_type = (
                    "Trend-Based Preventive Inspection"
                )

                recommended_action = (
                    "Review historical machine behavior "
                    "and perform preventive inspection "
                    "before risk escalates."
                )

                inspection_scope = (
                    "Historical risk, operating behavior "
                    "and recent machine trends"
                )

            # ------------------------------------------------
            # ANOMALY RESPONSE
            # ------------------------------------------------

            if anomaly_count > 0:

                anomaly_response = (
                    "Investigate detected anomalous "
                    "operating events before returning "
                    "machine to normal monitoring."
                )

            else:

                anomaly_response = (
                    "No detected anomaly requiring "
                    "immediate anomaly investigation."
                )

            # ------------------------------------------------
            # ESTIMATED DOWNTIME
            # ------------------------------------------------

            if priority == "P1 - Emergency":

                estimated_downtime = 8.0

            elif priority == "P2 - High":

                estimated_downtime = 5.0

            elif priority == "P3 - Planned":

                estimated_downtime = 3.0

            else:

                estimated_downtime = 1.5

            # ------------------------------------------------
            # ESTIMATED MAINTENANCE COST
            # ------------------------------------------------

            base_cost = {
                "P1 - Emergency": 15000,
                "P2 - High": 9000,
                "P3 - Planned": 5000,
                "P4 - Routine": 2500
            }

            estimated_cost = base_cost[
                priority
            ]

            # ------------------------------------------------
            # PRODUCTION IMPACT
            # ------------------------------------------------

            production_speed = intervention_numeric(
                machine_group,
                "Production_Speed_units_per_hr"
            )

            production_impact = (
                production_speed
                * estimated_downtime
            )

            # ------------------------------------------------
            # MACHINE HEALTH
            # ------------------------------------------------

            health_score = (
                100 - avg_risk
            )

            health_score = max(
                0,
                min(
                    100,
                    health_score
                )
            )

            # ------------------------------------------------
            # BUSINESS IMPACT
            # ------------------------------------------------

            if priority == "P1 - Emergency":

                business_impact = (
                    "Very High â€” immediate operational "
                    "attention recommended"
                )

            elif priority == "P2 - High":

                business_impact = (
                    "High â€” prioritize maintenance planning"
                )

            elif priority == "P3 - Planned":

                business_impact = (
                    "Moderate â€” include in planned maintenance"
                )

            else:

                business_impact = (
                    "Low â€” continue routine monitoring"
                )

            # ------------------------------------------------
            # MONITORING FREQUENCY
            # ------------------------------------------------

            if priority == "P1 - Emergency":

                monitoring_frequency = (
                    "Continuous / Real-Time"
                )

            elif priority == "P2 - High":

                monitoring_frequency = (
                    "Every 1-4 Hours"
                )

            elif priority == "P3 - Planned":

                monitoring_frequency = (
                    "Daily"
                )

            else:

                monitoring_frequency = (
                    "Weekly"
                )

            intervention_rows.append(
                {
                    "Machine_ID":
                        machine_id,
                    "Priority":
                        priority,
                    "Urgency":
                        urgency,
                    "Intervention Window":
                        intervention_window,
                    "Machine Health":
                        round(
                            health_score,
                            2
                        ),
                    "Average Risk":
                        round(
                            avg_risk,
                            2
                        ),
                    "Maximum Risk":
                        round(
                            max_risk,
                            2
                        ),
                    "Detected Anomalies":
                        anomaly_count,
                    "Dominant Risk Driver":
                        dominant_driver,
                    "Driver Score":
                        round(
                            dominant_score,
                            2
                        ),
                    "Intervention Type":
                        intervention_type,
                    "Recommended Action":
                        recommended_action,
                    "Inspection Scope":
                        inspection_scope,
                    "Anomaly Response":
                        anomaly_response,
                    "Estimated Downtime (Hours)":
                        estimated_downtime,
                    "Estimated Cost (INR)":
                        estimated_cost,
                    "Estimated Production Impact":
                        round(
                            production_impact,
                            2
                        ),
                    "Business Impact":
                        business_impact,
                    "Monitoring Frequency":
                        monitoring_frequency
                }
            )

        detailed_intervention_df = pd.DataFrame(
            intervention_rows
        )

        if not detailed_intervention_df.empty:

            # ------------------------------------------------
            # PRIORITY FILTER
            # ------------------------------------------------

            selected_intervention_priority = st.selectbox(
                "Filter Maintenance Recommendations by Priority",
                [
                    "All",
                    "P1 - Emergency",
                    "P2 - High",
                    "P3 - Planned",
                    "P4 - Routine"
                ],
                key="maintenance_intervention_priority"
            )

            filtered_intervention_df = (
                detailed_intervention_df.copy()
            )

            if (
                selected_intervention_priority
                != "All"
            ):

                filtered_intervention_df = (
                    filtered_intervention_df[
                        filtered_intervention_df[
                            "Priority"
                        ]
                        == selected_intervention_priority
                    ]
                )

            # ------------------------------------------------
            # DETAILED TABLE
            # ------------------------------------------------

            st.markdown(
                "#### Recommended Maintenance Actions"
            )

            display_intervention_columns = [
                "Machine_ID",
                "Priority",
                "Urgency",
                "Intervention Window",
                "Machine Health",
                "Average Risk",
                "Maximum Risk",
                "Detected Anomalies",
                "Dominant Risk Driver",
                "Intervention Type",
                "Recommended Action",
                "Estimated Downtime (Hours)",
                "Estimated Cost (INR)",
                "Business Impact",
                "Monitoring Frequency"
            ]

            available_intervention_columns = [
                column
                for column
                in display_intervention_columns
                if column
                in filtered_intervention_df.columns
            ]

            st.dataframe(
                filtered_intervention_df[
                    available_intervention_columns
                ],
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # SELECT MACHINE
            # ------------------------------------------------

            st.markdown(
                "#### Maintenance Action Detail"
            )

            intervention_machine_list = sorted(
                detailed_intervention_df[
                    "Machine_ID"
                ]
                .dropna()
                .unique()
                .tolist(),
                key=lambda value: str(value)
            )

            if intervention_machine_list:

                selected_intervention_machine = st.selectbox(
                    "Select Machine",
                    intervention_machine_list,
                    key="maintenance_action_machine"
                )

                selected_action = (
                    detailed_intervention_df[
                        detailed_intervention_df[
                            "Machine_ID"
                        ]
                        == selected_intervention_machine
                    ]
                )

                if not selected_action.empty:

                    action_row = (
                        selected_action
                        .iloc[0]
                    )

                    detail_col1, detail_col2 = (
                        st.columns(2)
                    )

                    with detail_col1:

                        st.metric(
                            "Priority",
                            str(
                                action_row[
                                    "Priority"
                                ]
                            )
                        )

                        st.metric(
                            "Average Risk",
                            f"{float(action_row['Average Risk']):.2f}/100"
                        )

                        st.metric(
                            "Machine Health",
                            f"{float(action_row['Machine Health']):.2f}/100"
                        )

                        st.metric(
                            "Estimated Downtime",
                            f"{float(action_row['Estimated Downtime (Hours)']):.1f} hrs"
                        )

                    with detail_col2:

                        st.metric(
                            "Dominant Driver",
                            str(
                                action_row[
                                    "Dominant Risk Driver"
                                ]
                            )
                        )

                        st.metric(
                            "Detected Anomalies",
                            f"{int(action_row['Detected Anomalies']):,}"
                        )

                        st.metric(
                            "Estimated Cost",
                            f"â‚¹{float(action_row['Estimated Cost (INR)']):,.0f}"
                        )

                        st.metric(
                            "Monitoring",
                            str(
                                action_row[
                                    "Monitoring Frequency"
                                ]
                            )
                        )

                    st.markdown(
                        "##### Recommended Action"
                    )

                    st.info(
                        str(
                            action_row[
                                "Recommended Action"
                            ]
                        )
                    )

                    st.markdown(
                        "##### Inspection Scope"
                    )

                    st.write(
                        str(
                            action_row[
                                "Inspection Scope"
                            ]
                        )
                    )

                    st.markdown(
                        "##### Anomaly Response"
                    )

                    st.write(
                        str(
                            action_row[
                                "Anomaly Response"
                            ]
                        )
                    )

                    st.markdown(
                        "##### Business Impact"
                    )

                    st.write(
                        str(
                            action_row[
                                "Business Impact"
                            ]
                        )
                    )

                    st.caption(
                        "Maintenance costs and downtime "
                        "values are planning estimates for "
                        "dashboard prioritization, not actual "
                        "vendor quotations or confirmed outage costs."
                    )

    else:

        st.info(
            "Machine_ID is required for "
            "maintenance intervention planning."
        )

else:

    st.warning(
        "Maintenance recommendation data "
        "is not available."
    )
# ============================================================
# FLEET COMPARISON & MACHINE LEADERBOARD - PART 1
# ============================================================

st.markdown("### Fleet Comparison & Machine Leaderboard")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    fleet_comparison_df = maintenance_df.copy()

    # --------------------------------------------------------
    # HELPER
    # --------------------------------------------------------

    def fleet_numeric_mean(
        dataframe,
        column,
        default=0.0
    ):

        if column not in dataframe.columns:

            return default

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        ).dropna()

        if values.empty:

            return default

        return float(
            values.mean()
        )

    # --------------------------------------------------------
    # MACHINE GROUPING
    # --------------------------------------------------------

    if "Machine_ID" in fleet_comparison_df.columns:

        fleet_rows = []

        for machine_id, machine_group in (
            fleet_comparison_df.groupby(
                "Machine_ID"
            )
        ):

            average_risk = fleet_numeric_mean(
                machine_group,
                "risk_score"
            )

            maximum_risk = 0.0

            if "risk_score" in machine_group.columns:

                risk_values = pd.to_numeric(
                    machine_group[
                        "risk_score"
                    ],
                    errors="coerce"
                ).dropna()

                if not risk_values.empty:

                    maximum_risk = float(
                        risk_values.max()
                    )

            if average_risk <= 1:

                average_risk *= 100

            if maximum_risk <= 1:

                maximum_risk *= 100

            # ------------------------------------------------
            # HEALTH
            # ------------------------------------------------

            if "health_score" in machine_group.columns:

                average_health = fleet_numeric_mean(
                    machine_group,
                    "health_score"
                )

                if average_health <= 1:

                    average_health *= 100

            else:

                average_health = (
                    100 - average_risk
                )

            average_health = max(
                0,
                min(
                    100,
                    average_health
                )
            )

            # ------------------------------------------------
            # NETWORK
            # ------------------------------------------------

            network_risk = fleet_numeric_mean(
                machine_group,
                "network_risk"
            )

            if network_risk <= 1:

                network_risk *= 100

            # ------------------------------------------------
            # SENSOR
            # ------------------------------------------------

            sensor_risk = fleet_numeric_mean(
                machine_group,
                "sensor_risk"
            )

            if sensor_risk <= 1:

                sensor_risk *= 100

            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            quality_risk = fleet_numeric_mean(
                machine_group,
                "quality_risk"
            )

            if quality_risk <= 1:

                quality_risk *= 100

            # ------------------------------------------------
            # ANOMALIES
            # ------------------------------------------------

            anomaly_count = 0

            if "is_anomaly" in machine_group.columns:

                anomaly_count = int(
                    pd.to_numeric(
                        machine_group[
                            "is_anomaly"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            # ------------------------------------------------
            # PRODUCTION
            # ------------------------------------------------

            production_speed = fleet_numeric_mean(
                machine_group,
                "Production_Speed_units_per_hr"
            )

            # ------------------------------------------------
            # DEFECT RATE
            # ------------------------------------------------

            defect_rate = fleet_numeric_mean(
                machine_group,
                "Quality_Control_Defect_Rate_%"
            )

            # ------------------------------------------------
            # ERROR RATE
            # ------------------------------------------------

            error_rate = fleet_numeric_mean(
                machine_group,
                "Error_Rate_%"
            )

            # ------------------------------------------------
            # NETWORK LATENCY
            # ------------------------------------------------

            network_latency = fleet_numeric_mean(
                machine_group,
                "Network_Latency_ms"
            )

            # ------------------------------------------------
            # PACKET LOSS
            # ------------------------------------------------

            packet_loss = fleet_numeric_mean(
                machine_group,
                "Packet_Loss_%"
            )

            # ------------------------------------------------
            # EFFICIENCY
            # ------------------------------------------------

            efficiency_status = "Unknown"

            if "Efficiency_Status" in machine_group.columns:

                mode_values = (
                    machine_group[
                        "Efficiency_Status"
                    ]
                    .astype(str)
                    .str.strip()
                    .mode()
                )

                if not mode_values.empty:

                    efficiency_status = str(
                        mode_values.iloc[0]
                    )

            # ------------------------------------------------
            # MACHINE STATUS
            # ------------------------------------------------

            if average_risk >= 75:

                machine_status = "Critical"

            elif average_risk >= 50:

                machine_status = "High Risk"

            elif average_risk >= 30:

                machine_status = "Medium Risk"

            else:

                machine_status = "Healthy"

            # ------------------------------------------------
            # PERFORMANCE SCORE
            # ------------------------------------------------

            performance_score = (
                (
                    average_health
                    + (100 - network_risk)
                    + (100 - sensor_risk)
                    + (100 - quality_risk)
                )
                / 4
            )

            performance_score = max(
                0,
                min(
                    100,
                    performance_score
                )
            )

            fleet_rows.append(
                {
                    "Machine_ID":
                        machine_id,
                    "Average Risk":
                        round(
                            average_risk,
                            2
                        ),
                    "Maximum Risk":
                        round(
                            maximum_risk,
                            2
                        ),
                    "Machine Health":
                        round(
                            average_health,
                            2
                        ),
                    "Performance Score":
                        round(
                            performance_score,
                            2
                        ),
                    "Sensor Risk":
                        round(
                            sensor_risk,
                            2
                        ),
                    "Network Risk":
                        round(
                            network_risk,
                            2
                        ),
                    "Quality Risk":
                        round(
                            quality_risk,
                            2
                        ),
                    "Anomalies":
                        anomaly_count,
                    "Production Speed":
                        round(
                            production_speed,
                            2
                        ),
                    "Defect Rate":
                        round(
                            defect_rate,
                            2
                        ),
                    "Error Rate":
                        round(
                            error_rate,
                            2
                        ),
                    "Network Latency":
                        round(
                            network_latency,
                            2
                        ),
                    "Packet Loss":
                        round(
                            packet_loss,
                            2
                        ),
                    "Efficiency Status":
                        efficiency_status,
                    "Machine Status":
                        machine_status
                }
            )

        fleet_machine_df = pd.DataFrame(
            fleet_rows
        )

        if not fleet_machine_df.empty:

            # ------------------------------------------------
            # MACHINE RANKINGS
            # ------------------------------------------------

            fleet_machine_df[
                "Risk Rank"
            ] = (
                fleet_machine_df[
                    "Average Risk"
                ]
                .rank(
                    ascending=False,
                    method="min"
                )
                .astype(int)
            )

            fleet_machine_df[
                "Health Rank"
            ] = (
                fleet_machine_df[
                    "Machine Health"
                ]
                .rank(
                    ascending=False,
                    method="min"
                )
                .astype(int)
            )

            fleet_machine_df[
                "Performance Rank"
            ] = (
                fleet_machine_df[
                    "Performance Score"
                ]
                .rank(
                    ascending=False,
                    method="min"
                )
                .astype(int)
            )

            # ------------------------------------------------
            # FLEET KPIs
            # ------------------------------------------------

            total_machines = len(
                fleet_machine_df
            )

            fleet_avg_health = float(
                fleet_machine_df[
                    "Machine Health"
                ].mean()
            )

            fleet_avg_risk = float(
                fleet_machine_df[
                    "Average Risk"
                ].mean()
            )

            best_machine = fleet_machine_df.loc[
                fleet_machine_df[
                    "Performance Score"
                ].idxmax()
            ]

            worst_machine = fleet_machine_df.loc[
                fleet_machine_df[
                    "Average Risk"
                ].idxmax()
            ]

            fleet_col1, fleet_col2, fleet_col3, fleet_col4 = (
                st.columns(4)
            )

            with fleet_col1:

                st.metric(
                    "Machines Analyzed",
                    f"{total_machines:,}"
                )

            with fleet_col2:

                st.metric(
                    "Fleet Avg Health",
                    f"{fleet_avg_health:.2f}/100"
                )

            with fleet_col3:

                st.metric(
                    "Fleet Avg Risk",
                    f"{fleet_avg_risk:.2f}/100"
                )

            with fleet_col4:

                st.metric(
                    "Highest-Risk Machine",
                    str(
                        worst_machine[
                            "Machine_ID"
                        ]
                    )
                )

            # ------------------------------------------------
            # BEST / WORST MACHINES
            # ------------------------------------------------

            best_col, worst_col = st.columns(2)

            with best_col:

                st.markdown(
                    "#### ðŸŸ¢ Best Performing Machine"
                )

                st.write(
                    f"Machine **{best_machine['Machine_ID']}**"
                )

                st.metric(
                    "Performance Score",
                    f"{float(best_machine['Performance Score']):.2f}/100"
                )

                st.write(
                    f"Health: "
                    f"{float(best_machine['Machine Health']):.2f}/100"
                )

            with worst_col:

                st.markdown(
                    "#### ðŸ”´ Highest-Risk Machine"
                )

                st.write(
                    f"Machine **{worst_machine['Machine_ID']}**"
                )

                st.metric(
                    "Average Risk",
                    f"{float(worst_machine['Average Risk']):.2f}/100"
                )

                st.write(
                    f"Health: "
                    f"{float(worst_machine['Machine Health']):.2f}/100"
                )

            # ------------------------------------------------
            # RISK LEADERBOARD
            # ------------------------------------------------

            st.markdown(
                "#### Highest-Risk Machine Leaderboard"
            )

            risk_leaderboard = (
                fleet_machine_df[
                    [
                        "Risk Rank",
                        "Machine_ID",
                        "Average Risk",
                        "Maximum Risk",
                        "Machine Health",
                        "Anomalies",
                        "Machine Status"
                    ]
                ]
                .sort_values(
                    "Average Risk",
                    ascending=False
                )
                .head(15)
            )

            st.dataframe(
                risk_leaderboard,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # HEALTH LEADERBOARD
            # ------------------------------------------------

            st.markdown(
                "#### Health Leaderboard"
            )

            health_leaderboard = (
                fleet_machine_df[
                    [
                        "Health Rank",
                        "Machine_ID",
                        "Machine Health",
                        "Average Risk",
                        "Performance Score",
                        "Machine Status"
                    ]
                ]
                .sort_values(
                    "Machine Health",
                    ascending=False
                )
                .head(15)
            )

            st.dataframe(
                health_leaderboard,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # PERFORMANCE LEADERBOARD
            # ------------------------------------------------

            st.markdown(
                "#### Overall Performance Leaderboard"
            )

            performance_leaderboard = (
                fleet_machine_df[
                    [
                        "Performance Rank",
                        "Machine_ID",
                        "Performance Score",
                        "Machine Health",
                        "Average Risk",
                        "Production Speed",
                        "Efficiency Status"
                    ]
                ]
                .sort_values(
                    "Performance Score",
                    ascending=False
                )
                .head(15)
            )

            st.dataframe(
                performance_leaderboard,
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "Machine_ID is required for "
            "fleet comparison."
        )

else:

    st.warning(
        "Fleet comparison data is not available."
    )
# ============================================================
# FLEET COMPARISON & MACHINE LEADERBOARD - PART 2
# INTERACTIVE MACHINE COMPARISON & FLEET RISK MATRIX
# ============================================================

st.markdown("### Interactive Fleet Comparison")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    comparison_df = maintenance_df.copy()

    # --------------------------------------------------------
    # MACHINE LEVEL AGGREGATION
    # --------------------------------------------------------

    if "Machine_ID" in comparison_df.columns:

        comparison_rows = []

        for machine_id, machine_group in (
            comparison_df.groupby("Machine_ID")
        ):

            def comparison_mean(
                column,
                default=0.0
            ):

                if column not in machine_group.columns:

                    return default

                values = pd.to_numeric(
                    machine_group[column],
                    errors="coerce"
                ).dropna()

                if values.empty:

                    return default

                return float(
                    values.mean()
                )

            # ------------------------------------------------
            # CORE METRICS
            # ------------------------------------------------

            avg_risk = comparison_mean(
                "risk_score"
            )

            max_risk = 0.0

            if "risk_score" in machine_group.columns:

                risk_values = pd.to_numeric(
                    machine_group["risk_score"],
                    errors="coerce"
                ).dropna()

                if not risk_values.empty:

                    max_risk = float(
                        risk_values.max()
                    )

            if avg_risk <= 1:

                avg_risk *= 100

            if max_risk <= 1:

                max_risk *= 100

            if "health_score" in machine_group.columns:

                avg_health = comparison_mean(
                    "health_score"
                )

                if avg_health <= 1:

                    avg_health *= 100

            else:

                avg_health = (
                    100 - avg_risk
                )

            avg_health = max(
                0,
                min(
                    100,
                    avg_health
                )
            )

            # ------------------------------------------------
            # RISK DRIVERS
            # ------------------------------------------------

            sensor_risk = comparison_mean(
                "sensor_risk"
            )

            network_risk = comparison_mean(
                "network_risk"
            )

            quality_risk = comparison_mean(
                "quality_risk"
            )

            historical_risk = comparison_mean(
                "historical_risk"
            )

            for driver_name in [
                "sensor_risk",
                "network_risk",
                "quality_risk",
                "historical_risk"
            ]:

                driver_value = comparison_mean(
                    driver_name
                )

                if driver_value <= 1:

                    driver_value *= 100

                if driver_name == "sensor_risk":

                    sensor_risk = driver_value

                elif driver_name == "network_risk":

                    network_risk = driver_value

                elif driver_name == "quality_risk":

                    quality_risk = driver_value

                elif driver_name == "historical_risk":

                    historical_risk = driver_value

            # ------------------------------------------------
            # ANOMALY COUNT
            # ------------------------------------------------

            anomaly_count = 0

            if "is_anomaly" in machine_group.columns:

                anomaly_count = int(
                    pd.to_numeric(
                        machine_group[
                            "is_anomaly"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            # ------------------------------------------------
            # PRODUCTION
            # ------------------------------------------------

            production_speed = comparison_mean(
                "Production_Speed_units_per_hr"
            )

            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            defect_rate = comparison_mean(
                "Quality_Control_Defect_Rate_%"
            )

            error_rate = comparison_mean(
                "Error_Rate_%"
            )

            # ------------------------------------------------
            # NETWORK
            # ------------------------------------------------

            latency = comparison_mean(
                "Network_Latency_ms"
            )

            packet_loss = comparison_mean(
                "Packet_Loss_%"
            )

            # ------------------------------------------------
            # OVERALL PERFORMANCE
            # ------------------------------------------------

            performance_score = (
                avg_health
                + (100 - sensor_risk)
                + (100 - network_risk)
                + (100 - quality_risk)
            ) / 4

            performance_score = max(
                0,
                min(
                    100,
                    performance_score
                )
            )

            # ------------------------------------------------
            # MACHINE CLASS
            # ------------------------------------------------

            if avg_risk >= 75:

                machine_class = "Critical"

            elif avg_risk >= 50:

                machine_class = "High Risk"

            elif avg_risk >= 30:

                machine_class = "Watch"

            else:

                machine_class = "Healthy"

            comparison_rows.append(
                {
                    "Machine_ID":
                        machine_id,
                    "Average Risk":
                        round(
                            avg_risk,
                            2
                        ),
                    "Maximum Risk":
                        round(
                            max_risk,
                            2
                        ),
                    "Machine Health":
                        round(
                            avg_health,
                            2
                        ),
                    "Performance Score":
                        round(
                            performance_score,
                            2
                        ),
                    "Sensor Risk":
                        round(
                            sensor_risk,
                            2
                        ),
                    "Network Risk":
                        round(
                            network_risk,
                            2
                        ),
                    "Quality Risk":
                        round(
                            quality_risk,
                            2
                        ),
                    "Historical Risk":
                        round(
                            historical_risk,
                            2
                        ),
                    "Anomalies":
                        anomaly_count,
                    "Production Speed":
                        round(
                            production_speed,
                            2
                        ),
                    "Defect Rate":
                        round(
                            defect_rate,
                            2
                        ),
                    "Error Rate":
                        round(
                            error_rate,
                            2
                        ),
                    "Network Latency":
                        round(
                            latency,
                            2
                        ),
                    "Packet Loss":
                        round(
                            packet_loss,
                            2
                        ),
                    "Machine Class":
                        machine_class
                }
            )

        comparison_machine_df = pd.DataFrame(
            comparison_rows
        )

        if not comparison_machine_df.empty:

            # =================================================
            # MACHINE COMPARISON SELECTOR
            # =================================================

            st.markdown(
                "#### Compare Machines"
            )

            available_machine_ids = sorted(
                comparison_machine_df[
                    "Machine_ID"
                ]
                .dropna()
                .unique()
                .tolist(),
                key=lambda value: str(value)
            )

            comparison_selection = st.multiselect(
                "Select 2â€“5 Machines",
                available_machine_ids,
                default=available_machine_ids[
                    :min(
                        3,
                        len(
                            available_machine_ids
                        )
                    )
                ],
                max_selections=5,
                key="fleet_machine_comparison"
            )

            if comparison_selection:

                selected_comparison = (
                    comparison_machine_df[
                        comparison_machine_df[
                            "Machine_ID"
                        ].isin(
                            comparison_selection
                        )
                    ]
                    .copy()
                )

                # ---------------------------------------------
                # COMPARISON TABLE
                # ---------------------------------------------

                comparison_columns = [
                    "Machine_ID",
                    "Average Risk",
                    "Maximum Risk",
                    "Machine Health",
                    "Performance Score",
                    "Sensor Risk",
                    "Network Risk",
                    "Quality Risk",
                    "Anomalies",
                    "Production Speed",
                    "Defect Rate",
                    "Error Rate"
                ]

                available_comparison_columns = [
                    column
                    for column
                    in comparison_columns
                    if column
                    in selected_comparison.columns
                ]

                st.dataframe(
                    selected_comparison[
                        available_comparison_columns
                    ],
                    width="stretch",
                    hide_index=True
                )

                # ---------------------------------------------
                # RISK COMPARISON
                # ---------------------------------------------

                comparison_risk_chart = (
                    selected_comparison[
                        [
                            "Machine_ID",
                            "Average Risk"
                        ]
                    ]
                    .set_index(
                        "Machine_ID"
                    )
                )

                st.markdown(
                    "#### Machine Risk Comparison"
                )

                st.bar_chart(
                    comparison_risk_chart
                )

                # ---------------------------------------------
                # HEALTH COMPARISON
                # ---------------------------------------------

                comparison_health_chart = (
                    selected_comparison[
                        [
                            "Machine_ID",
                            "Machine Health"
                        ]
                    ]
                    .set_index(
                        "Machine_ID"
                    )
                )

                st.markdown(
                    "#### Machine Health Comparison"
                )

                st.bar_chart(
                    comparison_health_chart
                )

                # ---------------------------------------------
                # PERFORMANCE COMPARISON
                # ---------------------------------------------

                comparison_performance_chart = (
                    selected_comparison[
                        [
                            "Machine_ID",
                            "Performance Score"
                        ]
                    ]
                    .set_index(
                        "Machine_ID"
                    )
                )

                st.markdown(
                    "#### Performance Comparison"
                )

                st.bar_chart(
                    comparison_performance_chart
                )

            else:

                st.info(
                    "Select machines to compare."
                )

            # =================================================
            # FLEET RISK VS HEALTH MATRIX
            # =================================================

            st.markdown(
                "#### Fleet Risk vs Health Matrix"
            )

            matrix_columns = [
                "Machine_ID",
                "Average Risk",
                "Machine Health"
            ]

            matrix_df = comparison_machine_df[
                matrix_columns
            ].copy()

            matrix_df = matrix_df.set_index(
                "Machine_ID"
            )

            st.scatter_chart(
                matrix_df,
                x="Average Risk",
                y="Machine Health"
            )

            # =================================================
            # RISK DRIVER COMPARISON
            # =================================================

            st.markdown(
                "#### Fleet Risk Driver Comparison"
            )

            driver_matrix = comparison_machine_df[
                [
                    "Machine_ID",
                    "Sensor Risk",
                    "Network Risk",
                    "Quality Risk",
                    "Historical Risk"
                ]
            ].copy()

            driver_matrix = driver_matrix.set_index(
                "Machine_ID"
            )

            st.bar_chart(
                driver_matrix
            )

            # =================================================
            # MACHINE CLASS DISTRIBUTION
            # =================================================

            st.markdown(
                "#### Fleet Machine Classification"
            )

            machine_class_distribution = (
                comparison_machine_df[
                    "Machine Class"
                ]
                .value_counts()
                .reindex(
                    [
                        "Healthy",
                        "Watch",
                        "High Risk",
                        "Critical"
                    ],
                    fill_value=0
                )
            )

            st.bar_chart(
                machine_class_distribution
            )

            # =================================================
            # FLEET PERFORMANCE TABLE
            # =================================================

            st.markdown(
                "#### Complete Fleet Performance Profile"
            )

            fleet_profile_columns = [
                "Machine_ID",
                "Machine Class",
                "Machine Health",
                "Average Risk",
                "Performance Score",
                "Anomalies",
                "Production Speed",
                "Defect Rate",
                "Error Rate",
                "Network Latency",
                "Packet Loss"
            ]

            available_fleet_profile_columns = [
                column
                for column
                in fleet_profile_columns
                if column
                in comparison_machine_df.columns
            ]

            st.dataframe(
                comparison_machine_df[
                    available_fleet_profile_columns
                ]
                .sort_values(
                    "Performance Score",
                    ascending=False
                ),
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "Machine_ID is required for "
            "interactive fleet comparison."
        )

else:

    st.warning(
        "Fleet comparison data is not available."
    )
# ============================================================
# GLOBAL FILTERS & DASHBOARD CONTROLS - PART 1
# ============================================================

st.markdown("### Global Dashboard Filters")

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    filter_df = maintenance_df.copy()

    # --------------------------------------------------------
    # DATE PREPARATION
    # --------------------------------------------------------

    if "Date" in filter_df.columns:

        filter_df["Date"] = pd.to_datetime(
            filter_df["Date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # FILTER CONTAINER
    # --------------------------------------------------------

    with st.expander(
        "Open Global Filters",
        expanded=False
    ):

        # ----------------------------------------------------
        # DATE FILTER
        # ----------------------------------------------------

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            if "Date" in filter_df.columns:

                valid_dates = (
                    filter_df[
                        "Date"
                    ]
                    .dropna()
                )

                if not valid_dates.empty:

                    min_date = (
                        valid_dates.min()
                        .date()
                    )

                    max_date = (
                        valid_dates.max()
                        .date()
                    )

                    selected_date_range = st.date_input(
                        "Date Range",
                        value=(
                            min_date,
                            max_date
                        ),
                        min_value=min_date,
                        max_value=max_date,
                        key="global_date_range"
                    )

                else:

                    selected_date_range = None

            else:

                selected_date_range = None

        # ----------------------------------------------------
        # MACHINE FILTER
        # ----------------------------------------------------

        with filter_col2:

            if "Machine_ID" in filter_df.columns:

                filter_machine_options = sorted(
                    filter_df[
                        "Machine_ID"
                    ]
                    .dropna()
                    .unique()
                    .tolist(),
                    key=lambda value: str(value)
                )

                selected_filter_machines = st.multiselect(
                    "Machine",
                    filter_machine_options,
                    default=[],
                    key="global_machine_filter"
                )

            else:

                selected_filter_machines = []

        # ----------------------------------------------------
        # RISK FILTER
        # ----------------------------------------------------

        filter_col3, filter_col4 = st.columns(2)

        with filter_col3:

            if "risk_score" in filter_df.columns:

                risk_filter_options = [
                    "All",
                    "Low",
                    "Medium",
                    "High",
                    "Critical"
                ]

                selected_risk_level = st.selectbox(
                    "Risk Level",
                    risk_filter_options,
                    key="global_risk_filter"
                )

            else:

                selected_risk_level = "All"

        # ----------------------------------------------------
        # ANOMALY FILTER
        # ----------------------------------------------------

        with filter_col4:

            if "is_anomaly" in filter_df.columns:

                anomaly_filter = st.selectbox(
                    "Anomaly Status",
                    [
                        "All",
                        "Normal",
                        "Anomaly"
                    ],
                    key="global_anomaly_filter"
                )

            else:

                anomaly_filter = "All"

        # ----------------------------------------------------
        # OPERATION MODE
        # ----------------------------------------------------

        filter_col5, filter_col6 = st.columns(2)

        with filter_col5:

            if "Operation_Mode" in filter_df.columns:

                operation_filter_options = [
                    "All"
                ] + sorted(
                    filter_df[
                        "Operation_Mode"
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                selected_operation_mode = st.selectbox(
                    "Operation Mode",
                    operation_filter_options,
                    key="global_operation_filter"
                )

            else:

                selected_operation_mode = "All"

        # ----------------------------------------------------
        # EFFICIENCY STATUS
        # ----------------------------------------------------

        with filter_col6:

            if "Efficiency_Status" in filter_df.columns:

                efficiency_filter_options = [
                    "All"
                ] + sorted(
                    filter_df[
                        "Efficiency_Status"
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                selected_efficiency_status = st.selectbox(
                    "Efficiency Status",
                    efficiency_filter_options,
                    key="global_efficiency_filter"
                )

            else:

                selected_efficiency_status = "All"

        # ----------------------------------------------------
        # ANOMALY SEVERITY
        # ----------------------------------------------------

        if "anomaly_severity" in filter_df.columns:

            anomaly_severity_options = [
                "All"
            ] + sorted(
                filter_df[
                    "anomaly_severity"
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            selected_anomaly_severity = st.selectbox(
                "Anomaly Severity",
                anomaly_severity_options,
                key="global_anomaly_severity"
            )

        else:

            selected_anomaly_severity = "All"

        # ----------------------------------------------------
        # APPLY FILTERS
        # ----------------------------------------------------

        filtered_dashboard_df = filter_df.copy()

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if (
            selected_date_range
            and len(selected_date_range) == 2
            and "Date" in filtered_dashboard_df.columns
        ):

            start_date = pd.Timestamp(
                selected_date_range[0]
            )

            end_date = (
                pd.Timestamp(
                    selected_date_range[1]
                )
                + pd.Timedelta(
                    days=1
                )
                - pd.Timedelta(
                    seconds=1
                )
            )

            filtered_dashboard_df = (
                filtered_dashboard_df[
                    (
                        filtered_dashboard_df[
                            "Date"
                        ] >= start_date
                    )
                    &
                    (
                        filtered_dashboard_df[
                            "Date"
                        ] <= end_date
                    )
                ]
            )

        # ----------------------------------------------------
        # MACHINE
        # ----------------------------------------------------

        if (
            selected_filter_machines
            and "Machine_ID"
            in filtered_dashboard_df.columns
        ):

            filtered_dashboard_df = (
                filtered_dashboard_df[
                    filtered_dashboard_df[
                        "Machine_ID"
                    ].isin(
                        selected_filter_machines
                    )
                ]
            )

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if (
            selected_risk_level != "All"
            and "risk_score"
            in filtered_dashboard_df.columns
        ):

            risk_values = pd.to_numeric(
                filtered_dashboard_df[
                    "risk_score"
                ],
                errors="coerce"
            )

            if (
                not risk_values.dropna().empty
                and risk_values.dropna().mean() <= 1
            ):

                risk_values = (
                    risk_values * 100
                )

            if selected_risk_level == "Low":

                risk_mask = (
                    risk_values < 30
                )

            elif selected_risk_level == "Medium":

                risk_mask = (
                    (risk_values >= 30)
                    &
                    (risk_values < 50)
                )

            elif selected_risk_level == "High":

                risk_mask = (
                    (risk_values >= 50)
                    &
                    (risk_values < 75)
                )

            else:

                risk_mask = (
                    risk_values >= 75
                )

            filtered_dashboard_df = (
                filtered_dashboard_df[
                    risk_mask.fillna(False)
                ]
            )

        # ----------------------------------------------------
        # ANOMALY
        # ----------------------------------------------------

        if (
            anomaly_filter != "All"
            and "is_anomaly"
            in filtered_dashboard_df.columns
        ):

            anomaly_values = pd.to_numeric(
                filtered_dashboard_df[
                    "is_anomaly"
                ],
                errors="coerce"
            ).fillna(0)

            if anomaly_filter == "Anomaly":

                filtered_dashboard_df = (
                    filtered_dashboard_df[
                        anomaly_values == 1
                    ]
                )

            else:

                filtered_dashboard_df = (
                    filtered_dashboard_df[
                        anomaly_values == 0
                    ]
                )

        # ----------------------------------------------------
        # OPERATION MODE
        # ----------------------------------------------------

        if (
            selected_operation_mode != "All"
            and "Operation_Mode"
            in filtered_dashboard_df.columns
        ):

            filtered_dashboard_df = (
                filtered_dashboard_df[
                    filtered_dashboard_df[
                        "Operation_Mode"
                    ]
                    .astype(str)
                    .str.strip()
                    == selected_operation_mode
                ]
            )

        # ----------------------------------------------------
        # EFFICIENCY
        # ----------------------------------------------------

        if (
            selected_efficiency_status != "All"
            and "Efficiency_Status"
            in filtered_dashboard_df.columns
        ):

            filtered_dashboard_df = (
                filtered_dashboard_df[
                    filtered_dashboard_df[
                        "Efficiency_Status"
                    ]
                    .astype(str)
                    .str.strip()
                    == selected_efficiency_status
                ]
            )

        # ----------------------------------------------------
        # ANOMALY SEVERITY
        # ----------------------------------------------------

        if (
            selected_anomaly_severity != "All"
            and "anomaly_severity"
            in filtered_dashboard_df.columns
        ):

            filtered_dashboard_df = (
                filtered_dashboard_df[
                    filtered_dashboard_df[
                        "anomaly_severity"
                    ]
                    .astype(str)
                    .str.strip()
                    == selected_anomaly_severity
                ]
            )

        # ----------------------------------------------------
        # FILTER SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "#### Active Filter Summary"
        )

        summary_col1, summary_col2, summary_col3 = (
            st.columns(3)
        )

        with summary_col1:

            st.metric(
                "Filtered Records",
                f"{len(filtered_dashboard_df):,}"
            )

        with summary_col2:

            if "Machine_ID" in filtered_dashboard_df.columns:

                filtered_machine_count = (
                    filtered_dashboard_df[
                        "Machine_ID"
                    ]
                    .nunique()
                )

            else:

                filtered_machine_count = 0

            st.metric(
                "Machines",
                f"{filtered_machine_count:,}"
            )

        with summary_col3:

            if "is_anomaly" in filtered_dashboard_df.columns:

                filtered_anomalies = int(
                    pd.to_numeric(
                        filtered_dashboard_df[
                            "is_anomaly"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            else:

                filtered_anomalies = 0

            st.metric(
                "Detected Anomalies",
                f"{filtered_anomalies:,}"
            )

        # ----------------------------------------------------
        # FILTERED DATA PREVIEW
        # ----------------------------------------------------

        st.markdown(
            "#### Filtered Dataset Preview"
        )

        if not filtered_dashboard_df.empty:

            preview_columns = [
                "Date",
                "Machine_ID",
                "Operation_Mode",
                "risk_score",
                "is_anomaly",
                "anomaly_severity",
                "anomaly_domain",
                "maintenance_priority"
            ]

            available_preview_columns = [
                column
                for column in preview_columns
                if column
                in filtered_dashboard_df.columns
            ]

            st.dataframe(
                filtered_dashboard_df[
                    available_preview_columns
                ].head(100),
                width="stretch",
                hide_index=True
            )

        else:

            st.warning(
                "No records match the selected filters."
            )

    # --------------------------------------------------------
    # FILTER STATE INFORMATION
    # --------------------------------------------------------

    st.caption(
        "Global filters currently provide an analytical "
        "view of the selected records. Existing dashboard "
        "sections continue to use their dedicated datasets "
        "until filter propagation is enabled in the final "
        "dashboard integration stage."
    )

else:

    st.warning( "Dashboard filter data is not available")
# ============================================================
# GLOBAL FILTERS - PART 2
# FILTERED INTELLIGENCE + ALERT CENTER
# ============================================================

if (
    maintenance_df is not None
    and not maintenance_df.empty
):

    # --------------------------------------------------------
    # USE FILTERED DATASET WHEN AVAILABLE
    # --------------------------------------------------------

    try:

        active_analysis_df = filtered_dashboard_df.copy()

    except NameError:

        active_analysis_df = maintenance_df.copy()

    if active_analysis_df.empty:

        st.info(
            "No data available for the current dashboard filters."
        )

    else:

        st.markdown("---")
        st.markdown("## Filtered Intelligence Overview")

        # ----------------------------------------------------
        # NORMALIZE RISK SCORE
        # ----------------------------------------------------

        if "risk_score" in active_analysis_df.columns:

            active_analysis_df["risk_score_numeric"] = pd.to_numeric(
                active_analysis_df["risk_score"],
                errors="coerce"
            )

            valid_risk = (
                active_analysis_df[
                    "risk_score_numeric"
                ]
                .dropna()
            )

            if (
                not valid_risk.empty
                and valid_risk.mean() <= 1
            ):

                active_analysis_df[
                    "risk_score_numeric"
                ] *= 100

        else:

            active_analysis_df[
                "risk_score_numeric"
            ] = np.nan

        # ----------------------------------------------------
        # NORMALIZE HEALTH
        # ----------------------------------------------------

        health_columns = [
            "health_score",
            "machine_health_score",
            "average_health_score"
        ]

        health_column = next(
            (
                column
                for column in health_columns
                if column in active_analysis_df.columns
            ),
            None
        )

        if health_column:

            active_analysis_df[
                "health_numeric"
            ] = pd.to_numeric(
                active_analysis_df[
                    health_column
                ],
                errors="coerce"
            )

            valid_health = (
                active_analysis_df[
                    "health_numeric"
                ]
                .dropna()
            )

            if (
                not valid_health.empty
                and valid_health.mean() <= 1
            ):

                active_analysis_df[
                    "health_numeric"
                ] *= 100

        else:

            active_analysis_df[
                "health_numeric"
            ] = np.nan

        # ----------------------------------------------------
        # NORMALIZE NETWORK HEALTH
        # ----------------------------------------------------

        if "network_health" in active_analysis_df.columns:

            active_analysis_df[
                "network_health_numeric"
            ] = pd.to_numeric(
                active_analysis_df[
                    "network_health"
                ],
                errors="coerce"
            )

            valid_network_health = (
                active_analysis_df[
                    "network_health_numeric"
                ]
                .dropna()
            )

            if (
                not valid_network_health.empty
                and valid_network_health.mean() <= 1
            ):

                active_analysis_df[
                    "network_health_numeric"
                ] *= 100

        else:

            active_analysis_df[
                "network_health_numeric"
            ] = np.nan

        # ----------------------------------------------------
        # KPI CALCULATIONS
        # ----------------------------------------------------

        avg_filtered_risk = (
            active_analysis_df[
                "risk_score_numeric"
            ]
            .mean()
        )

        avg_filtered_health = (
            active_analysis_df[
                "health_numeric"
            ]
            .mean()
        )

        if "is_anomaly" in active_analysis_df.columns:

            filtered_anomaly_count = int(
                pd.to_numeric(
                    active_analysis_df[
                        "is_anomaly"
                    ],
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            )

        else:

            filtered_anomaly_count = 0

        if "Machine_ID" in active_analysis_df.columns:

            filtered_machine_count = (
                active_analysis_df[
                    "Machine_ID"
                ]
                .nunique()
            )

        else:

            filtered_machine_count = 0

        if "network_risk" in active_analysis_df.columns:

            network_risk_value = pd.to_numeric(
                active_analysis_df[
                    "network_risk"
                ],
                errors="coerce"
            ).mean()

            if (
                pd.notna(network_risk_value)
                and network_risk_value <= 1
            ):

                network_risk_value *= 100

        else:

            network_risk_value = np.nan

        # ----------------------------------------------------
        # KPI ROW
        # ----------------------------------------------------

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:

            st.metric(
                "Filtered Records",
                f"{len(active_analysis_df):,}"
            )

        with k2:

            st.metric(
                "Machines",
                f"{filtered_machine_count:,}"
            )

        with k3:

            st.metric(
                "Average Risk",
                (
                    f"{avg_filtered_risk:.2f}/100"
                    if pd.notna(avg_filtered_risk)
                    else "N/A"
                )
            )

        with k4:

            st.metric(
                "Average Health",
                (
                    f"{avg_filtered_health:.2f}/100"
                    if pd.notna(avg_filtered_health)
                    else "N/A"
                )
            )

        with k5:

            st.metric(
                "Anomalies",
                f"{filtered_anomaly_count:,}"
            )

        # ----------------------------------------------------
        # FILTERED RISK DISTRIBUTION
        # ----------------------------------------------------

        st.markdown(
            "### Filtered Risk Distribution"
        )

        if "risk_score_numeric" in active_analysis_df.columns:

            risk_distribution_filtered = pd.cut(
                active_analysis_df[
                    "risk_score_numeric"
                ],
                bins=[
                    -np.inf,
                    30,
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
            risk_distribution_filtered = risk_distribution_filtered.astype(str)

            risk_distribution_filtered = (
                risk_distribution_filtered
                .value_counts()
                .reindex(
                    [
                        "Low",
                        "Medium",
                        "High",
                        "Critical"
                    ],
                    fill_value=0
                )
                .reset_index()
            )

            risk_distribution_filtered.columns = [
                "Risk Level",
                "Records"
            ]

            st.bar_chart(
                risk_distribution_filtered.set_index(
                    "Risk Level"
                )
            )

        # ----------------------------------------------------
        # FILTERED ANOMALY DISTRIBUTION
        # ----------------------------------------------------

        if "anomaly_severity" in active_analysis_df.columns:

            st.markdown(
                "### Filtered Anomaly Severity"
            )

            severity_filtered = (
                active_analysis_df[
                    "anomaly_severity"
                ]
                .astype(str)
                .str.strip()
                .value_counts()
                .reset_index()
            )

            severity_filtered.columns = [
                "Severity",
                "Records"
            ]

            st.bar_chart(
                severity_filtered.set_index(
                    "Severity"
                )
            )

        # ----------------------------------------------------
        # FILTERED MACHINE RISK LEADERBOARD
        # ----------------------------------------------------

        if (
            "Machine_ID" in active_analysis_df.columns
            and "risk_score_numeric"
            in active_analysis_df.columns
        ):

            st.markdown(
                "### Filtered Machine Risk Leaderboard"
            )

            filtered_machine_risk = (
                active_analysis_df
                .groupby(
                    "Machine_ID",
                    as_index=False
                )
                .agg(
                    Average_Risk=(
                        "risk_score_numeric",
                        "mean"
                    ),
                    Maximum_Risk=(
                        "risk_score_numeric",
                        "max"
                    ),
                    Records=(
                        "risk_score_numeric",
                        "count"
                    )
                )
                .sort_values(
                    "Average_Risk",
                    ascending=False
                )
            )

            filtered_machine_risk[
                "Rank"
            ] = range(
                1,
                len(filtered_machine_risk) + 1
            )

            st.dataframe(
                filtered_machine_risk.head(20),
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # FILTERED NETWORK PERFORMANCE
        # ----------------------------------------------------

        if (
            "Network_Latency_ms"
            in active_analysis_df.columns
            and "Packet_Loss_%"
            in active_analysis_df.columns
        ):

            st.markdown(
                "### Filtered Network / 6G Performance"
            )

            network_filtered = (
                active_analysis_df[
                    [
                        "Network_Latency_ms",
                        "Packet_Loss_%"
                    ]
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            network_summary = pd.DataFrame(
                {
                    "Metric": [
                        "Average Latency",
                        "95th Percentile Latency",
                        "Maximum Latency",
                        "Average Packet Loss",
                        "95th Percentile Packet Loss",
                        "Maximum Packet Loss"
                    ],
                    "Value": [
                        network_filtered[
                            "Network_Latency_ms"
                        ].mean(),

                        network_filtered[
                            "Network_Latency_ms"
                        ].quantile(0.95),

                        network_filtered[
                            "Network_Latency_ms"
                        ].max(),

                        network_filtered[
                            "Packet_Loss_%"
                        ].mean(),

                        network_filtered[
                            "Packet_Loss_%"
                        ].quantile(0.95),

                        network_filtered[
                            "Packet_Loss_%"
                        ].max()
                    ]
                }
            )

            st.dataframe(
                network_summary,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # FILTERED QUALITY PERFORMANCE
        # ----------------------------------------------------

        quality_columns = [
            "Quality_Control_Defect_Rate_%",
            "Error_Rate_%",
            "Production_Speed_units_per_hr"
        ]

        available_quality = [
            column
            for column in quality_columns
            if column in active_analysis_df.columns
        ]

        if available_quality:

            st.markdown(
                "### Filtered Quality & Production"
            )

            quality_summary_data = []

            for column in available_quality:

                values = pd.to_numeric(
                    active_analysis_df[column],
                    errors="coerce"
                )

                quality_summary_data.append(
                    {
                        "Metric": column,
                        "Average": values.mean(),
                        "Minimum": values.min(),
                        "Maximum": values.max(),
                        "95th Percentile": values.quantile(
                            0.95
                        )
                    }
                )

            quality_summary_filtered = pd.DataFrame(
                quality_summary_data
            )

            st.dataframe(
                quality_summary_filtered,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # FILTERED DAILY TREND
        # ----------------------------------------------------

        if (
            "Date" in active_analysis_df.columns
            and "risk_score_numeric"
            in active_analysis_df.columns
        ):

            st.markdown(
                "### Filtered Daily Risk Trend"
            )

            trend_filtered = (
                active_analysis_df[
                    [
                        "Date",
                        "risk_score_numeric"
                    ]
                ]
                .dropna(
                    subset=[
                        "Date",
                        "risk_score_numeric"
                    ]
                )
                .groupby(
                    "Date",
                    as_index=False
                )[
                    "risk_score_numeric"
                ]
                .mean()
            )

            if not trend_filtered.empty:

                trend_filtered = (
                    trend_filtered
                    .sort_values("Date")
                    .set_index("Date")
                )

                st.line_chart(
                    trend_filtered[
                        "risk_score_numeric"
                    ]
                )

        # ----------------------------------------------------
        # FILTERED MACHINE HEALTH
        # ----------------------------------------------------

        if (
            "Machine_ID"
            in active_analysis_df.columns
            and "health_numeric"
            in active_analysis_df.columns
        ):

            st.markdown(
                "### Filtered Machine Health Profile"
            )

            machine_health_filtered = (
                active_analysis_df
                .groupby(
                    "Machine_ID",
                    as_index=False
                )[
                    "health_numeric"
                ]
                .mean()
                .rename(
                    columns={
                        "health_numeric":
                        "Average Health"
                    }
                )
                .sort_values(
                    "Average Health"
                )
            )

            st.dataframe(
                machine_health_filtered.head(20),
                width="stretch",
                hide_index=True
            )

# ============================================================
# ALERT CENTER
# ============================================================

st.markdown("---")
st.markdown("## ðŸš¨ Alert Center")

try:

    alert_source_df = active_analysis_df.copy()

except NameError:

    alert_source_df = maintenance_df.copy()

if (
    alert_source_df is not None
    and not alert_source_df.empty
):

    # --------------------------------------------------------
    # ALERT RISK NORMALIZATION
    # --------------------------------------------------------

    if "risk_score" in alert_source_df.columns:

        alert_source_df[
            "alert_risk_score"
        ] = pd.to_numeric(
            alert_source_df[
                "risk_score"
            ],
            errors="coerce"
        )

        valid_alert_risk = (
            alert_source_df[
                "alert_risk_score"
            ]
            .dropna()
        )

        if (
            not valid_alert_risk.empty
            and valid_alert_risk.mean() <= 1
        ):

            alert_source_df[
                "alert_risk_score"
            ] *= 100

    else:

        alert_source_df[
            "alert_risk_score"
        ] = np.nan

    # --------------------------------------------------------
    # ALERT TYPE GENERATION
    # --------------------------------------------------------

    alert_records = []

    for index, row in alert_source_df.iterrows():

        machine_id = row.get(
            "Machine_ID",
            "Unknown"
        )

        risk_score = row.get(
            "alert_risk_score",
            np.nan
        )

        anomaly_flag = int(
            pd.to_numeric(
                row.get(
                    "is_anomaly",
                    0
                ),
                errors="coerce"
            )
            if pd.notna(
                pd.to_numeric(
                    row.get(
                        "is_anomaly",
                        0
                    ),
                    errors="coerce"
                )
            )
            else 0
        )

        severity = str(
            row.get(
                "anomaly_severity",
                "Normal"
            )
        )

        domain = str(
            row.get(
                "anomaly_domain",
                "General"
            )
        )

        maintenance_priority = str(
            row.get(
                "maintenance_priority",
                ""
            )
        )

        # ----------------------------------------------------
        # CRITICAL RISK ALERT
        # ----------------------------------------------------

        if (
            pd.notna(risk_score)
            and risk_score >= 75
        ):

            alert_records.append(
                {
                    "Machine_ID": machine_id,
                    "Alert_Type": "Critical Risk",
                    "Severity": "Critical",
                    "Risk_Score": risk_score,
                    "Domain": domain,
                    "Priority": "P1 - Emergency",
                    "Message":
                        "Machine risk is in the critical range.",
                    "Recommended_Action":
                        "Immediate inspection and controlled intervention."
                }
            )

        # ----------------------------------------------------
        # HIGH RISK ALERT
        # ----------------------------------------------------

        elif (
            pd.notna(risk_score)
            and risk_score >= 50
        ):

            alert_records.append(
                {
                    "Machine_ID": machine_id,
                    "Alert_Type": "High Risk",
                    "Severity": "High",
                    "Risk_Score": risk_score,
                    "Domain": domain,
                    "Priority": "P2 - High",
                    "Message":
                        "Machine risk exceeds the high-risk threshold.",
                    "Recommended_Action":
                        "Schedule priority inspection and corrective action."
                }
            )

        # ----------------------------------------------------
        # ANOMALY ALERT
        # ----------------------------------------------------

        if anomaly_flag == 1:

            anomaly_priority = (
                "P1 - Emergency"
                if severity.lower() == "critical"
                else "P2 - High"
                if severity.lower() == "high"
                else "P3 - Planned"
            )

            alert_records.append(
                {
                    "Machine_ID": machine_id,
                    "Alert_Type": "Anomaly Detected",
                    "Severity": severity,
                    "Risk_Score": risk_score,
                    "Domain": domain,
                    "Priority": anomaly_priority,
                    "Message":
                        f"Anomaly detected in {domain} domain.",
                    "Recommended_Action":
                        "Investigate contributing sensor, network, "
                        "quality or operational signals."
                }
            )

        # ----------------------------------------------------
        # NETWORK ALERT
        # ----------------------------------------------------

        latency = pd.to_numeric(
            row.get(
                "Network_Latency_ms",
                np.nan
            ),
            errors="coerce"
        )

        packet_loss = pd.to_numeric(
            row.get(
                "Packet_Loss_%",
                np.nan
            ),
            errors="coerce"
        )

        if (
            pd.notna(latency)
            and latency >= 95
        ):

            alert_records.append(
                {
                    "Machine_ID": machine_id,
                    "Alert_Type": "Network Latency",
                    "Severity": "High",
                    "Risk_Score": risk_score,
                    "Domain": "Network / 6G",
                    "Priority": "P2 - High",
                    "Message":
                        "Network latency is above the stress threshold.",
                    "Recommended_Action":
                        "Inspect network path, congestion and connectivity."
                }
            )

        if (
            pd.notna(packet_loss)
            and packet_loss >= 5
        ):

            alert_records.append(
                {
                    "Machine_ID": machine_id,
                    "Alert_Type": "Packet Loss",
                    "Severity": "High",
                    "Risk_Score": risk_score,
                    "Domain": "Network / 6G",
                    "Priority": "P2 - High",
                    "Message":
                        "Packet loss is above the operational threshold.",
                    "Recommended_Action":
                        "Inspect network reliability and communication quality."
                }
            )

        # ----------------------------------------------------
        # TEMPERATURE ALERT
        # ----------------------------------------------------

        temperature = pd.to_numeric(
            row.get(
                "Temperature_C",
                np.nan
            ),
            errors="coerce"
        )

        if (
            pd.notna(temperature)
            and temperature >= 85
        ):

            alert_records.append(
                {
                    "Machine_ID": machine_id,
                    "Alert_Type": "Temperature Stress",
                    "Severity": "High",
                    "Risk_Score": risk_score,
                    "Domain": "Sensor / Mechanical",
                    "Priority": "P2 - High",
                    "Message":
                        "Temperature exceeds the thermal stress threshold.",
                    "Recommended_Action":
                        "Inspect cooling system and thermal operating conditions."
                }
            )

        # ----------------------------------------------------
        # VIBRATION ALERT
        # ----------------------------------------------------

        vibration = pd.to_numeric(
            row.get(
                "Vibration_Hz",
                np.nan
            ),
            errors="coerce"
        )

        if (
            pd.notna(vibration)
            and vibration >= 8
        ):

            alert_records.append(
                {
                    "Machine_ID": machine_id,
                    "Alert_Type": "Vibration Stress",
                    "Severity": "High",
                    "Risk_Score": risk_score,
                    "Domain": "Sensor / Mechanical",
                    "Priority": "P2 - High",
                    "Message":
                        "Vibration level indicates mechanical stress.",
                    "Recommended_Action":
                        "Inspect bearings, alignment and rotating components."
                }
            )

    # --------------------------------------------------------
    # BUILD ALERT DATAFRAME
    # --------------------------------------------------------

    alerts_df = pd.DataFrame(
        alert_records
    )

    if not alerts_df.empty:

        alerts_df = (
            alerts_df
            .sort_values(
                [
                    "Severity",
                    "Risk_Score"
                ],
                ascending=[
                    True,
                    False
                ]
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # ALERT KPIs
        # ----------------------------------------------------

        total_alerts = len(alerts_df)

        critical_alerts = int(
            (
                alerts_df[
                    "Severity"
                ]
                .astype(str)
                .str.lower()
                == "critical"
            )
            .sum()
        )

        high_alerts = int(
            (
                alerts_df[
                    "Severity"
                ]
                .astype(str)
                .str.lower()
                == "high"
            )
            .sum()
        )

        unique_alert_machines = (
            alerts_df[
                "Machine_ID"
            ]
            .nunique()
        )

        alert_k1, alert_k2, alert_k3, alert_k4 = (
            st.columns(4)
        )

        with alert_k1:

            st.metric(
                "Total Alerts",
                f"{total_alerts:,}"
            )

        with alert_k2:

            st.metric(
                "Critical Alerts",
                f"{critical_alerts:,}"
            )

        with alert_k3:

            st.metric(
                "High Alerts",
                f"{high_alerts:,}"
            )

        with alert_k4:

            st.metric(
                "Affected Machines",
                f"{unique_alert_machines:,}"
            )

        # ----------------------------------------------------
        # ALERT SEVERITY
        # ----------------------------------------------------

        alert_col1, alert_col2 = st.columns(2)

        with alert_col1:

            st.markdown(
                "### Alert Severity"
            )

            alert_severity_chart = (
                alerts_df[
                    "Severity"
                ]
                .value_counts()
                .reset_index()
            )

            alert_severity_chart.columns = [
                "Severity",
                "Alerts"
            ]

            st.bar_chart(
                alert_severity_chart.set_index(
                    "Severity"
                )
            )

        with alert_col2:

            st.markdown(
                "### Alert Domain"
            )

            alert_domain_chart = (
                alerts_df[
                    "Domain"
                ]
                .value_counts()
                .reset_index()
            )

            alert_domain_chart.columns = [
                "Domain",
                "Alerts"
            ]

            st.bar_chart(
                alert_domain_chart.set_index(
                    "Domain"
                )
            )

        # ----------------------------------------------------
        # PRIORITY QUEUE
        # ----------------------------------------------------

        st.markdown(
            "### Active Alert Queue"
        )

        priority_filter = st.selectbox(
            "Alert Priority",
            [
                "All",
                "P1 - Emergency",
                "P2 - High",
                "P3 - Planned"
            ],
            key="alert_priority_filter"
        )

        alert_view = alerts_df.copy()

        if priority_filter != "All":

            alert_view = (
                alert_view[
                    alert_view[
                        "Priority"
                    ]
                    == priority_filter
                ]
            )

        display_alert_columns = [
            "Machine_ID",
            "Alert_Type",
            "Severity",
            "Risk_Score",
            "Domain",
            "Priority",
            "Message",
            "Recommended_Action"
        ]

        display_alert_columns = [
            column
            for column in display_alert_columns
            if column in alert_view.columns
        ]

        st.dataframe(
            alert_view[
                display_alert_columns
            ].head(100),
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # TOP AFFECTED MACHINES
        # ----------------------------------------------------

        st.markdown(
            "### Machines With Most Active Alerts"
        )

        machine_alert_summary = (
            alerts_df
            .groupby(
                "Machine_ID",
                as_index=False
            )
            .agg(
                Total_Alerts=(
                    "Alert_Type",
                    "count"
                ),
                Maximum_Risk=(
                    "Risk_Score",
                    "max"
                )
            )
            .sort_values(
                [
                    "Total_Alerts",
                    "Maximum_Risk"
                ],
                ascending=[
                    False,
                    False
                ]
            )
        )

        st.dataframe(
            machine_alert_summary.head(20),
            width="stretch",
            hide_index=True
        )

    else:

        st.success(
            "No active alerts were generated for the current filtered dataset."
        )

else:

    st.info(
        "Alert Center is waiting for dashboard data."
    )
        
                                                                  
# ============================================================
 # ============================================================
# EXECUTIVE INTELLIGENCE & DECISION SUPPORT
# ============================================================

st.markdown("---")
st.header("Executive Intelligence & Decision Support")

st.caption(
    "Management-level summary of fleet health, operational risk, "
    "network intelligence, anomalies and maintenance priorities."
)

# ------------------------------------------------------------
# EXECUTIVE DATA PREPARATION
# ------------------------------------------------------------

executive_df = maintenance_df.copy()

if executive_df.empty:
    st.warning(
        "Executive intelligence cannot be generated because "
        "maintenance data is unavailable."
    )
else:

    # --------------------------------------------------------
    # NUMERIC NORMALIZATION
    # --------------------------------------------------------

    executive_df["risk_score_exec"] = pd.to_numeric(
        executive_df.get("risk_score"),
        errors="coerce"
    )

    executive_df["health_score_exec"] = (
        100 - executive_df["risk_score_exec"]
    )

    executive_df["network_risk_exec"] = pd.to_numeric(
        executive_df.get("network_risk"),
        errors="coerce"
    )

    executive_df["sensor_risk_exec"] = pd.to_numeric(
        executive_df.get("sensor_risk"),
        errors="coerce"
    )

    executive_df["quality_risk_exec"] = pd.to_numeric(
        executive_df.get("quality_risk"),
        errors="coerce"
    )

    executive_df["anomaly_exec"] = pd.to_numeric(
        executive_df.get("is_anomaly"),
        errors="coerce"
    ).fillna(0)

    # Normalize risk values if they are fractions.
    for column in [
        "network_risk_exec",
        "sensor_risk_exec",
        "quality_risk_exec"
    ]:

        if column in executive_df.columns:

            valid_values = executive_df[column].dropna()

            if (
                not valid_values.empty
                and valid_values.mean() <= 1
            ):
                executive_df[column] = (
                    executive_df[column] * 100
                )

    # --------------------------------------------------------
    # BASIC EXECUTIVE METRICS
    # --------------------------------------------------------

    total_records_exec = len(executive_df)

    total_machines_exec = (
        executive_df["Machine_ID"]
        .nunique()
        if "Machine_ID" in executive_df.columns
        else 0
    )

    average_risk_exec = (
        executive_df["risk_score_exec"]
        .mean()
    )

    average_health_exec = (
        executive_df["health_score_exec"]
        .mean()
    )

    critical_records_exec = int(
        (
            executive_df["risk_score_exec"] >= 75
        )
        .sum()
    )

    high_records_exec = int(
        (
            (executive_df["risk_score_exec"] >= 50)
            &
            (executive_df["risk_score_exec"] < 75)
        )
        .sum()
    )

    anomaly_records_exec = int(
        (
            executive_df["anomaly_exec"] == 1
        )
        .sum()
    )

    anomaly_rate_exec = (
        anomaly_records_exec
        / total_records_exec
        * 100
        if total_records_exec > 0
        else 0
    )

    # --------------------------------------------------------
    # MACHINE LEVEL EXECUTIVE RISK
    # --------------------------------------------------------

    if "Machine_ID" in executive_df.columns:

        executive_machine_summary = (
            executive_df
            .groupby("Machine_ID", as_index=False)
            .agg(
                Average_Risk=(
                    "risk_score_exec",
                    "mean"
                ),
                Maximum_Risk=(
                    "risk_score_exec",
                    "max"
                ),
                Average_Health=(
                    "health_score_exec",
                    "mean"
                ),
                Anomaly_Count=(
                    "anomaly_exec",
                    "sum"
                )
            )
        )

    else:

        executive_machine_summary = pd.DataFrame()

    critical_machines_exec = 0
    high_risk_machines_exec = 0

    if not executive_machine_summary.empty:

        critical_machines_exec = int(
            (
                executive_machine_summary[
                    "Maximum_Risk"
                ] >= 75
            ).sum()
        )

        high_risk_machines_exec = int(
            (
                (
                    executive_machine_summary[
                        "Maximum_Risk"
                    ] >= 50
                )
                &
                (
                    executive_machine_summary[
                        "Maximum_Risk"
                    ] < 75
                )
            ).sum()
        )

    # --------------------------------------------------------
    # EXECUTIVE KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Fleet Health",
            f"{average_health_exec:.1f}/100"
        )

    with col2:
        st.metric(
            "Average Risk",
            f"{average_risk_exec:.1f}/100"
        )

    with col3:
        st.metric(
            "Critical Machines",
            f"{critical_machines_exec:,}"
        )

    with col4:
        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate_exec:.2f}%"
        )

    # --------------------------------------------------------
    # EXECUTIVE STATUS
    # --------------------------------------------------------

    if average_risk_exec >= 75:

        executive_status = "CRITICAL"
        executive_message = (
            "Immediate maintenance intervention is required "
            "across the fleet."
        )

    elif average_risk_exec >= 50:

        executive_status = "HIGH ATTENTION"
        executive_message = (
            "The fleet shows elevated operational risk. "
            "Priority maintenance planning is recommended."
        )

    elif average_risk_exec >= 30:

        executive_status = "MONITOR"

        executive_message = (
            "The fleet is operational but requires continuous "
            "condition monitoring and preventive action."
        )

    else:

        executive_status = "STABLE"

        executive_message = (
            "Overall fleet condition is stable with relatively "
            "low predictive maintenance risk."
        )

    st.markdown("### Executive Fleet Status")

    st.info(
        f"**Fleet Status: {executive_status}**\n\n"
        f"{executive_message}"
    )

    # --------------------------------------------------------
    # RISK DRIVER ANALYSIS
    # --------------------------------------------------------

    driver_columns_exec = {
        "Sensor / Mechanical": "sensor_risk_exec",
        "Network / 6G": "network_risk_exec",
        "Quality / Production": "quality_risk_exec"
    }

    driver_summary_exec = []

    for driver_name, driver_column in driver_columns_exec.items():

        if driver_column in executive_df.columns:

            driver_value = (
                executive_df[driver_column]
                .mean()
            )

            if pd.notna(driver_value):

                driver_summary_exec.append(
                    {
                        "Risk Driver": driver_name,
                        "Average Risk": float(
                            driver_value
                        )
                    }
                )

    driver_summary_exec = pd.DataFrame(
        driver_summary_exec
    )

    if not driver_summary_exec.empty:

        driver_summary_exec = (
            driver_summary_exec
            .sort_values(
                "Average Risk",
                ascending=False
            )
            .reset_index(drop=True)
        )

        dominant_driver_exec = (
            driver_summary_exec.iloc[0]["Risk Driver"]
        )

        dominant_driver_score_exec = (
            driver_summary_exec.iloc[0]["Average Risk"]
        )

    else:

        dominant_driver_exec = "Unavailable"
        dominant_driver_score_exec = 0

    st.markdown("### Dominant Fleet Risk Driver")

    driver_col1, driver_col2 = st.columns(2)

    with driver_col1:

        st.metric(
            "Primary Driver",
            dominant_driver_exec
        )

    with driver_col2:

        st.metric(
            "Driver Risk",
            f"{dominant_driver_score_exec:.1f}/100"
        )

    if not driver_summary_exec.empty:

        st.bar_chart(
            driver_summary_exec.set_index(
                "Risk Driver"
            )["Average Risk"]
        )

    # --------------------------------------------------------
    # MAINTENANCE WORKLOAD
    # --------------------------------------------------------

    st.markdown("### Maintenance Workload")

    if "maintenance_priority" in executive_df.columns:

        priority_exec = (
            executive_df[
                "maintenance_priority"
            ]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        p1_exec = int(
            priority_exec.get(
                "P1 - Emergency",
                0
            )
        )

        p2_exec = int(
            priority_exec.get(
                "P2 - High",
                0
            )
        )

        p3_exec = int(
            priority_exec.get(
                "P3 - Planned",
                0
            )
        )

        p4_exec = int(
            priority_exec.get(
                "P4 - Routine",
                0
            )
        )

        w1, w2, w3, w4 = st.columns(4)

        with w1:
            st.metric(
                "P1 Emergency",
                f"{p1_exec:,}"
            )

        with w2:
            st.metric(
                "P2 High",
                f"{p2_exec:,}"
            )

        with w3:
            st.metric(
                "P3 Planned",
                f"{p3_exec:,}"
            )

        with w4:
            st.metric(
                "P4 Routine",
                f"{p4_exec:,}"
            )

    # --------------------------------------------------------
    # NETWORK / 6G EXECUTIVE SIGNAL
    # --------------------------------------------------------

    st.markdown("### Network / 6G Operational Signal")

    if "network_risk_exec" in executive_df.columns:

        network_exec_mean = (
            executive_df[
                "network_risk_exec"
            ].mean()
        )

        network_exec_high = int(
            (
                executive_df[
                    "network_risk_exec"
                ] >= 75
            ).sum()
        )

        n1, n2 = st.columns(2)

        with n1:

            st.metric(
                "Average Network Risk",
                f"{network_exec_mean:.1f}/100"
            )

        with n2:

            st.metric(
                "High Network Stress Records",
                f"{network_exec_high:,}"
            )

        if network_exec_mean >= 75:

            st.warning(
                "Network conditions represent a critical "
                "operational risk signal."
            )

        elif network_exec_mean >= 50:

            st.warning(
                "Network conditions show elevated stress "
                "and should be monitored closely."
            )

        else:

            st.success(
                "Network conditions are currently within "
                "the monitored operating range."
            )

    # --------------------------------------------------------
    # ANOMALY EXECUTIVE SIGNAL
    # --------------------------------------------------------

    st.markdown("### Anomaly Intelligence")

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Anomalous Records",
            f"{anomaly_records_exec:,}"
        )

    with a2:

        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate_exec:.2f}%"
        )

    with a3:

        st.metric(
            "Normal Records",
            f"{total_records_exec - anomaly_records_exec:,}"
        )

    if anomaly_rate_exec >= 10:

        st.error(
            "Anomaly frequency is elevated and requires "
            "immediate investigation."
        )

    elif anomaly_rate_exec >= 5:

        st.warning(
            "Anomaly frequency is significant and should "
            "be included in maintenance prioritization."
        )

    else:

        st.success(
            "Anomaly frequency remains within the configured "
            "monitoring threshold."
        )

    # --------------------------------------------------------
    # TOP EXECUTIVE PRIORITY MACHINES
    # --------------------------------------------------------

    st.markdown("### Immediate Management Attention")

    if not executive_machine_summary.empty:

        top_exec_machines = (
            executive_machine_summary
            .sort_values(
                [
                    "Maximum_Risk",
                    "Average_Risk"
                ],
                ascending=False
            )
            .head(10)
            .copy()
        )

        top_exec_machines["Machine_ID"] = (
            top_exec_machines[
                "Machine_ID"
            ].astype(str)
        )

        top_exec_machines = (
            top_exec_machines.rename(
                columns={
                    "Machine_ID": "Machine",
                    "Average_Risk": "Avg Risk",
                    "Maximum_Risk": "Max Risk",
                    "Average_Health": "Avg Health",
                    "Anomaly_Count": "Anomalies"
                }
            )
        )

        st.dataframe(
            top_exec_machines,
            width="stretch",
            hide_index=True
        )

    # --------------------------------------------------------
    # MANAGEMENT RECOMMENDATION
    # --------------------------------------------------------

    st.markdown("### Management Recommendation")

    recommendation_points = []

    if critical_machines_exec > 0:

        recommendation_points.append(
            f"Prioritize {critical_machines_exec} machine(s) "
            "with critical risk exposure."
        )

    if high_risk_machines_exec > 0:

        recommendation_points.append(
            f"Schedule focused preventive interventions "
            f"for {high_risk_machines_exec} high-risk machine(s)."
        )

    if dominant_driver_exec != "Unavailable":

        recommendation_points.append(
            f"Investigate **{dominant_driver_exec}** as the "
            "dominant fleet-level risk driver."
        )

    if anomaly_records_exec > 0:

        recommendation_points.append(
            f"Review {anomaly_records_exec:,} anomalous "
            "records for abnormal operating conditions."
        )

    if "network_risk_exec" in executive_df.columns:

        if network_exec_mean >= 50:

            recommendation_points.append(
                "Review network latency and packet-loss "
                "conditions because network stress is elevated."
            )

    if not recommendation_points:

        recommendation_points.append(
            "Continue preventive monitoring and periodic "
            "maintenance review."
        )

    for point in recommendation_points:

        st.markdown(
            f"- {point}"
        )

    # --------------------------------------------------------
    # PROJECT SCOPE DISCLAIMER
    # --------------------------------------------------------

    st.caption(
        "Decision-support note: risk scores and maintenance "
        "recommendations are analytical estimates derived from "
        "available sensor, network, quality, anomaly and "
        "historical signals. The dataset does not contain a "
        "verified machine-failure target or remaining-useful-life "
        "ground truth; therefore this system should not be "
        "interpreted as guaranteed failure prediction or actual "
        "RUL forecasting."
    )

# BUSINESS IMPACT
# ============================================================
# ============================================================
# BUSINESS IMPACT
# ============================================================


st.markdown(
    "### Estimated Business Impact"
)


impact_col1, impact_col2, impact_col3 = st.columns(3)


if "estimated_downtime_hours" in maintenance_df.columns:

    downtime = pd.to_numeric(
        maintenance_df[
            "estimated_downtime_hours"
        ],
        errors="coerce"
    ).sum()

else:

    downtime = 0


if "estimated_maintenance_cost" in maintenance_df.columns:

    maintenance_cost = pd.to_numeric(
        maintenance_df[
            "estimated_maintenance_cost"
        ],
        errors="coerce"
    ).sum()

else:

    maintenance_cost = 0


if "estimated_production_impact_units" in maintenance_df.columns:

    production_impact = pd.to_numeric(
        maintenance_df[
            "estimated_production_impact_units"
        ],
        errors="coerce"
    ).sum()

else:

    production_impact = 0


with impact_col1:

    st.metric(
        "Estimated Downtime",
        f"{downtime:,.2f} hrs"
    )


with impact_col2:

    st.metric(
        "Estimated Maintenance Cost",
        f"â‚¹{maintenance_cost:,.0f}"
    )


with impact_col3:

    st.metric(
        "Production Impact",
        f"{production_impact:,.0f} units"
    )


st.divider()
# ============================================================
# ============================================================
# EXPORT & REPORTING CENTER
# ============================================================

st.markdown("---")
st.header("Export & Reporting Center")

st.caption(
    "Generate downloadable operational, maintenance and "
    "executive reports from the currently available intelligence."
)

# ------------------------------------------------------------
# REPORT DATA PREPARATION
# ------------------------------------------------------------

export_maintenance_df = (
    maintenance_df.copy()
    if (
        maintenance_df is not None
        and not maintenance_df.empty
    )
    else pd.DataFrame()
)

export_machine_df = (
    machine_df.copy()
    if (
        machine_df is not None
        and not machine_df.empty
    )
    else pd.DataFrame()
)

export_priority_df = (
    priority_df.copy()
    if (
        priority_df is not None
        and not priority_df.empty
    )
    else pd.DataFrame()
)

export_fleet_df = (
    fleet_df.copy()
    if (
        fleet_df is not None
        and not fleet_df.empty
    )
    else pd.DataFrame()
)

def prepare_export_dataframe(df):

    if df is None or df.empty:
        return pd.DataFrame()

    export_df = df.copy()

    # Convert datetime columns into Excel/CSV-friendly strings.
    for column in export_df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            export_df[column]
        ):

            export_df[column] = (
                export_df[column]
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

    return export_df


def dataframe_to_csv_bytes(df):

    if df is None or df.empty:
        return b""

    return df.to_csv(
        index=False
    ).encode("utf-8")


def dataframe_to_excel_bytes(
    dataframes,
    sheet_names
):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for df, sheet_name in zip(
            dataframes,
            sheet_names
        ):

            if df is None or df.empty:

                empty_df = pd.DataFrame(
                    {
                        "Information": [
                            "No data available"
                        ]
                    }
                )

                empty_df.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False
                )

            else:

                prepared_df = (
                    prepare_export_dataframe(df)
                )

                prepared_df.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False
                )

    output.seek(0)

    return output.getvalue()


# ------------------------------------------------------------
# EXPORT SUMMARY
# ------------------------------------------------------------

st.markdown("### Available Report Data")

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)

with summary_col1:

    st.metric(
        "Detailed Records",
        f"{len(export_maintenance_df):,}"
    )

with summary_col2:

    st.metric(
        "Machines",
        (
            f"{export_machine_df['Machine_ID'].nunique():,}"
            if (
                not export_machine_df.empty
                and "Machine_ID"
                in export_machine_df.columns
            )
            else "0"
        )
    )

with summary_col3:

    st.metric(
        "Priority Records",
        f"{len(export_priority_df):,}"
    )

with summary_col4:

    st.metric(
        "Fleet Summary Rows",
        f"{len(export_fleet_df):,}"
    )

# ------------------------------------------------------------
# CSV EXPORTS
# ------------------------------------------------------------

st.markdown("### CSV Reports")

csv_col1, csv_col2 = st.columns(2)

with csv_col1:

    if not export_maintenance_df.empty:

        st.download_button(
            label="Download Detailed Maintenance CSV",
            data=dataframe_to_csv_bytes(
                prepare_export_dataframe(
                    export_maintenance_df
                )
            ),
            file_name=(
                "maintenance_recommendation_results.csv"
            ),
            mime="text/csv",
            key="download_maintenance_csv"
        )

    else:

        st.info(
            "Detailed maintenance data is unavailable."
        )

with csv_col2:

    if not export_machine_df.empty:

        st.download_button(
            label="Download Machine Summary CSV",
            data=dataframe_to_csv_bytes(
                prepare_export_dataframe(
                    export_machine_df
                )
            ),
            file_name=(
                "machine_maintenance_summary.csv"
            ),
            mime="text/csv",
            key="download_machine_summary_csv"
        )

    else:

        st.info(
            "Machine maintenance summary is unavailable."
        )


csv_col3, csv_col4 = st.columns(2)

with csv_col3:

    if not export_priority_df.empty:

        st.download_button(
            label="Download Priority Queue CSV",
            data=dataframe_to_csv_bytes(
                prepare_export_dataframe(
                    export_priority_df
                )
            ),
            file_name=(
                "maintenance_priority_queue.csv"
            ),
            mime="text/csv",
            key="download_priority_queue_csv"
        )

    else:

        st.info(
            "Maintenance priority queue is unavailable."
        )

with csv_col4:

    if not export_fleet_df.empty:

        st.download_button(
            label="Download Fleet Summary CSV",
            data=dataframe_to_csv_bytes(
                prepare_export_dataframe(
                    export_fleet_df
                )
            ),
            file_name=(
                "fleet_maintenance_summary.csv"
            ),
            mime="text/csv",
            key="download_fleet_summary_csv"
        )

    else:

        st.info(
            "Fleet maintenance summary is unavailable."
        )

# ------------------------------------------------------------
# EXCEL REPORT
# ------------------------------------------------------------

st.markdown("### Excel Management Report")

excel_priority_export_df = (
    export_priority_df
    .sort_values(
        by="maintenance_risk_score",
        ascending=False
    )
    .head(5000)
    .copy()
    if (
        not export_priority_df.empty
        and "maintenance_risk_score" in export_priority_df.columns
    )
    else export_priority_df.head(5000).copy()
)

excel_report = dataframe_to_excel_bytes(
    [
        export_machine_df,
        excel_priority_export_df,
        export_fleet_df
    ],
    [
        "Machine Summary",
        "Priority Queue Top 5000",
        "Fleet Summary"
    ]
)

if excel_report:

    st.download_button(
        label="Download Complete Excel Report",
        data=excel_report,
        file_name=(
            "predictive_maintenance_management_report.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        key="download_complete_excel_report"
    )

else:

    st.warning(
        "Excel report could not be generated because "
        "report data is unavailable."
    )

# ------------------------------------------------------------
# FILTERED DATA EXPORT
# ------------------------------------------------------------

st.markdown("### Analysis Dataset Export")

if (
    "active_analysis_df" in globals()
    and active_analysis_df is not None
    and not active_analysis_df.empty
):

    filtered_export_df = active_analysis_df

    st.success(
        f"Current filtered analysis contains "
        f"{len(filtered_export_df):,} records."
    )

    st.download_button(
        label="Download Current Filtered Analysis CSV",
        data=dataframe_to_csv_bytes(
            prepare_export_dataframe(
                filtered_export_df
            )
        ),
        file_name=(
            "filtered_predictive_maintenance_analysis.csv"
        ),
        mime="text/csv",
        key="download_filtered_analysis_csv"
    )

else:

    st.info(
        "No active filtered analysis dataset is available."
    )

# ------------------------------------------------------------
# EXECUTIVE REPORT
# ------------------------------------------------------------

st.markdown("### Executive Summary Export")

executive_report_rows = []

if not export_maintenance_df.empty:

    risk_column = None

    for candidate in [
        "risk_score",
        "maintenance_risk_score"
    ]:

        if candidate in export_maintenance_df.columns:

            risk_column = candidate
            break

    if risk_column is not None:

        risk_values = pd.to_numeric(
            export_maintenance_df[risk_column],
            errors="coerce"
        )

        average_risk_report = (
            risk_values.mean()
        )

        critical_records_report = int(
            (
                risk_values >= 75
            ).sum()
        )

        high_records_report = int(
            (
                (risk_values >= 50)
                &
                (risk_values < 75)
            ).sum()
        )

    else:

        average_risk_report = 0
        critical_records_report = 0
        high_records_report = 0

    machine_count_report = (
        export_maintenance_df[
            "Machine_ID"
        ].nunique()
        if "Machine_ID"
        in export_maintenance_df.columns
        else 0
    )

    anomaly_count_report = (
        int(
            (
                pd.to_numeric(
                    export_maintenance_df[
                        "is_anomaly"
                    ],
                    errors="coerce"
                ).fillna(0)
                == 1
            ).sum()
        )
        if "is_anomaly"
        in export_maintenance_df.columns
        else 0
    )

    anomaly_rate_report = (
        anomaly_count_report
        / len(export_maintenance_df)
        * 100
        if len(export_maintenance_df) > 0
        else 0
    )

    executive_report_rows = [
        {
            "Metric": "Total Records",
            "Value": len(
                export_maintenance_df
            )
        },
        {
            "Metric": "Total Machines",
            "Value": machine_count_report
        },
        {
            "Metric": "Average Risk Score",
            "Value": round(
                average_risk_report,
                2
            )
        },
        {
            "Metric": "Critical Risk Records",
            "Value": critical_records_report
        },
        {
            "Metric": "High Risk Records",
            "Value": high_records_report
        },
        {
            "Metric": "Detected Anomalies",
            "Value": anomaly_count_report
        },
        {
            "Metric": "Anomaly Rate (%)",
            "Value": round(
                anomaly_rate_report,
                2
            )
        }
    ]

executive_report_df = pd.DataFrame(
    executive_report_rows
)

if not executive_report_df.empty:

    st.dataframe(
        executive_report_df,
        width="stretch",
        hide_index=True
    )

    executive_excel = dataframe_to_excel_bytes(
        [
            executive_report_df,
            export_machine_df,
            export_priority_df
        ],
        [
            "Executive Summary",
            "Machine Summary",
            "Priority Queue"
        ]
    )

    st.download_button(
        label="Download Executive Excel Report",
        data=executive_excel,
        file_name=(
            "executive_maintenance_report.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        key="download_executive_excel_report"
    )

# ------------------------------------------------------------
# REPORTING NOTES
# ------------------------------------------------------------

st.markdown("### Reporting Notes")

st.info(
    "Exported reports contain analytical results generated "
    "from the available sensor, network, quality, anomaly "
    "and maintenance-risk signals. These reports are intended "
    "for operational decision support and should be reviewed "
    "by maintenance and engineering teams before executing "
    "physical interventions."
)

st.caption(
    "The system does not contain verified machine-failure "
    "or remaining-useful-life ground truth. Exported risk "
    "scores and maintenance recommendations should therefore "
    "not be interpreted as guaranteed failure predictions."
)

# ============================================================
# SYSTEM HEALTH & DATA QUALITY CENTER
# ============================================================

st.markdown("---")
st.header("System Health & Data Quality Center")

st.caption(
    "Operational health monitoring for dataset quality, "
    "analytics engines, generated outputs and dashboard readiness."
)

# ------------------------------------------------------------
# SYSTEM PATH CONFIGURATION
# ------------------------------------------------------------

system_health_base = Path(__file__).resolve().parent

system_health_data_dir = (
    system_health_base / "data"
)

system_health_output_dir = (
    system_health_base / "outputs"
)

system_health_dataset_path = (
    system_health_data_dir
    / "Thales_Group_Manufacturing.csv"
)

system_health_required_outputs = {
    "Engineered Dataset":
        "engineered_dataset.csv",

    "Anomaly Detection":
        "anomaly_detection_results.csv",

    "Risk Engine":
        "risk_engine_results.csv",

    "Maintenance Recommendations":
        "maintenance_recommendation_results.csv",

    "Machine Maintenance Summary":
        "machine_maintenance_summary.csv",

    "Maintenance Priority Queue":
        "maintenance_priority_queue.csv",

    "Fleet Maintenance Summary":
        "fleet_maintenance_summary.csv"
}

# ------------------------------------------------------------
# DATASET HEALTH
# ------------------------------------------------------------

system_health_dataset_exists = (
    system_health_dataset_path.exists()
)

system_health_output_dir_exists = (
    system_health_output_dir.exists()
)

system_health_dataset_size_mb = 0.0

if system_health_dataset_exists:

    try:

        system_health_dataset_size_mb = (
            system_health_dataset_path.stat().st_size
            / (1024 * 1024)
        )

    except Exception:

        system_health_dataset_size_mb = 0.0

# ------------------------------------------------------------
# CURRENT DATASET STATISTICS
# ------------------------------------------------------------

system_health_records = 0
system_health_columns = 0
system_health_missing = 0
system_health_duplicates = 0
system_health_machine_count = 0

system_health_date_min = None
system_health_date_max = None

system_health_quality_score = 0.0

system_health_active_df = (
    maintenance_df.copy()
    if (
        "maintenance_df" in globals()
        and maintenance_df is not None
    )
    else pd.DataFrame()
)

if not system_health_active_df.empty:

    system_health_records = (
        len(system_health_active_df)
    )

    system_health_columns = (
        len(system_health_active_df.columns)
    )

    system_health_missing = int(
        system_health_active_df
        .isna()
        .sum()
        .sum()
    )

    system_health_duplicates = int(
        system_health_active_df
        .duplicated()
        .sum()
    )

    if "Machine_ID" in system_health_active_df.columns:

        system_health_machine_count = int(
            system_health_active_df[
                "Machine_ID"
            ]
            .nunique()
        )

    if "Date" in system_health_active_df.columns:

        system_health_dates = pd.to_datetime(
            system_health_active_df[
                "Date"
            ],
            errors="coerce"
        )

        valid_dates = (
            system_health_dates
            .dropna()
        )

        if not valid_dates.empty:

            system_health_date_min = (
                valid_dates.min()
            )

            system_health_date_max = (
                valid_dates.max()
            )

    # --------------------------------------------------------
    # DATA QUALITY SCORE
    # --------------------------------------------------------

    quality_components = []

    # Missing-value quality
    if system_health_records > 0:

        missing_rate = (
            system_health_missing
            / (
                system_health_records
                * max(system_health_columns, 1)
            )
            * 100
        )

        missing_quality = max(
            0,
            100 - missing_rate
        )

        quality_components.append(
            missing_quality
        )

    # Duplicate quality
    if system_health_records > 0:

        duplicate_rate = (
            system_health_duplicates
            / system_health_records
            * 100
        )

        duplicate_quality = max(
            0,
            100 - duplicate_rate
        )

        quality_components.append(
            duplicate_quality
        )

    if quality_components:

        system_health_quality_score = (
            sum(quality_components)
            / len(quality_components)
        )

# ------------------------------------------------------------
# OUTPUT ENGINE HEALTH
# ------------------------------------------------------------

system_health_engine_rows = []

for engine_name, output_filename in (
    system_health_required_outputs.items()
):

    output_path = (
        system_health_output_dir
        / output_filename
    )

    exists = output_path.exists()

    size_mb = 0.0

    if exists:

        try:

            size_mb = (
                output_path.stat().st_size
                / (1024 * 1024)
            )

        except Exception:

            size_mb = 0.0

    system_health_engine_rows.append(
        {
            "Engine": engine_name,
            "Output File": output_filename,
            "Status": (
                "READY"
                if exists
                else "MISSING"
            ),
            "Size (MB)": round(
                size_mb,
                2
            )
        }
    )

system_health_engine_df = pd.DataFrame(
    system_health_engine_rows
)

system_health_ready_engines = int(
    (
        system_health_engine_df[
            "Status"
        ]
        == "READY"
    ).sum()
)

system_health_total_engines = len(
    system_health_engine_df
)

# ------------------------------------------------------------
# SYSTEM READINESS
# ------------------------------------------------------------

if not system_health_dataset_exists:

    system_health_status = "DEGRADED"

    system_health_status_message = (
        "The primary manufacturing dataset is unavailable."
    )

elif system_health_total_engines == 0:

    system_health_status = "ATTENTION"

    system_health_status_message = (
        "No analytics engine outputs are available."
    )

elif system_health_ready_engines == system_health_total_engines:

    if system_health_quality_score >= 99:

        system_health_status = "READY"

        system_health_status_message = (
            "All required analytics outputs are available "
            "and dataset quality is within the expected range."
        )

    else:

        system_health_status = "ATTENTION"

        system_health_status_message = (
            "All analytics outputs are available, but "
            "data-quality indicators require review."
        )

elif system_health_ready_engines >= (
    system_health_total_engines * 0.70
):

    system_health_status = "ATTENTION"

    system_health_status_message = (
        "Most analytics outputs are available, but "
        "one or more pipeline components require attention."
    )

else:

    system_health_status = "DEGRADED"

    system_health_status_message = (
        "Multiple analytics outputs are unavailable. "
        "Pipeline execution should be reviewed."
    )

# ------------------------------------------------------------
# SYSTEM STATUS HEADER
# ------------------------------------------------------------

if system_health_status == "READY":

    st.success(
        f"System Status: {system_health_status} â€” "
        f"{system_health_status_message}"
    )

elif system_health_status == "ATTENTION":

    st.warning(
        f"System Status: {system_health_status} â€” "
        f"{system_health_status_message}"
    )

else:

    st.error(
        f"System Status: {system_health_status} â€” "
        f"{system_health_status_message}"
    )

# ------------------------------------------------------------
# SYSTEM KPI CARDS
# ------------------------------------------------------------

system_col1, system_col2, system_col3, system_col4 = (
    st.columns(4)
)

with system_col1:

    st.metric(
        "Data Quality",
        f"{system_health_quality_score:.1f}%"
    )

with system_col2:

    st.metric(
        "Records",
        f"{system_health_records:,}"
    )

with system_col3:

    st.metric(
        "Machines",
        f"{system_health_machine_count:,}"
    )

with system_col4:

    st.metric(
        "Pipeline Engines",
        f"{system_health_ready_engines}/"
        f"{system_health_total_engines}"
    )

# ------------------------------------------------------------
# DATASET PROFILE
# ------------------------------------------------------------

st.markdown("### Dataset Profile")

dataset_profile_col1, dataset_profile_col2 = (
    st.columns(2)
)

with dataset_profile_col1:

    dataset_profile_df = pd.DataFrame(
        {
            "Metric": [
                "Dataset Available",
                "Records",
                "Columns",
                "Machines",
                "Missing Values",
                "Duplicate Records"
            ],
            "Value": [
                (
                    "YES"
                    if system_health_dataset_exists
                    else "NO"
                ),
                f"{system_health_records:,}",
                f"{system_health_columns:,}",
                f"{system_health_machine_count:,}",
                f"{system_health_missing:,}",
                f"{system_health_duplicates:,}"
            ]
        }
    )

    st.dataframe(
        dataset_profile_df,
        width="stretch",
        hide_index=True
    )

with dataset_profile_col2:

    date_profile_df = pd.DataFrame(
        {
            "Metric": [
                "Data Start",
                "Data End",
                "Dataset Size"
            ],
            "Value": [
                (
                    system_health_date_min.strftime(
                        "%Y-%m-%d"
                    )
                    if system_health_date_min is not None
                    else "Unavailable"
                ),
                (
                    system_health_date_max.strftime(
                        "%Y-%m-%d"
                    )
                    if system_health_date_max is not None
                    else "Unavailable"
                ),
                f"{system_health_dataset_size_mb:.2f} MB"
            ]
        }
    )

    st.dataframe(
        date_profile_df,
        width="stretch",
        hide_index=True
    )

# ------------------------------------------------------------
# DATA QUALITY ASSESSMENT
# ------------------------------------------------------------

st.markdown("### Data Quality Assessment")

quality_col1, quality_col2, quality_col3 = (
    st.columns(3)
)

with quality_col1:

    if system_health_missing == 0:

        st.success(
            "Missing Values: CLEAN"
        )

    else:

        st.warning(
            f"Missing Values: "
            f"{system_health_missing:,}"
        )

with quality_col2:

    if system_health_duplicates == 0:

        st.success(
            "Duplicates: CLEAN"
        )

    else:

        st.warning(
            f"Duplicates: "
            f"{system_health_duplicates:,}"
        )

with quality_col3:

    if system_health_quality_score >= 99:

        st.success(
            "Overall Quality: EXCELLENT"
        )

    elif system_health_quality_score >= 95:

        st.warning(
            "Overall Quality: GOOD"
        )

    else:

        st.error(
            "Overall Quality: REVIEW REQUIRED"
        )

# ------------------------------------------------------------
# ANALYTICS ENGINE STATUS
# ------------------------------------------------------------

st.markdown("### Analytics Engine Status")

if not system_health_engine_df.empty:

    st.dataframe(
        system_health_engine_df,
        width="stretch",
        hide_index=True
    )

# ------------------------------------------------------------
# ENGINE READINESS VISUALIZATION
# ------------------------------------------------------------

if not system_health_engine_df.empty:

    engine_status_chart = (
        system_health_engine_df
        .copy()
    )

    engine_status_chart[
        "Ready"
    ] = (
        engine_status_chart[
            "Status"
        ]
        == "READY"
    ).astype(int)

    engine_status_chart = (
        engine_status_chart
        .set_index("Engine")[
            "Ready"
        ]
    )

    st.markdown(
        "#### Pipeline Readiness"
    )

    st.bar_chart(
        engine_status_chart
    )

# ------------------------------------------------------------
# PIPELINE STAGE MONITOR
# ------------------------------------------------------------

st.markdown("### Pipeline Stage Monitor")

pipeline_stage_data = pd.DataFrame(
    {
        "Stage": [
            "Raw Dataset",
            "Feature Engineering",
            "Anomaly Detection",
            "Risk Engine",
            "Maintenance Engine",
            "Dashboard"
        ],
        "Status": [
            (
                "READY"
                if system_health_dataset_exists
                else "MISSING"
            ),
            (
                "READY"
                if (
                    system_health_output_dir
                    / "engineered_dataset.csv"
                ).exists()
                else "MISSING"
            ),
            (
                "READY"
                if (
                    system_health_output_dir
                    / "anomaly_detection_results.csv"
                ).exists()
                else "MISSING"
            ),
            (
                "READY"
                if (
                    system_health_output_dir
                    / "risk_engine_results.csv"
                ).exists()
                else "MISSING"
            ),
            (
                "READY"
                if (
                    system_health_output_dir
                    / "maintenance_recommendation_results.csv"
                ).exists()
                else "MISSING"
            ),
            "READY"
        ]
    }
)

st.dataframe(
    pipeline_stage_data,
    width="stretch",
    hide_index=True
)

# ------------------------------------------------------------
# SYSTEM HEALTH SCORE
# ------------------------------------------------------------

pipeline_ready_ratio = (
    system_health_ready_engines
    / system_health_total_engines
    * 100
    if system_health_total_engines > 0
    else 0
)

system_health_score = (
    system_health_quality_score * 0.60
    + pipeline_ready_ratio * 0.40
)

system_health_score = max(
    0,
    min(
        100,
        system_health_score
    )
)

st.markdown("### Overall System Health Score")

health_score_col1, health_score_col2 = (
    st.columns(2)
)

with health_score_col1:

    st.metric(
        "System Health Score",
        f"{system_health_score:.1f}/100"
    )

with health_score_col2:

    if system_health_score >= 95:

        st.success(
            "System health is excellent."
        )

    elif system_health_score >= 85:

        st.info(
            "System health is good."
        )

    elif system_health_score >= 70:

        st.warning(
            "System health requires monitoring."
        )

    else:

        st.error(
            "System health requires immediate review."
        )

# ------------------------------------------------------------
# DATA QUALITY WARNINGS
# ------------------------------------------------------------

st.markdown("### Data Quality & System Warnings")

system_health_warnings = []

if not system_health_dataset_exists:

    system_health_warnings.append(
        "Primary manufacturing dataset is missing."
    )

if system_health_missing > 0:

    system_health_warnings.append(
        f"{system_health_missing:,} missing values "
        "were detected."
    )

if system_health_duplicates > 0:

    system_health_warnings.append(
        f"{system_health_duplicates:,} duplicate records "
        "were detected."
    )

if (
    system_health_ready_engines
    < system_health_total_engines
):

    missing_engines = (
        system_health_engine_df[
            system_health_engine_df[
                "Status"
            ] != "READY"
        ]["Engine"]
        .tolist()
    )

    if missing_engines:

        system_health_warnings.append(
            "Unavailable pipeline outputs: "
            + ", ".join(
                missing_engines
            )
        )

if not system_health_warnings:

    st.success(
        "No active data-quality or pipeline warnings detected."
    )

else:

    for warning_message in system_health_warnings:

        st.warning(
            warning_message
        )

# ------------------------------------------------------------
# SYSTEM READINESS CHECKLIST
# ------------------------------------------------------------

st.markdown("### System Readiness Checklist")

system_readiness_rows = [
    {
        "Check": "Dataset Available",
        "Status": (
            "PASS"
            if system_health_dataset_exists
            else "FAIL"
        )
    },
    {
        "Check": "No Missing Values",
        "Status": (
            "PASS"
            if system_health_missing == 0
            else "REVIEW"
        )
    },
    {
        "Check": "No Duplicate Records",
        "Status": (
            "PASS"
            if system_health_duplicates == 0
            else "REVIEW"
        )
    },
    {
        "Check": "Analytics Outputs Available",
        "Status": (
            "PASS"
            if (
                system_health_ready_engines
                == system_health_total_engines
            )
            else "REVIEW"
        )
    },
    {
        "Check": "Dashboard Operational",
        "Status": "PASS"
    }
]

system_readiness_df = pd.DataFrame(
    system_readiness_rows
)

st.dataframe(
    system_readiness_df,
    width="stretch",
    hide_index=True
)

# ------------------------------------------------------------
# SYSTEM HEALTH DISCLAIMER
# ------------------------------------------------------------

st.caption(
    "System health metrics describe data and analytical "
    "pipeline readiness. They do not represent physical "
    "machine health or guaranteed equipment reliability."
)

# ============================================================
# ADVANCED 6G INTELLIGENCE CENTER
# ============================================================

st.markdown("---")
st.header("ðŸ“¡ Advanced 6G Intelligence Center")
st.caption(
    "Dataset-based communication intelligence layer for "
    "6G-integrated smart manufacturing operations."
)

# ------------------------------------------------------------
# 6G DATA PREPARATION
# ------------------------------------------------------------

sixg_df = maintenance_df.copy() if (
    maintenance_df is not None and not maintenance_df.empty
) else pd.DataFrame()

if not sixg_df.empty:

    # --------------------------------------------------------
    # REQUIRED NUMERIC COLUMNS
    # --------------------------------------------------------

    sixg_numeric_columns = [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "network_risk",
        "network_health",
        "risk_score",
        "Production_Speed_units_per_hr",
        "Quality_Control_Defect_Rate_%",
        "Temperature_C",
        "Vibration_Hz",
        "anomaly_score",
        "is_anomaly",
    ]

    for col in sixg_numeric_columns:
        if col in sixg_df.columns:
            sixg_df[col] = pd.to_numeric(
                sixg_df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # NETWORK HEALTH
    # --------------------------------------------------------

    if "network_health" not in sixg_df.columns:

        latency_series = pd.to_numeric(
            sixg_df.get(
                "Network_Latency_ms",
                pd.Series(index=sixg_df.index, dtype=float)
            ),
            errors="coerce"
        ).fillna(0)

        packet_series = pd.to_numeric(
            sixg_df.get(
                "Packet_Loss_%",
                pd.Series(index=sixg_df.index, dtype=float)
            ),
            errors="coerce"
        ).fillna(0)

        latency_ref = max(float(latency_series.quantile(0.95)), 1.0)
        packet_ref = max(float(packet_series.quantile(0.95)), 0.01)

        latency_stress = (
            latency_series / latency_ref
        ).clip(0, 1)

        packet_stress = (
            packet_series / packet_ref
        ).clip(0, 1)

        sixg_df["network_health"] = (
            100
            * (
                1
                - (
                    0.60 * latency_stress
                    + 0.40 * packet_stress
                )
            )
        ).clip(0, 100)

    # --------------------------------------------------------
    # NETWORK RISK
    # --------------------------------------------------------

    if "network_risk" not in sixg_df.columns:

        sixg_df["network_risk"] = (
            100
            - pd.to_numeric(
                sixg_df["network_health"],
                errors="coerce"
            ).fillna(50)
        ).clip(0, 100)

    # --------------------------------------------------------
    # COMMUNICATION RELIABILITY
    # --------------------------------------------------------

    sixg_df["communication_reliability"] = (
        100
        - pd.to_numeric(
            sixg_df["network_risk"],
            errors="coerce"
        ).fillna(50)
    ).clip(0, 100)

    # --------------------------------------------------------
    # COMMUNICATION STRESS
    # --------------------------------------------------------

    sixg_df["communication_stress"] = (
        pd.to_numeric(
            sixg_df["network_risk"],
            errors="coerce"
        ).fillna(50)
    ).clip(0, 100)

    # --------------------------------------------------------
    # NETWORK CONDITION
    # --------------------------------------------------------

    def classify_network_condition(value):
        try:
            value = float(value)

            if value >= 80:
                return "Healthy"

            if value >= 60:
                return "Stable"

            if value >= 40:
                return "Stressed"

            if value >= 20:
                return "Critical"

            return "Severe"

        except Exception:
            return "Unknown"

    sixg_df["network_condition"] = (
        sixg_df["communication_reliability"]
        .apply(classify_network_condition)
    )

    # --------------------------------------------------------
    # LATENCY CONDITION
    # --------------------------------------------------------

    latency_95 = (
        pd.to_numeric(
            sixg_df.get(
                "Network_Latency_ms",
                pd.Series(index=sixg_df.index, dtype=float)
            ),
            errors="coerce"
        )
        .quantile(0.95)
    )

    packet_95 = (
        pd.to_numeric(
            sixg_df.get(
                "Packet_Loss_%",
                pd.Series(index=sixg_df.index, dtype=float)
            ),
            errors="coerce"
        )
        .quantile(0.95)
    )

    sixg_df["latency_alert"] = (
        pd.to_numeric(
            sixg_df.get(
                "Network_Latency_ms",
                pd.Series(index=sixg_df.index, dtype=float)
            ),
            errors="coerce"
        ) >= latency_95
    )

    sixg_df["packet_loss_alert"] = (
        pd.to_numeric(
            sixg_df.get(
                "Packet_Loss_%",
                pd.Series(index=sixg_df.index, dtype=float)
            ),
            errors="coerce"
        ) >= packet_95
    )

    # --------------------------------------------------------
    # 6G OPERATIONAL STATUS
    # --------------------------------------------------------

    avg_network_health = float(
        sixg_df["network_health"].mean()
    )

    avg_network_risk = float(
        sixg_df["network_risk"].mean()
    )

    if avg_network_health >= 75:
        sixg_status = "ðŸŸ¢ HEALTHY"

    elif avg_network_health >= 55:
        sixg_status = "ðŸŸ¡ STABLE"

    elif avg_network_health >= 40:
        sixg_status = "ðŸŸ  STRESSED"

    else:
        sixg_status = "ðŸ”´ CRITICAL"

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "6G Network Health",
            f"{avg_network_health:.1f}/100"
        )

    with k2:
        st.metric(
            "Communication Reliability",
            f"{100 - avg_network_risk:.1f}%"
        )

    with k3:
        st.metric(
            "Avg Network Risk",
            f"{avg_network_risk:.1f}/100"
        )

    with k4:
        st.metric(
            "6G Operational Status",
            sixg_status
        )

    # --------------------------------------------------------
    # NETWORK TELEMETRY SUMMARY
    # --------------------------------------------------------

    st.subheader("ðŸ“Š Communication Telemetry")

    telemetry_cols = st.columns(4)

    avg_latency = pd.to_numeric(
        sixg_df.get(
            "Network_Latency_ms",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    ).mean()

    avg_packet_loss = pd.to_numeric(
        sixg_df.get(
            "Packet_Loss_%",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    ).mean()

    max_latency = pd.to_numeric(
        sixg_df.get(
            "Network_Latency_ms",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    ).max()

    max_packet_loss = pd.to_numeric(
        sixg_df.get(
            "Packet_Loss_%",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    ).max()

    with telemetry_cols[0]:
        st.metric(
            "Average Latency",
            f"{avg_latency:.2f} ms"
        )

    with telemetry_cols[1]:
        st.metric(
            "Average Packet Loss",
            f"{avg_packet_loss:.3f}%"
        )

    with telemetry_cols[2]:
        st.metric(
            "Peak Latency",
            f"{max_latency:.2f} ms"
        )

    with telemetry_cols[3]:
        st.metric(
            "Peak Packet Loss",
            f"{max_packet_loss:.3f}%"
        )

    # --------------------------------------------------------
    # NETWORK CONDITION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("ðŸ“¡ Network Condition Distribution")

    condition_distribution = (
        sixg_df["network_condition"]
        .value_counts()
        .rename_axis("Network Condition")
        .reset_index(name="Records")
    )

    if not condition_distribution.empty:

        st.bar_chart(
            condition_distribution.set_index(
                "Network Condition"
            )["Records"]
        )

    # --------------------------------------------------------
    # LATENCY VS PACKET LOSS
    # --------------------------------------------------------

    st.subheader("ðŸ“¶ Latency vs Packet Loss")

    latency_packet_df = sixg_df[
        [
            col for col in [
                "Network_Latency_ms",
                "Packet_Loss_%",
                "network_risk",
                "Machine_ID"
            ]
            if col in sixg_df.columns
        ]
    ].dropna()

    if (
        not latency_packet_df.empty
        and "Network_Latency_ms" in latency_packet_df.columns
        and "Packet_Loss_%" in latency_packet_df.columns
    ):

        if "network_risk" in latency_packet_df.columns:

            fig_latency_packet = px.scatter(
                latency_packet_df.sample(
                    min(10000, len(latency_packet_df)),
                    random_state=42
                ),
                x="Network_Latency_ms",
                y="Packet_Loss_%",
                color="network_risk",
                hover_data=[
                    c for c in ["Machine_ID", "network_risk"]
                    if c in latency_packet_df.columns
                ],
                title="Communication Stress Relationship"
            )

        else:

            fig_latency_packet = px.scatter(
                latency_packet_df.sample(
                    min(10000, len(latency_packet_df)),
                    random_state=42
                ),
                x="Network_Latency_ms",
                y="Packet_Loss_%",
                title="Latency vs Packet Loss"
            )

        st.plotly_chart(
            fig_latency_packet,
            width="stretch"
        )

    # --------------------------------------------------------
    # NETWORK HEALTH DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("ðŸŸ¢ 6G Network Health Distribution")

    health_distribution = (
        pd.cut(
            sixg_df["network_health"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=[
                "0-20 Critical",
                "20-40 Poor",
                "40-60 Stressed",
                "60-80 Stable",
                "80-100 Healthy"
            ],
            include_lowest=True
        )
        .astype(str)
        .value_counts()
        .reindex(
            [
                "0-20 Critical",
                "20-40 Poor",
                "40-60 Stressed",
                "60-80 Stable",
                "80-100 Healthy"
            ],
            fill_value=0
        )
    )

    st.bar_chart(health_distribution)

    # --------------------------------------------------------
    # MACHINE-WISE NETWORK INTELLIGENCE
    # --------------------------------------------------------

    st.subheader("ðŸ­ Machine-wise 6G Network Intelligence")

    if "Machine_ID" in sixg_df.columns:

        machine_network = (
            sixg_df
            .groupby("Machine_ID", as_index=False)
            .agg(
                average_network_health=(
                    "network_health",
                    "mean"
                ),
                average_network_risk=(
                    "network_risk",
                    "mean"
                ),
                average_latency_ms=(
                    "Network_Latency_ms",
                    "mean"
                ),
                average_packet_loss=(
                    "Packet_Loss_%",
                    "mean"
                ),
                peak_latency_ms=(
                    "Network_Latency_ms",
                    "max"
                ),
                peak_packet_loss=(
                    "Packet_Loss_%",
                    "max"
                ),
                communication_reliability=(
                    "communication_reliability",
                    "mean"
                ),
                latency_alerts=(
                    "latency_alert",
                    "sum"
                ),
                packet_loss_alerts=(
                    "packet_loss_alert",
                    "sum"
                )
            )
        )

        machine_network["network_rank"] = (
            machine_network["average_network_risk"]
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )

        machine_network["network_status"] = (
            machine_network["average_network_health"]
            .apply(classify_network_condition)
        )

        machine_network = machine_network.sort_values(
            "average_network_risk",
            ascending=False
        )

        display_network = machine_network[
            [
                "network_rank",
                "Machine_ID",
                "average_network_health",
                "average_network_risk",
                "average_latency_ms",
                "average_packet_loss",
                "communication_reliability",
                "network_status"
            ]
        ].head(15).copy()

        display_network.columns = [
            "Rank",
            "Machine",
            "Network Health",
            "Network Risk",
            "Avg Latency (ms)",
            "Avg Packet Loss (%)",
            "Reliability (%)",
            "Status"
        ]

        st.dataframe(
            display_network,
            width="stretch",
            hide_index=True
        )

    # --------------------------------------------------------
    # NETWORK RISK VS MACHINE HEALTH
    # --------------------------------------------------------

    st.subheader("âš ï¸ Network Risk vs Machine Health")

    risk_health_columns = [
        c for c in [
            "Machine_ID",
            "network_risk",
            "network_health",
            "risk_score"
        ]
        if c in sixg_df.columns
    ]

    risk_health_df = sixg_df[risk_health_columns].dropna()

    if (
        not risk_health_df.empty
        and "network_risk" in risk_health_df.columns
        and "risk_score" in risk_health_df.columns
    ):

        fig_risk_health = px.scatter(
            risk_health_df.sample(
                min(10000, len(risk_health_df)),
                random_state=42
            ),
            x="network_risk",
            y="risk_score",
            color="network_health",
            hover_data=[
                c for c in ["Machine_ID", "network_health"]
                if c in risk_health_df.columns
            ],
            title="Network Risk vs Overall Machine Risk"
        )

        st.plotly_chart(
            fig_risk_health,
            width="stretch"
        )

    # --------------------------------------------------------
    # NETWORK RISK VS PRODUCTION
    # --------------------------------------------------------

    st.subheader("ðŸ­ Network Risk vs Production Performance")

    production_network_cols = [
        c for c in [
            "network_risk",
            "Production_Speed_units_per_hr",
            "Machine_ID"
        ]
        if c in sixg_df.columns
    ]

    production_network_df = sixg_df[
        production_network_cols
    ].dropna()

    if (
        not production_network_df.empty
        and "network_risk" in production_network_df.columns
        and "Production_Speed_units_per_hr"
        in production_network_df.columns
    ):

        fig_production_network = px.scatter(
            production_network_df.sample(
                min(10000, len(production_network_df)),
                random_state=42
            ),
            x="network_risk",
            y="Production_Speed_units_per_hr",
            color="network_risk",
            hover_data=[
                c for c in ["Machine_ID"]
                if c in production_network_df.columns
            ],
            title="Network Risk vs Production Speed"
        )

        st.plotly_chart(
            fig_production_network,
            width="stretch"
        )

    # --------------------------------------------------------
    # NETWORK RISK VS QUALITY
    # --------------------------------------------------------

    st.subheader("ðŸŽ¯ Network Risk vs Quality Performance")

    quality_network_cols = [
        c for c in [
            "network_risk",
            "Quality_Control_Defect_Rate_%",
            "Error_Rate_%",
            "Machine_ID"
        ]
        if c in sixg_df.columns
    ]

    quality_network_df = sixg_df[
        quality_network_cols
    ].dropna()

    if (
        not quality_network_df.empty
        and "network_risk" in quality_network_df.columns
        and "Quality_Control_Defect_Rate_%"
        in quality_network_df.columns
    ):

        fig_quality_network = px.scatter(
            quality_network_df.sample(
                min(10000, len(quality_network_df)),
                random_state=42
            ),
            x="network_risk",
            y="Quality_Control_Defect_Rate_%",
            color="network_risk",
            hover_data=[
                c for c in [
                    "Machine_ID",
                    "Error_Rate_%"
                ]
                if c in quality_network_df.columns
            ],
            title="Network Risk vs Defect Rate"
        )

        st.plotly_chart(
            fig_quality_network,
            width="stretch"
        )

    # --------------------------------------------------------
    # NETWORK ANOMALY RELATIONSHIP
    # --------------------------------------------------------

    st.subheader("ðŸš¨ Communication Anomaly Intelligence")

    if "is_anomaly" in sixg_df.columns:

        anomaly_network = (
            sixg_df
            .groupby("is_anomaly", as_index=False)
            .agg(
                records=("is_anomaly", "size"),
                average_network_risk=(
                    "network_risk",
                    "mean"
                ),
                average_latency=(
                    "Network_Latency_ms",
                    "mean"
                ),
                average_packet_loss=(
                    "Packet_Loss_%",
                    "mean"
                ),
                average_network_health=(
                    "network_health",
                    "mean"
                )
            )
        )

        anomaly_network["Anomaly Status"] = (
            anomaly_network["is_anomaly"]
            .map({
                0: "Normal",
                1: "Anomalous"
            })
            .fillna("Unknown")
        )

        st.dataframe(
            anomaly_network[
                [
                    "Anomaly Status",
                    "records",
                    "average_network_risk",
                    "average_latency",
                    "average_packet_loss",
                    "average_network_health"
                ]
            ].rename(
                columns={
                    "records": "Records",
                    "average_network_risk": "Avg Network Risk",
                    "average_latency": "Avg Latency (ms)",
                    "average_packet_loss": "Avg Packet Loss (%)",
                    "average_network_health": "Avg Network Health"
                }
            ),
            width="stretch",
            hide_index=True
        )

    # --------------------------------------------------------
    # DAILY NETWORK TREND
    # --------------------------------------------------------

    st.subheader("ðŸ“ˆ Daily 6G Communication Trend")

    if "Date" in sixg_df.columns:

        daily_network = (
            sixg_df
            .assign(
                Date=pd.to_datetime(
                    sixg_df["Date"],
                    errors="coerce"
                )
            )
            .dropna(subset=["Date"])
            .groupby("Date", as_index=False)
            .agg(
                network_health=(
                    "network_health",
                    "mean"
                ),
                network_risk=(
                    "network_risk",
                    "mean"
                ),
                latency=(
                    "Network_Latency_ms",
                    "mean"
                ),
                packet_loss=(
                    "Packet_Loss_%",
                    "mean"
                )
            )
            .sort_values("Date")
        )

        if not daily_network.empty:

            st.line_chart(
                daily_network.set_index("Date")[
                    [
                        "network_health",
                        "network_risk"
                    ]
                ]
            )

            st.line_chart(
                daily_network.set_index("Date")[
                    [
                        "latency",
                        "packet_loss"
                    ]
                ]
            )

    # --------------------------------------------------------
    # NETWORK ALERT CENTER
    # --------------------------------------------------------

    st.subheader("ðŸ”” Communication Alert Center")

    network_alerts = []

    for _, row in sixg_df.iterrows():

        try:

            machine = row.get(
                "Machine_ID",
                "Unknown"
            )

            latency = float(
                row.get(
                    "Network_Latency_ms",
                    0
                )
            )

            packet_loss = float(
                row.get(
                    "Packet_Loss_%",
                    0
                )
            )

            network_risk = float(
                row.get(
                    "network_risk",
                    0
                )
            )

            if network_risk >= 75:

                network_alerts.append({
                    "Machine_ID": machine,
                    "Alert": "Severe Network Risk",
                    "Severity": "Critical",
                    "Network Risk": network_risk,
                    "Latency (ms)": latency,
                    "Packet Loss (%)": packet_loss,
                    "Recommended Response":
                        "Immediate communication diagnostics"
                })

            elif network_risk >= 60:

                network_alerts.append({
                    "Machine_ID": machine,
                    "Alert": "Elevated Network Risk",
                    "Severity": "High",
                    "Network Risk": network_risk,
                    "Latency (ms)": latency,
                    "Packet Loss (%)": packet_loss,
                    "Recommended Response":
                        "Inspect communication performance"
                })

            elif latency >= latency_95:

                network_alerts.append({
                    "Machine_ID": machine,
                    "Alert": "High Communication Latency",
                    "Severity": "Medium",
                    "Network Risk": network_risk,
                    "Latency (ms)": latency,
                    "Packet Loss (%)": packet_loss,
                    "Recommended Response":
                        "Review network congestion"
                })

            elif packet_loss >= packet_95:

                network_alerts.append({
                    "Machine_ID": machine,
                    "Alert": "Elevated Packet Loss",
                    "Severity": "Medium",
                    "Network Risk": network_risk,
                    "Latency (ms)": latency,
                    "Packet Loss (%)": packet_loss,
                    "Recommended Response":
                        "Inspect communication reliability"
                })

        except Exception:
            continue

    if network_alerts:

        network_alert_df = pd.DataFrame(
            network_alerts
        ).sort_values(
            "Network Risk",
            ascending=False
        )

        st.dataframe(
            network_alert_df.head(100),
            width="stretch",
            hide_index=True
        )

        st.caption(
            f"Showing top {min(100, len(network_alert_df))} "
            f"communication alerts."
        )

    else:

        st.success(
            "No significant communication alerts detected "
            "under the current network-risk thresholds."
        )

    # --------------------------------------------------------
    # 6G MACHINE DRILL-DOWN
    # --------------------------------------------------------

    st.subheader("ðŸ”Ž Machine Communication Drill-down")

    if "Machine_ID" in sixg_df.columns:

        sixg_machines = sorted(
            sixg_df["Machine_ID"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if sixg_machines:

            selected_6g_machine = st.selectbox(
                "Select Machine for 6G Analysis",
                sixg_machines,
                key="advanced_6g_machine_selector"
            )

            selected_machine_6g = sixg_df[
                sixg_df["Machine_ID"]
                .astype(str)
                == str(selected_6g_machine)
            ].copy()

            if not selected_machine_6g.empty:

                m1, m2, m3, m4 = st.columns(4)

                machine_health = float(
                    selected_machine_6g[
                        "network_health"
                    ].mean()
                )

                machine_risk = float(
                    selected_machine_6g[
                        "network_risk"
                    ].mean()
                )

                machine_latency = float(
                    selected_machine_6g[
                        "Network_Latency_ms"
                    ].mean()
                )

                machine_packet = float(
                    selected_machine_6g[
                        "Packet_Loss_%"
                    ].mean()
                )

                with m1:
                    st.metric(
                        "Network Health",
                        f"{machine_health:.1f}/100"
                    )

                with m2:
                    st.metric(
                        "Network Risk",
                        f"{machine_risk:.1f}/100"
                    )

                with m3:
                    st.metric(
                        "Avg Latency",
                        f"{machine_latency:.2f} ms"
                    )

                with m4:
                    st.metric(
                        "Avg Packet Loss",
                        f"{machine_packet:.3f}%"
                    )

                st.dataframe(
                    selected_machine_6g[
                        [
                            c for c in [
                                "Date",
                                "Timestamp",
                                "Network_Latency_ms",
                                "Packet_Loss_%",
                                "network_health",
                                "network_risk",
                                "communication_reliability",
                                "network_condition",
                                "is_anomaly"
                            ]
                            if c in selected_machine_6g.columns
                        ]
                    ].tail(100),
                    width="stretch",
                    hide_index=True
                )

    # --------------------------------------------------------
    # 6G DECISION SUPPORT
    # --------------------------------------------------------

    st.subheader("ðŸ§  6G Communication Decision Support")

    decision_messages = []

    if avg_network_health < 40:

        decision_messages.append(
            "ðŸ”´ Critical: communication health is significantly degraded. "
            "Prioritize network diagnostics for high-risk machines."
        )

    elif avg_network_health < 60:

        decision_messages.append(
            "ðŸŸ  Attention: communication layer shows measurable stress. "
            "Review latency and packet-loss hotspots."
        )

    else:

        decision_messages.append(
            "ðŸŸ¢ Communication layer is broadly stable. "
            "Continue monitoring high-risk machines and network outliers."
        )

    if avg_latency >= latency_95:

        decision_messages.append(
            "âš ï¸ Average latency is elevated relative to the "
            "fleet distribution."
        )

    if avg_packet_loss >= packet_95:

        decision_messages.append(
            "âš ï¸ Packet-loss performance requires additional "
            "communication reliability monitoring."
        )

    if "is_anomaly" in sixg_df.columns:

        network_anomaly_count = int(
            sixg_df.loc[
                sixg_df["is_anomaly"] == 1,
                "is_anomaly"
            ].count()
        )

        if network_anomaly_count > 0:

            decision_messages.append(
                f"ðŸš¨ {network_anomaly_count:,} anomalous records "
                "should be cross-checked against communication stress."
            )

    for message in decision_messages:
        st.info(message)

    # --------------------------------------------------------
    # 6G SCOPE DISCLAIMER
    # --------------------------------------------------------

    st.caption(
        "Scope note: This 6G Intelligence Center analyzes the "
        "network telemetry available in the supplied manufacturing "
        "dataset. It represents a dataset-based/simulated "
        "6G-integrated communication intelligence layer and does "
        "not claim direct measurement of a physical 6G network."
    )

else:

    st.warning(
        "6G Intelligence Center is unavailable because the "
        "maintenance dataset is not loaded."
    )

# ============================================================
# ADVANCED MAINTENANCE PLANNING & RESOURCE INTELLIGENCE
# ============================================================

st.markdown("---")
st.header("ðŸ› ï¸ Advanced Maintenance Planning & Resource Intelligence")
st.caption(
    "Operational planning layer for maintenance workload, "
    "intervention priority, downtime, cost and production impact."
)

# ------------------------------------------------------------
# MAINTENANCE PLANNING DATA PREPARATION
# ------------------------------------------------------------

planning_df = maintenance_df.copy() if (
    maintenance_df is not None and not maintenance_df.empty
) else pd.DataFrame()

if not planning_df.empty:

    # --------------------------------------------------------
    # NUMERIC NORMALIZATION
    # --------------------------------------------------------

    planning_numeric_columns = [
        "risk_score",
        "maintenance_risk_score",
        "average_health_score",
        "machine_health_score",
        "estimated_downtime_hours",
        "estimated_maintenance_cost",
        "estimated_production_impact",
        "production_impact",
        "anomaly_score",
        "network_risk",
        "sensor_risk",
        "quality_risk",
        "historical_risk",
        "Production_Speed_units_per_hr",
        "Quality_Control_Defect_Rate_%",
        "Error_Rate_%",
        "Network_Latency_ms",
        "Packet_Loss_%",
    ]

    for col in planning_numeric_columns:

        if col in planning_df.columns:

            planning_df[col] = pd.to_numeric(
                planning_df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # RISK SCORE COMPATIBILITY
    # --------------------------------------------------------

    if "risk_score" not in planning_df.columns:

        if "maintenance_risk_score" in planning_df.columns:

            planning_df["risk_score"] = pd.to_numeric(
                planning_df["maintenance_risk_score"],
                errors="coerce"
            )

        else:

            planning_df["risk_score"] = 0.0

    planning_df["risk_score"] = (
        planning_df["risk_score"]
        .fillna(0)
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # HEALTH SCORE
    # --------------------------------------------------------

    if "average_health_score" not in planning_df.columns:

        if "machine_health_score" in planning_df.columns:

            planning_df["average_health_score"] = (
                pd.to_numeric(
                    planning_df["machine_health_score"],
                    errors="coerce"
                )
            )

        else:

            planning_df["average_health_score"] = (
                100
                - planning_df["risk_score"]
            )

    planning_df["average_health_score"] = (
        planning_df["average_health_score"]
        .fillna(
            100 - planning_df["risk_score"]
        )
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # MAINTENANCE PRIORITY
    # --------------------------------------------------------

    if "maintenance_priority" not in planning_df.columns:

        def build_priority(score):

            try:
                score = float(score)

                if score >= 80:
                    return "P1 - Emergency"

                if score >= 60:
                    return "P2 - High"

                if score >= 40:
                    return "P3 - Planned"

                return "P4 - Routine"

            except Exception:

                return "P4 - Routine"

        planning_df["maintenance_priority"] = (
            planning_df["risk_score"]
            .apply(build_priority)
        )

    planning_df["maintenance_priority"] = (
        planning_df["maintenance_priority"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # URGENCY
    # --------------------------------------------------------

    def planning_urgency(priority):

        priority = str(priority).strip()

        if priority == "P1 - Emergency":
            return "Immediate"

        if priority == "P2 - High":
            return "Within 24 Hours"

        if priority == "P3 - Planned":
            return "Within 7 Days"

        return "Routine Monitoring"

    planning_df["maintenance_urgency"] = (
        planning_df["maintenance_priority"]
        .apply(planning_urgency)
    )

    # --------------------------------------------------------
    # ESTIMATED DOWNTIME
    # --------------------------------------------------------

    if "estimated_downtime_hours" not in planning_df.columns:

        def estimate_downtime(row):

            try:

                risk = float(
                    row.get("risk_score", 0)
                )

                priority = str(
                    row.get(
                        "maintenance_priority",
                        ""
                    )
                )

                if priority == "P1 - Emergency":
                    base = 6.0

                elif priority == "P2 - High":
                    base = 4.0

                elif priority == "P3 - Planned":
                    base = 2.0

                else:
                    base = 0.5

                return round(
                    base
                    + max(risk - 40, 0) * 0.03,
                    2
                )

            except Exception:

                return 0.5

        planning_df["estimated_downtime_hours"] = (
            planning_df.apply(
                estimate_downtime,
                axis=1
            )
        )

    planning_df["estimated_downtime_hours"] = (
        pd.to_numeric(
            planning_df["estimated_downtime_hours"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # ESTIMATED MAINTENANCE COST
    # --------------------------------------------------------

    if "estimated_maintenance_cost" not in planning_df.columns:

        def estimate_cost(row):

            try:

                priority = str(
                    row.get(
                        "maintenance_priority",
                        ""
                    )
                )

                downtime = float(
                    row.get(
                        "estimated_downtime_hours",
                        0
                    )
                )

                if priority == "P1 - Emergency":
                    hourly_rate = 12000

                elif priority == "P2 - High":
                    hourly_rate = 9000

                elif priority == "P3 - Planned":
                    hourly_rate = 6500

                else:
                    hourly_rate = 3000

                return round(
                    downtime * hourly_rate,
                    2
                )

            except Exception:

                return 0.0

        planning_df["estimated_maintenance_cost"] = (
            planning_df.apply(
                estimate_cost,
                axis=1
            )
        )

    planning_df["estimated_maintenance_cost"] = (
        pd.to_numeric(
            planning_df[
                "estimated_maintenance_cost"
            ],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # PRODUCTION IMPACT
    # --------------------------------------------------------

    if "estimated_production_impact" not in planning_df.columns:

        if "production_impact" in planning_df.columns:

            planning_df[
                "estimated_production_impact"
            ] = pd.to_numeric(
                planning_df[
                    "production_impact"
                ],
                errors="coerce"
            )

        elif "Production_Speed_units_per_hr" in planning_df.columns:

            planning_df[
                "estimated_production_impact"
            ] = (
                planning_df[
                    "Production_Speed_units_per_hr"
                ]
                .fillna(0)
                * planning_df[
                    "estimated_downtime_hours"
                ]
            )

        else:

            planning_df[
                "estimated_production_impact"
            ] = 0.0

    planning_df[
        "estimated_production_impact"
    ] = pd.to_numeric(
        planning_df[
            "estimated_production_impact"
        ],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    # --------------------------------------------------------
    # RESOURCE LOAD
    # --------------------------------------------------------

    resource_load_map = {
        "P1 - Emergency": 4.0,
        "P2 - High": 3.0,
        "P3 - Planned": 2.0,
        "P4 - Routine": 1.0,
    }

    planning_df["resource_load_units"] = (
        planning_df["maintenance_priority"]
        .map(resource_load_map)
        .fillna(1.0)
    )

    # --------------------------------------------------------
    # BUSINESS IMPACT SCORE
    # --------------------------------------------------------

    downtime_norm = (
        planning_df[
            "estimated_downtime_hours"
        ]
        / max(
            float(
                planning_df[
                    "estimated_downtime_hours"
                ].quantile(0.95)
            ),
            1.0
        )
    ).clip(0, 1)

    production_norm = (
        planning_df[
            "estimated_production_impact"
        ]
        / max(
            float(
                planning_df[
                    "estimated_production_impact"
                ].quantile(0.95)
            ),
            1.0
        )
    ).clip(0, 1)

    risk_norm = (
        planning_df["risk_score"]
        / 100
    ).clip(0, 1)

    planning_df["business_impact_score"] = (
        (
            0.50 * risk_norm
            + 0.25 * downtime_norm
            + 0.25 * production_norm
        )
        * 100
    ).clip(0, 100)

    # --------------------------------------------------------
    # PLANNING PRIORITY SCORE
    # --------------------------------------------------------

    planning_df["planning_priority_score"] = (
        (
            0.60 * risk_norm
            + 0.20 * downtime_norm
            + 0.20 * production_norm
        )
        * 100
    ).clip(0, 100)

    # --------------------------------------------------------
    # INTERVENTION WINDOW
    # --------------------------------------------------------

    def intervention_window(priority):

        priority = str(priority).strip()

        if priority == "P1 - Emergency":
            return "Immediate Intervention"

        if priority == "P2 - High":
            return "0â€“24 Hours"

        if priority == "P3 - Planned":
            return "1â€“7 Days"

        return "Next Scheduled Maintenance"

    planning_df["intervention_window"] = (
        planning_df[
            "maintenance_priority"
        ]
        .apply(intervention_window)
    )

    # --------------------------------------------------------
    # RESOURCE REQUIREMENT
    # --------------------------------------------------------

    def resource_requirement(priority):

        priority = str(priority).strip()

        if priority == "P1 - Emergency":
            return "Senior Team + Specialist"

        if priority == "P2 - High":
            return "Dedicated Maintenance Team"

        if priority == "P3 - Planned":
            return "Scheduled Maintenance Team"

        return "Routine Inspection Team"

    planning_df["resource_requirement"] = (
        planning_df[
            "maintenance_priority"
        ]
        .apply(resource_requirement)
    )

    # --------------------------------------------------------
    # PLANNING STATUS
    # --------------------------------------------------------

    def planning_status(row):

        try:

            score = float(
                row.get(
                    "planning_priority_score",
                    0
                )
            )

            if score >= 80:
                return "Critical Action"

            if score >= 60:
                return "High Priority"

            if score >= 40:
                return "Planned Action"

            return "Monitor"

        except Exception:

            return "Monitor"

    planning_df["planning_status"] = (
        planning_df.apply(
            planning_status,
            axis=1
        )
    )

    # ========================================================
    # EXECUTIVE PLANNING KPIs
    # ========================================================

    st.subheader("ðŸ“‹ Maintenance Planning Overview")

    total_planning_records = len(
        planning_df
    )

    p1_total = int(
        (
            planning_df[
                "maintenance_priority"
            ]
            == "P1 - Emergency"
        ).sum()
    )

    p2_total = int(
        (
            planning_df[
                "maintenance_priority"
            ]
            == "P2 - High"
        ).sum()
    )

    total_downtime = float(
        planning_df[
            "estimated_downtime_hours"
        ].sum()
    )

    total_cost = float(
        planning_df[
            "estimated_maintenance_cost"
        ].sum()
    )

    total_production_impact = float(
        planning_df[
            "estimated_production_impact"
        ].sum()
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "P1 Emergency Jobs",
            f"{p1_total:,}"
        )

    with k2:
        st.metric(
            "P2 High Priority Jobs",
            f"{p2_total:,}"
        )

    with k3:
        st.metric(
            "Estimated Downtime",
            f"{total_downtime:,.1f} hrs"
        )

    with k4:
        st.metric(
            "Estimated Maintenance Cost",
            f"â‚¹{total_cost:,.0f}"
        )

    st.metric(
        "Estimated Production Impact",
        f"{total_production_impact:,.0f} units"
    )

    # ========================================================
    # MAINTENANCE WORKLOAD DISTRIBUTION
    # ========================================================

    st.subheader("ðŸ“Š Maintenance Workload Distribution")

    workload_distribution = (
        planning_df[
            "maintenance_priority"
        ]
        .value_counts()
        .reindex(
            [
                "P1 - Emergency",
                "P2 - High",
                "P3 - Planned",
                "P4 - Routine"
            ],
            fill_value=0
        )
    )

    st.bar_chart(
        workload_distribution
    )

    # ========================================================
    # RESOURCE LOAD DISTRIBUTION
    # ========================================================

    st.subheader("ðŸ‘· Maintenance Resource Load")

    resource_summary = (
        planning_df
        .groupby(
            "maintenance_priority",
            as_index=False
        )
        .agg(
            records=(
                "maintenance_priority",
                "size"
            ),
            resource_load_units=(
                "resource_load_units",
                "sum"
            ),
            estimated_downtime_hours=(
                "estimated_downtime_hours",
                "sum"
            ),
            estimated_cost=(
                "estimated_maintenance_cost",
                "sum"
            ),
            production_impact=(
                "estimated_production_impact",
                "sum"
            )
        )
    )

    resource_summary["maintenance_priority"] = (
        pd.Categorical(
            resource_summary[
                "maintenance_priority"
            ],
            categories=[
                "P1 - Emergency",
                "P2 - High",
                "P3 - Planned",
                "P4 - Routine"
            ],
            ordered=True
        )
    )

    resource_summary = (
        resource_summary
        .sort_values(
            "maintenance_priority"
        )
    )

    st.dataframe(
        resource_summary,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # MACHINE-LEVEL PLANNING
    # ========================================================

    st.subheader("ðŸ­ Machine Maintenance Planning")

    if "Machine_ID" in planning_df.columns:

        machine_planning = (
            planning_df
            .groupby(
                "Machine_ID",
                as_index=False
            )
            .agg(
                average_risk=(
                    "risk_score",
                    "mean"
                ),
                maximum_risk=(
                    "risk_score",
                    "max"
                ),
                average_health=(
                    "average_health_score",
                    "mean"
                ),
                maintenance_records=(
                    "maintenance_priority",
                    "size"
                ),
                resource_load=(
                    "resource_load_units",
                    "sum"
                ),
                downtime_hours=(
                    "estimated_downtime_hours",
                    "sum"
                ),
                maintenance_cost=(
                    "estimated_maintenance_cost",
                    "sum"
                ),
                production_impact=(
                    "estimated_production_impact",
                    "sum"
                ),
                planning_priority_score=(
                    "planning_priority_score",
                    "mean"
                ),
                business_impact_score=(
                    "business_impact_score",
                    "mean"
                )
            )
        )

        machine_planning["planning_rank"] = (
            machine_planning[
                "planning_priority_score"
            ]
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )

        machine_planning["planning_class"] = (
            pd.cut(
                machine_planning[
                    "planning_priority_score"
                ],
                bins=[
                    -0.01,
                    40,
                    60,
                    80,
                    100
                ],
                labels=[
                    "Monitor",
                    "Planned",
                    "High Priority",
                    "Critical Action"
                ],
                include_lowest=True
            )
            .astype(str)
        )

        machine_planning = (
            machine_planning
            .sort_values(
                "planning_priority_score",
                ascending=False
            )
        )

        st.dataframe(
            machine_planning.head(25),
            width="stretch",
            hide_index=True
        )

    # ========================================================
    # BUSINESS IMPACT MATRIX
    # ========================================================

    st.subheader("ðŸŽ¯ Maintenance Decision Matrix")

    matrix_cols = [
        c for c in [
            "Machine_ID",
            "risk_score",
            "estimated_downtime_hours",
            "estimated_maintenance_cost",
            "estimated_production_impact",
            "business_impact_score",
            "planning_priority_score",
            "maintenance_priority"
        ]
        if c in planning_df.columns
    ]

    matrix_df = planning_df[
        matrix_cols
    ].copy()

    if (
        not matrix_df.empty
        and "risk_score" in matrix_df.columns
        and "estimated_downtime_hours"
        in matrix_df.columns
        and "business_impact_score"
        in matrix_df.columns
    ):

        fig_matrix = px.scatter(
            matrix_df.sample(
                min(10000, len(matrix_df)),
                random_state=42
            ),
            x="estimated_downtime_hours",
            y="risk_score",
            size="business_impact_score",
            color="maintenance_priority",
            hover_data=[
                c for c in [
                    "Machine_ID",
                    "estimated_maintenance_cost",
                    "estimated_production_impact"
                ]
                if c in matrix_df.columns
            ],
            title="Risk Ã— Downtime Ã— Business Impact"
        )

        st.plotly_chart(
            fig_matrix,
            width="stretch"
        )

    # ========================================================
    # HIGH IMPACT MACHINES
    # ========================================================

    st.subheader("ðŸ”¥ High-Impact Maintenance Machines")

    if "Machine_ID" in planning_df.columns:

        high_impact_machines = (
            planning_df
            .groupby(
                "Machine_ID",
                as_index=False
            )
            .agg(
                average_risk=(
                    "risk_score",
                    "mean"
                ),
                business_impact=(
                    "business_impact_score",
                    "mean"
                ),
                downtime_hours=(
                    "estimated_downtime_hours",
                    "sum"
                ),
                maintenance_cost=(
                    "estimated_maintenance_cost",
                    "sum"
                ),
                production_impact=(
                    "estimated_production_impact",
                    "sum"
                ),
                resource_load=(
                    "resource_load_units",
                    "sum"
                )
            )
            .sort_values(
                [
                    "business_impact",
                    "average_risk"
                ],
                ascending=False
            )
        )

        high_impact_machines["impact_rank"] = (
            range(
                1,
                len(high_impact_machines) + 1
            )
        )

        st.dataframe(
            high_impact_machines.head(15),
            width="stretch",
            hide_index=True
        )

    # ========================================================
    # PRIORITY QUEUE
    # ========================================================

    st.subheader("ðŸš¨ Maintenance Execution Queue")

    queue_columns = [
        c for c in [
            "Machine_ID",
            "maintenance_priority",
            "maintenance_urgency",
            "intervention_window",
            "risk_score",
            "average_health_score",
            "business_impact_score",
            "estimated_downtime_hours",
            "estimated_maintenance_cost",
            "estimated_production_impact",
            "resource_requirement",
            "planning_status"
        ]
        if c in planning_df.columns
    ]

    execution_queue = (
        planning_df[
            queue_columns
        ]
        .sort_values(
            [
                "business_impact_score",
                "risk_score"
            ],
            ascending=False
        )
        .head(100)
        .copy()
    )

    st.dataframe(
        execution_queue,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # URGENCY DISTRIBUTION
    # ========================================================

    st.subheader("â±ï¸ Maintenance Urgency")

    urgency_distribution = (
        planning_df[
            "maintenance_urgency"
        ]
        .value_counts()
    )

    st.bar_chart(
        urgency_distribution
    )

    # ========================================================
    # COST VS PRODUCTION IMPACT
    # ========================================================

    st.subheader("ðŸ’° Maintenance Cost vs Production Impact")

    cost_impact_columns = [
        c for c in [
            "Machine_ID",
            "estimated_maintenance_cost",
            "estimated_production_impact",
            "risk_score"
        ]
        if c in planning_df.columns
    ]

    cost_impact_df = planning_df[
        cost_impact_columns
    ].dropna()

    if (
        not cost_impact_df.empty
        and "estimated_maintenance_cost"
        in cost_impact_df.columns
        and "estimated_production_impact"
        in cost_impact_df.columns
    ):

        fig_cost_impact = px.scatter(
            cost_impact_df.sample(
                min(10000, len(cost_impact_df)),
                random_state=42
            ),
            x="estimated_maintenance_cost",
            y="estimated_production_impact",
            color="risk_score",
            hover_data=[
                c for c in [
                    "Machine_ID",
                    "risk_score"
                ]
                if c in cost_impact_df.columns
            ],
            title="Estimated Maintenance Cost vs Production Impact"
        )

        st.plotly_chart(
            fig_cost_impact,
            width="stretch"
        )

    # ========================================================
    # MACHINE PLANNING DRILL-DOWN
    # ========================================================

    st.subheader("ðŸ”Ž Maintenance Planning Drill-down")

    if "Machine_ID" in planning_df.columns:

        planning_machines = sorted(
            planning_df[
                "Machine_ID"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if planning_machines:

            selected_planning_machine = st.selectbox(
                "Select Machine",
                planning_machines,
                key="advanced_maintenance_machine_selector"
            )

            selected_planning = planning_df[
                planning_df[
                    "Machine_ID"
                ].astype(str)
                == str(
                    selected_planning_machine
                )
            ].copy()

            if not selected_planning.empty:

                pm1, pm2, pm3, pm4 = st.columns(4)

                with pm1:

                    st.metric(
                        "Average Risk",
                        f"{selected_planning['risk_score'].mean():.1f}"
                    )

                with pm2:

                    st.metric(
                        "Health",
                        f"{selected_planning['average_health_score'].mean():.1f}"
                    )

                with pm3:

                    st.metric(
                        "Downtime",
                        f"{selected_planning['estimated_downtime_hours'].sum():.1f} hrs"
                    )

                with pm4:

                    st.metric(
                        "Maintenance Cost",
                        f"â‚¹{selected_planning['estimated_maintenance_cost'].sum():,.0f}"
                    )

                st.dataframe(
                    selected_planning[
                        [
                            c for c in [
                                "Date",
                                "Timestamp",
                                "maintenance_priority",
                                "maintenance_urgency",
                                "intervention_window",
                                "risk_score",
                                "average_health_score",
                                "estimated_downtime_hours",
                                "estimated_maintenance_cost",
                                "estimated_production_impact",
                                "resource_requirement",
                                "planning_status"
                            ]
                            if c in selected_planning.columns
                        ]
                    ].tail(100),
                    width="stretch",
                    hide_index=True
                )

    # ========================================================
    # MANAGEMENT DECISION SUPPORT
    # ========================================================

    st.subheader("ðŸ§  Maintenance Management Decision Support")

    decision_points = []

    if p1_total > 0:

        decision_points.append(
            f"ðŸ”´ {p1_total:,} emergency maintenance "
            "records require immediate operational review."
        )

    if p2_total > 0:

        decision_points.append(
            f"ðŸŸ  {p2_total:,} high-priority records "
            "should be scheduled within the defined high-priority window."
        )

    if total_downtime > 0:

        decision_points.append(
            f"â±ï¸ Estimated maintenance downtime across "
            f"the analyzed records is {total_downtime:,.1f} hours."
        )

    if total_cost > 0:

        decision_points.append(
            f"ðŸ’° Estimated maintenance expenditure is "
            f"â‚¹{total_cost:,.0f} under the planning assumptions."
        )

    if total_production_impact > 0:

        decision_points.append(
            f"ðŸ­ Estimated production impact is "
            f"{total_production_impact:,.0f} units."
        )

    high_business_impact = int(
        (
            planning_df[
                "business_impact_score"
            ] >= 75
        ).sum()
    )

    if high_business_impact > 0:

        decision_points.append(
            f"ðŸ”¥ {high_business_impact:,} records have "
            "high combined risk, downtime and production impact."
        )

    for point in decision_points:

        st.info(point)

    # ========================================================
    # PLANNING ASSUMPTIONS
    # ========================================================

    st.caption(
        "Planning note: downtime, maintenance cost and production "
        "impact are operational estimates derived from the available "
        "risk data and planning assumptions. They are not measured "
        "financial or maintenance records."
    )

    st.caption(
        "This planning layer prioritizes interventions based on "
        "risk and estimated operational impact. It does not claim "
        "guaranteed equipment failure dates or exact remaining "
        "useful life."
    )

else:

    st.warning(
        "Advanced Maintenance Planning is unavailable because "
        "the maintenance dataset is not loaded."
    )

# FOOTER
# ============================================================

st.divider()

st.caption(
    f"{APP_NAME} | Version {APP_VERSION} | "
    f"Developed by {APP_AUTHOR}"
)





