# ============================================================
# FULL PROJECT DIAGNOSTIC
# Predictive Maintenance & Anomaly Detection System
# Author: Narendra Suthar
# ============================================================

from pathlib import Path
import ast
import importlib.util
import sys
import traceback

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

APP_FILE = PROJECT_ROOT / "app.py"

REQUIRED_OUTPUTS = {
    "engineered_dataset.csv": [
        "Machine_ID",
        "Date",
    ],
    "anomaly_detection_results.csv": [
        "Machine_ID",
        "anomaly_score",
        "is_anomaly",
        "anomaly_label",
        "anomaly_severity",
        "anomaly_domain",
    ],
    "risk_engine_results.csv": [
        "Machine_ID",
        "risk_score",
    ],
    "maintenance_recommendation_results.csv": [
        "Machine_ID",
        "maintenance_priority",
    ],
    "machine_maintenance_summary.csv": [
        "Machine_ID",
    ],
    "maintenance_priority_queue.csv": [
        "Machine_ID",
        "maintenance_priority",
    ],
    "fleet_maintenance_summary.csv": [],
}


# ============================================================
# REPORT STORAGE
# ============================================================

PASS_COUNT = 0
WARNING_COUNT = 0
ERROR_COUNT = 0

PASS_ITEMS = []
WARNING_ITEMS = []
ERROR_ITEMS = []


def report_pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    PASS_ITEMS.append(message)
    print(f"[PASS] {message}")


def report_warning(message):
    global WARNING_COUNT
    WARNING_COUNT += 1
    WARNING_ITEMS.append(message)
    print(f"[WARNING] {message}")


def report_error(message):
    global ERROR_COUNT
    ERROR_COUNT += 1
    ERROR_ITEMS.append(message)
    print(f"[ERROR] {message}")


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# BASIC PROJECT STRUCTURE
# ============================================================

section("1. PROJECT STRUCTURE")

report_pass(f"Project root: {PROJECT_ROOT}")

required_files = [
    "app.py",
    "feature_engineering.py",
    "anomaly_detection.py",
    "risk_engine.py",
    "maintenance_recommendation.py",
    "check_dataset.py",
    "analyze_dataset.py",
    "requirements.txt",
    "README.md",
]

for filename in required_files:
    path = PROJECT_ROOT / filename

    if path.exists():
        size_kb = path.stat().st_size / 1024
        report_pass(f"{filename} exists ({size_kb:.1f} KB)")
    else:
        report_error(f"{filename} is missing")

for directory in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    if directory.exists():
        report_pass(f"Directory exists: {directory.name}/")
    else:
        report_warning(f"Directory missing: {directory.name}/")


# ============================================================
# DATASET CHECK
# ============================================================

section("2. SOURCE DATASET CHECK")

dataset_files = list(DATA_DIR.glob("*.csv"))

if not dataset_files:
    report_error("No CSV dataset found inside data/")
else:
    for dataset in dataset_files:
        try:
            df = pd.read_csv(dataset)

            report_pass(
                f"Dataset loaded: {dataset.name} | "
                f"Rows={len(df):,} | Columns={len(df.columns)}"
            )

            missing_total = int(df.isna().sum().sum())
            duplicate_total = int(df.duplicated().sum())

            if missing_total == 0:
                report_pass(f"{dataset.name}: no missing values")
            else:
                report_warning(
                    f"{dataset.name}: {missing_total:,} missing values"
                )

            if duplicate_total == 0:
                report_pass(f"{dataset.name}: no duplicate rows")
            else:
                report_warning(
                    f"{dataset.name}: {duplicate_total:,} duplicate rows"
                )

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_columns = df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            report_pass(
                f"{dataset.name}: "
                f"{len(numeric_columns)} numeric + "
                f"{len(categorical_columns)} non-numeric columns"
            )

            for col in numeric_columns:
                invalid = int(
                    (~np.isfinite(df[col].astype(float))).sum()
                )

                if invalid > 0:
                    report_warning(
                        f"{dataset.name}: {col} has "
                        f"{invalid:,} NaN/Inf values"
                    )

        except Exception as exc:
            report_error(
                f"Dataset failed to load: {dataset.name} -> {exc}"
            )


# ============================================================
# OUTPUT CSV CHECK
# ============================================================

section("3. OUTPUT CSV CHECK")

for filename, required_columns in REQUIRED_OUTPUTS.items():

    path = OUTPUT_DIR / filename

    if not path.exists():
        report_error(f"Missing output: outputs/{filename}")
        continue

    try:
        df = pd.read_csv(path)

        size_mb = path.stat().st_size / (1024 * 1024)

        report_pass(
            f"{filename}: loaded | "
            f"Rows={len(df):,} | Columns={len(df.columns)} | "
            f"Size={size_mb:.1f} MB"
        )

        if len(df) == 0:
            report_warning(f"{filename}: CSV is EMPTY")

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            report_error(
                f"{filename}: missing required columns -> "
                f"{missing_columns}"
            )
        else:
            if required_columns:
                report_pass(
                    f"{filename}: required columns present"
                )

        missing_total = int(df.isna().sum().sum())

        if missing_total == 0:
            report_pass(
                f"{filename}: no missing values"
            )
        else:
            report_warning(
                f"{filename}: {missing_total:,} missing values"
            )

        duplicate_total = int(df.duplicated().sum())

        if duplicate_total > 0:
            report_warning(
                f"{filename}: {duplicate_total:,} duplicate rows"
            )

    except Exception as exc:
        report_error(
            f"{filename}: FAILED TO READ -> {exc}"
        )


# ============================================================
# GRAPH DATA-TYPE CHECK
# ============================================================

section("4. GRAPH / CHART DATA-TYPE CHECK")

graph_files = [
    "engineered_dataset.csv",
    "anomaly_detection_results.csv",
    "risk_engine_results.csv",
    "maintenance_recommendation_results.csv",
]

for filename in graph_files:

    path = OUTPUT_DIR / filename

    if not path.exists():
        continue

    try:
        df = pd.read_csv(path)

        report_pass(
            f"Graph inspection loaded: {filename}"
        )

        problematic_columns = []

        for col in df.columns:

            dtype = str(df[col].dtype)

            if "interval" in dtype.lower():
                problematic_columns.append(
                    f"{col}=Interval"
                )

            if df[col].dtype == "object":
                unique_sample = df[col].dropna().head(100)

                if any(
                    "Interval" in type(value).__name__
                    for value in unique_sample
                ):
                    problematic_columns.append(
                        f"{col}=Interval objects"
                    )

            if pd.api.types.is_numeric_dtype(df[col]):

                try:
                    invalid = int(
                        (~np.isfinite(
                            pd.to_numeric(
                                df[col],
                                errors="coerce"
                            )
                        )).sum()
                    )

                    if invalid > 0:
                        problematic_columns.append(
                            f"{col}=NaN/Inf"
                        )

                except Exception:
                    pass

        if problematic_columns:
            report_warning(
                f"{filename}: possible chart issues -> "
                f"{problematic_columns[:20]}"
            )
        else:
            report_pass(
                f"{filename}: no obvious chart datatype problems"
            )

    except Exception as exc:
        report_error(
            f"Graph inspection failed for {filename}: {exc}"
        )


# ============================================================
# IMPORTANT COLUMN CHECK
# ============================================================

section("5. DASHBOARD COLUMN AVAILABILITY")

dashboard_columns = [
    "Machine_ID",
    "Date",
    "Timestamp",
    "Operation_Mode",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Error_Rate_%",
    "Efficiency_Status",
    "risk_score",
    "sensor_risk",
    "network_risk",
    "quality_risk",
    "historical_risk",
    "anomaly_score",
    "is_anomaly",
    "anomaly_label",
    "anomaly_severity",
    "anomaly_domain",
    "maintenance_priority",
    "recommended_action",
    "root_cause",
]


combined_sources = []

for filename in [
    "engineered_dataset.csv",
    "anomaly_detection_results.csv",
    "risk_engine_results.csv",
    "maintenance_recommendation_results.csv",
]:

    path = OUTPUT_DIR / filename

    if path.exists():
        try:
            temp = pd.read_csv(path, nrows=5)
            combined_sources.append(temp)
        except Exception:
            pass

if combined_sources:

    combined_columns = set()

    for df in combined_sources:
        combined_columns.update(df.columns.tolist())

    for col in dashboard_columns:

        if col in combined_columns:
            report_pass(f"Dashboard column available: {col}")
        else:
            report_warning(
                f"Dashboard column NOT found anywhere: {col}"
            )

else:
    report_error(
        "No usable output CSV available for dashboard column check"
    )


# ============================================================
# VALUE DISTRIBUTION CHECK
# ============================================================

section("6. IMPORTANT VALUE DISTRIBUTION CHECK")


def check_value_distribution(filename, column):

    path = OUTPUT_DIR / filename

    if not path.exists():
        return

    try:
        df = pd.read_csv(path, usecols=[column])

        counts = df[column].value_counts(dropna=False)

        if len(counts) == 0:
            report_warning(
                f"{filename}: {column} has no values"
            )
            return

        preview = counts.head(10).to_dict()

        report_pass(
            f"{filename}: {column} -> {preview}"
        )

    except ValueError:
        report_warning(
            f"{filename}: column not found -> {column}"
        )

    except Exception as exc:
        report_error(
            f"{filename}: distribution check failed "
            f"for {column} -> {exc}"
        )


check_value_distribution(
    "anomaly_detection_results.csv",
    "is_anomaly"
)

check_value_distribution(
    "anomaly_detection_results.csv",
    "anomaly_severity"
)

check_value_distribution(
    "anomaly_detection_results.csv",
    "anomaly_domain"
)

check_value_distribution(
    "risk_engine_results.csv",
    "risk_level"
)

check_value_distribution(
    "maintenance_recommendation_results.csv",
    "maintenance_priority"
)

check_value_distribution(
    "maintenance_recommendation_results.csv",
    "root_cause"
)

check_value_distribution(
    "engineered_dataset.csv",
    "Efficiency_Status"
)

check_value_distribution(
    "engineered_dataset.csv",
    "Operation_Mode"
)


# ============================================================
# MACHINE CONSISTENCY CHECK
# ============================================================

section("7. MACHINE CONSISTENCY CHECK")

machine_sets = {}

for filename in [
    "engineered_dataset.csv",
    "anomaly_detection_results.csv",
    "risk_engine_results.csv",
    "maintenance_recommendation_results.csv",
    "machine_maintenance_summary.csv",
    "maintenance_priority_queue.csv",
]:

    path = OUTPUT_DIR / filename

    if not path.exists():
        continue

    try:
        df = pd.read_csv(
            path,
            usecols=["Machine_ID"]
        )

        machines = set(
            df["Machine_ID"]
            .dropna()
            .astype(str)
            .unique()
        )

        machine_sets[filename] = machines

        report_pass(
            f"{filename}: {len(machines)} unique machines"
        )

    except Exception as exc:
        report_warning(
            f"{filename}: Machine_ID check failed -> {exc}"
        )


if machine_sets:

    reference_name = next(iter(machine_sets))
    reference_set = machine_sets[reference_name]

    for filename, machines in machine_sets.items():

        missing_from_file = reference_set - machines
        extra_in_file = machines - reference_set

        if not missing_from_file and not extra_in_file:
            report_pass(
                f"Machine consistency: {filename} matches reference"
            )
        else:

            if missing_from_file:
                report_warning(
                    f"{filename}: missing machines -> "
                    f"{sorted(list(missing_from_file))[:10]}"
                )

            if extra_in_file:
                report_warning(
                    f"{filename}: extra machines -> "
                    f"{sorted(list(extra_in_file))[:10]}"
                )


# ============================================================
# APP.PY SYNTAX CHECK
# ============================================================

section("8. APP.PY SYNTAX CHECK")

if APP_FILE.exists():

    try:
        source = APP_FILE.read_text(
            encoding="utf-8",
            errors="replace"
        )

        ast.parse(source)

        report_pass(
            "app.py AST syntax check passed"
        )

    except SyntaxError as exc:

        report_error(
            f"app.py syntax error at line "
            f"{exc.lineno}: {exc.msg}"
        )

    except Exception as exc:

        report_error(
            f"app.py syntax inspection failed -> {exc}"
        )

else:
    report_error("app.py does not exist")


# ============================================================
# PYTHON MODULE SYNTAX CHECK
# ============================================================

section("9. PYTHON MODULE SYNTAX CHECK")

python_files = [
    "app.py",
    "feature_engineering.py",
    "anomaly_detection.py",
    "risk_engine.py",
    "maintenance_recommendation.py",
    "check_dataset.py",
    "analyze_dataset.py",
]

for filename in python_files:

    path = PROJECT_ROOT / filename

    if not path.exists():
        continue

    try:

        source = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        ast.parse(source)

        report_pass(
            f"{filename}: syntax OK"
        )

    except SyntaxError as exc:

        report_error(
            f"{filename}: syntax error at line "
            f"{exc.lineno}: {exc.msg}"
        )

    except Exception as exc:

        report_error(
            f"{filename}: syntax inspection failed -> {exc}"
        )


# ============================================================
# IMPORT AVAILABILITY CHECK
# ============================================================

section("10. PYTHON PACKAGE CHECK")

packages = [
    "streamlit",
    "pandas",
    "numpy",
    "sklearn",
    "plotly",
    "openpyxl",
]

for package in packages:

    try:

        module = importlib.import_module(package)

        version = getattr(
            module,
            "__version__",
            "version unavailable"
        )

        report_pass(
            f"{package}: available ({version})"
        )

    except Exception as exc:

        report_error(
            f"{package}: import failed -> {exc}"
        )


# ============================================================
# APP IMPORT SCAN
# ============================================================

section("11. APP IMPORT / REFERENCE SCAN")

if APP_FILE.exists():

    try:

        source = APP_FILE.read_text(
            encoding="utf-8",
            errors="replace"
        )

        tree = ast.parse(source)

        imported_names = set()
        imported_modules = set()

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])

                    if alias.asname:
                        imported_names.add(alias.asname)
                    else:
                        imported_names.add(
                            alias.name.split(".")[0]
                        )

            elif isinstance(node, ast.ImportFrom):

                if node.module:
                    imported_modules.add(
                        node.module.split(".")[0]
                    )

                for alias in node.names:

                    if alias.asname:
                        imported_names.add(alias.asname)
                    else:
                        imported_names.add(alias.name)

        report_pass(
            f"app.py imports detected: "
            f"{len(imported_modules)} modules"
        )

        for module in sorted(imported_modules):

            if module in {
                "os",
                "sys",
                "json",
                "math",
                "time",
                "datetime",
                "pathlib",
                "warnings",
                "traceback",
                "typing",
                "re",
                "csv",
                "io",
                "base64",
                "statistics",
                "collections",
                "itertools",
                "functools",
                "numpy",
                "pandas",
                "plotly",
                "streamlit",
                "sklearn",
                "openpyxl",
            }:
                continue

            module_path = PROJECT_ROOT / f"{module}.py"

            if module_path.exists():
                report_pass(
                    f"Local imported module exists: {module}.py"
                )
            else:
                report_warning(
                    f"Possible missing local module: {module}"
                )

    except Exception as exc:

        report_error(
            f"App import scan failed -> {exc}"
        )


# ============================================================
# COMMON STREAMLIT CHART RISK SCAN
# ============================================================

section("12. STREAMLIT CHART RISK SCAN")

if APP_FILE.exists():

    try:

        source = APP_FILE.read_text(
            encoding="utf-8",
            errors="replace"
        )

        chart_calls = [
            "st.bar_chart",
            "st.line_chart",
            "st.area_chart",
            "st.scatter_chart",
            "st.plotly_chart",
            "px.bar",
            "px.line",
            "px.scatter",
            "px.pie",
        ]

        chart_counts = {}

        for chart in chart_calls:

            count = source.count(chart)
            chart_counts[chart] = count

            if count > 0:
                report_pass(
                    f"{chart}: {count} occurrence(s)"
                )

        if "pd.cut(" in source:
            report_warning(
                "app.py uses pd.cut(); categorical Interval "
                "objects can break native Streamlit charts "
                "unless converted to strings"
            )

        if "use_container_width" in source:
            report_warning(
                "app.py still uses deprecated "
                "use_container_width parameter"
            )

        if "plotly.express as px" in source:
            report_pass(
                "Plotly Express import detected"
            )

        elif "px." in source:
            report_error(
                "px.* usage detected but "
                "plotly.express import was not found"
            )

    except Exception as exc:

        report_error(
            f"Streamlit chart scan failed -> {exc}"
        )


# ============================================================
# APP FILE SIZE / PERFORMANCE CHECK
# ============================================================

section("13. PERFORMANCE / FILE SIZE CHECK")

if APP_FILE.exists():

    app_size_mb = APP_FILE.stat().st_size / (
        1024 * 1024
    )

    report_pass(
        f"app.py size: {app_size_mb:.2f} MB"
    )

    if app_size_mb > 1:
        report_warning(
            "app.py is larger than 1 MB; "
            "performance/maintenance complexity may be high"
        )


large_files = []

for directory in [
    DATA_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
]:

    if not directory.exists():
        continue

    for path in directory.rglob("*"):

        if path.is_file():

            size_mb = path.stat().st_size / (
                1024 * 1024
            )

            if size_mb >= 50:

                large_files.append(
                    (path.relative_to(PROJECT_ROOT), size_mb)
                )

if large_files:

    for relative_path, size_mb in sorted(
        large_files,
        key=lambda x: x[1],
        reverse=True
    ):

        report_warning(
            f"Large file: {relative_path} "
            f"({size_mb:.1f} MB)"
        )

else:

    report_pass(
        "No files larger than 50 MB detected"
    )


# ============================================================
# OUTPUT PIPELINE CHECK
# ============================================================

section("14. PIPELINE OUTPUT CHECK")

pipeline = [
    (
        "Feature Engineering",
        "engineered_dataset.csv"
    ),
    (
        "Anomaly Detection",
        "anomaly_detection_results.csv"
    ),
    (
        "Risk Engine",
        "risk_engine_results.csv"
    ),
    (
        "Maintenance Recommendation",
        "maintenance_recommendation_results.csv"
    ),
    (
        "Machine Maintenance Summary",
        "machine_maintenance_summary.csv"
    ),
    (
        "Maintenance Priority Queue",
        "maintenance_priority_queue.csv"
    ),
    (
        "Fleet Maintenance Summary",
        "fleet_maintenance_summary.csv"
    ),
]

for stage, filename in pipeline:

    path = OUTPUT_DIR / filename

    if path.exists() and path.stat().st_size > 0:

        report_pass(
            f"{stage}: output available"
        )

    else:

        report_error(
            f"{stage}: output missing or empty"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

section("15. FINAL DIAGNOSTIC SUMMARY")

print()
print(f"PASS     : {PASS_COUNT}")
print(f"WARNING  : {WARNING_COUNT}")
print(f"ERROR    : {ERROR_COUNT}")

print()

if ERROR_COUNT == 0 and WARNING_COUNT == 0:

    print("OVERALL STATUS: EXCELLENT")
    print("No diagnostic problems were detected.")

elif ERROR_COUNT == 0:

    print("OVERALL STATUS: FUNCTIONAL WITH WARNINGS")
    print(
        "The project appears functional, but "
        "some warnings should be reviewed."
    )

else:

    print("OVERALL STATUS: ACTION REQUIRED")
    print(
        "One or more real errors were detected. "
        "Fix these before final dashboard testing."
    )


# ============================================================
# ERROR LIST
# ============================================================

if ERROR_ITEMS:

    print()
    print("=" * 72)
    print("ERRORS REQUIRING FIX")
    print("=" * 72)

    for index, item in enumerate(
        ERROR_ITEMS,
        start=1
    ):
        print(f"{index}. {item}")


# ============================================================
# WARNING LIST
# ============================================================

if WARNING_ITEMS:

    print()
    print("=" * 72)
    print("WARNINGS TO REVIEW")
    print("=" * 72)

    for index, item in enumerate(
        WARNING_ITEMS,
        start=1
    ):
        print(f"{index}. {item}")


# ============================================================
# END
# ============================================================

print()
print("=" * 72)
print("DIAGNOSTIC COMPLETE")
print("=" * 72)
print()
print(
    "Please copy the COMPLETE terminal output "
    "and send it to ChatGPT."
)
print()