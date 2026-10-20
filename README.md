# Predictive Maintenance and Anomaly Detection in 6G-Integrated Smart Manufacturing Systems

An intelligent manufacturing analytics project designed to analyze machine operating conditions, identify abnormal behavior, assess maintenance risk, and support maintenance planning through data analytics, machine learning, and an interactive Streamlit dashboard.

## Project Overview

Unexpected machine downtime can interrupt production, reduce operational efficiency, and increase maintenance costs.

This project explores how machine sensor readings, production indicators, quality metrics, and network performance measurements can be analyzed together to support predictive maintenance decisions in a connected manufacturing environment.

The system combines feature engineering, anomaly detection, maintenance risk assessment, maintenance recommendations, and interactive analytics in one application.

**Important:** The application provides risk assessment and maintenance decision support. It does not guarantee the exact date or time when a machine will fail.

## Problem Statement

Traditional maintenance approaches may rely on fixed schedules or respond only after a machine develops a problem.

An analytics-driven approach can help identify unusual operating conditions, monitor changing machine behavior, prioritize inspections, and provide more structured maintenance recommendations.

This project addresses the following questions:

* Which machines or observations require attention?
* What unusual sensor or operating conditions are present?
* How do temperature, vibration, power consumption, and production quality contribute to operational risk?
* How can network latency and packet loss be included in manufacturing-system monitoring?
* Which maintenance actions should receive higher priority?
* How can these findings be communicated through an interactive dashboard?

## Project Objectives

1. Inspect and validate the manufacturing dataset.
2. Analyze machine behavior, operating conditions, and production indicators.
3. Engineer features representing current conditions and historical behavior.
4. Identify unusual observations using machine learning.
5. Assess maintenance risk and categorize observations by risk level.
6. Generate maintenance recommendations and priority rankings.
7. Examine network-related indicators, including latency and packet loss.
8. Present key findings through an interactive Streamlit dashboard.
9. Export analytical results and maintenance reports for further review.

## Key Features

### 1. Dataset Inspection and Validation

* Dataset structure and column inspection.
* Data-type verification.
* Missing-value and duplicate checks.
* Numerical and categorical feature identification.
* Statistical summaries.

### 2. Exploratory Data Analysis

* Machine operating-condition analysis.
* Maintenance-score distribution.
* Machine-level and fleet-level comparisons.
* Production efficiency and quality analysis.
* Network performance analysis.
* Correlation and trend exploration.

### 3. Feature Engineering

* Temporal features derived from date and timestamp.
* Cyclical time representations.
* Machine operating-mode indicators.
* Normalized sensor and network measurements.
* Sensor stress and network issue indicators.
* Previous-observation values and change calculations.
* Rolling statistics for historical behavior.
* Machine-health and preliminary-risk indicators.

### 4. Machine Learning-Based Anomaly Detection

* Isolation Forest-based anomaly detection.
* Identification of unusual operating observations.
* Anomaly severity categorization.
* Machine-level anomaly summaries.

### 5. Maintenance Risk Assessment

* Maintenance risk categorization.
* Sensor, network, quality, and historical indicators.
* Machine-level risk summaries.
* Risk-based prioritization.

### 6. Maintenance Recommendations

* Action-oriented maintenance recommendations.
* Priority queue generation.
* Machine-level maintenance summaries.
* Fleet-level maintenance reporting.

### 7. Network and 6G-Oriented Analytics

* Network latency analysis.
* Packet-loss analysis.
* Network-health indicators.
* Integration of network-related measurements into manufacturing risk analysis.

The 6G aspect represents the project's connected-manufacturing and network-performance analysis perspective. The presence of network indicators alone does not establish that the application has been deployed on an actual 6G network.

### 8. Interactive Dashboard and Reporting

* Streamlit-based analytical interface.
* Machine and operational insights.
* Risk and anomaly summaries.
* Maintenance planning information.
* Exportable analytical results, where supported by the application.

## Technology Stack

* **Programming Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Libraries used by the application, such as Matplotlib or Plotly, where configured
* **Machine Learning:** Scikit-learn
* **Dashboard:** Streamlit
* **Data Storage:** CSV files
* **Model Storage:** Pickle and JSON files, where applicable

The exact installed dependencies are defined in `requirements.txt`.

## Dataset Information

The project dataset is located at:

`data/Thales_Group_Manufacturing.csv`

The dataset analyzed during development contains:

* 100,000 records
* 50 machine identifiers
* 14 original columns
* Observations covering January 1, 2025, through March 10, 2025

The original columns include:

* `Date`
* `Timestamp`
* `Machine_ID`
* `Operation_Mode`
* `Temperature_C`
* `Vibration_Hz`
* `Power_Consumption_kW`
* `Network_Latency_ms`
* `Packet_Loss_%`
* `Quality_Control_Defect_Rate_%`
* `Production_Speed_units_per_hr`
* `Predictive_Maintenance_Score`
* `Error_Rate_%`
* `Efficiency_Status`

The dataset contains sensor, production, quality, operating-mode, and network-related measurements.

**Dataset note:** These figures describe the dataset used during development. Before redistribution, verify the dataset's source, ownership, licensing, and permission for public sharing. Do not imply that this dataset or project is officially endorsed by Thales Group unless such endorsement has been explicitly granted.

## Project Architecture

The main analytical workflow is organized into separate modules.

1. `check_dataset.py` — inspects and validates the input dataset.
2. `analyze_dataset.py` — explores statistical patterns and machine behavior.
3. `feature_engineering.py` — creates derived features from operating and historical data.
4. `anomaly_detection.py` — identifies unusual observations using the configured anomaly-detection approach.
5. `risk_engine.py` — calculates and summarizes maintenance risk.
6. `maintenance_recommendation.py` — produces maintenance recommendations and priority information.
7. `app.py` — provides the interactive Streamlit application.

Generated datasets, summaries, and model artifacts are stored in the `outputs/` and `models/` directories, as applicable.

## Installation and Setup

### Prerequisites

* Python installed on your system.
* Git, if you want to clone the repository.
* A terminal such as PowerShell, Command Prompt, or a VS Code terminal.

### 1. Clone the Repository

Replace `YOUR-USERNAME` with the GitHub username that owns this repository.

```bash
git clone https://github.com/YOUR-USERNAME/Predictive-Maintenance-6G.git
```

### 2. Enter the Project Directory

```bash
cd Predictive-Maintenance-6G
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly instead.

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If dependency installation fails, check the Python version and the compatibility of the packages listed in `requirements.txt`.

## Running the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit should display a local application address in the terminal. Open that address in your browser.

The application may depend on the dataset, saved model, or generated result files. Ensure that all required files are present and that the paths configured in the code match the repository structure.

## Running Individual Modules

Run the following commands from the project root when you want to execute the corresponding analytical modules individually:

```bash
python check_dataset.py
```

```bash
python analyze_dataset.py
```

```bash
python feature_engineering.py
```

```bash
python anomaly_detection.py
```

```bash
python risk_engine.py
```

```bash
python maintenance_recommendation.py
```

Run each module only after confirming its input files, dependencies, and expected execution order. The Streamlit application may use existing outputs rather than automatically executing every module.

## Project Outputs

Depending on the modules executed, the project can generate artifacts such as:

* Engineered dataset.
* Anomaly-detection results.
* Risk-engine results.
* Maintenance recommendations.
* Machine-level summaries.
* Fleet-level maintenance summaries.
* Maintenance priority queues.
* Exportable reports.

Generated files are stored under `outputs/`, where configured.

Large intermediate files do not necessarily need to be committed to GitHub. The README, source code, dependency list, required model artifacts, and instructions for reproducing the results should remain clear and consistent.

## Limitations

* Maintenance risk categories are analytical indicators and should not automatically be interpreted as calibrated probabilities of machine failure.
* Anomaly detection identifies unusual patterns; an anomaly does not always mean a machine is faulty.
* Recommendations depend on the available measurements, feature-engineering logic, configured thresholds, and model behavior.
* Historical patterns may not generalize to other machines, factories, or operating environments.
* Network latency and packet loss are analyzed as available dataset features; the project does not by itself demonstrate deployment on a real 6G network.
* Production use would require validation against real maintenance records, failure events, domain-expert feedback, and operational requirements.

## Future Improvements

* Validate risk predictions against confirmed machine failures and maintenance logs.
* Add time-based model validation and performance monitoring.
* Evaluate anomaly-detection precision and false-positive rates.
* Add automated data-quality and model-health monitoring.
* Improve maintenance-cost and downtime-impact estimation.
* Integrate real-time sensor streams where available.
* Add role-based access, deployment configuration, and automated tests where required.

## Author

**Narendra Suthar**

Project developed as a portfolio project in predictive maintenance, machine learning, and smart-manufacturing analytics.

## Disclaimer

This is an analytical and educational project. It is not a certified industrial safety system and should not be used as the sole basis for critical maintenance or safety decisions.
