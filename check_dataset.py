import pandas as pd
from pathlib import Path

# ============================================
# DATASET INSPECTION
# ============================================

DATA_PATH = Path("data") / "Thales_Group_Manufacturing.csv"

print("=" * 70)
print("PREDICTIVE MAINTENANCE - DATASET INSPECTION")
print("=" * 70)

# --------------------------------------------
# 1. Check file
# --------------------------------------------

if not DATA_PATH.exists():
    print("\nERROR: Dataset file not found!")
    print(f"Expected path: {DATA_PATH}")
    raise SystemExit

print(f"\nDataset found: {DATA_PATH}")

# --------------------------------------------
# 2. Load dataset
# --------------------------------------------

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print("\nERROR while reading dataset:")
    print(e)
    raise SystemExit

# --------------------------------------------
# 3. Basic information
# --------------------------------------------

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")

# --------------------------------------------
# 4. Column names
# --------------------------------------------

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

# --------------------------------------------
# 5. Data types
# --------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes.to_string())

# --------------------------------------------
# 6. Missing values
# --------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_percentage.round(2)
})

missing_table = missing_table[
    missing_table["Missing_Count"] > 0
].sort_values(
    "Missing_Count",
    ascending=False
)

if missing_table.empty:
    print("No missing values found.")
else:
    print(missing_table.to_string())

# --------------------------------------------
# 7. Duplicate rows
# --------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")

# --------------------------------------------
# 8. Numerical columns
# --------------------------------------------

numerical_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

print("\n" + "=" * 70)
print("NUMERICAL COLUMNS")
print("=" * 70)

for column in numerical_columns:
    print(f"- {column}")

print(f"\nTotal numerical columns: {len(numerical_columns)}")

# --------------------------------------------
# 9. Categorical columns
# --------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\n" + "=" * 70)
print("CATEGORICAL COLUMNS")
print("=" * 70)

for column in categorical_columns:
    print(f"- {column}")

print(f"\nTotal categorical columns: {len(categorical_columns)}")

# --------------------------------------------
# 10. Unique values
# --------------------------------------------

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:
    unique_count = df[column].nunique(dropna=False)

    print(
        f"{column:<40} "
        f"{unique_count:>10,} unique values"
    )

# --------------------------------------------
# 11. First 5 rows
# --------------------------------------------

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head().to_string())

# --------------------------------------------
# 12. Statistical summary
# --------------------------------------------

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe(include="all").transpose().to_string())

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)