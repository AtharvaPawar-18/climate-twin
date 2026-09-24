import pandas as pd

# Load Conduit data
df = pd.read_csv("data/raw/conduit_data.csv")

print("=" * 60)
print("CLIMATETWIN AI - DATA ANALYSIS")
print("=" * 60)

# 1. Basic information
print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. COLUMNS")
print(df.columns.tolist())

# 2. Data types
print("\n3. DATA TYPES")
print(df.dtypes)

# 3. Missing values
print("\n4. MISSING VALUES")
missing = df.isnull().sum()
print(missing)
print("\nTotal missing values:", missing.sum())

# 4. Duplicates
print("\n5. DUPLICATE ROWS")
print("Duplicate rows:", df.duplicated().sum())

# 5. Timestamp analysis
print("\n6. TIMESTAMP ANALYSIS")

df["ts"] = pd.to_datetime(df["ts"])

print("First timestamp:")
print(df["ts"].min())

print("Last timestamp:")
print(df["ts"].max())

time_diff = df["ts"].diff().dropna()

print("\nMost common sampling intervals:")
print(time_diff.value_counts().head(10))

# 6. Convert sensor columns to numeric
print("\n7. CONVERTING SENSOR COLUMNS TO NUMERIC")

sensor_columns = [
    column for column in df.columns
    if column != "ts"
]

for column in sensor_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("Conversion complete.")

# 7. Statistical summary
print("\n8. STATISTICAL SUMMARY")
print(df[sensor_columns].describe().T)

# 8. Constant columns
print("\n9. CONSTANT COLUMNS")

constant_columns = []

for column in sensor_columns:
    if df[column].nunique() <= 1:
        constant_columns.append(column)

if constant_columns:
    for column in constant_columns:
        print(
            f"{column} -> constant "
            f"(unique values: {df[column].nunique()})"
        )
else:
    print("No constant columns found.")

# 9. Unique values
print("\n10. NUMBER OF UNIQUE VALUES")

for column in sensor_columns:
    print(
        f"{column}: "
        f"{df[column].nunique()} unique values"
    )

# 10. Correlation
print("\n11. CORRELATION MATRIX")

correlation = df[sensor_columns].corr()

print(correlation.round(2))

# 11. Save cleaned dataset
output_path = "data/processed/conduit_cleaned.csv"

df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"\nCleaned dataset saved to:\n{output_path}"
)