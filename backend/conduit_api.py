import pandas as pd

# Load Conduit data
df = pd.read_csv("data/raw/conduit_data.csv")

print("=" * 60)
print("CLIMATETWIN AI - DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. BASIC INFORMATION
# --------------------------------------------------

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. COLUMNS")
print(df.columns.tolist())

# --------------------------------------------------
# 2. DATA TYPES
# --------------------------------------------------

print("\n3. DATA TYPES")
print(df.dtypes)

# --------------------------------------------------
# 3. MISSING VALUES
# --------------------------------------------------

print("\n4. MISSING VALUES")

missing = df.isnull().sum()

print(missing)

print("\nTotal missing values:", missing.sum())

# --------------------------------------------------
# 4. DUPLICATES
# --------------------------------------------------

print("\n5. DUPLICATE ROWS")

print("Duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 5. TIMESTAMP
# --------------------------------------------------

print("\n6. TIMESTAMP ANALYSIS")

df["ts"] = pd.to_datetime(df["ts"])

print("First timestamp:")
print(df["ts"].min())

print("Last timestamp:")
print(df["ts"].max())

# Time difference between observations
time_diff = df["ts"].diff().dropna()

print("\nMost common sampling intervals:")

print(
    time_diff
    .value_counts()
    .head(10)
)

# --------------------------------------------------
# 6. NUMERIC CONVERSION
# --------------------------------------------------

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

# --------------------------------------------------
# 7. STATISTICAL SUMMARY
# --------------------------------------------------

print("\n8. STATISTICAL SUMMARY")

print(
    df[sensor_columns].describe().T
)

# --------------------------------------------------
# 8. CONSTANT COLUMNS
# --------------------------------------------------

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

# --------------------------------------------------
# 9. UNIQUE VALUES
# --------------------------------------------------

print("\n10. NUMBER OF UNIQUE VALUES")

for column in sensor_columns:

    print(
        f"{column}: "
        f"{df[column].nunique()} unique values"
    )

# --------------------------------------------------
# 10. CORRELATION
# --------------------------------------------------

print("\n11. CORRELATION MATRIX")

correlation = df[sensor_columns].corr()

print(correlation.round(2))

# --------------------------------------------------
# 11. SAVE CLEANED VERSION
# --------------------------------------------------

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