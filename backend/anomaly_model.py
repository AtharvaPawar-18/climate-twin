import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "data/processed/conduit_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("Data loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# 2. PREPARE TIMESTAMP
# ==========================================

df["ts"] = pd.to_datetime(df["ts"])


# ==========================================
# 3. SELECT ENVIRONMENTAL FEATURES
# ==========================================

features = [
    "temp_bmx",
    "press_bmx",
    "temp_mcp",
    "temp_sht",
    "humidity_sht",
    "si1145_vis",
    "si1145_ir",
    "wind_spd",
    "wind_dir",
    "wind_gust",
    "heat_idx",
    "wet_bulb_temp",
    "wet_bulb_globe_temp"
]

X = df[features].copy()


# ==========================================
# 4. CHECK DATA
# ==========================================

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nMissing values:")
print(X.isnull().sum().sum())


# ==========================================
# 5. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# 6. TRAIN ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)


# ==========================================
# 7. GENERATE ANOMALY RESULTS
# ==========================================

df["anomaly_prediction"] = model.predict(X_scaled)

df["anomaly_score"] = model.decision_function(X_scaled)


# Isolation Forest:
# -1 = anomaly
#  1 = normal

df["is_anomaly"] = (
    df["anomaly_prediction"] == -1
)


# ==========================================
# 8. CREATE RISK LEVEL
# ==========================================

def classify_risk(score):

    if score < -0.15:
        return "CRITICAL"

    elif score < -0.05:
        return "WARNING"

    else:
        return "NORMAL"


df["risk_level"] = df["anomaly_score"].apply(
    classify_risk
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n========================================")
print("ANOMALY DETECTION RESULTS")
print("========================================")

print("\nTotal observations:", len(df))

print(
    "Anomalies detected:",
    df["is_anomaly"].sum()
)

print(
    "Normal observations:",
    (~df["is_anomaly"]).sum()
)

print("\nRisk distribution:")

print(
    df["risk_level"].value_counts()
)


# ==========================================
# 10. SHOW MOST EXTREME ANOMALIES
# ==========================================

print("\n========================================")
print("TOP 10 MOST UNUSUAL OBSERVATIONS")
print("========================================")

top_anomalies = df.sort_values(
    "anomaly_score"
).head(10)

print(
    top_anomalies[
        [
            "ts",
            "temp_sht",
            "humidity_sht",
            "wind_spd",
            "wind_gust",
            "heat_idx",
            "anomaly_score",
            "risk_level"
        ]
    ].to_string(index=False)
)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

output_path = (
    "data/processed/"
    "conduit_anomaly_results.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 12. SAVE MODEL + SCALER
# ==========================================

joblib.dump(
    model,
    "backend/isolation_forest.joblib"
)

joblib.dump(
    scaler,
    "backend/scaler.joblib"
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    "Results:",
    output_path
)

print(
    "Model:",
    "backend/isolation_forest.joblib"
)

print(
    "Scaler:",
    "backend/scaler.joblib"
)