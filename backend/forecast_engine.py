import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "conduit_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "forecast_models")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df = df.sort_values("ts")

target_columns = [
    "temp_sht",
    "humidity_sht",
    "heat_idx",
    "wind_spd"
]

df = df[["ts"] + target_columns].copy()

for col in target_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.set_index("ts").resample("15min").mean()

df[target_columns] = df[target_columns].interpolate(method="time")
df = df.dropna()

for col in target_columns:
    for lag in [1, 2, 3, 4, 8, 12]:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

for col in target_columns:
    df[f"{col}_roll_4"] = df[col].rolling(4).mean()
    df[f"{col}_roll_8"] = df[col].rolling(8).mean()

df["hour"] = df.index.hour
df["minute"] = df.index.minute
df["day_of_week"] = df.index.dayofweek

df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] * 60 + df["minute"]) / 1440)
df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] * 60 + df["minute"]) / 1440)

horizon = 4

for col in target_columns:
    df[f"{col}_target"] = df[col].shift(-horizon)

df = df.dropna()

feature_columns = [
    col for col in df.columns
    if not col.endswith("_target")
]

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[feature_columns]
X_test = test[feature_columns]

results = {}

for target in target_columns:
    y_train = train[f"{target}_target"]
    y_test = test[f"{target}_target"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[target] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, f"{target}_forecast.joblib")
    )

print("\nForecast Model Results\n")

for target, metrics in results.items():
    print(
        f"{target}: "
        f"MAE={metrics['MAE']:.4f}, "
        f"RMSE={metrics['RMSE']:.4f}, "
        f"R2={metrics['R2']:.4f}"
    )

print("\nModels saved to:", MODEL_DIR)
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Forecast horizon: 1 hour")