import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "conduit_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "forecast_models")

TARGET_COLUMNS = [
    "temp_sht",
    "humidity_sht",
    "heat_idx",
    "wind_spd"
]

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")

    df = df[["ts"] + TARGET_COLUMNS].copy()

    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("ts").resample("15min").mean()
    df[TARGET_COLUMNS] = df[TARGET_COLUMNS].interpolate(method="time")
    return df.dropna()

def create_features(df):
    data = df.copy()

    for col in TARGET_COLUMNS:
        for lag in [1, 2, 3, 4, 8, 12]:
            data[f"{col}_lag_{lag}"] = data[col].shift(lag)

    for col in TARGET_COLUMNS:
        data[f"{col}_roll_4"] = data[col].rolling(4).mean()
        data[f"{col}_roll_8"] = data[col].rolling(8).mean()

    data["hour"] = data.index.hour
    data["minute"] = data.index.minute
    data["day_of_week"] = data.index.dayofweek

    data["hour_sin"] = np.sin(
        2 * np.pi * (data["hour"] * 60 + data["minute"]) / 1440
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * (data["hour"] * 60 + data["minute"]) / 1440
    )

    return data

def load_models():
    return {
        target: joblib.load(
            os.path.join(MODEL_DIR, f"{target}_forecast.joblib")
        )
        for target in TARGET_COLUMNS
    }

def predict_next_hour():
    df = load_data()
    features = create_features(df).dropna()

    latest = features.iloc[[-1]]

    model_input = latest.copy()

    models = load_models()

    predictions = {}

    for target, model in models.items():
        predictions[target] = float(model.predict(model_input)[0])

    current = {
        target: float(df.iloc[-1][target])
        for target in TARGET_COLUMNS
    }

    timestamp = df.index[-1]

    return {
        "timestamp": timestamp,
        "current": current,
        "predicted": predictions
    }

if __name__ == "__main__":
    result = predict_next_hour()

    print("\nCurrent Conditions")
    for key, value in result["current"].items():
        print(f"{key}: {value:.2f}")

    print("\nPredicted Conditions (+1 hour)")
    for key, value in result["predicted"].items():
        print(f"{key}: {value:.2f}")

    print("\nPrediction generated from:", result["timestamp"])