import os
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "conduit_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "forecast_models_multi")

TARGET_COLUMNS = [
    "temp_sht",
    "humidity_sht",
    "heat_idx",
    "wind_spd"
]

HORIZONS = {
    "1h": 4,
    "2h": 8,
    "3h": 12
}

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

@lru_cache(maxsize=1)
def load_models():
    models = {}

    for horizon in HORIZONS:
        models[horizon] = {}

        for target in TARGET_COLUMNS:
            path = os.path.join(
                MODEL_DIR,
                f"{target}_{horizon}.joblib"
            )

            models[horizon][target] = joblib.load(path)

    return models

def build_prediction(latest_features, models):
    forecasts = {}

    for horizon in HORIZONS:
        forecasts[horizon] = {}

        for target in TARGET_COLUMNS:
            prediction = models[horizon][target].predict(
                latest_features
            )[0]

            forecasts[horizon][target] = float(prediction)

    return forecasts

def predict_all():
    df = load_data()
    features = create_features(df).dropna()

    latest_features = features.iloc[[-1]]
    models = load_models()

    current = {
        target: float(df.iloc[-1][target])
        for target in TARGET_COLUMNS
    }

    forecasts = build_prediction(
        latest_features,
        models
    )

    return {
        "mode": "live",
        "timestamp": str(df.index[-1]),
        "current": current,
        "forecasts": forecasts
    }

def predict_custom(
    temperature,
    humidity,
    heat_index,
    wind_speed
):
    df = load_data()
    features = create_features(df).dropna()

    latest_features = features.iloc[[-1]].copy()

    latest_features["temp_sht"] = temperature
    latest_features["humidity_sht"] = humidity
    latest_features["heat_idx"] = heat_index
    latest_features["wind_spd"] = wind_speed

    models = load_models()

    forecasts = build_prediction(
        latest_features,
        models
    )

    current = {
        "temp_sht": float(temperature),
        "humidity_sht": float(humidity),
        "heat_idx": float(heat_index),
        "wind_spd": float(wind_speed)
    }

    return {
        "mode": "what_if",
        "timestamp": str(df.index[-1]),
        "current": current,
        "forecasts": forecasts
    }

def print_forecast(result):
    print("\nClimateTwin Prediction")
    print("=" * 45)

    print(f"\nMode: {result['mode']}")
    print(f"Reference observation: {result['timestamp']}")

    print("\nInput Conditions")

    for target, value in result["current"].items():
        print(f"{target}: {value:.2f}")

    for horizon, values in result["forecasts"].items():
        print(f"\nPrediction {horizon}")

        for target, value in values.items():
            print(f"{target}: {value:.2f}")

if __name__ == "__main__":
    result = predict_all()
    print_forecast(result)