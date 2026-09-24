import os
import json
from forecast_predictor import predict_next_hour

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "latest_forecast.json")

def classify_change(delta, positive_label, negative_label, stable_label, threshold):
    if delta > threshold:
        return positive_label
    if delta < -threshold:
        return negative_label
    return stable_label

def generate_outlook():
    result = predict_next_hour()

    current = result["current"]
    predicted = result["predicted"]

    temp_delta = predicted["temp_sht"] - current["temp_sht"]
    humidity_delta = predicted["humidity_sht"] - current["humidity_sht"]
    heat_delta = predicted["heat_idx"] - current["heat_idx"]
    wind_delta = predicted["wind_spd"] - current["wind_spd"]

    temp_trend = classify_change(
        temp_delta,
        "Rising temperature",
        "Falling temperature",
        "Stable temperature",
        0.3
    )

    humidity_trend = classify_change(
        humidity_delta,
        "Increasing humidity",
        "Decreasing humidity",
        "Stable humidity",
        2.0
    )

    heat_trend = classify_change(
        heat_delta,
        "Increasing heat exposure",
        "Decreasing heat exposure",
        "Stable heat exposure",
        0.3
    )

    wind_trend = classify_change(
        wind_delta,
        "Increasing wind",
        "Decreasing wind",
        "Stable wind",
        0.3
    )

    if temp_delta > 0.5 and humidity_delta < -3:
        outlook = "Warmer and drier conditions are expected."
        outlook_level = "INCREASING HEAT"
    elif temp_delta < -0.5 and humidity_delta > 3:
        outlook = "Cooler and more humid conditions are expected."
        outlook_level = "COOLING TREND"
    elif heat_delta > 0.5:
        outlook = "Heat exposure is expected to increase."
        outlook_level = "HEAT INCREASE"
    elif heat_delta < -0.5:
        outlook = "Heat exposure is expected to decrease."
        outlook_level = "HEAT DECREASE"
    else:
        outlook = "No major change is expected in the short-term environmental state."
        outlook_level = "STABLE"

    forecast = {
        "generated_at": str(result["timestamp"]),
        "horizon": "1 hour",
        "current": current,
        "predicted": predicted,
        "changes": {
            "temperature": round(temp_delta, 2),
            "humidity": round(humidity_delta, 2),
            "heat_index": round(heat_delta, 2),
            "wind_speed": round(wind_delta, 2)
        },
        "trends": {
            "temperature": temp_trend,
            "humidity": humidity_trend,
            "heat_index": heat_trend,
            "wind_speed": wind_trend
        },
        "outlook": outlook,
        "outlook_level": outlook_level,
        "model_notes": {
            "temperature": "High validation performance in current prototype dataset",
            "humidity": "High validation performance in current prototype dataset",
            "heat_index": "High validation performance in current prototype dataset",
            "wind_speed": "Lower validation performance; interpret cautiously"
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(forecast, f, indent=4)

    return forecast

if __name__ == "__main__":
    forecast = generate_outlook()

    print("\nClimateTwin Environmental Outlook")
    print("=" * 40)

    print(f"\nCurrent Time: {forecast['generated_at']}")
    print(f"Forecast Horizon: {forecast['horizon']}")

    print("\nCurrent Conditions")
    for key, value in forecast["current"].items():
        print(f"{key}: {value:.2f}")

    print("\nPredicted Conditions")
    for key, value in forecast["predicted"].items():
        print(f"{key}: {value:.2f}")

    print("\nChanges")
    for key, value in forecast["changes"].items():
        print(f"{key}: {value:+.2f}")

    print("\nTrends")
    for key, value in forecast["trends"].items():
        print(f"{key}: {value}")

    print("\nOutlook")
    print(f"{forecast['outlook_level']}: {forecast['outlook']}")

    print(f"\nSaved to: {OUTPUT_PATH}")