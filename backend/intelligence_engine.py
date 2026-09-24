import pandas as pd


# ==========================================
# 1. LOAD DATA
# ==========================================

INPUT_PATH = "data/processed/conduit_anomaly_results.csv"
OUTPUT_PATH = "data/processed/climate_intelligence.csv"

df = pd.read_csv(INPUT_PATH)

df["ts"] = pd.to_datetime(df["ts"])


# ==========================================
# 2. HISTORICAL THRESHOLDS
# ==========================================

heat_90 = df["heat_idx"].quantile(0.90)
heat_95 = df["heat_idx"].quantile(0.95)

humidity_10 = df["humidity_sht"].quantile(0.10)
humidity_20 = df["humidity_sht"].quantile(0.20)

wind_90 = df["wind_gust"].quantile(0.90)
wind_95 = df["wind_gust"].quantile(0.95)

temp_95 = df["temp_sht"].quantile(0.95)
temp_05 = df["temp_sht"].quantile(0.05)


# ==========================================
# 3. INTELLIGENCE ENGINE
# ==========================================

def analyze_environment(row):

    # --------------------------------------
    # If ML says normal
    # --------------------------------------

    if not row["is_anomaly"]:

        return (
            "Normal environmental conditions",
            "None",
            "Continue monitoring environmental conditions.",
            "LOW"
        )


    # --------------------------------------
    # ML detected an anomaly
    # --------------------------------------

    temperature = row["temp_sht"]
    humidity = row["humidity_sht"]
    heat_index = row["heat_idx"]
    wind_gust = row["wind_gust"]

    signals = []
    actions = []


    # ======================================
    # HEAT
    # ======================================

    if heat_index >= heat_95:

        signals.append("Elevated heat conditions")

        actions.append(
            "Monitor heat exposure and maintain hydration."
        )

    elif heat_index >= heat_90:

        signals.append("Above-normal heat conditions")

        actions.append(
            "Monitor heat conditions."
        )


    # ======================================
    # LOW HUMIDITY
    # ======================================

    if humidity <= humidity_10:

        signals.append("Unusually low humidity")

        actions.append(
            "Monitor dry conditions and potential vegetation stress."
        )

    elif humidity <= humidity_20:

        signals.append("Below-normal humidity")


    # ======================================
    # HIGH WIND
    # ======================================

    if wind_gust >= wind_95:

        signals.append("Elevated wind gust")

        actions.append(
            "Monitor exposed areas and unsecured infrastructure."
        )

    elif wind_gust >= wind_90:

        signals.append("Above-normal wind gust")


    # ======================================
    # TEMPERATURE
    # ======================================

    if temperature >= temp_95:

        signals.append("Unusually high temperature")

    elif temperature <= temp_05:

        signals.append("Unusually low temperature")


    # ======================================
    # CLASSIFY EVENT
    # ======================================

    if (
        heat_index >= heat_90
        and humidity <= humidity_20
    ):

        event = "Heat and dryness anomaly"

        if not actions:

            actions.append(
                "Monitor heat exposure and dry conditions."
            )


    elif (
        heat_index >= heat_90
        and humidity <= humidity_20
        and wind_gust >= wind_90
    ):

        event = "Heat, dryness and wind anomaly"


    elif wind_gust >= wind_90:

        event = "Wind anomaly"


    elif humidity <= humidity_20:

        event = "Dryness anomaly"


    elif temperature >= temp_95:

        event = "Temperature anomaly"


    elif temperature <= temp_05:

        event = "Temperature anomaly"


    else:

        event = "Environmental anomaly"


    # ======================================
    # REMOVE DUPLICATE ACTIONS
    # ======================================

    actions = list(dict.fromkeys(actions))

    if actions:

        action = " ".join(actions)

    else:

        action = (
            "Continue monitoring environmental conditions."
        )


    # ======================================
    # PRIORITY
    # ======================================

    if len(signals) >= 2:

        priority = "HIGH"

    else:

        priority = "MEDIUM"


    return (
        event,
        ", ".join(signals),
        action,
        priority
    )


# ==========================================
# 4. APPLY ENGINE
# ==========================================

results = df.apply(
    analyze_environment,
    axis=1
)


df[
    [
        "environmental_event",
        "signals",
        "recommended_action",
        "priority"
    ]
] = pd.DataFrame(
    results.tolist(),
    index=df.index
)


# ==========================================
# 5. RESULTS
# ==========================================

print("\n========================================")
print("CLIMATE INTELLIGENCE RESULTS")
print("========================================")

print("\nEnvironmental events:")

print(
    df["environmental_event"]
    .value_counts()
)


print("\nPriority distribution:")

print(
    df["priority"]
    .value_counts()
)


# ==========================================
# 6. SHOW ONLY ACTUAL ML ANOMALIES
# ==========================================

anomalies = df[
    df["is_anomaly"] == True
]

print("\n========================================")
print("DETECTED ANOMALIES")
print("========================================")

print(
    anomalies[
        [
            "ts",
            "temp_sht",
            "humidity_sht",
            "heat_idx",
            "wind_gust",
            "anomaly_score",
            "environmental_event",
            "signals",
            "recommended_action",
            "priority"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ==========================================
# 7. SAVE
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n========================================")
print("SUCCESS")
print("========================================")

print(
    f"Saved to:\n{OUTPUT_PATH}"
)
