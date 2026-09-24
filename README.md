# 🌍 ClimateTwin AI

## AI-Powered Environmental Intelligence

**Monitor • Detect • Predict • Explain**

> **From Data → Insight → Action → Impact**

ClimateTwin AI is an environmental intelligence platform that transforms real environmental observations into **anomaly detection, environmental event intelligence, short-term forecasting, What-If analysis, and evidence-grounded AI explanations**.

The system is built around real **JKUAT Conduit environmental data** and combines machine learning, retrieval-augmented generation, and interactive visualization into one end-to-end platform.

---

## 🚀 Live Demo

**ClimateTwin AI:**  
https://climatetwinai.streamlit.app

---

## 🎯 Problem

Environmental sensors generate large volumes of measurements, but raw measurements do not immediately answer:

- What is unusual?
- Why is the condition unusual?
- What could happen next?
- What should a user investigate?
- How can the data be understood without manually analyzing thousands of records?

ClimateTwin AI addresses this gap by connecting **observation → detection → prediction → explanation → action**.

---

## 💡 Solution

ClimateTwin AI acts as an **Environmental Black Box**.

Instead of being only a visualization dashboard, it creates an intelligence pipeline that:

1. Ingests real Conduit environmental observations
2. Cleans and prepares the data
3. Detects unusual multivariable observations
4. Converts anomalies into environmental events
5. Assigns contextual priorities and signals
6. Generates short-term environmental forecasts
7. Allows users to test hypothetical scenarios
8. Explains results through an AI Copilot
9. Provides direct access to the underlying dataset

---

## 📊 Conduit Dataset

The current ClimateTwin dataset contains:

- **2,089 observations**
- **30 dataset columns**
- Observation period: **September 1–22, 2026**

Environmental measurements include variables such as:

- Temperature
- Humidity
- Wind speed
- Wind gust
- Heat index
- Pressure
- UV measurements
- Wet-bulb temperature
- Wet-bulb globe temperature
- Other sensor observations

The Conduit observations are the foundation of the anomaly detection, intelligence, forecasting, visualization, and Copilot workflows.

---

# 🧠 Core Features

## 📊 1. Intelligence Overview

The Overview dashboard provides a high-level view of environmental conditions.

It displays:

- Total observations
- High-priority observations
- Medium-priority observations
- Normal observations
- Current temperature
- Humidity
- Heat index
- Wind speed
- Wind gust
- Climate risk intelligence
- Detected environmental events
- Environmental trends
- Risk timeline
- High-priority events
- Climate impact summary

The dashboard can be filtered dynamically by:

- Date range
- Priority
- Environmental event

The displayed statistics and visualizations update according to the selected filters.

---

## 🚨 2. AI Anomaly Detection

ClimateTwin AI uses **Isolation Forest** for multivariable anomaly detection.

The anomaly pipeline analyzes environmental variables together rather than relying on a single measurement.

Detected observations are then passed to an environmental intelligence layer.

The system can classify observations into events such as:

- Heat and dryness anomaly
- Temperature anomaly
- Wind anomaly
- Dryness anomaly
- Environmental anomaly

The intelligence layer also generates:

**Event → Priority → Signals → Recommended Action**

An anomaly is treated as a statistically unusual observation and is not automatically interpreted as proof of a dangerous real-world hazard.

---

## 🔮 3. Short-Term Environmental Forecasting

The Prediction Lab provides short-term predictions for:

- Temperature
- Humidity
- Heat index
- Wind speed

Forecast horizons:

- **+1 hour**
- **+2 hours**
- **+3 hours**

The forecasting pipeline uses historical environmental observations and temporal features with **Random Forest regression models**.

The models are evaluated using a temporal train/test split so that later observations are not randomly mixed into the training data.

Example three-hour validation results from the current prototype:

| Variable | R² @ +3h |
|---|---:|
| Temperature | 0.862 |
| Humidity | 0.804 |
| Heat Index | 0.860 |

Wind forecasting is included in the system, but its predictive performance is weaker than the temperature, humidity, and heat-index models.

These forecasts are **short-term model outputs**, not guaranteed future conditions.

---

## 🧪 4. What-If Scenario Analysis

ClimateTwin AI allows users to test hypothetical environmental conditions.

For example:

```text
Temperature = 30°C
Humidity = 35%
