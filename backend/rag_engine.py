import os
import re
import glob
import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from backend.forecast_predictor_multi import (
    predict_all,
    predict_custom
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

KNOWLEDGE_DIR = os.path.join(
    BASE_DIR,
    "knowledge"
)

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "conduit_data.csv"
)

ANOMALY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "conduit_anomaly_results.csv"
)

INTELLIGENCE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "climate_intelligence.csv"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GENERATION_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite"
]


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        EMBEDDING_MODEL
    )


@st.cache_resource
def build_knowledge_index():

    embedding_model = load_embedding_model()

    documents = []
    metadata = []

    files = sorted(
        glob.glob(
            os.path.join(
                KNOWLEDGE_DIR,
                "*.txt"
            )
        )
    )

    for file_path in files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        chunks = [
            chunk.strip()
            for chunk in text.split("\n\n")
            if chunk.strip()
        ]

        for chunk in chunks:

            documents.append(
                chunk
            )

            metadata.append(
                os.path.basename(
                    file_path
                )
            )

    if not documents:

        raise ValueError(
            "No knowledge documents were found in the knowledge folder."
        )

    embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    faiss.normalize_L2(
        embeddings
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return (
        index,
        documents,
        metadata
    )


@st.cache_resource
def get_gemini_client():

    api_key = st.secrets[
        "GEMINI_API_KEY"
    ]

    return genai.Client(
        api_key=api_key
    )


def retrieve(
    query,
    top_k=3
):

    embedding_model = load_embedding_model()

    index, documents, metadata = (
        build_knowledge_index()
    )

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype(
        "float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        min(
            top_k,
            len(documents)
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        results.append(
            {
                "text": documents[idx],
                "source": metadata[idx],
                "score": float(score)
            }
        )

    return results


@st.cache_data
def load_raw_data():

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    df["ts"] = pd.to_datetime(
        df["ts"],
        utc=True,
        errors="coerce"
    )

    return df


@st.cache_data
def load_anomaly_data():

    if not os.path.exists(
        ANOMALY_PATH
    ):

        return None

    df = pd.read_csv(
        ANOMALY_PATH
    )

    if "ts" in df.columns:

        df["ts"] = pd.to_datetime(
            df["ts"],
            utc=True,
            errors="coerce"
        )

    return df


@st.cache_data
def load_intelligence_data():

    if not os.path.exists(
        INTELLIGENCE_PATH
    ):

        return None

    df = pd.read_csv(
        INTELLIGENCE_PATH
    )

    if "ts" in df.columns:

        df["ts"] = pd.to_datetime(
            df["ts"],
            utc=True,
            errors="coerce"
        )

    return df


def estimate_heat_index(
    temperature_c,
    humidity
):

    if (
        temperature_c < 26.7
        or humidity < 40
    ):

        return float(
            temperature_c
        )

    temperature_f = (
        temperature_c * 9 / 5
    ) + 32

    t = temperature_f
    r = humidity

    heat_index_f = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * r
        - 0.22475541 * t * r
        - 0.00683783 * t * t
        - 0.05481717 * r * r
        + 0.00122874 * t * t * r
        + 0.00085282 * t * r * r
        - 0.00000199 * t * t * r * r
    )

    return (
        heat_index_f - 32
    ) * 5 / 9


def extract_what_if_values(
    question
):

    q = question.lower()

    temperature_match = re.search(
        r"(?:temperature|temp)\s*(?:becomes?|become|is|=|to|rises?\s+to|increases?\s+to|changes?\s+to|drops?\s+to|decreases?\s+to|falls?\s+to)?\s*(-?\d+(?:\.\d+)?)\s*°?\s*c",
        q
    )

    humidity_match = re.search(
        r"(?:humidity|relative humidity|rh)\s*(?:drops?\s+to|decreases?\s+to|falls?\s+to|rises?\s+to|increases?\s+to|becomes?|become|is|=|to|changes?\s+to)?\s*(\d+(?:\.\d+)?)\s*%",
        q
    )

    heat_index_match = re.search(
        r"(?:heat index)\s*(?:becomes?|become|is|=|to|rises?\s+to|increases?\s+to|drops?\s+to|decreases?\s+to|changes?\s+to|falls?\s+to)?\s*(-?\d+(?:\.\d+)?)\s*°?\s*c",
        q
    )

    wind_match = re.search(
        r"(?:wind speed|wind)\s*(?:becomes?|become|is|=|to|rises?\s+to|increases?\s+to|drops?\s+to|decreases?\s+to|changes?\s+to|falls?\s+to)?\s*(\d+(?:\.\d+)?)\s*(?:m/s)?",
        q
    )

    temperature = (
        float(
            temperature_match.group(1)
        )
        if temperature_match
        else None
    )

    humidity = (
        float(
            humidity_match.group(1)
        )
        if humidity_match
        else None
    )

    heat_index = (
        float(
            heat_index_match.group(1)
        )
        if heat_index_match
        else None
    )

    wind_speed = (
        float(
            wind_match.group(1)
        )
        if wind_match
        else None
    )

    return {
        "temperature": temperature,
        "humidity": humidity,
        "heat_index": heat_index,
        "wind_speed": wind_speed
    }


def get_what_if_prediction_context(
    question
):

    scenario = extract_what_if_values(
        question
    )

    current_forecast = predict_all()

    current = (
        current_forecast["current"]
    )

    temperature = (
        scenario["temperature"]
        if scenario["temperature"] is not None
        else current["temp_sht"]
    )

    humidity = (
        scenario["humidity"]
        if scenario["humidity"] is not None
        else current["humidity_sht"]
    )

    wind_speed = (
        scenario["wind_speed"]
        if scenario["wind_speed"] is not None
        else current["wind_spd"]
    )

    if scenario["heat_index"] is not None:

        heat_index = scenario[
            "heat_index"
        ]

    else:

        heat_index = estimate_heat_index(
            temperature,
            humidity
        )

    prediction = predict_custom(
        temperature,
        humidity,
        heat_index,
        wind_speed
    )

    parts = []

    parts.append(
        "WHAT-IF SCENARIO"
    )

    parts.append(
        f"Scenario temperature: "
        f"{temperature:.2f} °C"
    )

    parts.append(
        f"Scenario humidity: "
        f"{humidity:.2f}%"
    )

    parts.append(
        f"Scenario heat index input: "
        f"{heat_index:.2f} °C"
    )

    parts.append(
        f"Scenario wind speed: "
        f"{wind_speed:.2f} m/s"
    )

    for horizon in [
        "1h",
        "2h",
        "3h"
    ]:

        values = prediction[
            "forecasts"
        ][horizon]

        parts.append(
            f"{horizon} WHAT-IF PREDICTION:"
        )

        parts.append(
            f"Temperature: "
            f"{values['temp_sht']:.2f} °C"
        )

        parts.append(
            f"Humidity: "
            f"{values['humidity_sht']:.2f}%"
        )

        parts.append(
            f"Heat index: "
            f"{values['heat_idx']:.2f} °C"
        )

        parts.append(
            f"Wind speed: "
            f"{values['wind_spd']:.2f} m/s"
        )

    parts.append(
        "These values are hypothetical model outputs "
        "generated from the supplied scenario."
    )

    return "\n".join(
        parts
    )


def get_data_context(
    question
):

    df = load_raw_data()

    q = question.lower()

    parts = []

    temp = pd.to_numeric(
        df["temp_sht"],
        errors="coerce"
    )

    humidity = pd.to_numeric(
        df["humidity_sht"],
        errors="coerce"
    )

    heat = pd.to_numeric(
        df["heat_idx"],
        errors="coerce"
    )

    wind = pd.to_numeric(
        df["wind_spd"],
        errors="coerce"
    )

    if (
        "highest temperature" in q
        or "maximum temperature" in q
        or "max temperature" in q
        or "highest temp" in q
    ):

        idx = temp.idxmax()

        parts.append(
            f"Maximum temperature: "
            f"{temp.loc[idx]:.2f} °C"
        )

        parts.append(
            f"Timestamp: {df.loc[idx, 'ts']}"
        )

    elif (
        "lowest temperature" in q
        or "minimum temperature" in q
        or "min temperature" in q
        or "lowest temp" in q
    ):

        idx = temp.idxmin()

        parts.append(
            f"Minimum temperature: "
            f"{temp.loc[idx]:.2f} °C"
        )

        parts.append(
            f"Timestamp: {df.loc[idx, 'ts']}"
        )

    elif (
        "highest humidity" in q
        or "maximum humidity" in q
        or "max humidity" in q
    ):

        idx = humidity.idxmax()

        parts.append(
            f"Maximum humidity: "
            f"{humidity.loc[idx]:.2f}%"
        )

        parts.append(
            f"Timestamp: {df.loc[idx, 'ts']}"
        )

    elif (
        "lowest humidity" in q
        or "minimum humidity" in q
        or "min humidity" in q
    ):

        idx = humidity.idxmin()

        parts.append(
            f"Minimum humidity: "
            f"{humidity.loc[idx]:.2f}%"
        )

        parts.append(
            f"Timestamp: {df.loc[idx, 'ts']}"
        )

    elif (
        "highest heat index" in q
        or "maximum heat index" in q
        or "max heat index" in q
    ):

        idx = heat.idxmax()

        parts.append(
            f"Maximum heat index: "
            f"{heat.loc[idx]:.2f} °C"
        )

        parts.append(
            f"Timestamp: {df.loc[idx, 'ts']}"
        )

    elif (
        "highest wind" in q
        or "maximum wind" in q
        or "max wind" in q
        or "strongest wind" in q
    ):

        idx = wind.idxmax()

        parts.append(
            f"Maximum wind speed: "
            f"{wind.loc[idx]:.2f} m/s"
        )

        parts.append(
            f"Timestamp: {df.loc[idx, 'ts']}"
        )

    elif (
        "how many observations" in q
        or "how many records" in q
        or "dataset size" in q
    ):

        parts.append(
            f"Total observations: "
            f"{len(df):,}"
        )

        parts.append(
            f"First observation: "
            f"{df['ts'].min()}"
        )

        parts.append(
            f"Last observation: "
            f"{df['ts'].max()}"
        )

    else:

        latest = (
            df.sort_values("ts")
            .iloc[-1]
        )

        parts.append(
            f"Latest observation: "
            f"{latest['ts']}"
        )

        parts.append(
            f"Temperature: "
            f"{float(latest['temp_sht']):.2f} °C"
        )

        parts.append(
            f"Humidity: "
            f"{float(latest['humidity_sht']):.2f}%"
        )

        parts.append(
            f"Heat index: "
            f"{float(latest['heat_idx']):.2f} °C"
        )

        parts.append(
            f"Wind speed: "
            f"{float(latest['wind_spd']):.2f} m/s"
        )

    return "\n".join(
        parts
    )


def get_anomaly_context(
    question
):

    anomaly_df = load_anomaly_data()

    intelligence_df = (
        load_intelligence_data()
    )

    parts = []

    if anomaly_df is not None:

        if "is_anomaly" in anomaly_df.columns:

            anomaly_values = (
                anomaly_df["is_anomaly"]
                .astype(bool)
            )

            anomaly_count = int(
                anomaly_values.sum()
            )

            parts.append(
                f"Anomaly observations: "
                f"{anomaly_count}"
            )

        if "risk_level" in anomaly_df.columns:

            risk_counts = (
                anomaly_df[
                    "risk_level"
                ]
                .astype(str)
                .value_counts()
            )

            parts.append(
                "Risk-level distribution:"
            )

            for risk, count in (
                risk_counts.items()
            ):

                parts.append(
                    f"{risk}: {count}"
                )

    if intelligence_df is not None:

        if (
            "environmental_event"
            in intelligence_df.columns
        ):

            event_counts = (
                intelligence_df[
                    "environmental_event"
                ]
                .astype(str)
                .value_counts()
            )

            parts.append(
                "Environmental event distribution:"
            )

            for event, count in (
                event_counts.items()
            ):

                parts.append(
                    f"{event}: {count}"
                )

        if "priority" in intelligence_df.columns:

            priority_counts = (
                intelligence_df[
                    "priority"
                ]
                .astype(str)
                .str.upper()
                .value_counts()
            )

            parts.append(
                "Priority distribution:"
            )

            for priority, count in (
                priority_counts.items()
            ):

                parts.append(
                    f"{priority}: {count}"
                )

    if not parts:

        parts.append(
            "No anomaly or intelligence data is available."
        )

    return "\n".join(
        parts
    )


@st.cache_data(
    ttl=300
)
def get_prediction_context():

    forecast = predict_all()

    current = (
        forecast["current"]
    )

    forecasts = (
        forecast["forecasts"]
    )

    parts = []

    parts.append(
        "Latest forecast reference timestamp:"
    )

    parts.append(
        forecast["timestamp"]
    )

    parts.append(
        "Current environmental state:"
    )

    parts.append(
        f"Temperature: "
        f"{current['temp_sht']:.2f} °C"
    )

    parts.append(
        f"Humidity: "
        f"{current['humidity_sht']:.2f}%"
    )

    parts.append(
        f"Heat index: "
        f"{current['heat_idx']:.2f} °C"
    )

    parts.append(
        f"Wind speed: "
        f"{current['wind_spd']:.2f} m/s"
    )

    for horizon in [
        "1h",
        "2h",
        "3h"
    ]:

        values = forecasts[
            horizon
        ]

        parts.append(
            f"{horizon} model prediction:"
        )

        parts.append(
            f"Temperature: "
            f"{values['temp_sht']:.2f} °C"
        )

        parts.append(
            f"Humidity: "
            f"{values['humidity_sht']:.2f}%"
        )

        parts.append(
            f"Heat index: "
            f"{values['heat_idx']:.2f} °C"
        )

        parts.append(
            f"Wind speed: "
            f"{values['wind_spd']:.2f} m/s"
        )

    return "\n".join(
        parts
    )


def is_what_if_question(
    question
):

    q = question.lower()

    explicit_terms = [
        "what if",
        "what happens if",
        "what would happen if",
        "suppose",
        "assuming",
        "hypothetical",
        "scenario"
    ]

    change_terms = [
        "becomes",
        "become",
        "drops to",
        "decreases to",
        "falls to",
        "rises to",
        "increases to",
        "changes to"
    ]

    parsed_values = (
        extract_what_if_values(
            question
        )
    )

    has_explicit_term = any(
        term in q
        for term in explicit_terms
    )

    has_change_term = any(
        term in q
        for term in change_terms
    )

    has_numeric_scenario = any(
        value is not None
        for value in parsed_values.values()
    )

    return (
        (
            has_explicit_term
            or has_change_term
        )
        and has_numeric_scenario
    )


def classify_question(
    question
):

    q = question.lower().strip()

    prediction_terms = [
        "predict",
        "prediction",
        "forecast",
        "future",
        "next hour",
        "next 1 hour",
        "next 2 hours",
        "next 3 hours",
        "next three hours",
        "what will happen",
        "expected",
        "expect",
        "over the next",
        "in the next",
        "later"
    ]

    data_terms = [
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "max",
        "min",
        "recorded",
        "observation",
        "observed",
        "dataset",
        "record",
        "how many",
        "when did",
        "latest"
    ]

    anomaly_terms = [
        "anomaly",
        "anomalies",
        "unusual",
        "abnormal",
        "event",
        "risk",
        "priority",
        "outlier"
    ]

    knowledge_terms = [
        "what is",
        "define",
        "definition",
        "meaning",
        "explain",
        "how does",
        "what does",
        "methodology",
        "isolation forest",
        "wet bulb",
        "wet-bulb",
        "heat index",
        "relative humidity",
        "humidity mean",
        "temperature mean"
    ]

    explanation_terms = [
        "why",
        "how",
        "reason",
        "because",
        "explain"
    ]

    what_if = is_what_if_question(
        question
    )

    prediction_match = any(
        term in q
        for term in prediction_terms
    )

    data_match = any(
        term in q
        for term in data_terms
    )

    anomaly_match = any(
        term in q
        for term in anomaly_terms
    )

    knowledge_match = any(
        term in q
        for term in knowledge_terms
    )

    explanation_match = any(
        term in q
        for term in explanation_terms
    )

    if what_if and (
        explanation_match
        or knowledge_match
    ):

        return "hybrid"

    if what_if:

        return "what_if"

    if prediction_match and (
        explanation_match
        or knowledge_match
    ):

        return "hybrid"

    if prediction_match:

        return "prediction"

    if anomaly_match and (
        explanation_match
        or knowledge_match
    ):

        return "hybrid"

    if data_match and anomaly_match:

        return "hybrid"

    if data_match:

        return "data"

    if anomaly_match:

        return "hybrid"

    return "rag"


def prepare_question(
    question
):

    route = classify_question(
        question
    )

    knowledge_context = ""
    data_context = ""
    anomaly_context = ""
    prediction_context = ""
    sources = []

    if route in [
        "rag",
        "hybrid"
    ]:

        retrieved = retrieve(
            question,
            top_k=3
        )

        knowledge_context = "\n\n".join(
            [
                f"Source: {item['source']}\n"
                f"{item['text']}"
                for item in retrieved
            ]
        )

        sources = retrieved

    if route in [
        "data",
        "hybrid"
    ]:

        data_context = (
            get_data_context(
                question
            )
        )

    if (
        route == "hybrid"
        or any(
            word in question.lower()
            for word in [
                "anomaly",
                "unusual",
                "abnormal",
                "event",
                "risk",
                "priority",
                "outlier"
            ]
        )
    ):

        anomaly_context = (
            get_anomaly_context(
                question
            )
        )

    if route in [
        "prediction",
        "hybrid"
    ]:

        if is_what_if_question(
            question
        ):

            prediction_context = (
                get_what_if_prediction_context(
                    question
                )
            )

        else:

            prediction_context = (
                get_prediction_context()
            )

    if route == "what_if":

        prediction_context = (
            get_what_if_prediction_context(
                question
            )
        )

    return {
        "route": route,
        "knowledge_context": knowledge_context,
        "data_context": data_context,
        "anomaly_context": anomaly_context,
        "prediction_context": prediction_context,
        "sources": sources
    }


def build_prompt(
    question,
    route,
    knowledge_context,
    data_context,
    anomaly_context,
    prediction_context
):

    return f"""
You are ClimateTwin Copilot, an environmental intelligence assistant.

Answer the user's question using only the supplied evidence.

Question route:
{route}

Evidence categories:

Observed data:
These are actual measurements from the Conduit dataset.

Anomaly intelligence:
These are outputs generated by ClimateTwin's anomaly detection and intelligence pipeline.

Prediction:
These are outputs from ClimateTwin's short-term forecasting models.

What-if scenario:
These are hypothetical model outputs generated from user-supplied environmental conditions.

Knowledge:
These are documents retrieved from the ClimateTwin knowledge base.

Important rules:

Never invent environmental measurements.

Never present a prediction as an observed fact.

Always describe forecasts as model predictions.

Always describe what-if results as hypothetical model outputs.

Never claim that an anomaly automatically represents a dangerous real-world hazard.

Do not create unsupported meteorological conclusions.

When evidence is insufficient, clearly say so.

Use the actual numerical values supplied in the evidence.

When explaining predictions, distinguish model output from environmental interpretation.

For a what-if scenario, use the user's supplied values exactly when they are present in the prediction evidence.

Do not replace a user-supplied what-if value with the current observed value.

Answer the user directly and avoid unnecessary technical detail.

User question:
{question}

Knowledge evidence:
{knowledge_context}

Environmental data evidence:
{data_context}

Anomaly and intelligence evidence:
{anomaly_context}

Prediction evidence:
{prediction_context}

For what-if questions, clearly mention the scenario conditions before discussing the predicted results.

End with:

Evidence used:
- relevant knowledge sources
- relevant observed data
- relevant anomaly/intelligence outputs
- relevant prediction model outputs
"""


def generate_response_stream(
    question,
    route,
    knowledge_context,
    data_context,
    anomaly_context,
    prediction_context
):

    client = get_gemini_client()

    prompt = build_prompt(
        question,
        route,
        knowledge_context,
        data_context,
        anomaly_context,
        prediction_context
    )

    last_error = None

    for model in GENERATION_MODELS:

        try:

            stream = (
                client.models.generate_content_stream(
                    model=model,
                    contents=prompt
                )
            )

            emitted = False

            for chunk in stream:

                if chunk.text:

                    emitted = True

                    yield chunk.text

            if emitted:

                return

        except Exception as e:

            last_error = e

    if last_error is not None:

        raise last_error

    raise RuntimeError(
        "No Gemini model produced a response."
    )


def answer_question(
    question
):

    prepared = prepare_question(
        question
    )

    client = get_gemini_client()

    prompt = build_prompt(
        question,
        prepared["route"],
        prepared["knowledge_context"],
        prepared["data_context"],
        prepared["anomaly_context"],
        prepared["prediction_context"]
    )

    last_error = None

    for model in GENERATION_MODELS:

        try:

            response = (
                client.models.generate_content(
                    model=model,
                    contents=prompt
                )
            )

            return {
                "answer": response.text.strip(),
                "route": prepared["route"],
                "model": model,
                "sources": prepared[
                    "sources"
                ]
            }

        except Exception as e:

            last_error = e

    if last_error is not None:

        raise last_error

    raise RuntimeError(
        "No Gemini model produced a response."
    )