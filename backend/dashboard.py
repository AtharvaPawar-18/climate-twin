import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.forecast_predictor_multi import (
    predict_all,
    predict_custom
)

RAW_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "conduit_data.csv"
)

INTELLIGENCE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "climate_intelligence.csv"
)

ANOMALY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "conduit_anomaly_results.csv"
)

st.set_page_config(
    page_title="ClimateTwin AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b0f14;
        --surface: #10161d;
        --surface-2: #141b23;
        --border: rgba(255,255,255,0.075);
        --muted: #9aa5b4;
        --text: #f4f7fa;
        --accent: #43d17d;
    }

    html,
    body,
    [class*="css"],
    .stApp {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(67,209,125,0.055),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(80,160,255,0.035),
                transparent 28%
            ),
            var(--bg);
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #11161d 0%,
                #0f141b 100%
            );
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] * {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 24px !important;
        font-weight: 750 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.2rem !important;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 17px !important;
        font-weight: 650 !important;
        letter-spacing: -0.01em !important;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #9aa5b4;
        font-size: 14px;
        line-height: 1.6;
    }

    [data-testid="stSidebar"] hr {
        margin: 1.1rem 0;
    }

    [data-testid="stSidebar"] label {
        font-size: 14px !important;
        font-weight: 550 !important;
        color: #c0c8d3 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        border-radius: 10px !important;
    }

    [data-testid="stSidebarNav"] {
        padding-top: 0.2rem;
    }

    [data-testid="stSidebarNav"] span {
        font-size: 15px !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: 13.5px !important;
        line-height: 1.55 !important;
        color: #9aa5b4 !important;
    }

    h1,
    h2,
    h3,
    h4 {
        color: var(--text) !important;
    }

    h1 {
        font-size: 42px !important;
        line-height: 1.12 !important;
        font-weight: 800 !important;
        letter-spacing: -0.045em !important;
        margin-top: 0 !important;
        margin-bottom: 0.45rem !important;
    }

    h2 {
        font-size: 30px !important;
        line-height: 1.25 !important;
        font-weight: 720 !important;
        letter-spacing: -0.03em !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        font-size: 21px !important;
        line-height: 1.3 !important;
        font-weight: 650 !important;
        letter-spacing: -0.018em !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.65rem !important;
    }

    h4 {
        font-size: 17px !important;
        font-weight: 650 !important;
    }

    p {
        font-size: 16px;
        line-height: 1.7;
        color: #cbd3dd;
    }

    [data-testid="stCaptionContainer"] {
        color: #9aa5b4 !important;
        font-size: 14px !important;
        line-height: 1.55 !important;
    }

    h1 a,
    h2 a,
    h3 a,
    h4 a {
        display: none !important;
    }

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                180deg,
                #121922 0%,
                #0f151c 100%
            );
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.1rem 1.15rem;
        min-height: 128px;
        box-shadow:
            0 10px 28px rgba(0,0,0,0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #a7b1be !important;
        font-size: 13px !important;
        font-weight: 650 !important;
        text-transform: uppercase;
        letter-spacing: 0.055em !important;
    }

    [data-testid="stMetricValue"] {
        color: #f6f8fa !important;
        font-size: 34px !important;
        line-height: 1.08 !important;
        font-weight: 760 !important;
        letter-spacing: -0.04em !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                180deg,
                rgba(17,24,32,0.96),
                rgba(14,20,27,0.96)
            ) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow:
            0 12px 34px rgba(0,0,0,0.13);
    }

    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        font-size: 15px !important;
    }

    [data-testid="stAlert"] p {
        font-size: 15px !important;
        line-height: 1.65 !important;
    }

    [data-testid="stInfo"] {
        background: rgba(76,132,255,0.055) !important;
    }

    [data-testid="stSuccess"] {
        background: rgba(67,209,125,0.075) !important;
    }

    [data-testid="stWarning"] {
        background: rgba(242,180,66,0.075) !important;
    }

    [data-testid="stError"] {
        background: rgba(255,90,90,0.075) !important;
    }

    .stButton > button {
        min-height: 47px !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        font-weight: 650 !important;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        transition:
            transform 0.18s ease,
            border-color 0.18s ease,
            background 0.18s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(67,209,125,0.32) !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        border-radius: 11px !important;
        background: #111820 !important;
        border-color: rgba(255,255,255,0.08) !important;
    }

    .stTextInput input,
    .stNumberInput input,
    textarea {
        font-size: 15px !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.045);
        background: rgba(16,22,29,0.55);
        margin-bottom: 0.7rem;
    }

    [data-testid="stChatMessage"] p {
        font-size: 16px !important;
        line-height: 1.72 !important;
    }

    [data-testid="stChatMessage"] li {
        font-size: 16px !important;
        line-height: 1.65 !important;
    }

    [data-testid="stChatInput"] {
        margin-top: 0.7rem;
    }

    [data-testid="stRadio"] label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    [data-testid="stRadio"] p {
        font-size: 14px !important;
    }

    [data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
    }

    [data-testid="stExpander"] p {
        font-size: 15px !important;
        line-height: 1.65 !important;
    }

    hr {
        border-color: rgba(255,255,255,0.065) !important;
        margin: 1.6rem 0 !important;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #b8c2cf;
        font-weight: 500;
        line-height: 1.55;
        margin-bottom: 0.15rem;
    }

    .hero-caption {
        font-size: 14.5px;
        color: #8995a5;
        letter-spacing: 0.015em;
        line-height: 1.6;
    }

    .section-note {
        color: #9aa5b4;
        font-size: 14px;
        line-height: 1.55;
        margin-top: -0.35rem;
        margin-bottom: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


def load_data():

    raw_df = pd.read_csv(
        RAW_PATH
    )

    raw_df["ts"] = pd.to_datetime(
        raw_df["ts"],
        utc=True,
        errors="coerce"
    )

    intelligence_df = pd.read_csv(
        INTELLIGENCE_PATH
    )

    intelligence_df["ts"] = pd.to_datetime(
        intelligence_df["ts"],
        utc=True,
        errors="coerce"
    )

    return raw_df, intelligence_df


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


def find_column(
    df,
    candidates
):

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    return None


def format_value(
    value,
    suffix="",
    digits=1
):

    if pd.isna(value):
        return "—"

    return (
        f"{float(value):.{digits}f}{suffix}"
    )


def get_priority_column(df):

    return find_column(
        df,
        [
            "priority",
            "Priority",
            "priority_level"
        ]
    )


def get_event_column(df):

    return find_column(
        df,
        [
            "environmental_event",
            "event",
            "event_type",
            "event_label",
            "Environmental Event"
        ]
    )


def get_signal_column(df):

    return find_column(
        df,
        [
            "signals",
            "signal",
            "detected_signals"
        ]
    )


def get_action_column(df):

    return find_column(
        df,
        [
            "recommended_action",
            "recommendation",
            "action",
            "recommended"
        ]
    )


def get_risk_column(df):

    return find_column(
        df,
        [
            "risk_level",
            "risk",
            "Risk Level"
        ]
    )


raw_df, intelligence_df = load_data()
anomaly_df = load_anomaly_data()

filtered_df = intelligence_df.copy()

if "ts" in filtered_df.columns:

    filtered_df = filtered_df.sort_values(
        "ts"
    )

priority_col = get_priority_column(
    filtered_df
)

event_col = get_event_column(
    filtered_df
)

signal_col = get_signal_column(
    filtered_df
)

action_col = get_action_column(
    filtered_df
)

risk_col = get_risk_column(
    filtered_df
)


st.sidebar.title(
    "🌍 ClimateTwin"
)

st.sidebar.caption(
    "AI Environmental Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🌍 Overview",
        "🔮 Prediction Lab",
        "🤖 ClimateTwin Copilot",
        "📊 Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "Filters"
)

if "ts" in filtered_df.columns:

    min_date = (
        filtered_df["ts"]
        .min()
        .date()
    )

    max_date = (
        filtered_df["ts"]
        .max()
        .date()
    )

    selected_dates = st.sidebar.date_input(
        "Date Range",
        value=(
            min_date,
            max_date
        ),
        min_value=min_date,
        max_value=max_date
    )

    if (
        isinstance(
            selected_dates,
            tuple
        )
        and len(selected_dates) == 2
    ):

        start_date, end_date = (
            selected_dates
        )

        filtered_df = filtered_df[
            (
                filtered_df["ts"]
                .dt.date
                >= start_date
            )
            &
            (
                filtered_df["ts"]
                .dt.date
                <= end_date
            )
        ]


if priority_col:

    priorities = sorted(
        filtered_df[
            priority_col
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_priorities = (
        st.sidebar.multiselect(
            "Priority",
            priorities,
            default=priorities
        )
    )

    filtered_df = filtered_df[
        filtered_df[
            priority_col
        ]
        .astype(str)
        .isin(
            selected_priorities
        )
    ]


if event_col:

    events = sorted(
        filtered_df[
            event_col
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_events = (
        st.sidebar.multiselect(
            "Environmental Event",
            events,
            default=events
        )
    )

    filtered_df = filtered_df[
        filtered_df[
            event_col
        ]
        .astype(str)
        .isin(
            selected_events
        )
    ]


for col in [
    "temp_sht",
    "humidity_sht",
    "wind_spd",
    "heat_idx",
    "wind_gust"
]:

    if col in filtered_df.columns:

        filtered_df[col] = pd.to_numeric(
            filtered_df[col],
            errors="coerce"
        )


st.sidebar.caption(
    f"{len(filtered_df):,} observations match current filters"
)


if page == "🌍 Overview":

    st.title(
        "🌍 ClimateTwin AI"
    )

    st.markdown(
        '<div class="hero-subtitle">AI-Powered Environmental Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-caption">Monitor • Detect • Predict • Explain</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-caption">From Data → Insight → Action → Impact</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.success(
        f"Conduit dataset loaded — "
        f"{len(raw_df):,} observations | "
        f"{len(filtered_df):,} observations match current filters"
    )

    st.header(
        "📊 Intelligence Overview"
    )

    total_count = len(
        filtered_df
    )

    high_count = 0
    medium_count = 0
    normal_count = 0

    if priority_col:

        priority_values = (
            filtered_df[
                priority_col
            ]
            .astype(str)
            .str.upper()
        )

        high_count = int(
            (
                priority_values
                == "HIGH"
            ).sum()
        )

        medium_count = int(
            (
                priority_values
                == "MEDIUM"
            ).sum()
        )

        normal_count = int(
            (
                priority_values
                == "LOW"
            ).sum()
        )

    k1, k2, k3, k4 = (
        st.columns(4)
    )

    with k1:

        st.metric(
            "TOTAL OBSERVATIONS",
            f"{total_count:,}",
            "Filtered sensor records"
        )

    with k2:

        st.metric(
            "🔴 HIGH PRIORITY",
            f"{high_count:,}",
            "Requires attention"
        )

    with k3:

        st.metric(
            "🟠 MEDIUM PRIORITY",
            f"{medium_count:,}",
            "Monitor conditions"
        )

    with k4:

        st.metric(
            "🟢 NORMAL",
            f"{normal_count:,}",
            "Normal conditions"
        )

    st.header(
        "🌤️ Current Environmental Conditions"
    )

    st.markdown(
        '<div class="section-note">Latest available environmental state in the filtered dataset</div>',
        unsafe_allow_html=True
    )

    if len(filtered_df) > 0:

        latest = (
            filtered_df
            .sort_values("ts")
            .iloc[-1]
        )

        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )

        with c1:

            st.metric(
                "Temperature",
                format_value(
                    latest.get(
                        "temp_sht"
                    ),
                    " °C"
                )
            )

        with c2:

            st.metric(
                "Humidity",
                format_value(
                    latest.get(
                        "humidity_sht"
                    ),
                    "%"
                )
            )

        with c3:

            st.metric(
                "Heat Index",
                format_value(
                    latest.get(
                        "heat_idx"
                    ),
                    " °C"
                )
            )

        with c4:

            st.metric(
                "Wind Speed",
                format_value(
                    latest.get(
                        "wind_spd"
                    ),
                    "",
                    1
                )
            )

        with c5:

            st.metric(
                "Wind Gust",
                format_value(
                    latest.get(
                        "wind_gust"
                    ),
                    "",
                    1
                )
            )

    st.header(
        "🚨 Climate Risk Intelligence"
    )

    if len(filtered_df) > 0:

        latest = (
            filtered_df
            .sort_values("ts")
            .iloc[-1]
        )

        current_risk = (
            str(
                latest[risk_col]
            )
            if (
                risk_col
                and not pd.isna(
                    latest[risk_col]
                )
            )
            else "LOW RISK"
        )

        current_event = (
            str(
                latest[event_col]
            )
            if (
                event_col
                and not pd.isna(
                    latest[event_col]
                )
            )
            else "Normal environmental conditions"
        )

        current_signal = (
            str(
                latest[signal_col]
            )
            if (
                signal_col
                and not pd.isna(
                    latest[signal_col]
                )
            )
            else "No abnormal environmental signals detected."
        )

        current_action = (
            str(
                latest[action_col]
            )
            if (
                action_col
                and not pd.isna(
                    latest[action_col]
                )
            )
            else "Continue monitoring environmental conditions."
        )

        r1, r2 = (
            st.columns(
                2,
                gap="large"
            )
        )

        with r1:

            risk_upper = (
                current_risk.upper()
            )

            if "HIGH" in risk_upper:
                icon = "🔴"
            elif "MEDIUM" in risk_upper:
                icon = "🟠"
            else:
                icon = "🟢"

            with st.container(
                border=True
            ):

                st.subheader(
                    f"{icon} {current_risk}"
                )

                m1, m2 = (
                    st.columns(2)
                )

                with m1:

                    st.metric(
                        "Heat Index",
                        format_value(
                            latest.get(
                                "heat_idx"
                            ),
                            " °C"
                        )
                    )

                with m2:

                    st.metric(
                        "Humidity",
                        format_value(
                            latest.get(
                                "humidity_sht"
                            ),
                            "%"
                        )
                    )

                st.caption(
                    f"Observation timestamp: {latest['ts']}"
                )

        with r2:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Detected Event"
                )

                st.write(
                    current_event
                )

                st.subheader(
                    "Signals"
                )

                st.write(
                    current_signal
                )

                st.subheader(
                    "Recommended Action"
                )

                st.write(
                    current_action
                )

    st.header(
        "📈 Environmental Trends"
    )

    st.markdown(
        '<div class="section-note">Observed environmental changes across the selected time period</div>',
        unsafe_allow_html=True
    )

    if len(filtered_df) > 1:

        trend_df = (
            filtered_df
            .sort_values("ts")
        )

        if all(
            col in trend_df.columns
            for col in [
                "ts",
                "temp_sht",
                "heat_idx"
            ]
        ):

            fig = px.line(
                trend_df,
                x="ts",
                y=[
                    "temp_sht",
                    "heat_idx"
                ],
                title="Temperature & Heat Index",
                markers=False
            )

            fig.update_layout(
                height=420,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=15
                ),
                legend_title_text=""
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        if all(
            col in trend_df.columns
            for col in [
                "ts",
                "humidity_sht"
            ]
        ):

            fig = px.line(
                trend_df,
                x="ts",
                y="humidity_sht",
                title="Humidity Trend"
            )

            fig.update_layout(
                height=360,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=15
                ),
                legend_title_text=""
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        if all(
            col in trend_df.columns
            for col in [
                "ts",
                "wind_spd"
            ]
        ):

            fig = px.line(
                trend_df,
                x="ts",
                y="wind_spd",
                title="Wind Speed Trend"
            )

            fig.update_layout(
                height=360,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=15
                ),
                legend_title_text=""
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

    st.header(
        "🚦 Risk Timeline"
    )

    if (
        priority_col
        and "ts" in filtered_df.columns
    ):

        timeline_df = filtered_df[
            [
                "ts",
                priority_col
            ]
        ].copy()

        timeline_df[
            "priority_num"
        ] = (
            timeline_df[
                priority_col
            ]
            .astype(str)
            .str.upper()
            .map(
                {
                    "LOW": 1,
                    "MEDIUM": 2,
                    "HIGH": 3
                }
            )
        )

        timeline_df = (
            timeline_df
            .dropna(
                subset=[
                    "priority_num"
                ]
            )
        )

        fig = px.scatter(
            timeline_df,
            x="ts",
            y="priority_num",
            color=priority_col,
            title="Environmental Priority Timeline"
        )

        fig.update_yaxes(
            tickmode="array",
            tickvals=[
                1,
                2,
                3
            ],
            ticktext=[
                "LOW",
                "MEDIUM",
                "HIGH"
            ]
        )

        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=15
            ),
            legend_title_text=""
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.header(
        "🧠 Environmental Event Analysis"
    )

    if (
        event_col
        and len(filtered_df) > 0
    ):

        event_counts = (
            filtered_df[
                event_col
            ]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        event_counts.columns = [
            "Event",
            "Count"
        ]

        fig = px.bar(
            event_counts,
            x="Event",
            y="Count",
            title="Detected Environmental Events"
        )

        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=15
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.header(
        "🚨 High Priority Events"
    )

    if priority_col:

        high_df = filtered_df[
            filtered_df[
                priority_col
            ]
            .astype(str)
            .str.upper()
            == "HIGH"
        ].copy()

        if len(high_df) > 0:

            display_cols = [
                col
                for col in [
                    "ts",
                    event_col,
                    "temp_sht",
                    "humidity_sht",
                    "heat_idx",
                    "wind_spd",
                    signal_col,
                    action_col
                ]
                if (
                    col
                    and col in high_df.columns
                )
            ]

            st.dataframe(
                high_df[
                    display_cols
                ]
                .sort_values(
                    "ts",
                    ascending=False
                )
                .head(20),
                width="stretch",
                hide_index=True
            )

        else:

            st.info(
                "No high-priority events are present in the current filters."
            )

    st.header(
        "🗺️ Environmental Intelligence Map"
    )

    st.info(
        "The current Conduit dataset does not contain latitude/longitude "
        "fields. The geospatial layer can be activated when coordinates "
        "are added to the dataset."
    )

    st.header(
        "📊 Climate Impact Summary"
    )

    elevated_count = 0

    if priority_col:

        elevated = (
            filtered_df[
                priority_col
            ]
            .astype(str)
            .str.upper()
            .isin(
                [
                    "HIGH",
                    "MEDIUM"
                ]
            )
        )

        elevated_count = int(
            elevated.sum()
        )

    elevated_rate = (
        elevated_count
        / total_count
        * 100
        if total_count
        else 0
    )

    i1, i2, i3 = (
        st.columns(3)
    )

    with i1:

        st.metric(
            "Elevated Observations",
            elevated_count
        )

    with i2:

        st.metric(
            "Elevated Risk Rate",
            f"{elevated_rate:.2f}%"
        )

    with i3:

        st.metric(
            "Total Observations",
            total_count
        )

    st.info(
        "ClimateTwin AI transforms environmental observations into "
        "environmental events, priorities, signals, and recommended "
        "actions to support earlier interpretation and decision-making."
    )

    st.header(
        "🤖 Latest Climate Intelligence"
    )

    if len(filtered_df) > 0:

        latest = (
            filtered_df
            .sort_values("ts")
            .iloc[-1]
        )

        st.subheader(
            str(
                latest[event_col]
                if event_col
                else "Normal environmental conditions"
            )
        )

        col1, col2 = (
            st.columns(
                2,
                gap="large"
            )
        )

        with col1:

            with st.container(
                border=True
            ):

                st.write(
                    "**Priority**"
                )

                st.write(
                    str(
                        latest[
                            priority_col
                        ]
                        if priority_col
                        else "LOW"
                    )
                )

                st.write(
                    "**Signals**"
                )

                st.write(
                    str(
                        latest[
                            signal_col
                        ]
                        if signal_col
                        else "No abnormal environmental signals detected."
                    )
                )

        with col2:

            with st.container(
                border=True
            ):

                st.write(
                    "**Recommended Action**"
                )

                st.write(
                    str(
                        latest[
                            action_col
                        ]
                        if action_col
                        else "Continue monitoring environmental conditions."
                    )
                )

                st.write(
                    "**Timestamp**"
                )

                st.write(
                    str(
                        latest["ts"]
                    )
                )

    st.divider()

    st.caption(
        "ClimateTwin AI • Environmental Data → Intelligence → "
        "Prediction → Explain → Impact"
    )


elif page == "🔮 Prediction Lab":

    st.title(
        "🔮 Prediction Lab"
    )

    st.markdown(
        '<div class="hero-subtitle">Short-term environmental forecasting</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-caption">+1 hour • +2 hours • +3 hours</div>',
        unsafe_allow_html=True
    )

    st.write("")

    prediction_mode = st.radio(
        "Prediction Mode",
        [
            "Live Conduit Forecast",
            "What-If Scenario"
        ],
        horizontal=True
    )

    forecast = None

    if (
        prediction_mode
        == "Live Conduit Forecast"
    ):

        try:

            forecast = predict_all()

            st.success(
                "Forecast generated from the latest available Conduit observation."
            )

        except Exception as e:

            st.error(
                f"Prediction system error: {e}"
            )

    else:

        st.subheader(
            "🧪 What-If Scenario"
        )

        st.markdown(
            '<div class="section-note">Enter a hypothetical environmental state and estimate the model response.</div>',
            unsafe_allow_html=True
        )

        try:

            live_forecast = predict_all()

            defaults = (
                live_forecast["current"]
            )

        except Exception:

            defaults = {
                "temp_sht": 25.0,
                "humidity_sht": 60.0,
                "heat_idx": 25.0,
                "wind_spd": 2.0
            }

        with st.form(
            "prediction_form"
        ):

            i1, i2 = (
                st.columns(
                    2,
                    gap="large"
                )
            )

            with i1:

                input_temperature = (
                    st.number_input(
                        "Temperature (°C)",
                        min_value=-20.0,
                        max_value=60.0,
                        value=float(
                            defaults[
                                "temp_sht"
                            ]
                        ),
                        step=0.1
                    )
                )

                input_humidity = (
                    st.number_input(
                        "Humidity (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(
                            defaults[
                                "humidity_sht"
                            ]
                        ),
                        step=0.1
                    )
                )

            with i2:

                input_heat_index = (
                    st.number_input(
                        "Heat Index (°C)",
                        min_value=-20.0,
                        max_value=70.0,
                        value=float(
                            defaults[
                                "heat_idx"
                            ]
                        ),
                        step=0.1
                    )
                )

                input_wind = (
                    st.number_input(
                        "Wind Speed (m/s)",
                        min_value=0.0,
                        max_value=50.0,
                        value=float(
                            defaults[
                                "wind_spd"
                            ]
                        ),
                        step=0.1
                    )
                )

            predict_button = (
                st.form_submit_button(
                    "🔮 Predict Environment",
                    width="stretch"
                )
            )

        if predict_button:

            try:

                st.session_state[
                    "custom_forecast"
                ] = predict_custom(
                    input_temperature,
                    input_humidity,
                    input_heat_index,
                    input_wind
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

        if (
            "custom_forecast"
            in st.session_state
        ):

            forecast = (
                st.session_state[
                    "custom_forecast"
                ]
            )

        else:

            st.info(
                "Enter your scenario values and click Predict Environment."
            )

    if forecast is not None:

        current = (
            forecast["current"]
        )

        forecasts = (
            forecast["forecasts"]
        )

        st.divider()

        st.subheader(
            "Current Environmental State"
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Temperature",
                f"{current['temp_sht']:.2f} °C"
            )

        with c2:

            st.metric(
                "Humidity",
                f"{current['humidity_sht']:.2f} %"
            )

        with c3:

            st.metric(
                "Heat Index",
                f"{current['heat_idx']:.2f} °C"
            )

        with c4:

            st.metric(
                "Wind Speed",
                f"{current['wind_spd']:.2f} m/s"
            )

        timeline = [
            {
                "Horizon": "Current",
                "Temperature": current["temp_sht"],
                "Humidity": current["humidity_sht"],
                "Heat Index": current["heat_idx"],
                "Wind Speed": current["wind_spd"]
            }
        ]

        for horizon in [
            "1h",
            "2h",
            "3h"
        ]:

            timeline.append(
                {
                    "Horizon": (
                        f"+{horizon.replace('h', '')}h"
                    ),
                    "Temperature": (
                        forecasts[
                            horizon
                        ]["temp_sht"]
                    ),
                    "Humidity": (
                        forecasts[
                            horizon
                        ]["humidity_sht"]
                    ),
                    "Heat Index": (
                        forecasts[
                            horizon
                        ]["heat_idx"]
                    ),
                    "Wind Speed": (
                        forecasts[
                            horizon
                        ]["wind_spd"]
                    )
                }
            )

        forecast_table = pd.DataFrame(
            timeline
        )

        st.subheader(
            "🔮 Forecast Timeline"
        )

        table_display = (
            forecast_table.copy()
        )

        for col in [
            "Temperature",
            "Humidity",
            "Heat Index",
            "Wind Speed"
        ]:

            table_display[col] = (
                table_display[col]
                .round(2)
            )

        st.dataframe(
            table_display,
            width="stretch",
            hide_index=True
        )

        st.subheader(
            "🌡️ Temperature & Heat Index"
        )

        fig = px.line(
            forecast_table,
            x="Horizon",
            y=[
                "Temperature",
                "Heat Index"
            ],
            markers=True,
            title="Temperature and Heat Index Forecast"
        )

        fig.update_layout(
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=15
            ),
            legend_title_text=""
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        left, right = (
            st.columns(
                2,
                gap="large"
            )
        )

        with left:

            st.subheader(
                "💧 Humidity Forecast"
            )

            fig = px.line(
                forecast_table,
                x="Horizon",
                y="Humidity",
                markers=True,
                title="Humidity"
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=15
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with right:

            st.subheader(
                "💨 Wind Forecast"
            )

            fig = px.line(
                forecast_table,
                x="Horizon",
                y="Wind Speed",
                markers=True,
                title="Wind Speed"
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=15
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        st.subheader(
            "🌍 Environmental Outlook"
        )

        temp_change = (
            forecasts["3h"]["temp_sht"]
            - current["temp_sht"]
        )

        humidity_change = (
            forecasts["3h"]["humidity_sht"]
            - current["humidity_sht"]
        )

        heat_change = (
            forecasts["3h"]["heat_idx"]
            - current["heat_idx"]
        )

        if (
            temp_change > 0.5
            and humidity_change < -3
        ):

            outlook_title = (
                "INCREASING HEAT"
            )

            outlook_text = (
                "The model projects warmer and drier "
                "conditions over the next three hours."
            )

        elif (
            temp_change < -0.5
            and humidity_change > 3
        ):

            outlook_title = (
                "COOLING AND HUMIDITY INCREASE"
            )

            outlook_text = (
                "The model projects cooler and more humid "
                "conditions over the next three hours."
            )

        elif heat_change > 0.5:

            outlook_title = (
                "HEAT INCREASE"
            )

            outlook_text = (
                "The predicted heat index increases over "
                "the next three hours."
            )

        elif heat_change < -0.5:

            outlook_title = (
                "HEAT DECREASE"
            )

            outlook_text = (
                "The predicted heat index decreases over "
                "the next three hours."
            )

        else:

            outlook_title = "STABLE"

            outlook_text = (
                "The forecast indicates relatively stable "
                "short-term environmental conditions."
            )

        with st.container(
            border=True
        ):

            st.subheader(
                f"🌍 {outlook_title}"
            )

            st.write(
                outlook_text
            )

        st.subheader(
            "📈 Change Over 3 Hours"
        )

        q1, q2, q3, q4 = (
            st.columns(4)
        )

        with q1:

            st.metric(
                "Temperature",
                f"{temp_change:+.2f} °C"
            )

        with q2:

            st.metric(
                "Humidity",
                f"{humidity_change:+.2f} %"
            )

        with q3:

            st.metric(
                "Heat Index",
                f"{heat_change:+.2f} °C"
            )

        with q4:

            wind_change = (
                forecasts["3h"]["wind_spd"]
                - current["wind_spd"]
            )

            st.metric(
                "Wind",
                f"{wind_change:+.2f} m/s"
            )

        st.subheader(
            "🎯 Forecast Validation"
        )

        v1, v2, v3 = (
            st.columns(3)
        )

        with v1:

            st.metric(
                "Temperature",
                "R² 0.862 @ +3h"
            )

        with v2:

            st.metric(
                "Humidity",
                "R² 0.804 @ +3h"
            )

        with v3:

            st.metric(
                "Heat Index",
                "R² 0.860 @ +3h"
            )

        with st.expander(
            "Model limitations"
        ):

            st.write(
                "These validation values are measured on the "
                "current prototype dataset using a temporal "
                "test split. They should not be interpreted "
                "as general weather-forecast accuracy."
            )

            st.write(
                "Wind speed currently has substantially "
                "lower validation performance than the other "
                "forecast targets and should be interpreted cautiously."
            )

            st.write(
                "What-If predictions are hypothetical model "
                "outputs based on user-supplied conditions."
            )


elif page == "🤖 ClimateTwin Copilot":

    from backend.rag_engine import (
        prepare_question,
        generate_response_stream
    )

    st.title(
        "🤖 ClimateTwin Copilot"
    )

    st.markdown(
        '<div class="hero-subtitle">Ask • Investigate • Predict • Understand</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-caption">Data + Anomaly Intelligence + Prediction + RAG + Gemini</div>',
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "📊 DATA",
            "Connected"
        )

    with c2:

        st.metric(
            "🚨 ANOMALY",
            "Connected"
        )

    with c3:

        st.metric(
            "🔮 PREDICTION",
            "Connected"
        )

    with c4:

        st.metric(
            "📚 RAG",
            "Connected"
        )

    st.divider()

    st.subheader(
        "💬 Ask ClimateTwin"
    )

    st.caption(
        "Ask about observations, unusual conditions, forecasts, concepts, or hypothetical environmental scenarios."
    )

    sample_questions = [
        "What is wet-bulb temperature?",
        "What was the highest temperature recorded?",
        "What will happen in the next 3 hours?",
        "What will happen in the next 3 hours and why?",
        "Why should wind predictions be interpreted cautiously?",
        "Why is humidity expected to increase?",
        "What happens if temperature becomes 30°C and humidity drops to 35%?"
    ]

    selected_question = st.selectbox(
        "Quick questions",
        [
            "Choose a question"
        ] + sample_questions
    )

    if selected_question != "Choose a question":

        if st.button(
            "Use Selected Question",
            width="stretch"
        ):

            st.session_state[
                "copilot_question"
            ] = selected_question

    if (
        "chat_history"
        not in st.session_state
    ):

        st.session_state[
            "chat_history"
        ] = []

    for message in st.session_state[
        "chat_history"
    ]:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
            ):

                route_value = message.get(
                    "route"
                )

                if route_value:

                    st.caption(
                        f"Route: {route_value} • Gemini Flash"
                    )

                sources = message.get(
                    "sources",
                    []
                )

                if sources:

                    source_names = []

                    for source in sources:

                        source_name = source.get(
                            "source"
                        )

                        if (
                            source_name
                            and source_name
                            not in source_names
                        ):

                            source_names.append(
                                source_name
                            )

                    if source_names:

                        st.caption(
                            "RAG sources: "
                            + ", ".join(
                                source_names
                            )
                        )

    question = st.chat_input(
        "Ask ClimateTwin..."
    )

    if (
        question is None
        and "copilot_question"
        in st.session_state
    ):

        question = (
            st.session_state.pop(
                "copilot_question"
            )
        )

    if question:

        st.session_state[
            "chat_history"
        ].append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        try:

            prepared = prepare_question(
                question
            )

            route = prepared[
                "route"
            ]

            with st.chat_message(
                "assistant"
            ):

                with st.status(
                    "🧠 ClimateTwin is analyzing...",
                    expanded=False
                ):

                    answer = st.write_stream(
                        generate_response_stream(
                            question,
                            prepared[
                                "route"
                            ],
                            prepared[
                                "knowledge_context"
                            ],
                            prepared[
                                "data_context"
                            ],
                            prepared[
                                "anomaly_context"
                            ],
                            prepared[
                                "prediction_context"
                            ]
                        )
                    )

                st.caption(
                    f"Route: {route} • Gemini Flash"
                )

                sources = prepared.get(
                    "sources",
                    []
                )

                source_names = []

                for source in sources:

                    source_name = source.get(
                        "source"
                    )

                    if (
                        source_name
                        and source_name
                        not in source_names
                    ):

                        source_names.append(
                            source_name
                        )

                if source_names:

                    st.caption(
                        "RAG sources: "
                        + ", ".join(
                            source_names
                        )
                    )

            st.session_state[
                "chat_history"
            ].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "route": route,
                    "sources": sources
                }
            )

        except Exception as e:

            error_message = (
                f"ClimateTwin could not process the question: {e}"
            )

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    error_message
                )

            st.session_state[
                "chat_history"
            ].append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "route": "error",
                    "sources": []
                }
            )

    st.divider()

    st.subheader(
        "🔀 Copilot Intelligence Flow"
    )

    st.code(
        "User Question\n"
        "      ↓\n"
        "Question Router\n"
        "      ↓\n"
        " ┌────┼─────┬─────┐\n"
        " ↓    ↓     ↓     ↓\n"
        "Data Anomaly Prediction RAG\n"
        " └────┴─────┴─────┘\n"
        "          ↓\n"
        "    Evidence Builder\n"
        "          ↓\n"
        "        Gemini\n"
        "          ↓\n"
        "   Grounded Answer"
    )

    st.subheader(
        "🧠 Copilot Capabilities"
    )

    capability_columns = (
        st.columns(4)
    )

    with capability_columns[0]:

        with st.container(
            border=True
        ):

            st.subheader(
                "📊 Data"
            )

            st.write(
                "Answers questions using actual "
                "Conduit observations."
            )

    with capability_columns[1]:

        with st.container(
            border=True
        ):

            st.subheader(
                "🚨 Anomaly"
            )

            st.write(
                "Uses detected anomalies and "
                "environmental intelligence."
            )

    with capability_columns[2]:

        with st.container(
            border=True
        ):

            st.subheader(
                "🔮 Prediction"
            )

            st.write(
                "Uses +1h, +2h and +3h "
                "forecast models."
            )

    with capability_columns[3]:

        with st.container(
            border=True
        ):

            st.subheader(
                "📚 RAG"
            )

            st.write(
                "Retrieves relevant ClimateTwin "
                "knowledge before generating an answer."
            )


elif page == "📊 Data Explorer":

    st.title(
        "📊 Data Explorer"
    )

    st.markdown(
        '<div class="hero-subtitle">Explore the environmental intelligence dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-caption">Filter • Inspect • Download</div>',
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        st.metric(
            "FILTERED OBSERVATIONS",
            f"{len(filtered_df):,}"
        )

    with c2:

        st.metric(
            "DATASET OBSERVATIONS",
            f"{len(raw_df):,}"
        )

    with c3:

        st.metric(
            "DATASET COLUMNS",
            f"{len(filtered_df.columns):,}"
        )

    st.divider()

    st.subheader(
        "Complete Dataset"
    )

    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True
    )

    csv_data = (
        filtered_df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=csv_data,
        file_name="climatetwin_filtered_data.csv",
        mime="text/csv",
        width="stretch"
    )

    st.divider()

    st.subheader(
        "Dataset Structure"
    )

    structure_df = pd.DataFrame(
        {
            "Column": (
                filtered_df.columns
            ),
            "Data Type": [
                str(dtype)
                for dtype
                in filtered_df.dtypes
            ],
            "Non-Null Values": [
                int(
                    filtered_df[
                        col
                    ].notna().sum()
                )
                for col
                in filtered_df.columns
            ]
        }
    )

    st.dataframe(
        structure_df,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "ClimateTwin AI • Environmental Data → Intelligence → "
        "Prediction → Explain → Impact"
    )