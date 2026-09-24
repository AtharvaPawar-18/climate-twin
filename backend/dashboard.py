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


def find_column(df, candidates):
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
    "🌍 ClimateTwin AI"
)

st.sidebar.caption(
    "Environmental Intelligence System"
)

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
    "Dashboard Filters"
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

    st.subheader(
        "AI-Powered Environmental Intelligence"
    )

    st.write(
        "From Data → Insight → Action → Impact"
    )

    st.success(
        f"Dataset loaded successfully — "
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
            st.columns(2)
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

                st.metric(
                    "Current Heat Index",
                    format_value(
                        latest.get(
                            "heat_idx"
                        ),
                        " °C"
                    )
                )

                st.metric(
                    "Current Humidity",
                    format_value(
                        latest.get(
                            "humidity_sht"
                        ),
                        "%"
                    )
                )

                st.caption(
                    f"Observation: {latest['ts']}"
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
                title=(
                    "Temperature and Heat Index"
                )
            )

            fig.update_layout(
                height=430,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
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
                height=380,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
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
                height=380,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
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
            title=(
                "Environmental Priority Timeline"
            )
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
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
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
            title=(
                "Detected Environmental Events"
            )
        )

        fig.update_layout(
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
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
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No high-priority events are present "
                "in the current filters."
            )

    st.header(
        "🗺️ Environmental Intelligence Map"
    )

    st.info(
        "The current Conduit dataset does not contain "
        "latitude/longitude fields. The map layer will "
        "automatically activate when geospatial coordinates "
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
            "Detected Environmental Events",
            elevated_count
        )

    with i2:

        st.metric(
            "Elevated Risk Observations",
            elevated_count
        )

    with i3:

        st.metric(
            "Elevated Risk Rate",
            f"{elevated_rate:.2f}%"
        )

    st.info(
        "ClimateTwin AI converts environmental observations "
        "into environmental events, risk levels, signals, "
        "and recommended actions to support earlier "
        "decision-making."
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
            st.columns(2)
        )

        with col1:

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

    st.caption(
        "ClimateTwin AI • Environmental Data → Intelligence → "
        "Prediction → Action → Impact"
    )


elif page == "🔮 Prediction Lab":

    st.title(
        "🔮 ClimateTwin Prediction Lab"
    )

    st.write(
        "Forecast the short-term environmental state "
        "using Conduit data or create a custom "
        "what-if scenario."
    )

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
                "Forecast generated from the latest "
                "available Conduit observation."
            )

        except Exception as e:

            st.error(
                f"Prediction system error: {e}"
            )

    else:

        st.subheader(
            "🧪 What-If Scenario"
        )

        try:

            live_forecast = predict_all()

            defaults = (
                live_forecast["current"]
            )

        except Exception as e:

            st.error(
                f"Unable to load current conditions: {e}"
            )

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
                st.columns(2)
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
                    use_container_width=True
                )
            )

        if predict_button:

            try:

                custom_forecast = (
                    predict_custom(
                        input_temperature,
                        input_humidity,
                        input_heat_index,
                        input_wind
                    )
                )

                st.session_state[
                    "custom_forecast"
                ] = custom_forecast

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
                "Enter environmental conditions "
                "and click Predict Environment."
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
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🌡️ Temperature & Heat Index Forecast"
        )

        fig = px.line(
            forecast_table,
            x="Horizon",
            y=[
                "Temperature",
                "Heat Index"
            ],
            markers=True,
            title=(
                "Temperature and Heat Index Forecast"
            )
        )

        fig.update_layout(
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "💧 Humidity Forecast"
        )

        fig = px.line(
            forecast_table,
            x="Horizon",
            y="Humidity",
            markers=True,
            title="Humidity Forecast"
        )

        fig.update_layout(
            height=380,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "💨 Wind Forecast"
        )

        fig = px.line(
            forecast_table,
            x="Horizon",
            y="Wind Speed",
            markers=True,
            title="Wind Speed Forecast"
        )

        fig.update_layout(
            height=380,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "🌍 Predicted Environmental Outlook"
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
                "The model projects warmer and "
                "drier conditions over the next "
                "three hours."
            )

        elif (
            temp_change < -0.5
            and humidity_change > 3
        ):

            outlook_title = (
                "COOLING AND HUMIDITY INCREASE"
            )

            outlook_text = (
                "The model projects cooler and "
                "more humid conditions over the "
                "next three hours."
            )

        elif heat_change > 0.5:

            outlook_title = (
                "HEAT INCREASE"
            )

            outlook_text = (
                "The predicted heat index increases "
                "over the next three hours."
            )

        elif heat_change < -0.5:

            outlook_title = (
                "HEAT DECREASE"
            )

            outlook_text = (
                "The predicted heat index decreases "
                "over the next three hours."
            )

        else:

            outlook_title = "STABLE"

            outlook_text = (
                "The forecast indicates relatively "
                "stable short-term environmental "
                "conditions."
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
            "📈 Three-Hour Change"
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
            "🎯 Model Validation"
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

        st.caption(
            "Validation values come from the current "
            "prototype dataset and temporal test split. "
            "Wind prediction has substantially lower "
            "validation performance and should be "
            "interpreted cautiously."
        )

        if forecast.get(
            "mode"
        ) == "what_if":

            st.info(
                "This is a what-if scenario. The model "
                "uses the current Conduit history as "
                "context while replacing the current "
                "environmental state with your entered values."
            )


elif page == "🤖 ClimateTwin Copilot":

    from backend.rag_engine import (
        prepare_question,
        generate_response_stream
    )

    st.title(
        "🤖 ClimateTwin Copilot"
    )

    st.write(
        "Ask ClimateTwin about environmental observations, "
        "anomalies, forecasts, methodology, or environmental concepts."
    )

    st.info(
        "ClimateTwin Copilot combines Conduit observations, "
        "anomaly intelligence, forecasting models, retrieved "
        "knowledge, and Gemini."
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "📊 Data",
            "Connected"
        )

    with c2:

        st.metric(
            "🚨 Anomaly",
            "Connected"
        )

    with c3:

        st.metric(
            "🔮 Prediction",
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

    sample_questions = [
        "What is wet-bulb temperature?",
        "What was the highest temperature recorded?",
        "What will happen in the next 3 hours?",
        "What will happen in the next 3 hours and why?",
        "Why should wind predictions be interpreted cautiously?",
        "Why is humidity expected to increase?"
    ]

    selected_question = st.selectbox(
        "Try a question",
        [
            "Choose a question"
        ] + sample_questions
    )

    if selected_question != "Choose a question":

        if st.button(
            "Use Selected Question",
            use_container_width=True
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
                        f"Route: {route_value} | "
                        f"Gemini Flash"
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
                    f"Route: {route} | Gemini Flash"
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
                f"ClimateTwin could not process "
                f"the question: {e}"
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

    capability_columns = st.columns(4)

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
                "Uses the +1h, +2h and +3h "
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
        "📊 ClimateTwin Data Explorer"
    )

    st.write(
        "Explore the filtered environmental "
        "intelligence dataset."
    )

    st.subheader(
        f"{len(filtered_df):,} filtered observations"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
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
        file_name=(
            "climatetwin_filtered_data.csv"
        ),
        mime="text/csv",
        use_container_width=True
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
        use_container_width=True,
        hide_index=True
    )