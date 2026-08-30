import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.data_loader import load_data, merge_metadata


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IoT Sensor Analysis",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📡 IoT Sensor Analysis")

st.markdown(
    """
    Halaman ini digunakan untuk menganalisis data sensor IoT
    yang digunakan dalam sistem monitoring kesehatan sapi.

    Analisis meliputi parameter fisiologis, perilaku,
    konsumsi, kondisi lingkungan, korelasi antarvariabel,
    dan hubungan sensor dengan Disease Risk Score.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

sensor_path = Path(
    "data/raw/cattle_sensor_data.csv"
)

metadata_path = Path(
    "data/raw/cattle_metadata.csv"
)


sensor, metadata = load_data(
    sensor_path,
    metadata_path
)

df = merge_metadata(
    sensor,
    metadata
)


# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("🔎 Sensor Analysis Filter")


# Select cattle
cattle_list = [
    "All"
] + sorted(
    df["cattle_id"].unique()
)

selected_cattle = st.sidebar.selectbox(
    "Cattle ID",
    cattle_list
)


# Select observation period
day_min = int(
    df["day"].min()
)

day_max = int(
    df["day"].max()
)

selected_days = st.sidebar.slider(
    "Observation Day",
    min_value=day_min,
    max_value=day_max,
    value=(day_min, day_max)
)


# ============================================================
# FILTER DATA
# ============================================================

filtered = df[
    df["day"].between(
        selected_days[0],
        selected_days[1]
    )
].copy()


if selected_cattle != "All":

    filtered = filtered[
        filtered["cattle_id"]
        == selected_cattle
    ]


# ============================================================
# SENSOR CATEGORIES
# ============================================================

physiological = {
    "Body Temperature": "body_temperature_c",
    "Heart Rate": "heart_rate_bpm",
    "Respiration Rate": "respiration_rate_bpm",
    "Activity": "activity_percent"
}


consumption = {
    "Feed Intake": "feed_intake_kg_day",
    "Water Intake": "water_intake_l_day"
}


environment = {
    "Ambient Temperature": "ambient_temperature_c",
    "Humidity": "humidity_percent",
    "CO2": "co2_ppm",
    "NH3": "nh3_ppm"
}


# ============================================================
# KPI
# ============================================================

st.subheader("📊 Sensor Monitoring Summary")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Sensor Records",
        f"{len(filtered):,}"
    )


with col2:

    st.metric(
        "Cattle",
        filtered["cattle_id"].nunique()
    )


with col3:

    st.metric(
        "Observation Days",
        filtered["day"].nunique()
    )


with col4:

    avg_risk = filtered[
        "disease_risk_score"
    ].mean()

    st.metric(
        "Average Disease Risk",
        f"{avg_risk:.3f}"
    )


st.divider()


# ============================================================
# SENSOR VARIABLE SELECTION
# ============================================================

st.subheader(
    "📈 Sensor Trend Analysis"
)


sensor_variables = {
    **physiological,
    **consumption,
    **environment"
}

sensor_label = st.selectbox(
    "Select Sensor Variable",
    list(sensor_variables.keys())
)

sensor_column = sensor_variables[
    sensor_label
]


# ============================================================
# AGGREGATION
# ============================================================

daily_sensor = (
    filtered
    .groupby(
        "day",
        as_index=False
    )[sensor_column]
    .mean()
)


# ============================================================
# SENSOR TREND
# ============================================================

fig_sensor = px.line(
    daily_sensor,
    x="day",
    y=sensor_column,
    markers=True,
    title=f"Daily Average — {sensor_label}"
)


fig_sensor.update_layout(
    xaxis_title="Observation Day",
    yaxis_title=sensor_label
)


st.plotly_chart(
    fig_sensor,
    use_container_width=True
)


# ============================================================
# SENSOR STATISTICS
# ============================================================

st.subheader(
    "📋 Sensor Statistics"
)


stats = filtered[
    sensor_column
].describe().to_frame(
    name=sensor_label
)


st.dataframe(
    stats,
    use_container_width=True
)


# ============================================================
# PHYSIOLOGICAL PARAMETERS
# ============================================================

st.divider()

st.subheader(
    "❤️ Physiological Sensor Trends"
)


available_physiological = [
    col
    for col in physiological.values()
    if col in filtered.columns
]


if available_physiological:

    daily_physio = (
        filtered
        .groupby("day")[
            available_physiological
        ]
        .mean()
        .reset_index()
    )


    fig_physio = px.line(
        daily_physio,
        x="day",
        y=available_physiological,
        markers=True,
        title="Daily Physiological Parameters"
    )


    fig_physio.update_layout(
        xaxis_title="Observation Day",
        yaxis_title="Sensor Value"
    )


    st.plotly_chart(
        fig_physio,
        use_container_width=True
    )


# ============================================================
# ACTIVITY
# ============================================================

st.subheader(
    "🏃 Activity Monitoring"
)


if "activity_percent" in filtered.columns:

    activity_daily = (
        filtered
        .groupby(
            "day",
            as_index=False
        )["activity_percent"]
        .mean()
    )


    fig_activity = px.area(
        activity_daily,
        x="day",
        y="activity_percent",
        title="Daily Activity Level"
    )


    fig_activity.update_layout(
        xaxis_title="Observation Day",
        yaxis_title="Activity (%)"
    )


    st.plotly_chart(
        fig_activity,
        use_container_width=True
    )


# ============================================================
# FEED AND WATER
# ============================================================

st.divider()

st.subheader(
    "🍽️ Feed & Water Intake"
)


available_consumption = [
    col
    for col in consumption.values()
    if col in filtered.columns
]


if available_consumption:

    daily_consumption = (
        filtered
        .groupby("day")[
            available_consumption
        ]
        .mean()
        .reset_index()
    )


    fig_consumption = px.line(
        daily_consumption,
        x="day",
        y=available_consumption,
        markers=True,
        title="Feed and Water Intake"
    )


    fig_consumption.update_layout(
        xaxis_title="Observation Day",
        yaxis_title="Daily Intake"
    )


    st.plotly_chart(
        fig_consumption,
        use_container_width=True
    )


# ============================================================
# ENVIRONMENTAL PARAMETERS
# ============================================================

st.divider()

st.subheader(
    "🌡️ Environmental Sensor Analysis"
)


available_environment = [
    col
    for col in environment.values()
    if col in filtered.columns
]


if available_environment:

    daily_environment = (
        filtered
        .groupby("day")[
            available_environment
        ]
        .mean()
        .reset_index()
    )


    fig_environment = px.line(
        daily_environment,
        x="day",
        y=available_environment,
        markers=True,
        title="Environmental Conditions"
    )


    fig_environment.update_layout(
        xaxis_title="Observation Day",
        yaxis_title="Sensor Value"
    )


    st.plotly_chart(
        fig_environment,
        use_container_width=True
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "🔗 Sensor Correlation Analysis"
)


correlation_columns = [
    "body_temperature_c",
    "heart_rate_bpm",
    "respiration_rate_bpm",
    "activity_percent",
    "feed_intake_kg_day",
    "water_intake_l_day",
    "ambient_temperature_c",
    "humidity_percent",
    "co2_ppm",
    "nh3_ppm",
    "disease_risk_score"
]


available_correlation = [
    col
    for col in correlation_columns
    if col in filtered.columns
]


corr = filtered[
    available_correlation
].corr()


fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    title="Correlation Matrix"
)


st.plotly_chart(
    fig_corr,
    use_container_width=True
)


# ============================================================
# CORRELATION WITH DISEASE RISK
# ============================================================

st.subheader(
    "🚨 Correlation with Disease Risk Score"
)


risk_corr = (
    corr["disease_risk_score"]
    .drop("disease_risk_score")
    .sort_values()
)


risk_corr_df = (
    risk_corr
    .reset_index()
)


risk_corr_df.columns = [
    "Sensor",
    "Correlation"
]


fig_risk_corr = px.bar(
    risk_corr_df,
    x="Correlation",
    y="Sensor",
    orientation="h",
    title="Sensor Correlation with Disease Risk Score"
)


fig_risk_corr.add_vline(
    x=0,
    line_dash="dash"
)


st.plotly_chart(
    fig_risk_corr,
    use_container_width=True
)


# ============================================================
# SENSOR VS DISEASE RISK
# ============================================================

st.divider()

st.subheader(
    "⚠️ Sensor vs Disease Risk"
)


selected_sensor_for_risk = st.selectbox(
    "Select Sensor for Risk Analysis",
    [
        col
        for col in correlation_columns
        if col != "disease_risk_score"
        and col in filtered.columns
    ]
)


fig_risk = px.scatter(
    filtered.sample(
        min(5000, len(filtered)),
        random_state=42
    ),
    x=selected_sensor_for_risk,
    y="disease_risk_score",
    color="health_status",
    hover_data=[
        "cattle_id",
        "day"
    ],
    trendline="ols",
    title=(
        f"{selected_sensor_for_risk} "
        "vs Disease Risk Score"
    )
)


st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# ============================================================
# DISEASE ALERT ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "🚨 Disease Alert Sensor Comparison"
)


if "disease_alert" in filtered.columns:

    alert_summary = (
        filtered
        .groupby("disease_alert")[
            [
                "body_temperature_c",
                "heart_rate_bpm",
                "respiration_rate_bpm",
                "activity_percent",
                "ambient_temperature_c",
                "humidity_percent",
                "co2_ppm",
                "nh3_ppm"
            ]
        ]
        .mean()
        .reset_index()
    )


    alert_summary[
        "disease_alert"
    ] = alert_summary[
        "disease_alert"
    ].map({
        0: "Normal",
        1: "Disease Alert"
    })


    st.dataframe(
        alert_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DAILY SENSOR TABLE
# ============================================================

st.divider()

st.subheader(
    "📊 Daily Sensor Summary"
)


daily_summary_columns = [
    "body_temperature_c",
    "heart_rate_bpm",
    "respiration_rate_bpm",
    "activity_percent",
    "feed_intake_kg_day",
    "water_intake_l_day",
    "ambient_temperature_c",
    "humidity_percent",
    "co2_ppm",
    "nh3_ppm",
    "disease_risk_score"
]


available_daily = [
    col
    for col in daily_summary_columns
    if col in filtered.columns
]


daily_summary = (
    filtered
    .groupby("day")[
        available_daily
    ]
    .mean()
    .reset_index()
)


st.dataframe(
    daily_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RAW SENSOR DATA
# ============================================================

with st.expander(
    "🔍 Show Raw IoT Sensor Data"
):

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA QUALITY
# ============================================================

st.divider()

st.subheader(
    "🧪 Sensor Data Quality"
)


quality = pd.DataFrame({
    "Variable": filtered.columns,
    "Missing Values": [
        filtered[col].isna().sum()
        for col in filtered.columns
    ],
    "Missing (%)": [
        filtered[col].isna().mean() * 100
        for col in filtered.columns
    ]
})


st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "🧠 Automated Sensor Interpretation"
)


highest_risk_sensor = (
    risk_corr.abs()
    .sort_values(ascending=False)
    .index[0]
)


highest_corr = risk_corr[
    highest_risk_sensor
]


st.info(
    f"""
    Berdasarkan korelasi Pearson pada data yang sedang
    ditampilkan, variabel **{highest_risk_sensor}**
    memiliki hubungan paling kuat dengan
    **Disease Risk Score**, dengan koefisien korelasi
    **{highest_corr:.3f}**.

    Hasil korelasi menunjukkan hubungan statistik dan
    tidak dapat diinterpretasikan sebagai hubungan kausal.
    Untuk penelitian, hasil ini sebaiknya dikombinasikan
    dengan feature importance dan metode Explainable AI
    seperti SHAP.
    """
)


st.caption(
    "Cattle Health ML Dashboard — IoT Sensor Analysis"
)