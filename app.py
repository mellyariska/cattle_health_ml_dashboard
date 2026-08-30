import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.data_loader_v2 import load_data, merge_metadata


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cattle Health ML Dashboard",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🐄 Cattle Health Monitoring & Machine Learning Dashboard"
)

st.caption(
    "Machine Learning-Based Forecasting System for Early "
    "Disease Detection in Beef Cattle Using IoT Sensor Data"
)


# ============================================================
# DATA PATH
# ============================================================

DATA_DIR = Path("data")

sensor_path = (
    DATA_DIR / "cattle_sensor_data.csv"
)

meta_path = (
    DATA_DIR / "cattle_metadata.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():

    sensor, meta = load_data(
        sensor_path,
        meta_path
    )

    data = merge_metadata(
        sensor,
        meta
    )

    return data


try:

    df = get_data()

except Exception as e:

    st.error(
        "❌ Dataset gagal dimuat."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "cattle_id",
    "timestamp",
    "day",
    "body_temperature_c",
    "heart_rate_bpm",
    "respiration_rate_bpm",
    "activity_percent",
    "disease_risk_score",
    "health_status",
    "disease_alert"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "❌ Beberapa kolom wajib tidak ditemukan "
        "dalam dataset:"
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔎 Filter Data"
)


# Cattle selection

cattle_options = [
    "All"
] + sorted(
    df["cattle_id"]
    .dropna()
    .unique()
    .tolist()
)


selected_cattle = st.sidebar.selectbox(
    "🐄 Cattle ID",
    cattle_options
)


# Observation day

day_min = int(
    df["day"].min()
)

day_max = int(
    df["day"].max()
)


selected_days = st.sidebar.slider(
    "📅 Observation Day",
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
    ].copy()


# ============================================================
# CHECK FILTER RESULT
# ============================================================

if filtered.empty:

    st.warning(
        "⚠️ Tidak ada data yang sesuai dengan "
        "filter yang dipilih."
    )

    st.stop()


# ============================================================
# KPI
# ============================================================

latest = (
    filtered
    .sort_values("timestamp")
    .groupby("cattle_id")
    .tail(1)
)


alerts = int(
    (
        filtered["disease_alert"]
        == 1
    ).sum()
)


high_risk = int(
    (
        filtered["disease_risk_score"]
        >= 0.70
    ).sum()
)


cattle_count = (
    filtered["cattle_id"]
    .nunique()
)


avg_risk = (
    filtered["disease_risk_score"]
    .mean()
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "🐄 Cattle Monitored",
        cattle_count
    )


with c2:

    st.metric(
        "⚠️ Disease Alerts",
        alerts
    )


with c3:

    st.metric(
        "🔴 High-Risk Records",
        high_risk
    )


with c4:

    st.metric(
        "📊 Mean Risk Score",
        f"{avg_risk:.3f}"
    )


st.divider()


# ============================================================
# CURRENT CATTLE HEALTH STATUS
# ============================================================

st.subheader(
    "🐄 Current Cattle Health Status"
)


status_cols = [
    "cattle_id",
    "timestamp",
    "body_temperature_c",
    "heart_rate_bpm",
    "respiration_rate_bpm",
    "activity_percent",
    "disease_risk_score",
    "health_status",
    "disease_alert"
]


available_status_cols = [
    col
    for col in status_cols
    if col in latest.columns
]


show = (
    latest[
        available_status_cols
    ]
    .sort_values(
        "disease_risk_score",
        ascending=False
    )
)


st.dataframe(
    show,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DISEASE RISK TREND
# ============================================================

st.subheader(
    "📈 Disease Risk Over Time"
)


risk_daily = (
    filtered
    .groupby(
        "day",
        as_index=False
    )[
        "disease_risk_score"
    ]
    .mean()
)


fig = px.line(
    risk_daily,
    x="day",
    y="disease_risk_score",
    markers=True,
    title="Average Disease Risk by Observation Day"
)


fig.add_hline(
    y=0.70,
    line_dash="dash",
    annotation_text="High-risk threshold"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# HEALTH STATUS + PHYSIOLOGICAL PROFILE
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# HEALTH STATUS DISTRIBUTION
# ============================================================

with col1:

    status_counts = (
        filtered[
            "health_status"
        ]
        .value_counts()
        .reset_index()
    )


    status_counts.columns = [
        "health_status",
        "count"
    ]


    fig2 = px.bar(
        status_counts,
        x="health_status",
        y="count",
        title="Health Status Distribution",
        text_auto=True
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# PHYSIOLOGICAL PROFILE
# ============================================================

with col2:

    sample_size = min(
        5000,
        len(filtered)
    )


    sample_data = filtered.sample(
        sample_size,
        random_state=42
    )


    fig3 = px.scatter(
        sample_data,
        x="body_temperature_c",
        y="heart_rate_bpm",
        color="health_status",
        size="disease_risk_score",
        hover_data=[
            "cattle_id",
            "day",
            "disease_risk_score"
        ],
        title="Physiological Profile"
    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

st.subheader(
    "🌡️ Environmental Conditions"
)


environment_columns = [
    "ambient_temperature_c",
    "humidity_percent",
    "co2_ppm",
    "nh3_ppm"
]


available_environment_columns = [
    col
    for col in environment_columns
    if col in filtered.columns
]


env = (
    filtered
    .groupby(
        "day",
        as_index=False
    )[
        available_environment_columns
    ]
    .mean()
)


if (
    "ambient_temperature_c"
    in env.columns
    and
    "humidity_percent"
    in env.columns
):

    fig4 = px.line(
        env,
        x="day",
        y=[
            "ambient_temperature_c",
            "humidity_percent"
        ],
        markers=True,
        title="Average Temperature & Humidity"
    )


    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# ============================================================
# ENVIRONMENTAL SUMMARY
# ============================================================

if available_environment_columns:

    st.subheader(
        "🌱 Environmental Sensor Summary"
    )


    environmental_summary = (
        filtered[
            available_environment_columns
        ]
        .describe()
        .T
    )


    st.dataframe(
        environmental_summary,
        use_container_width=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📊 Dataset Information"
)


info1, info2, info3 = st.columns(3)


with info1:

    st.metric(
        "Total Records",
        f"{len(filtered):,}"
    )


with info2:

    st.metric(
        "Variables",
        len(filtered.columns)
    )


with info3:

    st.metric(
        "Cattle IDs",
        filtered["cattle_id"]
        .nunique()
    )


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "Gunakan menu halaman di sidebar untuk mengeksplorasi "
    "Individual Cattle Monitoring, IoT Sensor Analysis, "
    "Machine Learning Early Disease Detection, dan Data Quality."
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Cattle Health ML Dashboard | "
    "IoT Sensor Data + Machine Learning"
)
