import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.data_loader import load_data, merge_metadata
from src.ml_model import train_models, predict_risk

st.set_page_config(
    page_title="Cattle Health ML Dashboard",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐄 Cattle Health Monitoring & Machine Learning Dashboard")
st.caption(
    "Machine Learning-Based Forecasting System for Early Disease Detection "
    "in Beef Cattle Using IoT Sensor Data"
)

DATA_DIR = Path("data/raw")
sensor_path = DATA_DIR / "cattle_sensor_data.csv"
meta_path = DATA_DIR / "cattle_metadata.csv"


@st.cache_data
def get_data():
    sensor, meta = load_data(sensor_path, meta_path)
    return merge_metadata(sensor, meta)


df = get_data()


# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("🔎 Filter Data")

cattle_options = ["All"] + sorted(df["cattle_id"].unique().tolist())

selected_cattle = st.sidebar.selectbox(
    "Cattle ID",
    cattle_options
)

day_min = int(df["day"].min())
day_max = int(df["day"].max())

selected_days = st.sidebar.slider(
    "Observation Day",
    day_min,
    day_max,
    (day_min, day_max)
)


filtered = df[
    df["day"].between(
        selected_days[0],
        selected_days[1]
    )
].copy()


if selected_cattle != "All":
    filtered = filtered[
        filtered["cattle_id"] == selected_cattle
    ]


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
    (filtered["disease_alert"] == 1).sum()
)

high_risk = int(
    (filtered["disease_risk_score"] >= 0.70).sum()
)

cattle_count = filtered["cattle_id"].nunique()

avg_risk = filtered["disease_risk_score"].mean()


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🐄 Cattle Monitored",
    cattle_count
)

c2.metric(
    "⚠️ Disease Alerts",
    alerts
)

c3.metric(
    "🔴 High-Risk Records",
    high_risk
)

c4.metric(
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

if not latest.empty:

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

    show = (
        latest[status_cols]
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
    )["disease_risk_score"]
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


# ------------------------------------------------------------
# HEALTH STATUS DISTRIBUTION
# ------------------------------------------------------------

with col1:

    status_counts = (
        filtered["health_status"]
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


# ------------------------------------------------------------
# PHYSIOLOGICAL PROFILE
# ------------------------------------------------------------

with col2:

    sample_data = filtered.sample(
        min(5000, len(filtered)),
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

env = (
    filtered
    .groupby("day", as_index=False)[
        [
            "ambient_temperature_c",
            "humidity_percent",
            "co2_ppm",
            "nh3_ppm"
        ]
    ]
    .mean()
)


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
# INFORMATION
# ============================================================

st.info(
    "Gunakan menu halaman di sidebar untuk mengeksplorasi "
    "individual cattle monitoring, sensor IoT, machine learning "
    "early disease detection, feature importance, dan data quality."
)