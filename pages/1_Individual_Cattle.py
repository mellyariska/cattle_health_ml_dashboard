import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.data_loader_v2 import load_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Individual Cattle Monitoring",
    page_icon="🐄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🐄 Individual Cattle Monitoring")

st.markdown(
    """
    Halaman ini digunakan untuk memantau kondisi kesehatan
    masing-masing sapi berdasarkan data fisiologis, perilaku,
    konsumsi, lingkungan, dan risiko penyakit.
    """
)


# ============================================================
# DATA PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"

sensor_path = DATA_DIR / "cattle_sensor_data.csv"
metadata_path = DATA_DIR / "cattle_metadata.csv"


# ============================================================
# CHECK DATA FILE
# ============================================================

if not sensor_path.exists():

    st.error(
        f"Dataset sensor tidak ditemukan: {sensor_path}"
    )

    st.stop()


if not metadata_path.exists():

    st.error(
        f"Dataset metadata tidak ditemukan: {metadata_path}"
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():

    sensor, metadata = load_data(
        sensor_path,
        metadata_path
    )

    return sensor, metadata


try:

    sensor, metadata = get_data()

except Exception as e:

    st.error("Dataset gagal dimuat.")

    st.exception(e)

    st.stop()


# ============================================================
# MERGE DATA
# ============================================================

df = sensor.merge(
    metadata,
    on="cattle_id",
    how="left",
    suffixes=("", "_metadata")
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Cattle Selection")


cattle_list = sorted(
    df["cattle_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if len(cattle_list) == 0:

    st.warning(
        "Tidak ada cattle_id yang tersedia."
    )

    st.stop()


selected_cattle = st.sidebar.selectbox(
    "Pilih Cattle ID",
    cattle_list
)


# ============================================================
# FILTER INDIVIDUAL CATTLE
# ============================================================

cattle_df = df[
    df["cattle_id"].astype(str)
    == str(selected_cattle)
].copy()


if cattle_df.empty:

    st.warning(
        "Data untuk sapi yang dipilih tidak tersedia."
    )

    st.stop()


# ============================================================
# SORT
# ============================================================

if "timestamp" in cattle_df.columns:

    cattle_df = cattle_df.sort_values(
        "timestamp"
    )


# ============================================================
# LATEST RECORD
# ============================================================

latest = cattle_df.iloc[-1]


# ============================================================
# CATTLE INFORMATION
# ============================================================

st.subheader(
    f"🐄 Cattle ID: {selected_cattle}"
)


info1, info2, info3, info4 = st.columns(4)


info1.metric(
    "Observation Records",
    len(cattle_df)
)


if "day" in cattle_df.columns:

    info2.metric(
        "Observation Day",
        int(pd.to_numeric(
            latest["day"],
            errors="coerce"
        ))
    )


if "disease_risk_score" in cattle_df.columns:

    risk = pd.to_numeric(
        latest["disease_risk_score"],
        errors="coerce"
    )

    if pd.isna(risk):
        risk = 0

    info3.metric(
        "Disease Risk",
        f"{risk:.3f}"
    )


if "health_status" in cattle_df.columns:

    info4.metric(
        "Health Status",
        str(latest["health_status"])
    )


st.divider()


# ============================================================
# PHYSIOLOGICAL PARAMETERS
# ============================================================

st.subheader(
    "🌡️ Current Physiological Parameters"
)


p1, p2, p3, p4 = st.columns(4)


if "body_temperature_c" in cattle_df.columns:

    value = pd.to_numeric(
        latest["body_temperature_c"],
        errors="coerce"
    )

    p1.metric(
        "Body Temperature",
        f"{value:.2f} °C"
        if pd.notna(value)
        else "N/A"
    )


if "heart_rate_bpm" in cattle_df.columns:

    value = pd.to_numeric(
        latest["heart_rate_bpm"],
        errors="coerce"
    )

    p2.metric(
        "Heart Rate",
        f"{value:.1f} bpm"
        if pd.notna(value)
        else "N/A"
    )


if "respiration_rate_bpm" in cattle_df.columns:

    value = pd.to_numeric(
        latest["respiration_rate_bpm"],
        errors="coerce"
    )

    p3.metric(
        "Respiration Rate",
        f"{value:.1f} bpm"
        if pd.notna(value)
        else "N/A"
    )


if "activity_percent" in cattle_df.columns:

    value = pd.to_numeric(
        latest["activity_percent"],
        errors="coerce"
    )

    p4.metric(
        "Activity",
        f"{value:.1f}%"
        if pd.notna(value)
        else "N/A"
    )


# ============================================================
# DISEASE RISK TREND
# ============================================================

if (
    "day" in cattle_df.columns
    and "disease_risk_score" in cattle_df.columns
):

    st.subheader(
        "📈 Disease Risk Trend"
    )

    risk_data = cattle_df[
        ["day", "disease_risk_score"]
    ].copy()

    risk_data["day"] = pd.to_numeric(
        risk_data["day"],
        errors="coerce"
    )

    risk_data["disease_risk_score"] = pd.to_numeric(
        risk_data["disease_risk_score"],
        errors="coerce"
    )

    risk_data = risk_data.dropna()


    fig = px.line(
        risk_data,
        x="day",
        y="disease_risk_score",
        markers=True,
        title=f"Disease Risk - {selected_cattle}"
    )


    fig.add_hline(
        y=0.70,
        line_dash="dash",
        annotation_text="High Risk Threshold"
    )


    fig.add_hline(
        y=0.40,
        line_dash="dot",
        annotation_text="Moderate Risk Threshold"
    )


    fig.update_yaxes(
        range=[0, 1],
        title="Disease Risk Score"
    )

    fig.update_xaxes(
        title="Observation Day"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# BODY TEMPERATURE
# ============================================================

if (
    "timestamp" in cattle_df.columns
    and "body_temperature_c" in cattle_df.columns
):

    st.subheader(
        "🌡️ Body Temperature Trend"
    )


    temp_data = cattle_df[
        ["timestamp", "body_temperature_c"]
    ].copy()


    temp_data["body_temperature_c"] = pd.to_numeric(
        temp_data["body_temperature_c"],
        errors="coerce"
    )


    temp_data = temp_data.dropna()


    fig_temp = px.line(
        temp_data,
        x="timestamp",
        y="body_temperature_c",
        markers=True,
        title="Body Temperature Over Time"
    )


    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )


# ============================================================
# HEART RATE
# ============================================================

if (
    "timestamp" in cattle_df.columns
    and "heart_rate_bpm" in cattle_df.columns
):

    st.subheader(
        "❤️ Heart Rate Trend"
    )


    hr_data = cattle_df[
        ["timestamp", "heart_rate_bpm"]
    ].copy()


    hr_data["heart_rate_bpm"] = pd.to_numeric(
        hr_data["heart_rate_bpm"],
        errors="coerce"
    )


    hr_data = hr_data.dropna()


    fig_hr = px.line(
        hr_data,
        x="timestamp",
        y="heart_rate_bpm",
        markers=True,
        title="Heart Rate Over Time"
    )


    st.plotly_chart(
        fig_hr,
        use_container_width=True
    )


# ============================================================
# ACTIVITY
# ============================================================

if (
    "timestamp" in cattle_df.columns
    and "activity_percent" in cattle_df.columns
):

    st.subheader(
        "🏃 Activity Level"
    )


    activity_data = cattle_df[
        ["timestamp", "activity_percent"]
    ].copy()


    activity_data["activity_percent"] = pd.to_numeric(
        activity_data["activity_percent"],
        errors="coerce"
    )


    activity_data = activity_data.dropna()


    fig_activity = px.line(
        activity_data,
        x="timestamp",
        y="activity_percent",
        markers=True,
        title="Activity Over Time"
    )


    st.plotly_chart(
        fig_activity,
        use_container_width=True
    )


# ============================================================
# ENVIRONMENT
# ============================================================

environment_columns = [
    "ambient_temperature_c",
    "humidity_percent",
    "co2_ppm",
    "nh3_ppm"
]


available_environment = [
    col
    for col in environment_columns
    if col in cattle_df.columns
]


if available_environment:

    st.subheader(
        "🌱 Environmental Conditions"
    )


    env_data = cattle_df[
        ["timestamp"] + available_environment
    ].copy()


    for col in available_environment:

        env_data[col] = pd.to_numeric(
            env_data[col],
            errors="coerce"
        )


    env_data = env_data.dropna(
        subset=["timestamp"]
    )


    fig_env = px.line(
        env_data,
        x="timestamp",
        y=available_environment,
        title="Environmental Sensor Data"
    )


    st.plotly_chart(
        fig_env,
        use_container_width=True
    )


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "📋 Lihat Data Sensor Sapi"
):

    st.dataframe(
        cattle_df,
        use_container_width=True,
        hide_index=True
    )
