import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.data_loader_v2 import load_data


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
    yang digunakan dalam pemantauan kesehatan sapi.
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
# CHECK DATA
# ============================================================

if not sensor_path.exists():

    st.error(
        f"Sensor dataset tidak ditemukan: {sensor_path}"
    )

    st.stop()


if not metadata_path.exists():

    st.error(
        f"Metadata dataset tidak ditemukan: {metadata_path}"
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
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("🔎 Sensor Filter")


cattle_options = [
    "All"
] + sorted(
    df["cattle_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


selected_cattle = st.sidebar.selectbox(
    "Cattle ID",
    cattle_options
)


if selected_cattle != "All":

    df = df[
        df["cattle_id"].astype(str)
        == str(selected_cattle)
    ].copy()


# ============================================================
# KPI
# ============================================================

st.subheader("📊 Sensor Overview")


k1, k2, k3, k4 = st.columns(4)


k1.metric(
    "🐄 Cattle",
    df["cattle_id"].nunique()
)


k2.metric(
    "📡 Sensor Records",
    len(df)
)


if "body_temperature_c" in df.columns:

    temperature = pd.to_numeric(
        df["body_temperature_c"],
        errors="coerce"
    ).mean()

    k3.metric(
        "🌡️ Mean Body Temperature",
        f"{temperature:.2f} °C"
        if pd.notna(temperature)
        else "N/A"
    )


if "heart_rate_bpm" in df.columns:

    heart_rate = pd.to_numeric(
        df["heart_rate_bpm"],
        errors="coerce"
    ).mean()

    k4.metric(
        "❤️ Mean Heart Rate",
        f"{heart_rate:.1f} bpm"
        if pd.notna(heart_rate)
        else "N/A"
    )


st.divider()


# ============================================================
# BODY TEMPERATURE
# ============================================================

if (
    "timestamp" in df.columns
    and "body_temperature_c" in df.columns
):

    st.subheader(
        "🌡️ Body Temperature Monitoring"
    )

    temp = df[
        [
            "timestamp",
            "body_temperature_c",
            "cattle_id"
        ]
    ].copy()

    temp["body_temperature_c"] = pd.to_numeric(
        temp["body_temperature_c"],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=[
            "timestamp",
            "body_temperature_c"
        ]
    )

    fig_temp = px.line(
        temp,
        x="timestamp",
        y="body_temperature_c",
        color="cattle_id"
        if selected_cattle == "All"
        else None,
        title="Body Temperature Over Time"
    )

    fig_temp.update_yaxes(
        title="Body Temperature (°C)"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )


# ============================================================
# HEART RATE
# ============================================================

if (
    "timestamp" in df.columns
    and "heart_rate_bpm" in df.columns
):

    st.subheader(
        "❤️ Heart Rate Monitoring"
    )

    hr = df[
        [
            "timestamp",
            "heart_rate_bpm",
            "cattle_id"
        ]
    ].copy()

    hr["heart_rate_bpm"] = pd.to_numeric(
        hr["heart_rate_bpm"],
        errors="coerce"
    )

    hr = hr.dropna(
        subset=[
            "timestamp",
            "heart_rate_bpm"
        ]
    )

    fig_hr = px.line(
        hr,
        x="timestamp",
        y="heart_rate_bpm",
        color="cattle_id"
        if selected_cattle == "All"
        else None,
        title="Heart Rate Over Time"
    )

    fig_hr.update_yaxes(
        title="Heart Rate (bpm)"
    )

    st.plotly_chart(
        fig_hr,
        use_container_width=True
    )


# ============================================================
# RESPIRATION RATE
# ============================================================

if (
    "timestamp" in df.columns
    and "respiration_rate_bpm" in df.columns
):

    st.subheader(
        "🫁 Respiration Rate"
    )

    respiration = df[
        [
            "timestamp",
            "respiration_rate_bpm",
            "cattle_id"
        ]
    ].copy()

    respiration["respiration_rate_bpm"] = pd.to_numeric(
        respiration["respiration_rate_bpm"],
        errors="coerce"
    )

    respiration = respiration.dropna(
        subset=[
            "timestamp",
            "respiration_rate_bpm"
        ]
    )

    fig_resp = px.line(
        respiration,
        x="timestamp",
        y="respiration_rate_bpm",
        color="cattle_id"
        if selected_cattle == "All"
        else None,
        title="Respiration Rate Over Time"
    )

    fig_resp.update_yaxes(
        title="Respiration Rate (breaths/min)"
    )

    st.plotly_chart(
        fig_resp,
        use_container_width=True
    )


# ============================================================
# ACTIVITY
# ============================================================

if (
    "timestamp" in df.columns
    and "activity_percent" in df.columns
):

    st.subheader(
        "🏃 Activity Monitoring"
    )

    activity = df[
        [
            "timestamp",
            "activity_percent",
            "cattle_id"
        ]
    ].copy()

    activity["activity_percent"] = pd.to_numeric(
        activity["activity_percent"],
        errors="coerce"
    )

    activity = activity.dropna(
        subset=[
            "timestamp",
            "activity_percent"
        ]
    )

    fig_activity = px.line(
        activity,
        x="timestamp",
        y="activity_percent",
        color="cattle_id"
        if selected_cattle == "All"
        else None,
        title="Activity Level Over Time"
    )

    fig_activity.update_yaxes(
        title="Activity (%)"
    )

    st.plotly_chart(
        fig_activity,
        use_container_width=True
    )


# ============================================================
# ENVIRONMENTAL SENSOR
# ============================================================

environment_columns = [
    "ambient_temperature_c",
    "humidity_percent",
    "co2_ppm",
    "nh3_ppm"
]


available_environment = [
    column
    for column in environment_columns
    if column in df.columns
]


if (
    "timestamp" in df.columns
    and len(available_environment) > 0
):

    st.subheader(
        "🌱 Environmental Sensor Monitoring"
    )

    environment = df[
        ["timestamp"] + available_environment
    ].copy()

    for column in available_environment:

        environment[column] = pd.to_numeric(
            environment[column],
            errors="coerce"
        )

    fig_environment = px.line(
        environment,
        x="timestamp",
        y=available_environment,
        title="Environmental Conditions"
    )

    st.plotly_chart(
        fig_environment,
        use_container_width=True
    )


# ============================================================
# SENSOR CORRELATION
# ============================================================

st.subheader(
    "🔗 Sensor Correlation"
)


numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()


if len(numeric_columns) >= 2:

    correlation = df[
        numeric_columns
    ].corr()

    fig_corr = px.imshow(
        correlation,
        text_auto=".2f",
        title="Correlation Matrix of Sensor Variables"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True

    )

else:

    st.info(
        "Belum tersedia cukup variabel numerik "
        "untuk membuat correlation matrix."
    )


# ============================================================
# RAW SENSOR DATA
# ============================================================

with st.expander(
    "📋 Lihat Raw Sensor Data"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
