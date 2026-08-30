import streamlit as st
import plotly.express as px
from pathlib import Path

from src.data_loader import load_data, merge_metadata


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
    masing-masing sapi berdasarkan data fisiologis,
    perilaku, konsumsi, dan risiko penyakit.
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
# SELECT CATTLE
# ============================================================

st.sidebar.header("🐄 Cattle Selection")

cattle_list = sorted(
    df["cattle_id"].unique()
)

selected_cattle = st.sidebar.selectbox(
    "Select Cattle",
    cattle_list
)


# ============================================================
# FILTER SELECTED CATTLE
# ============================================================

cattle_data = df[
    df["cattle_id"] == selected_cattle
].copy()

cattle_data = cattle_data.sort_values(
    "timestamp"
)


# ============================================================
# CURRENT CONDITION
# ============================================================

latest = cattle_data.iloc[-1]


st.subheader(
    f"Current Health Status — {selected_cattle}"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🌡️ Body Temperature",
        f"{latest['body_temperature_c']:.2f} °C"
    )


with col2:

    st.metric(
        "❤️ Heart Rate",
        f"{latest['heart_rate_bpm']:.1f} bpm"
    )


with col3:

    st.metric(
        "🫁 Respiration Rate",
        f"{latest['respiration_rate_bpm']:.1f} bpm"
    )


with col4:

    st.metric(
        "⚠️ Disease Risk",
        f"{latest['disease_risk_score']:.3f}"
    )


# ============================================================
# HEALTH STATUS
# ============================================================

st.divider()

status = latest["health_status"]

disease_alert = latest["disease_alert"]

col1, col2 = st.columns(2)


with col1:

    st.subheader("Health Status")

    st.write(
        f"### {status}"
    )


with col2:

    st.subheader("Disease Alert")

    if disease_alert == 1:

        st.error(
            "⚠️ DISEASE ALERT"
        )

    else:

        st.success(
            "✅ NORMAL"
        )


# ============================================================
# CATTLE INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📋 Cattle Information"
)


info_columns = [
    "cattle_id",
    "breed",
    "sex",
    "age_months"
]


available_info = [
    col
    for col in info_columns
    if col in cattle_data.columns
]


if available_info:

    cattle_info = (
        cattle_data[
            available_info
        ]
        .drop_duplicates()
    )

    st.dataframe(
        cattle_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PHYSIOLOGICAL TRENDS
# ============================================================

st.divider()

st.subheader(
    "📈 Physiological Parameters"
)


physiological_columns = [
    "body_temperature_c",
    "heart_rate_bpm",
    "respiration_rate_bpm"
]


available_physiological = [
    col
    for col in physiological_columns
    if col in cattle_data.columns
]


fig1 = px.line(
    cattle_data,
    x="timestamp",
    y=available_physiological,
    title=f"Physiological Trends — {selected_cattle}",
    markers=False
)


fig1.update_layout(
    xaxis_title="Time",
    yaxis_title="Value"
)


st.plotly_chart(
    fig1,
    use_container_width=True
)


# ============================================================
# ACTIVITY
# ============================================================

st.subheader(
    "🏃 Activity Monitoring"
)


if "activity_percent" in cattle_data.columns:

    fig2 = px.line(
        cattle_data,
        x="timestamp",
        y="activity_percent",
        title="Activity Level"
    )

    fig2.update_layout(
        xaxis_title="Time",
        yaxis_title="Activity (%)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# FEED AND WATER INTAKE
# ============================================================

st.subheader(
    "🍽️ Feed & Water Intake"
)


intake_columns = [
    "feed_intake_kg_day",
    "water_intake_l_day"
]


available_intake = [
    col
    for col in intake_columns
    if col in cattle_data.columns
]


if available_intake:

    fig3 = px.line(
        cattle_data,
        x="timestamp",
        y=available_intake,
        title="Feed and Water Intake"
    )

    fig3.update_layout(
        xaxis_title="Time",
        yaxis_title="Daily Intake"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

st.divider()

st.subheader(
    "🌡️ Environmental Conditions"
)


environment_columns = [
    "ambient_temperature_c",
    "humidity_percent",
    "co2_ppm",
    "nh3_ppm"
]


available_environment = [
    col
    for col in environment_columns
    if col in cattle_data.columns
]


if available_environment:

    fig4 = px.line(
        cattle_data,
        x="timestamp",
        y=available_environment,
        title="Environmental Conditions"
    )

    fig4.update_layout(
        xaxis_title="Time",
        yaxis_title="Sensor Value"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# ============================================================
# DISEASE RISK SCORE
# ============================================================

st.divider()

st.subheader(
    "🚨 Disease Risk Score"
)


fig5 = px.line(
    cattle_data,
    x="timestamp",
    y="disease_risk_score",
    title=f"Disease Risk Trend — {selected_cattle}"
)


# Medium risk threshold
fig5.add_hline(
    y=0.30,
    line_dash="dash",
    annotation_text="Medium Risk"
)


# High risk threshold
fig5.add_hline(
    y=0.70,
    line_dash="dash",
    annotation_text="High Risk"
)


fig5.update_layout(
    xaxis_title="Time",
    yaxis_title="Disease Risk Score",
    yaxis_range=[0, 1]
)


st.plotly_chart(
    fig5,
    use_container_width=True
)


# ============================================================
# DAILY SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📊 Daily Health Summary"
)


daily_summary = (
    cattle_data
    .groupby("day")
    .agg(
        body_temperature_c=(
            "body_temperature_c",
            "mean"
        ),
        heart_rate_bpm=(
            "heart_rate_bpm",
            "mean"
        ),
        respiration_rate_bpm=(
            "respiration_rate_bpm",
            "mean"
        ),
        activity_percent=(
            "activity_percent",
            "mean"
        ),
        disease_risk_score=(
            "disease_risk_score",
            "mean"
        )
    )
    .reset_index()
)


st.dataframe(
    daily_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "🔍 Show Raw Sensor Data"
):

    st.dataframe(
        cattle_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "🧠 Automated Interpretation"
)


risk = float(
    latest["disease_risk_score"]
)


if risk < 0.30:

    st.success(
        f"""
        **{selected_cattle} berada pada kategori risiko rendah.**

        Disease Risk Score saat ini adalah
        **{risk:.3f}**.

        Kondisi sensor menunjukkan kondisi yang relatif
        normal berdasarkan model.
        """
    )

elif risk < 0.70:

    st.warning(
        f"""
        **{selected_cattle} berada pada kategori risiko sedang.**

        Disease Risk Score saat ini adalah
        **{risk:.3f}**.

        Disarankan melakukan pemantauan lebih lanjut
        terhadap perubahan parameter fisiologis,
        aktivitas, konsumsi pakan, dan kondisi lingkungan.
        """
    )

else:

    st.error(
        f"""
        **{selected_cattle} berada pada kategori risiko tinggi.**

        Disease Risk Score saat ini adalah
        **{risk:.3f}**.

        Sapi perlu mendapatkan pemeriksaan lebih lanjut
        oleh tenaga/dokter hewan. Hasil dashboard bukan
        merupakan diagnosis klinis.
        """
    )


st.caption(
    "Cattle Health ML Dashboard — Early Disease Detection "
    "Using IoT Sensor Data"
)