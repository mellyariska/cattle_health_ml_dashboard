import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from src.data_loader import load_data, merge_metadata


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Quality",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧪 Data Quality & Dataset Validation")

st.markdown(
    """
    Halaman ini digunakan untuk melakukan pemeriksaan kualitas
    dataset sebelum digunakan dalam proses Machine Learning.

    Pemeriksaan meliputi:
    - jumlah data
    - struktur variabel
    - missing values
    - duplicate records
    - range data
    - outlier
    - konsistensi cattle ID
    - konsistensi timestamp
    - distribusi disease alert
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
# DATASET OVERVIEW
# ============================================================

st.subheader(
    "📊 Dataset Overview"
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Sensor Records",
        f"{len(sensor):,}"
    )


with col2:

    st.metric(
        "Metadata Records",
        f"{len(metadata):,}"
    )


with col3:

    st.metric(
        "Number of Cattle",
        df["cattle_id"].nunique()
    )


with col4:

    st.metric(
        "Variables",
        len(df.columns)
    )


with col5:

    st.metric(
        "Observation Days",
        df["day"].nunique()
    )


st.divider()


# ============================================================
# DATASET SHAPE
# ============================================================

st.subheader(
    "📐 Dataset Dimensions"
)


col1, col2 = st.columns(2)


with col1:

    st.write(
        "**Sensor Dataset**"
    )

    st.write(
        f"Rows: {sensor.shape[0]:,}"
    )

    st.write(
        f"Columns: {sensor.shape[1]:,}"
    )


with col2:

    st.write(
        "**Merged Dataset**"
    )

    st.write(
        f"Rows: {df.shape[0]:,}"
    )

    st.write(
        f"Columns: {df.shape[1]:,}"
    )


# ============================================================
# DATA TYPES
# ============================================================

st.divider()

st.subheader(
    "🔤 Variable Data Types"
)


dtype_df = pd.DataFrame({

    "Variable":
        df.columns,

    "Data Type":
        [
            str(df[col].dtype)
            for col in df.columns
        ],

    "Non-Null":
        [
            df[col].notna().sum()
            for col in df.columns
        ],

    "Missing":
        [
            df[col].isna().sum()
            for col in df.columns
        ]
})


dtype_df[
    "Missing (%)"
] = (
    dtype_df["Missing"]
    / len(df)
    * 100
)


st.dataframe(
    dtype_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "❓ Missing Value Analysis"
)


missing = (
    df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


missing_df = pd.DataFrame({

    "Variable":
        missing.index,

    "Missing Values":
        missing.values
})


missing_df[
    "Missing (%)"
] = (
    missing_df["Missing Values"]
    / len(df)
    * 100
)


st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True
)


# Missing value chart

missing_chart = (
    missing_df[
        missing_df["Missing Values"] > 0
    ]
)


if not missing_chart.empty:

    fig_missing = px.bar(
        missing_chart,
        x="Variable",
        y="Missing Values",
        title="Missing Values by Variable"
    )

    fig_missing.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_missing,
        use_container_width=True
    )

else:

    st.success(
        "✅ Tidak ditemukan missing values."
    )


# ============================================================
# DUPLICATE RECORDS
# ============================================================

st.divider()

st.subheader(
    "♻️ Duplicate Records"
)


duplicate_count = df.duplicated().sum()


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Duplicate Records",
        f"{duplicate_count:,}"
    )


with col2:

    duplicate_percentage = (
        duplicate_count
        / len(df)
        * 100
    )

    st.metric(
        "Duplicate Percentage",
        f"{duplicate_percentage:.2f}%"
    )


if duplicate_count == 0:

    st.success(
        "✅ Tidak ditemukan duplicate records."
    )

else:

    st.warning(
        f"⚠️ Ditemukan {duplicate_count:,} "
        "duplicate records."
    )


# ============================================================
# CATTLE ID CONSISTENCY
# ============================================================

st.divider()

st.subheader(
    "🐄 Cattle ID Consistency"
)


cattle_sensor = set(
    sensor["cattle_id"].unique()
)

cattle_metadata = set(
    metadata["cattle_id"].unique()
)


sensor_only = (
    cattle_sensor
    - cattle_metadata
)

metadata_only = (
    cattle_metadata
    - cattle_sensor
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Sensor Cattle",
        len(cattle_sensor)
    )


with col2:

    st.metric(
        "Metadata Cattle",
        len(cattle_metadata)
    )


with col3:

    st.metric(
        "Matched Cattle",
        len(
            cattle_sensor
            & cattle_metadata
        )
    )


if not sensor_only and not metadata_only:

    st.success(
        "✅ Semua cattle ID pada sensor "
        "memiliki pasangan metadata."
    )

else:

    st.warning(
        "⚠️ Terdapat cattle ID yang tidak "
        "memiliki pasangan antara dataset sensor "
        "dan metadata."
    )


if sensor_only:

    st.write(
        "Cattle ID hanya terdapat pada sensor:"
    )

    st.write(
        sorted(sensor_only)
    )


if metadata_only:

    st.write(
        "Cattle ID hanya terdapat pada metadata:"
    )

    st.write(
        sorted(metadata_only)
    )


# ============================================================
# TIMESTAMP QUALITY
# ============================================================

st.divider()

st.subheader(
    "⏱️ Timestamp Quality"
)


if "timestamp" in df.columns:

    timestamp_min = df["timestamp"].min()

    timestamp_max = df["timestamp"].max()

    timestamp_count = df["timestamp"].nunique()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "First Timestamp",
            str(timestamp_min)
        )


    with col2:

        st.metric(
            "Last Timestamp",
            str(timestamp_max)
        )


    with col3:

        st.metric(
            "Unique Timestamp",
            f"{timestamp_count:,}"
        )


    invalid_timestamp = (
        df["timestamp"].isna().sum()
    )


    if invalid_timestamp == 0:

        st.success(
            "✅ Tidak terdapat timestamp kosong."
        )

    else:

        st.warning(
            f"⚠️ Terdapat {invalid_timestamp} "
            "timestamp kosong."
        )


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📈 Numerical Variable Summary"
)


numeric_columns = (
    df
    .select_dtypes(
        include=np.number
    )
    .columns
)


summary = (
    df[numeric_columns]
    .describe()
    .T
)


summary[
    "missing"
] = df[
    numeric_columns
].isna().sum()


st.dataframe(
    summary,
    use_container_width=True
)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "📦 Outlier Analysis"
)


outlier_variables = [
    "body_temperature_c",
    "heart_rate_bpm",
    "respiration_rate_bpm",
    "activity_percent",
    "feed_intake_kg_day",
    "water_intake_l_day",
    "ambient_temperature_c",
    "humidity_percent",
    "co2_ppm",
    "nh3_ppm"
]


available_outlier_variables = [
    col
    for col in outlier_variables
    if col in df.columns
]


selected_outlier_variable = st.selectbox(
    "Select Variable",
    available_outlier_variables
)


q1 = df[
    selected_outlier_variable
].quantile(0.25)


q3 = df[
    selected_outlier_variable
].quantile(0.75)


iqr = q3 - q1


lower_bound = (
    q1 - 1.5 * iqr
)


upper_bound = (
    q3 + 1.5 * iqr
)


outliers = df[
    (
        df[selected_outlier_variable]
        < lower_bound
    )
    |
    (
        df[selected_outlier_variable]
        > upper_bound
    )
]


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Q1",
        f"{q1:.3f}"
    )


with col2:

    st.metric(
        "Q3",
        f"{q3:.3f}"
    )


with col3:

    st.metric(
        "Outlier Records",
        f"{len(outliers):,}"
    )


fig_box = px.box(
    df,
    y=selected_outlier_variable,
    points="outliers",
    title=f"Outlier Detection — {selected_outlier_variable}"
)


st.plotly_chart(
    fig_box,
    use_container_width=True
)


# ============================================================
# DISTRIBUTION
# ============================================================

st.subheader(
    "📊 Variable Distribution"
)


fig_distribution = px.histogram(
    df,
    x=selected_outlier_variable,
    nbins=50,
    title=f"Distribution — {selected_outlier_variable}"
)


st.plotly_chart(
    fig_distribution,
    use_container_width=True
)


# ============================================================
# DISEASE ALERT DISTRIBUTION
# ============================================================

st.divider()

st.subheader(
    "🚨 Disease Alert Distribution"
)


if "disease_alert" in df.columns:

    alert_counts = (
        df["disease_alert"]
        .value_counts()
        .reset_index()
    )


    alert_counts.columns = [
        "Disease Alert",
        "Records"
    ]


    alert_counts[
        "Status"
    ] = alert_counts[
        "Disease Alert"
    ].map({
        0: "Normal",
        1: "Disease Alert"
    })


    st.dataframe(
        alert_counts,
        use_container_width=True,
        hide_index=True
    )


    fig_alert = px.pie(
        alert_counts,
        names="Status",
        values="Records",
        title="Disease Alert Distribution"
    )


    st.plotly_chart(
        fig_alert,
        use_container_width=True
    )


# ============================================================
# HEALTH STATUS DISTRIBUTION
# ============================================================

st.subheader(
    "🐄 Health Status Distribution"
)


if "health_status" in df.columns:

    health_counts = (
        df["health_status"]
        .value_counts()
        .reset_index()
    )


    health_counts.columns = [
        "Health Status",
        "Records"
    ]


    st.dataframe(
        health_counts,
        use_container_width=True,
        hide_index=True
    )


    fig_health = px.bar(
        health_counts,
        x="Health Status",
        y="Records",
        text_auto=True,
        title="Health Status Distribution"
    )


    st.plotly_chart(
        fig_health,
        use_container_width=True
    )


# ============================================================
# RISK SCORE DISTRIBUTION
# ============================================================

st.divider()

st.subheader(
    "⚠️ Disease Risk Score Distribution"
)


if "disease_risk_score" in df.columns:

    risk = df[
        "disease_risk_score"
    ]


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Minimum Risk",
            f"{risk.min():.3f}"
        )


    with col2:

        st.metric(
            "Average Risk",
            f"{risk.mean():.3f}"
        )


    with col3:

        st.metric(
            "Maximum Risk",
            f"{risk.max():.3f}"
        )


    fig_risk = px.histogram(
        df,
        x="disease_risk_score",
        nbins=50,
        title="Disease Risk Score Distribution"
    )


    fig_risk.add_vline(
        x=0.30,
        line_dash="dash",
        annotation_text="Medium Risk"
    )


    fig_risk.add_vline(
        x=0.70,
        line_dash="dash",
        annotation_text="High Risk"
    )


    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


# ============================================================
# DATA QUALITY SCORE
# ============================================================

st.divider()

st.subheader(
    "🏆 Overall Data Quality Assessment"
)


total_cells = (
    df.shape[0]
    * df.shape[1]
)


missing_cells = (
    df.isna().sum().sum()
)


duplicate_cells = (
    duplicate_count
)


missing_score = (
    1
    - (
        missing_cells
        / total_cells
    )
)


duplicate_score = (
    1
    - (
        duplicate_cells
        / len(df)
    )
)


cattle_consistency_score = (
    len(
        cattle_sensor
        & cattle_metadata
    )
    /
    max(
        len(cattle_sensor),
        len(cattle_metadata)
    )
)


quality_score = (
    (
        missing_score
        + duplicate_score
        + cattle_consistency_score
    )
    / 3
    * 100
)


st.metric(
    "Overall Data Quality Score",
    f"{quality_score:.2f}%"
)


if quality_score >= 95:

    st.success(
        "🟢 Excellent Data Quality"
    )

elif quality_score >= 85:

    st.info(
        "🟡 Good Data Quality"
    )

elif quality_score >= 70:

    st.warning(
        "🟠 Moderate Data Quality"
    )

else:

    st.error(
        "🔴 Poor Data Quality — "
        "Data cleaning is recommended."
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.divider()

st.subheader(
    "🔍 Dataset Preview"
)


preview_rows = st.slider(
    "Number of rows to display",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)


st.dataframe(
    df.head(preview_rows),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD CLEAN DATA
# ============================================================

st.divider()

st.subheader(
    "💾 Export Dataset"
)


csv_data = df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="⬇️ Download Merged Dataset",
    data=csv_data,
    file_name="cattle_health_merged_dataset.csv",
    mime="text/csv"
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "🧠 Data Quality Interpretation"
)


st.info(
    f"""
    **Ringkasan kualitas data:**

    • Dataset sensor memiliki **{len(sensor):,} records**.

    • Dataset mencakup **{df["cattle_id"].nunique()} sapi**.

    • Periode pengamatan mencakup
      **{df["day"].nunique()} hari**.

    • Jumlah missing values:
      **{missing_cells:,}**.

    • Jumlah duplicate records:
      **{duplicate_count:,}**.

    • Konsistensi cattle ID:
      **{cattle_consistency_score * 100:.2f}%**.

    • Overall Data Quality Score:
      **{quality_score:.2f}%**.

    Dataset yang telah melalui pemeriksaan kualitas dapat
    dilanjutkan ke tahap preprocessing, feature engineering,
    training Machine Learning, dan forecasting.
    """
)


st.caption(
    "Cattle Health ML Dashboard — Data Quality & Validation"
)