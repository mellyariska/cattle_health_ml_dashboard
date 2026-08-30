import pandas as pd
from pathlib import Path


def load_data(sensor_path, metadata_path):

    sensor_path = Path(sensor_path)
    metadata_path = Path(metadata_path)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not sensor_path.exists():
        raise FileNotFoundError(
            f"Sensor dataset not found: {sensor_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata dataset not found: {metadata_path}"
        )

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    sensor = pd.read_csv(
        sensor_path,
        low_memory=False
    )

    metadata = pd.read_csv(
        metadata_path,
        low_memory=False
    )

    # --------------------------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------------------------

    sensor.columns = sensor.columns.astype(str).str.strip()
    metadata.columns = metadata.columns.astype(str).str.strip()

    # --------------------------------------------------------
    # CHECK COLUMNS
    # --------------------------------------------------------

    if "cattle_id" not in sensor.columns:
        raise KeyError(
            "Column 'cattle_id' tidak ditemukan pada "
            "cattle_sensor_data.csv"
        )

    if "cattle_id" not in metadata.columns:
        raise KeyError(
            "Column 'cattle_id' tidak ditemukan pada "
            "cattle_metadata.csv"
        )

    if "timestamp" not in sensor.columns:
        raise KeyError(
            "Column 'timestamp' tidak ditemukan pada "
            "cattle_sensor_data.csv"
        )

    # --------------------------------------------------------
    # CATTLE ID
    # --------------------------------------------------------

    sensor["cattle_id"] = (
        sensor["cattle_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    metadata["cattle_id"] = (
        metadata["cattle_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # REMOVE EMPTY ID
    # --------------------------------------------------------

    sensor = sensor[
        sensor["cattle_id"] != ""
    ].copy()

    metadata = metadata[
        metadata["cattle_id"] != ""
    ].copy()

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    sensor["timestamp"] = pd.to_datetime(
        sensor["timestamp"],
        errors="coerce"
    )

    sensor = sensor[
        sensor["timestamp"].notna()
    ].copy()

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    sensor = sensor.sort_values(
        ["cattle_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # CREATE DAY
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Tidak menggunakan .astype("Int64")
    #

    first_date = sensor["timestamp"].min().normalize()

    sensor["day"] = (
        sensor["timestamp"]
        .dt.normalize()
        .sub(first_date)
        .dt.days
        + 1
    )

    # --------------------------------------------------------
    # NUMERIC VARIABLES
    # --------------------------------------------------------

    numeric_columns = [
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
        "disease_risk_score",
        "disease_alert"
    ]

    for column in numeric_columns:

        if column in sensor.columns:

            sensor[column] = pd.to_numeric(
                sensor[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # DISEASE RISK
    # --------------------------------------------------------

    if "disease_risk_score" in sensor.columns:

        sensor["disease_risk_score"] = (
            sensor["disease_risk_score"]
            .fillna(0)
            .clip(0, 1)
        )

    # --------------------------------------------------------
    # DISEASE ALERT
    # --------------------------------------------------------

    if "disease_alert" in sensor.columns:

        sensor["disease_alert"] = (
            sensor["disease_alert"]
            .fillna(0)
            .apply(
                lambda x: 1 if x >= 0.5 else 0
            )
        )

    # --------------------------------------------------------
    # HEALTH STATUS
    # --------------------------------------------------------

    if "health_status" in sensor.columns:

        sensor["health_status"] = (
            sensor["health_status"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    sensor = sensor.drop_duplicates()

    metadata = metadata.drop_duplicates(
        subset=["cattle_id"]
    )

    return sensor, metadata


# ============================================================
# MERGE METADATA
# ============================================================

def merge_metadata(sensor, metadata):

    if "cattle_id" not in sensor.columns:
        raise KeyError(
            "Column 'cattle_id' tidak ditemukan pada sensor."
        )

    if "cattle_id" not in metadata.columns:
        raise KeyError(
            "Column 'cattle_id' tidak ditemukan pada metadata."
        )

    df = sensor.merge(
        metadata,
        on="cattle_id",
        how="left",
        suffixes=("", "_metadata")
    )

    return df


# ============================================================
# DATA QUALITY
# ============================================================

def get_data_quality(df):

    quality = pd.DataFrame({
        "Variable": df.columns,

        "Data Type": [
            str(df[column].dtype)
            for column in df.columns
        ],

        "Missing Values": [
            df[column].isna().sum()
            for column in df.columns
        ],

        "Missing (%)": [
            df[column].isna().mean() * 100
            for column in df.columns
        ],

        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    return quality


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def get_numeric_summary(df):

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    if len(numeric_columns) == 0:
        return pd.DataFrame()

    return (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Variable"
            }
        )
    )


# ============================================================
# LATEST CATTLE DATA
# ============================================================

def get_latest_cattle_data(df):

    if "timestamp" not in df.columns:
        raise KeyError(
            "Column 'timestamp' is required."
        )

    latest = (
        df
        .sort_values("timestamp")
        .groupby("cattle_id")
        .tail(1)
        .reset_index(drop=True)
    )

    return latest


# ============================================================
# SINGLE CATTLE
# ============================================================

def get_cattle_data(df, cattle_id):

    cattle_id = str(cattle_id).strip()

    result = df[
        df["cattle_id"].astype(str).str.strip()
        == cattle_id
    ].copy()

    if "timestamp" in result.columns:

        result = result.sort_values(
            "timestamp"
        )

    return result


# ============================================================
# DAILY AVERAGE
# ============================================================

def daily_average(df, columns):

    if "day" not in df.columns:
        raise KeyError(
            "Column 'day' is required."
        )

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    return (
        df
        .groupby("day")[available_columns]
        .mean()
        .reset_index()
    )
