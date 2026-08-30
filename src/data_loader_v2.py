import pandas as pd
from pathlib import Path


def load_data(sensor_path, metadata_path):

    sensor_path = Path(sensor_path)
    metadata_path = Path(metadata_path)

    if not sensor_path.exists():
        raise FileNotFoundError(
            f"Sensor dataset not found: {sensor_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata dataset not found: {metadata_path}"
        )

    # Read CSV
    sensor = pd.read_csv(
        sensor_path,
        low_memory=False
    )

    metadata = pd.read_csv(
        metadata_path,
        low_memory=False
    )

    # Clean column names
    sensor.columns = [
        str(c).strip()
        for c in sensor.columns
    ]

    metadata.columns = [
        str(c).strip()
        for c in metadata.columns
    ]

    # Required columns
    if "cattle_id" not in sensor.columns:
        raise KeyError(
            "cattle_id tidak ditemukan pada sensor dataset."
        )

    if "cattle_id" not in metadata.columns:
        raise KeyError(
            "cattle_id tidak ditemukan pada metadata dataset."
        )

    if "timestamp" not in sensor.columns:
        raise KeyError(
            "timestamp tidak ditemukan pada sensor dataset."
        )

    # ------------------------------------------------------
    # CATTLE ID
    # ------------------------------------------------------

    sensor["cattle_id"] = (
        sensor["cattle_id"]
        .fillna("")
        .map(str)
        .str.strip()
    )

    metadata["cattle_id"] = (
        metadata["cattle_id"]
        .fillna("")
        .map(str)
        .str.strip()
    )

    sensor = sensor[
        sensor["cattle_id"] != ""
    ].copy()

    metadata = metadata[
        metadata["cattle_id"] != ""
    ].copy()

    # ------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------

    sensor["timestamp"] = pd.to_datetime(
        sensor["timestamp"],
        errors="coerce"
    )

    sensor = sensor[
        sensor["timestamp"].notna()
    ].copy()

    # ------------------------------------------------------
    # SORT
    # ------------------------------------------------------

    sensor = (
        sensor
        .sort_values(
            ["cattle_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # DAY
    # ------------------------------------------------------

    if len(sensor) > 0:

        first_date = (
            sensor["timestamp"]
            .min()
            .normalize()
        )

        # IMPORTANT:
        # Tidak menggunakan astype("Int64")

        sensor["day"] = (
            sensor["timestamp"]
            .dt.normalize()
            .sub(first_date)
            .dt.days
            + 1
        )

    else:

        sensor["day"] = pd.Series(
            index=sensor.index,
            dtype="float64"
        )

    # ------------------------------------------------------
    # NUMERIC COLUMNS
    # ------------------------------------------------------

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

    for col in numeric_columns:

        if col in sensor.columns:

            sensor[col] = pd.to_numeric(
                sensor[col],
                errors="coerce"
            )

    # ------------------------------------------------------
    # RISK SCORE
    # ------------------------------------------------------

    if "disease_risk_score" in sensor.columns:

        sensor["disease_risk_score"] = (
            sensor["disease_risk_score"]
            .fillna(0)
            .clip(lower=0, upper=1)
        )

    # ------------------------------------------------------
    # DISEASE ALERT
    # ------------------------------------------------------

    if "disease_alert" in sensor.columns:

        sensor["disease_alert"] = (
            pd.to_numeric(
                sensor["disease_alert"],
                errors="coerce"
            )
            .fillna(0)
            .ge(0.5)
            .astype("int8")
        )

    # ------------------------------------------------------
    # HEALTH STATUS
    # ------------------------------------------------------

    if "health_status" in sensor.columns:

        sensor["health_status"] = (
            sensor["health_status"]
            .fillna("Unknown")
            .map(str)
            .str.strip()
        )

    # ------------------------------------------------------
    # REMOVE DUPLICATES
    # ------------------------------------------------------

    sensor = sensor.drop_duplicates()

    metadata = metadata.drop_duplicates(
        subset=["cattle_id"]
    )

    return sensor, metadata


def merge_metadata(sensor, metadata):

    sensor = sensor.copy()
    metadata = metadata.copy()

    sensor["cattle_id"] = (
        sensor["cattle_id"]
        .map(str)
        .str.strip()
    )

    metadata["cattle_id"] = (
        metadata["cattle_id"]
        .map(str)
        .str.strip()
    )

    df = sensor.merge(
        metadata,
        on="cattle_id",
        how="left",
        suffixes=("", "_metadata")
    )

    return df


def get_data_quality(df):

    return pd.DataFrame({
        "Variable": df.columns,
        "Data Type": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Missing Values": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Missing (%)": [
            float(df[c].isna().mean() * 100)
            for c in df.columns
        ],
        "Unique Values": [
            int(df[c].nunique())
            for c in df.columns
        ]
    })


def get_numeric_summary(df):

    numeric = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric) == 0:
        return pd.DataFrame()

    return (
        df[numeric]
        .describe()
        .T
        .reset_index()
        .rename(
            columns={"index": "Variable"}
        )
    )


def get_latest_cattle_data(df):

    return (
        df
        .sort_values("timestamp")
        .groupby("cattle_id")
        .tail(1)
        .reset_index(drop=True)
    )


def get_cattle_data(df, cattle_id):

    cattle_id = str(cattle_id).strip()

    result = df[
        df["cattle_id"].map(str).str.strip()
        == cattle_id
    ].copy()

    if "timestamp" in result.columns:

        result = result.sort_values(
            "timestamp"
        )

    return result


def daily_average(df, columns):

    if "day" not in df.columns:
        raise KeyError(
            "Column 'day' tidak ditemukan."
        )

    available = [
        c for c in columns
        if c in df.columns
    ]

    if not available:
        return pd.DataFrame()

    return (
        df
        .groupby("day")[available]
        .mean()
        .reset_index()
    )