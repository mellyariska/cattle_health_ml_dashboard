"""
Data loading and preprocessing utilities
for the Cattle Health ML Dashboard.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# LOAD DATA
# ============================================================

def load_data(sensor_path, metadata_path):
    """
    Load cattle sensor data and cattle metadata.

    Parameters
    ----------
    sensor_path : str or Path
        Path to cattle_sensor_data.csv

    metadata_path : str or Path
        Path to cattle_metadata.csv

    Returns
    -------
    sensor : pandas.DataFrame
        Sensor dataset

    metadata : pandas.DataFrame
        Cattle metadata
    """

    sensor_path = Path(sensor_path)
    metadata_path = Path(metadata_path)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not sensor_path.exists():
        raise FileNotFoundError(
            f"Sensor dataset not found: {sensor_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata dataset not found: {metadata_path}"
        )

    # ========================================================
    # READ CSV
    # ========================================================

    sensor = pd.read_csv(sensor_path)
    metadata = pd.read_csv(metadata_path)

    # ========================================================
    # CLEAN COLUMN NAMES
    # ========================================================

    sensor.columns = (
        sensor.columns
        .astype(str)
        .str.strip()
    )

    metadata.columns = (
        metadata.columns
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # CATTLE ID PROCESSING
    # ========================================================

    if "cattle_id" in sensor.columns:

        sensor["cattle_id"] = (
            sensor["cattle_id"]
            .astype("string")
            .str.strip()
        )

    if "cattle_id" in metadata.columns:

        metadata["cattle_id"] = (
            metadata["cattle_id"]
            .astype("string")
            .str.strip()
        )

    # ========================================================
    # TIMESTAMP PROCESSING
    # ========================================================

    if "timestamp" in sensor.columns:

        sensor["timestamp"] = pd.to_datetime(
            sensor["timestamp"],
            errors="coerce"
        )

    # ========================================================
    # REMOVE INVALID TIMESTAMP
    # ========================================================

    if "timestamp" in sensor.columns:

        sensor = sensor[
            sensor["timestamp"].notna()
        ].copy()

    # ========================================================
    # REMOVE INVALID CATTLE ID
    # ========================================================

    if "cattle_id" in sensor.columns:

        sensor = sensor[
            sensor["cattle_id"].notna()
            &
            (sensor["cattle_id"] != "")
        ].copy()

    if "cattle_id" in metadata.columns:

        metadata = metadata[
            metadata["cattle_id"].notna()
            &
            (metadata["cattle_id"] != "")
        ].copy()

    # ========================================================
    # SORT SENSOR DATA
    # ========================================================

    if (
        "cattle_id" in sensor.columns
        and
        "timestamp" in sensor.columns
    ):

        sensor = (
            sensor
            .sort_values(
                ["cattle_id", "timestamp"]
            )
            .reset_index(drop=True)
        )

    # ========================================================
    # CREATE OBSERVATION DAY
    # ========================================================
    #
    # Day 1 = tanggal pertama dalam dataset
    # Day 2 = hari berikutnya
    # Day 3 = hari berikutnya, dst.
    #
    # Menggunakan .dt.days agar tidak terjadi
    # error casting float/object ke Int64.
    # ========================================================

    if "timestamp" in sensor.columns:

        if not sensor.empty:

            first_timestamp = (
                sensor["timestamp"].min()
            )

            sensor["day"] = (
                sensor["timestamp"]
                .dt.normalize()
                .sub(
                    first_timestamp.normalize()
                )
                .dt.days
                + 1
            )

            sensor["day"] = (
                pd.to_numeric(
                    sensor["day"],
                    errors="coerce"
                )
                .fillna(1)
                .astype(int)
            )

        else:

            sensor["day"] = pd.Series(
                dtype="int64"
            )

    # ========================================================
    # CONVERT NUMERICAL SENSOR VARIABLES
    # ========================================================

    numerical_columns = [
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

    for column in numerical_columns:

        if column in sensor.columns:

            sensor[column] = pd.to_numeric(
                sensor[column],
                errors="coerce"
            )

    # ========================================================
    # DISEASE ALERT PROCESSING
    # ========================================================

    if "disease_alert" in sensor.columns:

        sensor["disease_alert"] = (
            pd.to_numeric(
                sensor["disease_alert"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

    # ========================================================
    # DISEASE RISK SCORE PROCESSING
    # ========================================================

    if "disease_risk_score" in sensor.columns:

        sensor["disease_risk_score"] = (
            pd.to_numeric(
                sensor["disease_risk_score"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
        )

    # ========================================================
    # HEALTH STATUS PROCESSING
    # ========================================================

    if "health_status" in sensor.columns:

        sensor["health_status"] = (
            sensor["health_status"]
            .astype("string")
            .str.strip()
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    sensor = sensor.drop_duplicates()

    if "cattle_id" in metadata.columns:

        metadata = metadata.drop_duplicates(
            subset=["cattle_id"]
        )

    else:

        metadata = metadata.drop_duplicates()

    # ========================================================
    # RESET INDEX
    # ========================================================

    sensor = sensor.reset_index(
        drop=True
    )

    metadata = metadata.reset_index(
        drop=True
    )

    return sensor, metadata


# ============================================================
# MERGE SENSOR + METADATA
# ============================================================

def merge_metadata(sensor, metadata):
    """
    Merge sensor data with cattle metadata
    using cattle_id.
    """

    if "cattle_id" not in sensor.columns:

        raise KeyError(
            "Column 'cattle_id' is missing "
            "from sensor dataset."
        )

    if "cattle_id" not in metadata.columns:

        raise KeyError(
            "Column 'cattle_id' is missing "
            "from metadata dataset."
        )

    # Make sure both cattle_id columns have
    # the same data type.

    sensor = sensor.copy()
    metadata = metadata.copy()

    sensor["cattle_id"] = (
        sensor["cattle_id"]
        .astype("string")
        .str.strip()
    )

    metadata["cattle_id"] = (
        metadata["cattle_id"]
        .astype("string")
        .str.strip()
    )

    # ========================================================
    # MERGE
    # ========================================================

    df = sensor.merge(
        metadata,
        on="cattle_id",
        how="left",
        suffixes=(
            "",
            "_metadata"
        )
    )

    return df


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def get_data_quality(df):
    """
    Generate basic data quality statistics.
    """

    quality = pd.DataFrame({

        "Variable": df.columns,

        "Data Type": [
            str(df[col].dtype)
            for col in df.columns
        ],

        "Missing Values": [
            df[col].isna().sum()
            for col in df.columns
        ],

        "Missing (%)": [
            df[col].isna().mean() * 100
            for col in df.columns
        ],

        "Unique Values": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    return quality


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def get_numeric_summary(df):
    """
    Return descriptive statistics for numerical variables.
    """

    numeric_columns = (
        df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    if len(numeric_columns) == 0:

        return pd.DataFrame()

    summary = (
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

    return summary


# ============================================================
# GET LATEST CATTLE RECORD
# ============================================================

def get_latest_cattle_data(df):
    """
    Get the latest available sensor record
    for each cattle.
    """

    if "timestamp" not in df.columns:

        raise KeyError(
            "Column 'timestamp' is required."
        )

    if "cattle_id" not in df.columns:

        raise KeyError(
            "Column 'cattle_id' is required."
        )

    latest = (
        df
        .sort_values("timestamp")
        .groupby(
            "cattle_id"
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return latest


# ============================================================
# GET SINGLE CATTLE DATA
# ============================================================

def get_cattle_data(df, cattle_id):
    """
    Return all records belonging to one cattle.
    """

    if "cattle_id" not in df.columns:

        raise KeyError(
            "Column 'cattle_id' is required."
        )

    cattle_data = df[
        df["cattle_id"].astype("string")
        ==
        str(cattle_id)
    ].copy()

    if "timestamp" in cattle_data.columns:

        cattle_data = (
            cattle_data
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    return cattle_data


# ============================================================
# DAILY AGGREGATION
# ============================================================

def daily_average(df, columns):
    """
    Calculate daily average values for selected
    sensor variables.
    """

    available_columns = [
        col
        for col in columns
        if col in df.columns
    ]

    if "day" not in df.columns:

        raise KeyError(
            "Column 'day' is required."
        )

    if not available_columns:

        return pd.DataFrame()

    daily = (
        df
        .groupby("day")[
            available_columns
        ]
        .mean()
        .reset_index()
    )

    return daily


# ============================================================
# CATTLE SUMMARY
# ============================================================

def get_cattle_summary(df):
    """
    Generate summary statistics for each cattle.
    """

    if "cattle_id" not in df.columns:

        raise KeyError(
            "Column 'cattle_id' is required."
        )

    aggregation = {}

    numerical_columns = [
        "body_temperature_c",
        "heart_rate_bpm",
        "respiration_rate_bpm",
        "activity_percent",
        "feed_intake_kg_day",
        "water_intake_l_day",
        "disease_risk_score",
        "disease_alert"
    ]

    for column in numerical_columns:

        if column in df.columns:

            aggregation[column] = "mean"

    if not aggregation:

        return pd.DataFrame()

    summary = (
        df
        .groupby("cattle_id")
        .agg(aggregation)
        .reset_index()
    )

    return summary


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_info(df):
    """
    Return general information about the dataset.
    """

    information = {

        "total_records": len(df),

        "total_variables": len(
            df.columns
        ),

        "total_cattle": (
            df["cattle_id"].nunique()
            if "cattle_id" in df.columns
            else 0
        ),

        "missing_cells": int(
            df.isna().sum().sum()
        ),

        "duplicate_records": int(
            df.duplicated().sum()
        )
    }

    return information
