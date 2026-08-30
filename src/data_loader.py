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

    # Check sensor file
    if not sensor_path.exists():
        raise FileNotFoundError(
            f"Sensor dataset not found: {sensor_path}"
        )

    # Check metadata file
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata dataset not found: {metadata_path}"
        )

    # Read CSV
    sensor = pd.read_csv(sensor_path)
    metadata = pd.read_csv(metadata_path)

    # ========================================================
    # TIMESTAMP PROCESSING
    # ========================================================

    if "timestamp" in sensor.columns:

        sensor["timestamp"] = pd.to_datetime(
            sensor["timestamp"],
            errors="coerce"
        )

    # ========================================================
    # SORT SENSOR DATA
    # ========================================================

    if "timestamp" in sensor.columns:

        sensor = sensor.sort_values(
            ["cattle_id", "timestamp"]
        ).reset_index(drop=True)

    # ========================================================
    # CREATE OBSERVATION DAY
    # ========================================================

    if "timestamp" in sensor.columns:

        first_timestamp = sensor[
            "timestamp"
        ].min()

        sensor["day"] = (
            (
                sensor["timestamp"]
                - first_timestamp
            )
            .dt.total_seconds()
            / 86400
        ).astype("Int64") + 1

    # ========================================================
    # REMOVE INVALID CATTLE ID
    # ========================================================

    if "cattle_id" in sensor.columns:

        sensor = sensor[
            sensor["cattle_id"].notna()
        ].copy()

    if "cattle_id" in metadata.columns:

        metadata = metadata[
            metadata["cattle_id"].notna()
        ].copy()

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    sensor = sensor.drop_duplicates()

    metadata = metadata.drop_duplicates(
        subset=["cattle_id"]
        if "cattle_id" in metadata.columns
        else None
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

    df = sensor.merge(
        metadata,
        on="cattle_id",
        how="left",
        suffixes=("", "_metadata")
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

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

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

    latest = (
        df.sort_values("timestamp")
        .groupby("cattle_id")
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

    cattle_data = df[
        df["cattle_id"] == cattle_id
    ].copy()

    if "timestamp" in cattle_data.columns:

        cattle_data = cattle_data.sort_values(
            "timestamp"
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
        df.groupby("day")[
            available_columns
        ]
        .mean()
        .reset_index()
    )

    return daily