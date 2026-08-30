"""
Data loading and preprocessing utilities
for the Cattle Health ML Dashboard.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# HELPER: CLEAN CATTLE ID
# ============================================================

def clean_cattle_id(series):
    """
    Safely clean cattle_id without using pandas nullable
    StringDtype, which can cause casting problems with
    mixed-type CSV columns.
    """

    def clean_value(value):

        if pd.isna(value):
            return ""

        return str(value).strip()

    return series.map(clean_value)


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

    sensor = pd.read_csv(
        sensor_path,
        low_memory=False
    )

    metadata = pd.read_csv(
        metadata_path,
        low_memory=False
    )

    # ========================================================
    # CLEAN COLUMN NAMES
    # ========================================================

    sensor.columns = [
        str(col).strip()
        for col in sensor.columns
    ]

    metadata.columns = [
        str(col).strip()
        for col in metadata.columns
    ]

    # ========================================================
    # VALIDATE CATTLE ID
    # ========================================================

    if "cattle_id" not in sensor.columns:

        raise KeyError(
            "Column 'cattle_id' is missing "
            "from cattle_sensor_data.csv."
        )

    if "cattle_id" not in metadata.columns:

        raise KeyError(
            "Column 'cattle_id' is missing "
            "from cattle_metadata.csv."
        )

    # ========================================================
    # CLEAN CATTLE ID
    # ========================================================

    sensor["cattle_id"] = clean_cattle_id(
        sensor["cattle_id"]
    )

    metadata["cattle_id"] = clean_cattle_id(
        metadata["cattle_id"]
    )

    # ========================================================
    # REMOVE INVALID CATTLE ID
    # ========================================================

    sensor = sensor[
        sensor["cattle_id"] != ""
    ].copy()

    metadata = metadata[
        metadata["cattle_id"] != ""
    ].copy()

    # ========================================================
    # TIMESTAMP PROCESSING
    # ========================================================

    if "timestamp" in sensor.columns:

        sensor["timestamp"] = pd.to_datetime(
            sensor["timestamp"],
            errors="coerce"
        )

        # Remove invalid timestamps

        sensor = sensor[
            sensor["timestamp"].notna()
        ].copy()

    else:

        raise KeyError(
            "Column 'timestamp' is missing "
            "from cattle_sensor_data.csv."
        )

    # ========================================================
    # SORT DATA
    # ========================================================

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

    if not sensor.empty:

        first_date = (
            sensor["timestamp"]
            .min()
            .normalize()
        )

        sensor["day"] = (
            sensor["timestamp"]
            .dt.normalize()
            .sub(first_date)
            .dt.days
            + 1
        )

        sensor["day"] = (
            pd.to_numeric(
                sensor["day"],
                errors="coerce"
            )
            .fillna(1)
            .round()
            .astype(int)
        )

    else:

        sensor["day"] = pd.Series(
            index=sensor.index,
            dtype=int
        )

    # ========================================================
    # NUMERICAL SENSOR VARIABLES
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
    # DISEASE ALERT
    # ========================================================

    if "disease_alert" in sensor.columns:

        sensor["disease_alert"] = (
            pd.to_numeric(
                sensor["disease_alert"],
                errors="coerce"
            )
            .fillna(0)
            .round()
            .astype(int)
        )

    # ========================================================
    # DISEASE RISK SCORE
    # ========================================================

    if "disease_risk_score" in sensor.columns:

        sensor["disease_risk_score"] = (
            pd.to_numeric(
                sensor["disease_risk_score"],
                errors="coerce"
            )
            .fillna(0)
            .clip(
                lower=0,
                upper=1
            )
        )

    # ========================================================
    # HEALTH STATUS
    # ========================================================

    if "health_status" in sensor.columns:

        sensor["health_status"] = (
            sensor["health_status"]
            .fillna("Unknown")
            .map(
                lambda x: str(x).strip()
            )
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    sensor = sensor.drop_duplicates()

    metadata = metadata.drop_duplicates(
        subset=["cattle_id"]
    )

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

    sensor = sensor.copy()
    metadata = metadata.copy()

    # IMPORTANT:
    # Use ordinary Python object/string conversion
    # instead of pandas nullable StringDtype.

    sensor["cattle_id"] = clean_cattle_id(
        sensor["cattle_id"]
    )

    metadata["cattle_id"] = clean_cattle_id(
        metadata["cattle_id"]
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
            int(df[col].isna().sum())
            for col in df.columns
        ],

        "Missing (%)": [
            float(
                df[col].isna().mean() * 100
            )
            for col in df.columns
        ],

        "Unique Values": [
            int(df[col].nunique())
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
    )

    summary = summary.rename(
        columns={
            "index": "Variable"
        }
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

    cattle_id = str(cattle_id).strip()

    cattle_data = df[
        df["cattle_id"].map(
            lambda x: str(x).strip()
        )
        == cattle_id
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

    if "day" not in df.columns:

        raise KeyError(
            "Column 'day' is required."
        )

    available_columns = [

        col

        for col in columns

        if col in df.columns
    ]

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

        "total_records":
            int(len(df)),

        "total_variables":
            int(len(df.columns)),

        "total_cattle":
            int(
                df["cattle_id"].nunique()
            )
            if "cattle_id" in df.columns
            else 0,

        "missing_cells":
            int(
                df.isna().sum().sum()
            ),

        "duplicate_records":
            int(
                df.duplicated().sum()
            )
    }

    return information
