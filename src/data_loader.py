"""
Data loading and preprocessing utilities
for the Cattle Health ML Dashboard.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_cattle_id(series):
    """
    Safely convert cattle IDs to ordinary Python strings.
    """

    result = []

    for value in series:

        if pd.isna(value):
            result.append("")
        else:
            result.append(str(value).strip())

    return pd.Series(
        result,
        index=series.index,
        dtype=object
    )


def safe_number(value, default=0.0):
    """
    Safely convert one value to a Python float.
    """

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (ValueError, TypeError):

        return default


# ============================================================
# LOAD DATA
# ============================================================

def load_data(sensor_path, metadata_path):
    """
    Load cattle sensor data and cattle metadata.
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
        str(column).strip()
        for column in sensor.columns
    ]

    metadata.columns = [
        str(column).strip()
        for column in metadata.columns
    ]

    # ========================================================
    # CHECK REQUIRED COLUMNS
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

    if "timestamp" not in sensor.columns:

        raise KeyError(
            "Column 'timestamp' is missing "
            "from cattle_sensor_data.csv."
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
    # REMOVE EMPTY CATTLE ID
    # ========================================================

    sensor = sensor[
        sensor["cattle_id"] != ""
    ].copy()

    metadata = metadata[
        metadata["cattle_id"] != ""
    ].copy()

    # ========================================================
    # TIMESTAMP
    # ========================================================

    sensor["timestamp"] = pd.to_datetime(
        sensor["timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps

    sensor = sensor[
        sensor["timestamp"].notna()
    ].copy()

    # ========================================================
    # SORT DATA
    # ========================================================

    sensor = sensor.sort_values(
        ["timestamp", "cattle_id"]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # CREATE OBSERVATION DAY
    # ========================================================
    #
    # IMPORTANT:
    # Do NOT use pandas astype("Int64") here.
    #
    # Day 1 = first calendar day
    # Day 2 = second calendar day
    # etc.
    # ========================================================

    if len(sensor) > 0:

        first_date = sensor[
            "timestamp"
        ].min().normalize()

        day_values = []

        for timestamp in sensor["timestamp"]:

            current_date = timestamp.normalize()

            difference = (
                current_date - first_date
            ).days

            day_values.append(
                int(difference) + 1
            )

        # Assign ordinary Python integers

        sensor["day"] = day_values

    else:

        sensor["day"] = []

    # ========================================================
    # NUMERICAL COLUMNS
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

            sensor[column] = sensor[column].apply(
                safe_number
            )

    # ========================================================
    # DISEASE ALERT
    # ========================================================

    if "disease_alert" in sensor.columns:

        alert_values = []

        for value in sensor[
            "disease_alert"
        ]:

            number = safe_number(
                value,
                default=0
            )

            if number >= 0.5:
                alert_values.append(1)
            else:
                alert_values.append(0)

        sensor["disease_alert"] = alert_values

    # ========================================================
    # DISEASE RISK SCORE
    # ========================================================

    if "disease_risk_score" in sensor.columns:

        risk_values = []

        for value in sensor[
            "disease_risk_score"
        ]:

            number = safe_number(
                value,
                default=0
            )

            # Keep risk between 0 and 1

            if number < 0:
                number = 0.0

            if number > 1:
                number = 1.0

            risk_values.append(number)

        sensor["disease_risk_score"] = (
            risk_values
        )

    # ========================================================
    # HEALTH STATUS
    # ========================================================

    if "health_status" in sensor.columns:

        status_values = []

        for value in sensor[
            "health_status"
        ]:

            if pd.isna(value):

                status_values.append(
                    "Unknown"
                )

            else:

                status_values.append(
                    str(value).strip()
                )

        sensor["health_status"] = (
            status_values
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
    Merge sensor data with cattle metadata.
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

    # Clean IDs again before merge

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

        "Variable": list(df.columns),

        "Data Type": [
            str(df[column].dtype)
            for column in df.columns
        ],

        "Missing Values": [
            int(
                df[column].isna().sum()
            )
            for column in df.columns
        ],

        "Missing (%)": [
            float(
                df[column].isna().mean() * 100
            )
            for column in df.columns
        ],

        "Unique Values": [
            int(
                df[column].nunique()
            )
            for column in df.columns
        ]
    })

    return quality


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def get_numeric_summary(df):
    """
    Return descriptive statistics
    for numerical variables.
    """

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    if len(numeric_columns) == 0:

        return pd.DataFrame()

    summary = (
        df[
            numeric_columns
        ]
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
        .reset_index(
            drop=True
        )
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

    target_id = str(
        cattle_id
    ).strip()

    cattle_data = df[
        df["cattle_id"].apply(
            lambda x: str(x).strip()
        )
        == target_id
    ].copy()

    if "timestamp" in cattle_data.columns:

        cattle_data = (
            cattle_data
            .sort_values("timestamp")
            .reset_index(
                drop=True
            )
        )

    return cattle_data


# ============================================================
# DAILY AGGREGATION
# ============================================================

def daily_average(df, columns):
    """
    Calculate daily average values
    for selected sensor variables.
    """

    if "day" not in df.columns:

        raise KeyError(
            "Column 'day' is required."
        )

    available_columns = [

        column

        for column in columns

        if column in df.columns
    ]

    if len(available_columns) == 0:

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

    if len(aggregation) == 0:

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
