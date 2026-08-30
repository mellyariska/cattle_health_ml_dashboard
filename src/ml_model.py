"""
Machine Learning module for
Cattle Health ML Dashboard.

Purpose:
- Disease alert classification
- Disease risk score regression
- Feature importance
- Individual cattle prediction
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# MACHINE LEARNING FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS = {

    "body_temperature_c":
        "Body Temperature",

    "heart_rate_bpm":
        "Heart Rate",

    "respiration_rate_bpm":
        "Respiration Rate",

    "activity_percent":
        "Activity",

    "feed_intake_kg_day":
        "Feed Intake",

    "water_intake_l_day":
        "Water Intake",

    "ambient_temperature_c":
        "Ambient Temperature",

    "humidity_percent":
        "Humidity",

    "co2_ppm":
        "CO₂",

    "nh3_ppm":
        "NH₃"
}


# ============================================================
# GET AVAILABLE FEATURES
# ============================================================

def get_available_features(df):
    """
    Return ML features that are available in the dataset.
    """

    return [
        feature
        for feature in FEATURES
        if feature in df.columns
    ]


# ============================================================
# PREPARE CLASSIFICATION DATA
# ============================================================

def prepare_classification_data(
    df,
    target="disease_alert"
):
    """
    Prepare dataset for disease alert classification.

    Target:
        disease_alert
        0 = Normal
        1 = Disease Alert
    """

    available_features = get_available_features(df)

    required_columns = (
        available_features
        + [target]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    data = df[
        required_columns
    ].dropna().copy()

    X = data[
        available_features
    ]

    y = data[
        target
    ].astype(int)

    return X, y, available_features


# ============================================================
# PREPARE REGRESSION DATA
# ============================================================

def prepare_regression_data(
    df,
    target="disease_risk_score"
):
    """
    Prepare dataset for disease risk score regression.
    """

    available_features = get_available_features(df)

    required_columns = (
        available_features
        + [target]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    data = df[
        required_columns
    ].dropna().copy()

    X = data[
        available_features
    ]

    y = data[
        target
    ].astype(float)

    return X, y, available_features


# ============================================================
# TRAIN RANDOM FOREST CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=10
):
    """
    Train Random Forest Classifier.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# TRAIN RANDOM FOREST REGRESSOR
# ============================================================

def train_regressor(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=10
):
    """
    Train Random Forest Regressor.
    """

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classifier(
    model,
    X_test,
    y_test
):
    """
    Evaluate Random Forest classification model.
    """

    prediction = model.predict(
        X_test
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                prediction
            ),

        "Precision":
            precision_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "F1-Score":
            f1_score(
                y_test,
                prediction,
                zero_division=0
            )
    }

    try:

        metrics["ROC-AUC"] = (
            roc_auc_score(
                y_test,
                probability
            )
        )

    except ValueError:

        metrics["ROC-AUC"] = np.nan

    return (
        metrics,
        prediction,
        probability
    )


# ============================================================
# REGRESSION EVALUATION
# ============================================================

def evaluate_regressor(
    model,
    X_test,
    y_test
):
    """
    Evaluate Random Forest regression model.
    """

    prediction = model.predict(
        X_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    metrics = {

        "MAE":
            mean_absolute_error(
                y_test,
                prediction
            ),

        "RMSE":
            rmse,

        "R²":
            r2_score(
                y_test,
                prediction
            )
    }

    return (
        metrics,
        prediction
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    features
):
    """
    Return Random Forest feature importance.
    """

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance":
            model.feature_importances_
    })

    importance_df[
        "Feature Name"
    ] = importance_df[
        "Feature"
    ].map(
        FEATURE_LABELS
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    risk_score
):
    """
    Convert disease risk score into risk category.

    < 0.30  = Low Risk
    < 0.70  = Medium Risk
    >= 0.70 = High Risk
    """

    if risk_score < 0.30:

        return "Low Risk"

    elif risk_score < 0.70:

        return "Medium Risk"

    else:

        return "High Risk"


# ============================================================
# RISK LEVEL NUMERIC
# ============================================================

def risk_level_numeric(
    risk_score
):
    """
    Convert risk score into numeric risk level.
    """

    if risk_score < 0.30:

        return 0

    elif risk_score < 0.70:

        return 1

    return 2


# ============================================================
# PREDICT DISEASE ALERT
# ============================================================

def predict_disease_alert(
    model,
    X
):
    """
    Predict disease alert probability.
    """

    prediction = model.predict(X)

    probability = model.predict_proba(
        X
    )[:, 1]

    return prediction, probability


# ============================================================
# PREDICT DISEASE RISK
# ============================================================

def predict_disease_risk(
    model,
    X
):
    """
    Predict disease risk score.
    """

    prediction = model.predict(X)

    prediction = np.clip(
        prediction,
        0,
        1
    )

    return prediction


# ============================================================
# PREDICT INDIVIDUAL CATTLE
# ============================================================

def predict_cattle_risk(
    model_classifier,
    model_regressor,
    cattle_data,
    features=None
):
    """
    Generate disease prediction for individual cattle.
    """

    if features is None:

        features = get_available_features(
            cattle_data
        )

    if not features:

        return pd.DataFrame()

    prediction_data = cattle_data[
        features
    ].copy()

    valid_data = (
        prediction_data
        .dropna()
        .copy()
    )

    if valid_data.empty:

        return pd.DataFrame()

    # Disease probability

    valid_data[
        "disease_probability"
    ] = (
        model_classifier
        .predict_proba(
            valid_data[features]
        )[:, 1]
    )

    # Disease alert

    valid_data[
        "disease_alert_predicted"
    ] = (
        model_classifier
        .predict(
            valid_data[features]
        )
    )

    # Disease risk

    valid_data[
        "disease_risk_score_predicted"
    ] = (
        model_regressor
        .predict(
            valid_data[features]
        )
    )

    # Keep score within 0–1

    valid_data[
        "disease_risk_score_predicted"
    ] = (
        valid_data[
            "disease_risk_score_predicted"
        ]
        .clip(0, 1)
    )

    # Risk category

    valid_data[
        "risk_level"
    ] = (
        valid_data[
            "disease_risk_score_predicted"
        ]
        .apply(
            classify_risk
        )
    )

    return valid_data


# ============================================================
# PREDICT ALL CATTLE
# ============================================================

def predict_all_cattle(
    df,
    model_classifier,
    model_regressor,
    features=None
):
    """
    Generate latest disease risk prediction
    for every cattle.
    """

    if features is None:

        features = get_available_features(df)

    if "timestamp" not in df.columns:

        raise ValueError(
            "Column 'timestamp' is required."
        )

    # Get latest record

    latest = (
        df
        .sort_values("timestamp")
        .groupby("cattle_id")
        .tail(1)
        .copy()
    )

    # Remove records with missing features

    latest = latest.dropna(
        subset=features
    )

    if latest.empty:

        return pd.DataFrame()

    X_latest = latest[
        features
    ]

    # Classification probability

    latest[
        "disease_probability"
    ] = (
        model_classifier
        .predict_proba(
            X_latest
        )[:, 1]
    )

    # Classification prediction

    latest[
        "disease_alert_predicted"
    ] = (
        model_classifier
        .predict(
            X_latest
        )
    )

    # Regression prediction

    latest[
        "disease_risk_score_predicted"
    ] = (
        model_regressor
        .predict(
            X_latest
        )
    )

    # Clip score

    latest[
        "disease_risk_score_predicted"
    ] = (
        latest[
            "disease_risk_score_predicted"
        ]
        .clip(0, 1)
    )

    # Risk category

    latest[
        "risk_level"
    ] = (
        latest[
            "disease_risk_score_predicted"
        ]
        .apply(
            classify_risk
        )
    )

    # Sort high risk first

    latest = latest.sort_values(
        "disease_risk_score_predicted",
        ascending=False
    )

    return latest


# ============================================================
# SIMPLE MODEL SUMMARY
# ============================================================

def model_summary(
    classifier,
    regressor,
    features
):
    """
    Return basic model information.
    """

    summary = {

        "Classifier":
            type(classifier).__name__,

        "Regressor":
            type(regressor).__name__,

        "Number of Features":
            len(features),

        "Features":
            features
    }

    return summary