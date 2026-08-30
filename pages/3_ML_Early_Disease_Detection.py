import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)

from src.data_loader_v2 import load_data, merge_metadata
from src.ml_model import FEATURES


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Early Disease Detection",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 Machine Learning — Early Disease Detection"
)

st.markdown(
    """
    Halaman ini digunakan untuk membangun dan mengevaluasi
    model Machine Learning untuk mendeteksi risiko penyakit
    pada sapi berdasarkan data fisiologis, konsumsi, aktivitas,
    dan kondisi lingkungan.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

sensor_path = Path(
    "data/cattle_sensor_data.csv"
)

metadata_path = Path(
    "data/cattle_metadata.csv"
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
# DATA INFORMATION
# ============================================================

st.subheader(
    "📊 Dataset Information"
)


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )


with col2:
    st.metric(
        "Number of Cattle",
        df["cattle_id"].nunique()
    )


with col3:
    st.metric(
        "Observation Days",
        df["day"].nunique()
    )


with col4:
    alert_percentage = (
        df["disease_alert"].mean()
        * 100
    )

    st.metric(
        "Disease Alert (%)",
        f"{alert_percentage:.2f}%"
    )


st.divider()


# ============================================================
# MACHINE LEARNING FEATURES
# ============================================================

st.subheader(
    "🧬 Machine Learning Features"
)


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


available_features = [
    feature
    for feature in FEATURES
    if feature in df.columns
]


feature_labels = {
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


feature_table = pd.DataFrame({
    "Feature": [
        feature_labels.get(
            x,
            x
        )
        for x in available_features
    ],

    "Column": available_features
})


st.dataframe(
    feature_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA PREPARATION
# ============================================================

st.divider()

st.subheader(
    "⚙️ Data Preparation"
)


target_classification = "disease_alert"

target_regression = "disease_risk_score"


model_data = df.dropna(
    subset=
    available_features
    + [
        target_classification,
        target_regression
    ]
).copy()


X = model_data[
    available_features
]


y_class = model_data[
    target_classification
].astype(int)


y_reg = model_data[
    target_regression
].astype(float)


st.write(
    f"Records used for modeling: **{len(model_data):,}**"
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

st.subheader(
    "🔀 Train-Test Split"
)


test_size = st.slider(
    "Test Data Percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)


X_train, X_test, y_train_class, y_test_class = (
    train_test_split(
        X,
        y_class,
        test_size=test_size / 100,
        random_state=42,
        stratify=y_class
    )
)


X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    train_test_split(
        X,
        y_reg,
        test_size=test_size / 100,
        random_state=42
    )
)


st.write(
    f"""
    Training records: **{len(X_train):,}**

    Testing records: **{len(X_test):,}**
    """
)


# ============================================================
# MODEL TRAINING
# ============================================================

st.divider()

st.subheader(
    "🌲 Random Forest Model"
)


n_estimators = st.slider(
    "Number of Trees",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)


max_depth = st.slider(
    "Maximum Tree Depth",
    min_value=2,
    max_value=30,
    value=10
)


if st.button(
    "🚀 Train Machine Learning Models",
    type="primary"
):

    with st.spinner(
        "Training Random Forest models..."
    ):

        # ====================================================
        # CLASSIFICATION MODEL
        # ====================================================

        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )


        classifier.fit(
            X_train,
            y_train_class
        )


        class_prediction = (
            classifier.predict(X_test)
        )


        class_probability = (
            classifier.predict_proba(X_test)[:, 1]
        )


        # ====================================================
        # REGRESSION MODEL
        # ====================================================

        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )


        regressor.fit(
            X_train_reg,
            y_train_reg
        )


        regression_prediction = (
            regressor.predict(X_test_reg)
        )


        # ====================================================
        # CLASSIFICATION METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test_class,
            class_prediction
        )


        precision = precision_score(
            y_test_class,
            class_prediction,
            zero_division=0
        )


        recall = recall_score(
            y_test_class,
            class_prediction,
            zero_division=0
        )


        f1 = f1_score(
            y_test_class,
            class_prediction,
            zero_division=0
        )


        try:

            auc = roc_auc_score(
                y_test_class,
                class_probability
            )

        except ValueError:

            auc = 0


        # ====================================================
        # REGRESSION METRICS
        # ====================================================

        mae = mean_absolute_error(
            y_test_reg,
            regression_prediction
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test_reg,
                regression_prediction
            )
        )


        r2 = r2_score(
            y_test_reg,
            regression_prediction
        )


        # ====================================================
        # SAVE RESULTS TO SESSION
        # ====================================================

        st.session_state[
            "classifier"
        ] = classifier


        st.session_state[
            "regressor"
        ] = regressor


        st.session_state[
            "classification_metrics"
        ] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc
        }


        st.session_state[
            "regression_metrics"
        ] = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        }


        st.session_state[
            "class_prediction"
        ] = class_prediction


        st.session_state[
            "class_probability"
        ] = class_probability


        st.session_state[
            "y_test_class"
        ] = y_test_class


        st.session_state[
            "regression_prediction"
        ] = regression_prediction


        st.session_state[
            "y_test_reg"
        ] = y_test_reg


        st.success(
            "Machine Learning models successfully trained."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if "classifier" in st.session_state:

    st.divider()

    st.subheader(
        "📊 Classification Performance"
    )


    metrics = st.session_state[
        "classification_metrics"
    ]


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.3f}"
    )


    c2.metric(
        "Precision",
        f"{metrics['Precision']:.3f}"
    )


    c3.metric(
        "Recall",
        f"{metrics['Recall']:.3f}"
    )


    c4.metric(
        "F1-Score",
        f"{metrics['F1-Score']:.3f}"
    )


    c5.metric(
        "ROC-AUC",
        f"{metrics['ROC-AUC']:.3f}"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )


    cm = confusion_matrix(
        st.session_state["y_test_class"],
        st.session_state["class_prediction"]
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Normal",
            "Actual Alert"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Alert"
        ]
    )


    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Disease Alert Confusion Matrix"
    )


    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


    # ========================================================
    # REGRESSION PERFORMANCE
    # ========================================================

    st.subheader(
        "📈 Disease Risk Score Regression"
    )


    reg_metrics = st.session_state[
        "regression_metrics"
    ]


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "MAE",
        f"{reg_metrics['MAE']:.3f}"
    )


    c2.metric(
        "RMSE",
        f"{reg_metrics['RMSE']:.3f}"
    )


    c3.metric(
        "R²",
        f"{reg_metrics['R²']:.3f}"
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    prediction_df = pd.DataFrame({

        "Actual":
            st.session_state[
                "y_test_reg"
            ],

        "Predicted":
            st.session_state[
                "regression_prediction"
            ]
    })


    fig_prediction = px.scatter(
        prediction_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Disease Risk Score",
        trendline="ols"
    )


    fig_prediction.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1
    )


    st.plotly_chart(
        fig_prediction,
        use_container_width=True
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Feature Importance"
    )


    importance_df = pd.DataFrame({

        "Feature": [
            feature_labels.get(
                x,
                x
            )
            for x in available_features
        ],

        "Importance":
            st.session_state[
                "classifier"
            ].feature_importances_
    })


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )


    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )


    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )


    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREDICTION FOR ALL CATTLE
# ============================================================

if "classifier" in st.session_state:

    st.divider()

    st.subheader(
        "🐄 Current Disease Risk Prediction"
    )


    latest = (
        df
        .sort_values("timestamp")
        .groupby("cattle_id")
        .tail(1)
        .copy()
    )


    latest = latest.dropna(
        subset=available_features
    )


    latest_X = latest[
        available_features
    ]


    # Classification probability

    latest[
        "Predicted Alert Probability"
    ] = (
        st.session_state[
            "classifier"
        ]
        .predict_proba(
            latest_X
        )[:, 1]
    )


    # Disease risk score

    latest[
        "Predicted Risk Score"
    ] = (
        st.session_state[
            "regressor"
        ]
        .predict(
            latest_X
        )
        .clip(0, 1)
    )


    # Risk category

    def classify_risk(score):

        if score < 0.30:
            return "Low Risk"

        elif score < 0.70:
            return "Medium Risk"

        return "High Risk"


    latest[
        "Risk Level"
    ] = latest[
        "Predicted Risk Score"
    ].apply(
        classify_risk
    )


    # Alert

    latest[
        "Predicted Alert"
    ] = np.where(
        latest[
            "Predicted Alert Probability"
        ] >= 0.50,
        "Disease Alert",
        "Normal"
    )


    result_columns = [
        "cattle_id",
        "timestamp",
        "body_temperature_c",
        "heart_rate_bpm",
        "respiration_rate_bpm",
        "activity_percent",
        "Predicted Alert Probability",
        "Predicted Risk Score",
        "Risk Level",
        "Predicted Alert"
    ]


    result_columns = [
        col
        for col in result_columns
        if col in latest.columns
    ]


    prediction_result = latest[
        result_columns
    ].sort_values(
        "Predicted Risk Score",
        ascending=False
    )


    st.dataframe(
        prediction_result,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.subheader(
        "🚨 Risk Distribution Across Cattle"
    )


    risk_counts = (
        prediction_result[
            "Risk Level"
        ]
        .value_counts()
        .reset_index()
    )


    risk_counts.columns = [
        "Risk Level",
        "Number of Cattle"
    ]


    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Number of Cattle",
        text_auto=True,
        title="Predicted Risk Level Distribution"
    )


    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


    # ========================================================
    # HIGH RISK CATTLE
    # ========================================================

    high_risk = prediction_result[
        prediction_result[
            "Risk Level"
        ] == "High Risk"
    ]


    if len(high_risk) > 0:

        st.warning(
            f"⚠️ Terdapat **{len(high_risk)} sapi** "
            "yang masuk kategori High Risk."
        )

        st.dataframe(
            high_risk,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "✅ Tidak terdapat sapi yang "
            "masuk kategori High Risk."
        )


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "🧠 Model Interpretation"
)


st.info(
    """
    **Interpretasi sistem:**

    • Disease Alert digunakan untuk klasifikasi kondisi
      berpotensi abnormal.

    • Disease Risk Score menunjukkan estimasi tingkat
      risiko pada rentang 0–1.

    • Low Risk: < 0.30

    • Medium Risk: 0.30–0.70

    • High Risk: > 0.70

    • Feature Importance menunjukkan kontribusi relatif
      fitur terhadap keputusan model Random Forest.

    Hasil model merupakan **early warning system** dan
    bukan diagnosis klinis. Pemeriksaan lebih lanjut oleh
    dokter hewan tetap diperlukan untuk menentukan status
    kesehatan sebenarnya.
    """
)


# ============================================================
# RESEARCH NOTE
# ============================================================

with st.expander(
    "📚 Research Methodology Note"
):

    st.markdown(
        """
        ### Recommended Research Design

        Untuk penelitian forecasting, pembagian data secara
        random seperti prototype ini sebaiknya diganti dengan
        **time-based split**.

        Contoh:

        **Training**
        
        Hari 1–60

        **Validation**

        Hari 61–75

        **Testing**

        Hari 76–90

        Pendekatan ini mengurangi risiko data leakage dan
        lebih sesuai dengan tujuan forecasting penyakit.

        Model baseline Random Forest dapat dibandingkan
        dengan XGBoost, LSTM, GRU, dan ensemble learning.

        Untuk Explainable AI, SHAP dapat digunakan untuk
        menjelaskan prediksi individual setiap sapi.
        """
    )


st.caption(
    "Cattle Health ML Dashboard — Early Disease Detection "
    "Using IoT Sensor Data"
)
