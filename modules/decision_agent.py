
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


# ==========================================================
# 1. DETECT ID COLUMNS
# ==========================================================
def detect_id_columns(df):

    id_cols = []

    for col in df.columns:
        col_lower = str(col).lower()

        if any(word in col_lower for word in [
            "id", "index", "serial", "roll"
        ]):
            id_cols.append(col)
            continue

        if len(df) > 0 and pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)

            if unique_ratio > 0.90:
                id_cols.append(col)

    return id_cols


# ==========================================================
# 2. DETECT DATETIME COLUMNS
# ==========================================================
def detect_datetime_columns(df):

    date_cols = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue

        if df[col].dtype == "object":
            converted = pd.to_datetime(
                df[col], errors="coerce"
            )

            non_missing = df[col].notna().sum()

            if non_missing > 0:
                success_ratio = (
                    converted.notna().sum() / non_missing
                )

                if success_ratio >= 0.80:
                    date_cols.append(col)

    return date_cols


# ==========================================================
# 3. DETECT NUMERIC COLUMNS
# ==========================================================
def detect_numeric_columns(df, id_cols):

    return [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in id_cols
    ]


# ==========================================================
# 4. DETECT CATEGORICAL COLUMNS
# ==========================================================
def detect_categorical_columns(df):

    return [
        col for col in df.columns
        if (
            df[col].dtype == "object"
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        )
    ]


# ==========================================================
# 5. DETECT TARGET COLUMN
# ==========================================================
def detect_target(df, id_cols):

    priority_names = [
        "target", "class", "label", "output", "response",
        "survived", "species", "diagnosis", "price",
        "salary", "income", "saleprice"
    ]

    for col in df.columns:
        if str(col).lower() in priority_names:
            return col, 95

    candidates = [
        col for col in df.columns
        if col not in id_cols
    ]

    if candidates:
        return candidates[-1], 60

    return None, 0


# ==========================================================
# 6. DETECT PROBLEM TYPE
# ==========================================================
def detect_problem_type(df, target, date_cols):

    if target is None:
        return {
            "problem": "Clustering / Unsupervised Learning",
            "confidence": "Medium",
            "reason": "No obvious target variable was detected."
        }

    target_series = df[target]
    unique_values = target_series.nunique()

    if unique_values == 0:
        return {
            "problem": "Unable to determine",
            "confidence": "Low",
            "reason": "The target contains no usable values."
        }

    if date_cols and pd.api.types.is_numeric_dtype(target_series):
        return {
            "problem": "Time Series Forecasting",
            "confidence": "Medium",
            "reason": (
                "Datetime columns and a numeric target were detected. "
                "Confirm that the data represents observations over time."
            )
        }

    if pd.api.types.is_numeric_dtype(target_series):

        if unique_values == 2:
            return {
                "problem": "Binary Classification",
                "confidence": "Medium",
                "reason": "The numeric target contains two distinct values."
            }

        elif unique_values <= 10:
            return {
                "problem": "Multi-class Classification",
                "confidence": "Medium",
                "reason": (
                    f"The numeric target contains {unique_values} "
                    "distinct values."
                )
            }

        return {
            "problem": "Regression",
            "confidence": "High",
            "reason": "The numeric target has more than 10 distinct values."
        }

    if unique_values == 2:
        return {
            "problem": "Binary Classification",
            "confidence": "High",
            "reason": "The categorical target contains two classes."
        }

    return {
        "problem": "Multi-class Classification",
        "confidence": "Medium",
        "reason": (
            f"The categorical target contains {unique_values} classes."
        )
    }


# ==========================================================
# 7. RECOMMEND STATISTICAL TESTS
# ==========================================================
def recommend_statistical_tests(
    df, target, problem, numeric_cols, categorical_cols
):

    recommendations = []

    if target is None:
        return [
            (
                "Exploratory Data Analysis",
                "Start with descriptive statistics and visualizations."
            )
        ]

    if problem == "Binary Classification":

        if categorical_cols:
            recommendations.append((
                "Chi-Square Test",
                "Examine associations between categorical variables."
            ))

        if numeric_cols:
            recommendations.append((
                "Independent t-Test",
                "Compare numeric means between two target groups."
            ))

        if len(numeric_cols) > 1:
            recommendations.append((
                "Pearson Correlation",
                "Examine linear relationships between numeric variables."
            ))

    elif problem == "Multi-class Classification":

        if numeric_cols:
            recommendations.append((
                "One-Way ANOVA",
                "Compare numeric means across target groups."
            ))

        if categorical_cols:
            recommendations.append((
                "Chi-Square Test",
                "Examine associations between categorical variables."
            ))

    elif problem == "Regression":

        recommendations.extend([
            (
                "Pearson Correlation",
                "Measure linear relationships between numeric variables."
            ),
            (
                "Multiple Linear Regression",
                "Model a numeric outcome using multiple predictors."
            ),
            (
                "Variance Inflation Factor (VIF)",
                "Check for multicollinearity among predictors."
            )
        ])

    elif problem == "Time Series Forecasting":

        recommendations.extend([
            (
                "ADF Test",
                "Check whether the time series is stationary."
            ),
            (
                "ACF/PACF",
                "Examine autocorrelation and temporal dependencies."
            )
        ])

    else:
        recommendations.append((
            "Cluster Analysis",
            "Explore groups in the data without a target variable."
        ))

    return recommendations


# ==========================================================
# 8. RECOMMEND ML MODELS
# ==========================================================
def recommend_ml_models(problem):

    if problem == "Binary Classification":
        return [
            ("Logistic Regression", "An interpretable baseline classifier."),
            ("Random Forest", "Can capture nonlinear relationships."),
            ("XGBoost", "An ensemble model for structured data.")
        ]

    elif problem == "Multi-class Classification":
        return [
            ("Random Forest", "Supports multiple target classes."),
            ("XGBoost", "An ensemble model for structured data."),
            ("Support Vector Machine", "Useful for suitable classification tasks.")
        ]

    elif problem == "Regression":
        return [
            ("Linear Regression", "A simple regression baseline."),
            ("Random Forest Regressor", "Can capture nonlinear relationships."),
            ("XGBoost Regressor", "An ensemble model for numeric prediction.")
        ]

    elif problem == "Time Series Forecasting":
        return [
            ("ARIMA", "A classical time-series forecasting model."),
            ("Prophet", "Can model trends and seasonality."),
            ("LSTM", "A neural network for sequential data.")
        ]

    return [
        ("K-Means", "Groups observations into clusters."),
        ("DBSCAN", "Finds density-based clusters."),
        ("Hierarchical Clustering", "Builds a hierarchy of clusters.")
    ]


# ==========================================================
# 9. DATASET UNDERSTANDING REPORT
# ==========================================================
def dataset_understanding(df):

    st.header("🧠 Dataset Understanding")

    id_cols = detect_id_columns(df)
    date_cols = detect_datetime_columns(df)
    numeric_cols = detect_numeric_columns(df, id_cols)
    categorical_cols = detect_categorical_columns(df)

    target, target_confidence = detect_target(df, id_cols)

    problem = detect_problem_type(
        df, target, date_cols
    )

    tests = recommend_statistical_tests(
        df, target, problem["problem"],
        numeric_cols, categorical_cols
    )

    models = recommend_ml_models(problem["problem"])

    st.subheader("📋 Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

    with col2:
        st.metric("Numeric Features", len(numeric_cols))
        st.metric("Categorical Features", len(categorical_cols))

    st.divider()

    st.subheader("🆔 Detected ID Columns")
    st.write(", ".join(map(str, id_cols)) if id_cols
             else "No ID columns detected.")

    st.subheader("📅 Datetime Columns")
    st.write(", ".join(map(str, date_cols)) if date_cols
             else "No datetime columns detected.")

    st.subheader("🎯 Possible Target Variable")

    if target is not None:
        st.success(str(target))
        st.caption(f"Heuristic confidence score: {target_confidence}%")
    else:
        st.warning("No obvious target detected.")

    st.subheader("🧠 Detected Problem Type")
    st.success(problem["problem"])
    st.caption(f"Confidence: {problem['confidence']}")
    st.info(problem["reason"])

    st.subheader("📊 Recommended Statistical Tests")

    for name, desc in tests:
        st.write(f"**{name}:** {desc}")

    st.subheader("🤖 Recommended ML Models")

    for name, desc in models:
        st.write(f"**{name}:** {desc}")

    return {
        "problem": problem,
        "target": target,
        "confidence": target_confidence
    }


# ==========================================================
# 10. PLANNING AGENT
# ==========================================================
def planning_agent(problem_type):

    base_plan = [
        "clean_data",
        "encode_data",
        "split_data"
    ]

    if problem_type in [
        "Binary Classification",
        "Multi-class Classification"
    ]:
        base_plan.extend([
            "train_logistic",
            "train_random_forest",
            "evaluate_models",
            "select_best"
        ])

    elif problem_type == "Regression":
        base_plan.extend([
            "train_linear",
            "train_random_forest",
            "evaluate_models",
            "select_best"
        ])

    elif problem_type == "Time Series Forecasting":
        base_plan.extend([
            "train_arima",
            "evaluate_forecast",
            "select_best"
        ])

    else:
        base_plan.append("clustering")

    return base_plan


