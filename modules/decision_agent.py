import streamlit as st
import pandas as pd


# ==========================================================
# Detect ID Columns
# ==========================================================
def detect_id_columns(df):

    id_cols = []

    for col in df.columns:

        col_lower = col.lower()

        # Check by column name
        if (
            "id" in col_lower
            or "index" in col_lower
            or "serial" in col_lower
            or "roll" in col_lower
        ):
            id_cols.append(col)
            continue

        # Check by uniqueness
        if pd.api.types.is_numeric_dtype(df[col]):

            unique_ratio = df[col].nunique() / len(df)

            if unique_ratio > 0.90:
                id_cols.append(col)

    return id_cols


# ==========================================================
# Detect Datetime Columns
# ==========================================================
def detect_datetime_columns(df):

    date_cols = []

    for col in df.columns:

        # Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue

        # Only attempt conversion for text columns
        if df[col].dtype == "object":

            try:

                converted = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                # At least 80% of non-null values should parse as dates
                success_ratio = converted.notna().mean()

                if success_ratio >= 0.80:
                    date_cols.append(col)

            except:
                pass

    return date_cols


# ==========================================================
# Detect Numeric Columns
# ==========================================================
def detect_numeric_columns(df, id_cols):

    numeric = []

    for col in df.select_dtypes(include="number").columns:

        if col not in id_cols:
            numeric.append(col)

    return numeric


# ==========================================================
# Detect Categorical Columns
# ==========================================================
def detect_categorical_columns(df):

    categorical = []

    for col in df.columns:

        if (
            df[col].dtype == "object"
            or str(df[col].dtype) == "category"
        ):
            categorical.append(col)

    return categorical


# ==========================================================
# Guess Target Column
# ==========================================================
def detect_target(df, id_cols):

    priority_names = [

        "target",
        "class",
        "label",
        "output",
        "response",
        "survived",
        "species",
        "diagnosis",
        "price",
        "salary",
        "income",
        "saleprice"

    ]

    # Check common names first
    for col in df.columns:

        if col.lower() in priority_names:

            return col, 95

    # Otherwise choose last non-ID column
    candidates = [c for c in df.columns if c not in id_cols]

    if candidates:

        return candidates[-1], 60

    return None, 0

# ==========================================================
# Detect Problem Type
# ==========================================================
def detect_problem_type(df, target, date_cols):

    # No target found
    if target is None:
        return {
            "problem": "Clustering / Unsupervised Learning",
            "confidence": 80,
            "reason": "No obvious target variable was detected."
        }

    target_series = df[target]

    # -------------------------
    # Time Series
    # -------------------------
    if len(date_cols) > 0:

        if pd.api.types.is_numeric_dtype(target_series):
            return {
                "problem": "Time Series Forecasting",
                "confidence": 95,
                "reason": f"Datetime column(s) detected ({', '.join(date_cols)}) with a numeric target."
            }

    # -------------------------
    # Numeric Target
    # -------------------------
    if pd.api.types.is_numeric_dtype(target_series):

        unique_values = target_series.nunique()

        # Binary Classification
        if unique_values == 2:
            return {
                "problem": "Binary Classification",
                "confidence": 98,
                "reason": "Target contains exactly two unique values."
            }

        # Multi-class Classification
        elif unique_values <= 10:
            return {
                "problem": "Multi-class Classification",
                "confidence": 92,
                "reason": f"Target has {unique_values} distinct classes."
            }

        # Regression
        else:
            return {
                "problem": "Regression",
                "confidence": 95,
                "reason": "Target is continuous with many unique values."
            }

    # -------------------------
    # Categorical Target
    # -------------------------
    else:

        unique_values = target_series.nunique()

        if unique_values == 2:
            return {
                "problem": "Binary Classification",
                "confidence": 98,
                "reason": "Categorical target with two classes."
            }

        return {
            "problem": "Multi-class Classification",
            "confidence": 95,
            "reason": f"Categorical target with {unique_values} classes."
        }
def recommend_statistical_tests(df, target, problem, numeric_cols, categorical_cols):

    recommendations = []

    if target is None:
        recommendations.append((
            "Exploratory Data Analysis",
            "No target variable detected. Begin with visualization and descriptive statistics."
        ))
        return recommendations

    # -------------------------
    # Binary Classification
    # -------------------------
    if problem == "Binary Classification":

        if len(categorical_cols) > 0:
            recommendations.append((
                "Chi-Square Test",
                "Useful for studying relationships between categorical predictors and the binary target."
            ))

        if len(numeric_cols) > 0:
            recommendations.append((
                "Independent t-Test",
                "Compare the means of numeric variables between the two target classes."
            ))

        if len(numeric_cols) > 1:
            recommendations.append((
                "Pearson Correlation",
                "Examine linear relationships among numeric predictors."
            ))

    # -------------------------
    # Multi-class Classification
    # -------------------------
    elif problem == "Multi-class Classification":

        if len(numeric_cols) > 0:
            recommendations.append((
                "One-Way ANOVA",
                "Compare numeric variables across multiple target classes."
            ))

        if len(categorical_cols) > 0:
            recommendations.append((
                "Chi-Square Test",
                "Assess association between categorical variables."
            ))

    # -------------------------
    # Regression
    # -------------------------
    elif problem == "Regression":

        recommendations.append((
            "Pearson Correlation",
            "Measure relationships between numeric variables."
        ))

        recommendations.append((
            "Multiple Linear Regression",
            "Model the relationship between predictors and the continuous target."
        ))

        recommendations.append((
            "Variance Inflation Factor (VIF)",
            "Detect multicollinearity among predictors."
        ))

    # -------------------------
    # Time Series
    # -------------------------
    elif problem == "Time Series Forecasting":

        recommendations.append((
            "ADF Test",
            "Check stationarity before forecasting."
        ))

        recommendations.append((
            "Autocorrelation (ACF/PACF)",
            "Identify temporal dependencies."
        ))

    else:

        recommendations.append((
            "Cluster Analysis",
            "No supervised target detected."
        ))

    return recommendations   
# ==========================================================
# Recommend Machine Learning Models
# ==========================================================
def recommend_ml_models(problem):

    if problem == "Binary Classification":

        return [
            ("Logistic Regression", "Simple, interpretable baseline model."),
            ("Random Forest", "Handles nonlinear relationships and mixed features."),
            ("XGBoost", "High predictive accuracy for tabular datasets.")
        ]

    elif problem == "Multi-class Classification":

        return [
            ("Random Forest", "Excellent for multi-class datasets."),
            ("XGBoost", "Strong performance on structured data."),
            ("Support Vector Machine", "Effective for moderate-sized datasets.")
        ]

    elif problem == "Regression":

        return [
            ("Linear Regression", "Simple baseline regression model."),
            ("Random Forest Regressor", "Captures nonlinear relationships."),
            ("XGBoost Regressor", "Powerful ensemble model for regression.")
        ]

    elif problem == "Time Series Forecasting":

        return [
            ("ARIMA", "Classical forecasting model."),
            ("Prophet", "Handles trend and seasonality well."),
            ("LSTM", "Deep learning approach for sequential data.")
        ]

    else:

        return [
            ("K-Means", "Popular clustering algorithm."),
            ("DBSCAN", "Density-based clustering."),
            ("Hierarchical Clustering", "Produces interpretable cluster hierarchy.")
        ]
 
# ==========================================================
# Dataset Understanding Report
# ==========================================================
def dataset_understanding(df):

    st.header("🧠 Dataset Understanding")

    id_cols = detect_id_columns(df)
    date_cols = detect_datetime_columns(df)
    numeric_cols = detect_numeric_columns(df, id_cols)
    categorical_cols = detect_categorical_columns(df)

    target, confidence = detect_target(df, id_cols)
    problem = detect_problem_type(df, target, date_cols)

    tests = recommend_statistical_tests(
        df,
        target,
        problem["problem"],
        numeric_cols,
        categorical_cols
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
    st.success(", ".join(id_cols) if id_cols else "No ID columns detected.")

    st.divider()

    st.subheader("📅 Datetime Columns")
    st.success(", ".join(date_cols) if date_cols else "No datetime columns detected.")

    st.divider()

    st.subheader("🎯 Possible Target Variable")

    if target:
        st.success(f"{target}")
        st.caption(f"Confidence : {confidence}%")
    else:
        st.warning("No obvious target detected.")

    st.divider()

    st.subheader("🧠 Detected Problem Type")
    st.success(problem["problem"])
    st.metric("Confidence", f'{problem["confidence"]}%')
    st.info(problem["reason"])

    st.divider()

    st.subheader("📊 Recommended Statistical Tests")
    for name, desc in tests:
        st.write(f"**{name}**: {desc}")

    st.divider()

    st.subheader("🤖 Recommended ML Models")
    for name, desc in models:
        st.write(f"**{name}**: {desc}")

    # ✅ CRITICAL FIX: RETURN VALUES FOR AGENT
    return {
        "problem": problem,
        "target": target,
        "confidence": confidence
    }
def planning_agent(problem_type):

    base_plan = [
        "clean_data",
        "encode_data",
        "split_data"
    ]

    if problem_type in ["Binary Classification", "Multi-class Classification"]:
        base_plan += [
            "train_logistic",
            "train_random_forest",
            "evaluate_models",
            "select_best"
        ]

    elif problem_type == "Regression":
        base_plan += [
            "train_linear",
            "train_random_forest",
            "evaluate_models",
            "select_best"
        ]

    elif problem_type == "Time Series Forecasting":
        base_plan += [
            "train_arima",
            "evaluate_forecast",
            "select_best"
        ]

    else:
        base_plan += ["clustering"]

    return base_plan

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, r2_score


# ==========================================================
# BASIC CLEANING
# ==========================================================
def clean_data(df):

    df = df.copy()

    # drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # fill missing values (simple strategy)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "missing")
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


# ==========================================================
# ENCODING
# ==========================================================
def encode_data(df):

    df = df.copy()
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col].astype(str))

    return df


# ==========================================================
# SPLIT DATA
# ==========================================================
def split_data(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================================================
# MODEL TRAINING HELPERS
# ==========================================================
def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_rf_classifier(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def train_rf_regressor(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ==========================================================
# EXECUTION ENGINE
# ==========================================================
def execution_engine(df, target, problem_type, plan):

    df = clean_data(df)
    df = encode_data(df)

    X_train = X_test = y_train = y_test = None
    models = {}

    for step in plan:

        # -------------------------
        # SPLIT
        # -------------------------
        if step == "split_data":
            X_train, X_test, y_train, y_test = split_data(df, target)

        # -------------------------
        # CLASSIFICATION MODELS
        # -------------------------
        elif step == "train_logistic":
            models["logistic"] = train_logistic(X_train, y_train)

        elif step == "train_random_forest":
            if problem_type in ["Binary Classification", "Multi-class Classification"]:
                models["random_forest"] = train_rf_classifier(X_train, y_train)
            else:
                models["random_forest"] = train_rf_regressor(X_train, y_train)

        # -------------------------
        # REGRESSION
        # -------------------------
        elif step == "train_linear":
            models["linear"] = train_linear_regression(X_train, y_train)

    return models, y_test


# ==========================================================
# EVALUATION ENGINE
# ==========================================================
def evaluate_models(models, X_test, y_test, problem_type):

    scores = {}

    for name, model in models.items():

        preds = model.predict(X_test)

        if "Regression" in problem_type:
            scores[name] = r2_score(y_test, preds)
        else:
            scores[name] = accuracy_score(y_test, preds)

    return scores


# ==========================================================
# SELECT BEST MODEL
# ==========================================================
def select_best_model(scores):

    best_model = max(scores, key=scores.get)

    return {
        "best_model": best_model,
        "best_score": scores[best_model]
    }
