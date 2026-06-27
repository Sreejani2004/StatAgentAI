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

    df = df.dropna(axis=1, how="all")

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("missing")

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

    models = {}

    X_train = X_test = y_train = y_test = None

    # FIRST ensure split happens
    if "split_data" in plan:
        X_train, X_test, y_train, y_test = split_data(df, target)

    for step in plan:

        if step == "train_logistic":
            models["logistic"] = train_logistic(X_train, y_train)

        elif step == "train_random_forest":
            if problem_type in ["Binary Classification", "Multi-class Classification"]:
                models["rf"] = train_rf_classifier(X_train, y_train)
            else:
                models["rf"] = train_rf_regressor(X_train, y_train)

        elif step == "train_linear":
            models["linear"] = train_linear_regression(X_train, y_train)

    return models, X_test, y_test

# ==========================================================
# EVALUATION ENGINE
# ==========================================================
def evaluate_models(models, X_test, y_test, problem_type):

    model_names = {
        "rf": "Random Forest",
        "logistic": "Logistic Regression",
        "linear": "Linear Regression",
        "random_forest": "Random Forest",
        "xgb": "XGBoost",
        "svm": "Support Vector Machine",
        "arima": "ARIMA",
        "lstm": "LSTM"
    }

    scores = {}

    for key, model in models.items():

        preds = model.predict(X_test)

        if "Regression" in problem_type:
            score = r2_score(y_test, preds)
        else:
            score = accuracy_score(y_test, preds)

        scores[model_names.get(key, key)] = score

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


# ==========================================================
# REFLECTION AGENT
# ==========================================================
def reflection_agent(best, problem_type):

    model = best["best_model"]
    score = best["best_score"]

    # Metric name
    if "Regression" in problem_type:
        metric = "R² Score"
    else:
        metric = "Accuracy"

    # Model-specific explanation
    if model == "Random Forest":
        reason = (
            "Random Forest achieved the highest performance because it can capture "
            "non-linear relationships, is robust to noisy data, and reduces "
            "overfitting through ensemble learning."
        )

    elif model == "Logistic Regression":
        reason = (
            "Logistic Regression performed best because the relationship between "
            "the predictors and target appears relatively simple and linearly "
            "separable while remaining highly interpretable."
        )

    elif model == "Linear Regression":
        reason = (
            "Linear Regression was selected because the target variable appears "
            "to have a strong linear relationship with the predictors."
        )

    elif model == "XGBoost":
        reason = (
            "XGBoost achieved the highest score by combining multiple decision "
            "trees through gradient boosting to learn complex patterns."
        )

    elif model == "Support Vector Machine":
        reason = (
            "Support Vector Machine performed best because it identified an "
            "effective decision boundary between classes."
        )

    elif model == "ARIMA":
        reason = (
            "ARIMA was selected because it effectively models trends and "
            "temporal dependencies in time-series data."
        )

    elif model == "LSTM":
        reason = (
            "LSTM captured long-term sequential dependencies better than the "
            "other forecasting models."
        )

    else:
        reason = (
            "This model achieved the highest evaluation score among all "
            "candidate models."
        )

    return f"""
## 🤖 Reflection Agent

### 🏆 Best Model
**{model}**

### 📊 Performance
**{metric}: {score:.4f}**

### 💡 Why was this model selected?

{reason}

### 🚀 Recommended Next Steps

- Perform hyperparameter tuning (Grid Search or Random Search).
- Evaluate using cross-validation for more reliable performance estimates.
- Explore feature engineering to improve predictive power.
- Compare with additional advanced models if needed.
- Validate the model on unseen data before deployment.
"""
