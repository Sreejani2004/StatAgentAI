
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, r2_score


# ==========================================================
# 1. BASIC CLEANING
# ==========================================================
def clean_data(df):

    df = pd.DataFrame(df).copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Replace positive and negative infinity with missing values
    df = df.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Fill missing values
    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            median_val = df[col].median()

            if pd.isna(median_val):
                median_val = 0

            df[col] = df[col].fillna(median_val)

        else:

            mode_val = df[col].mode()

            fill_value = (
                mode_val.iloc[0]
                if not mode_val.empty
                else "missing"
            )

            df[col] = df[col].fillna(fill_value)

    return df


# ==========================================================
# 2. ENCODING
# ==========================================================
def encode_data(df):

    df = df.copy()

    for col in df.columns:

        if (
            df[col].dtype == "object"
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            ):
            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(
                df[col].astype(str)
            )

    return df


# ==========================================================
# 3. SPLIT DATA
# ==========================================================
def split_data(df, target):

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' was not found in the dataset."
        )

    X = df.drop(columns=[target])
    y = df[target]

    if X.shape[1] == 0:
        raise ValueError(
            "No feature columns are available for training."
        )

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain at least two distinct values."
        )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ==========================================================
# 4. MODEL TRAINING HELPERS
# ==========================================================
def train_logistic(X_train, y_train):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def train_rf_classifier(X_train, y_train):

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model


def train_rf_regressor(X_train, y_train):

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model


def train_linear_regression(X_train, y_train):

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# ==========================================================
# 5. EXECUTION ENGINE
# ==========================================================
def execution_engine(df, target, problem_type, plan):

    if target is None or target not in df.columns:
        raise ValueError(
            "A valid target column must be selected."
        )

    # Clean and encode the dataset
    df = clean_data(df)
    df = encode_data(df)

    models = {}

    X_train = X_test = y_train = y_test = None

    # Split data before training
    if "split_data" in plan:

        X_train, X_test, y_train, y_test = split_data(
            df, target
        )

        # Convert features to numeric values
        X_train = X_train.apply(
            pd.to_numeric, errors="coerce"
        )

        X_test = X_test.apply(
            pd.to_numeric, errors="coerce"
        )

        # Replace missing and infinite feature values
        X_train = X_train.replace(
            [float("inf"), float("-inf")],
            float("nan")
        ).fillna(0)

        X_test = X_test.replace(
            [float("inf"), float("-inf")],
            float("nan")
        ).fillna(0)

        # Ensure all feature values are finite
        if not (
            X_train.map(lambda x: pd.notna(x) and abs(x) != float("inf"))
            .all()
            .all()
        ):
            raise ValueError(
                "Training features contain invalid values."
            )

        if y_train.isna().any() or y_test.isna().any():
            raise ValueError(
                "The target contains missing values."
            )

    else:
        raise ValueError(
            "The execution plan must include 'split_data'."
        )

    # Train models according to the plan
    for step in plan:

        if step == "train_logistic":

            if problem_type in [
                "Binary Classification",
                "Multi-class Classification"
            ]:
                models["logistic"] = train_logistic(
                    X_train, y_train
                )

        elif step == "train_random_forest":

            if problem_type in [
                "Binary Classification",
                "Multi-class Classification"
            ]:
                models["rf"] = train_rf_classifier(
                    X_train, y_train
                )

            elif problem_type == "Regression":
                models["rf"] = train_rf_regressor(
                    X_train, y_train
                )

        elif step == "train_linear":

            if problem_type == "Regression":
                models["linear"] = train_linear_regression(
                    X_train, y_train
                )

    # Keep the original three-value return format
    return models, X_test, y_test


# ==========================================================
# 6. EVALUATION ENGINE
# ==========================================================
def evaluate_models(models, X_test, y_test, problem_type):

    model_names = {
        "rf": "Random Forest",
        "logistic": "Logistic Regression",
        "linear": "Linear Regression",
        "random_forest": "Random Forest"
    }

    scores = {}

    for key, model in models.items():

        predictions = model.predict(X_test)

        if problem_type == "Regression":

            score = r2_score(y_test, predictions)

        elif problem_type in [
            "Binary Classification",
            "Multi-class Classification"
        ]:

            score = accuracy_score(y_test, predictions)

        else:
            continue

        scores[model_names.get(key, key)] = score

    return scores


# ==========================================================
# 7. SELECT BEST MODEL
# ==========================================================
def select_best_model(scores):

    if not scores:
        return {
            "best_model": None,
            "best_score": None
        }

    best_model = max(scores, key=scores.get)

    return {
        "best_model": best_model,
        "best_score": scores[best_model]
    }


# ==========================================================
# 8. REFLECTION AGENT
# ==========================================================
def reflection_agent(best, problem_type):

    model = best.get("best_model")
    score = best.get("best_score")

    if model is None or score is None:
        return """
## 🤖 Reflection Agent

No model could be selected.

Please check the dataset, target column, and training plan.
"""

    if problem_type == "Regression":
        metric = "R² Score"
    else:
        metric = "Accuracy"

    reasons = {
        "Random Forest": (
            "Random Forest can capture nonlinear relationships "
            "and combines multiple decision trees."
        ),
        "Logistic Regression": (
            "Logistic Regression provides an interpretable "
            "baseline for classification."
        ),
        "Linear Regression": (
            "Linear Regression models a numeric outcome using "
            "a linear relationship with its predictors."
        )
    }

    reason = reasons.get(
        model,
        "This model achieved the highest evaluation score "
        "among the evaluated candidate models."
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

- Test the model using cross-validation.
- Explore hyperparameter tuning.
- Consider feature engineering.
- Validate performance on unseen data.
"""

