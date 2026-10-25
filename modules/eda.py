import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
# =========================================================
# 1. DATA QUALITY REPORT 
# =========================================================
def data_quality_report(df):

    st.header("🤖 Data Quality Dashboard")

    rows, cols = df.shape

    missing_total = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    missing_pct = (missing_total / (rows * cols)) * 100 if rows * cols > 0 else 0
    duplicate_pct = (duplicate_rows / rows) * 100 if rows > 0 else 0

    numeric_df = df.select_dtypes(include="number")


    # remove ID-like columns
    id_like_cols = []
    for col in numeric_df.columns:
        if numeric_df[col].nunique() / len(numeric_df) > 0.90:
            id_like_cols.append(col)

    numeric_df = numeric_df.drop(columns=id_like_cols, errors="ignore")
    # -------------------------
    # METRICS
    # -------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing (%)", f"{missing_pct:.2f}")
    c4.metric("Duplicates", duplicate_rows)

    st.divider()

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
    st.subheader("💡 Recommendations")

    if missing_total > 0:
        if missing_pct < 5:
            st.info("Low missing values → imputation recommended")
        elif missing_pct < 20:
            st.warning("Moderate missing values → advanced imputation needed")
        else:
            st.error("High missing values → consider dropping columns")

    if duplicate_rows > 0:
        st.warning("Duplicates detected → consider removing them")

    if missing_total == 0 and duplicate_rows == 0:
        st.success("Dataset is clean")

    st.divider()

    # -------------------------
    # HISTOGRAM
    # -------------------------
    st.subheader("📊 Feature Distribution")

    if not numeric_df.empty:

        col = st.selectbox("Select column", numeric_df.columns)

        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20, edgecolor="black")
        st.pyplot(fig)

    else:
        st.info("No numeric columns")

    st.divider()

    # -------------------------
    # CORRELATION
    # -------------------------
    st.subheader("🔥 Correlation Heatmap")
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
        numeric_df.corr(),
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax
        )
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns.")

# =========================================================
# 2. 🧠 FREE AI DATASET EXPLAINER (OLLAMA)
# =========================================================
# ==========================================================
# 🧠 AI DATASET EXPLAINER (FASTER)
# ==========================================================
# ==========================================================
# 🧠 AI DATASET EXPLAINER
# ==========================================================
def ai_dataset_explainer_llm(df):

    st.subheader("🧠 AI Dataset Insights")

    # Build dataset summary
    missing_cols = (
        df.isnull()
        .sum()
        .loc[lambda x: x > 0]
        .to_dict()
    )

    summary = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Column Names": list(df.columns),
        "Data Types": df.dtypes.astype(str).to_dict(),
        "Missing Columns": missing_cols,
        "Sample Rows": df.head(2).to_dict()
    }

    prompt = f"""
You are an expert data scientist.

Below is a summary of a dataset.

{summary}

Explain the dataset in simple English.

Do NOT repeat the statistics.

Instead explain:

• What the dataset is probably about.
• Possible prediction or analysis tasks.
• Possible data quality issues.
• Suggested preprocessing.
• Interesting observations.

Maximum 150 words.
Use bullet points.
"""

    try:

        response = ollama.chat(
            model="mistral",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"""
❌ Unable to generate AI insights.

**Error:**

{e}

Make sure:

• Ollama is running (`ollama serve`)
• The Mistral model is installed (`ollama pull mistral`)
"""
