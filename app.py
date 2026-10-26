import streamlit as st
from modules.data_loader import load_dataset
from modules.eda import (
    data_quality_report,
    ai_dataset_explainer_llm,
)
from modules.decision_agent import (
    dataset_understanding,
    planning_agent,
)
from modules.ml_engine import (
    execution_engine,
    evaluate_models,
    select_best_model,
    reflection_agent,
)
from modules.report_generator import generate_pdf_report

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="StatAgent AI",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state safely
if "df" not in st.session_state:
    st.session_state["df"] = None
if "problem" not in st.session_state:
    st.session_state["problem"] = None
if "target" not in st.session_state:
    st.session_state["target"] = None
if "plan" not in st.session_state:
    st.session_state["plan"] = None

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("🤖 StatAgent AI")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Upload Dataset",
        "📊 EDA Dashboard",
        "🧠 Statistical Decision Agent",
        "🚀 ML Model Selection Engine",
        "📄 Download Report",
    ],
)

# ==========================================================
# HOME
# ==========================================================
if page == "🏠 Home":
    st.title("🤖 StatAgent AI")
    st.markdown("""
## Welcome!

Your AI-powered data analysis assistant.

### Features

- 📂 Upload datasets (CSV / Excel)
- 📊 Exploratory Data Analysis (EDA)
- 🧠 AI Dataset Insights
- 🧠 Automatic Dataset Understanding
- 📈 Statistical Test Recommendations
- 🤖 ML Model Recommendations
- 🚀 Automated Model Selection Engine
- 📄 Download Report
""")

# ==========================================================
# UPLOAD DATASET
# ==========================================================
elif page == "📂 Upload Dataset":
    st.title("📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        if df is not None:
            st.success("✅ Dataset uploaded successfully!")
            st.session_state["df"] = df

            # Clear results associated with the previous dataset
            for key in (
                "problem",
                "target",
                "plan",
                "scores",
                "best_model",
                "reflection",
                "ai_analysis",
            ):
                st.session_state[key] = None
        else:
            st.error("❌ Failed to load dataset.")
    else:
        st.info("Upload a dataset to begin.")

    if st.session_state["df"] is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state["df"].head())

# ==========================================================
# EDA DASHBOARD
# ==========================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    if st.session_state["df"] is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]
    data_quality_report(df)
    st.divider()

    if st.button("🧠 Generate AI Analysis"):
        with st.spinner("Generating AI insights..."):
            st.session_state["ai_analysis"] = ai_dataset_explainer_llm(df)

    if st.session_state.get("ai_analysis") is not None:
        st.divider()
        with st.expander("🤖 AI Interpretation", expanded=True):
            st.write(st.session_state["ai_analysis"])

# ==========================================================
# DECISION AGENT
# ==========================================================
elif page == "🧠 Statistical Decision Agent":
    st.title("🧠 Statistical Decision Agent")

    if st.session_state["df"] is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    problem_info = dataset_understanding(st.session_state["df"])

    if (
        not isinstance(problem_info, dict)
        or not isinstance(problem_info.get("problem"), dict)
        or "problem" not in problem_info["problem"]
        or "target" not in problem_info
    ):
        st.error(
            "The Decision Agent did not return the expected problem type "
            "and target. Please check modules/decision_agent.py."
        )
        st.stop()

    problem_type = problem_info["problem"]["problem"]
    target = problem_info["target"]

    if target is None or target not in st.session_state["df"].columns:
        st.warning(
            "The Decision Agent could not identify a valid target column. "
            "Please check your dataset and target detection."
        )
        st.stop()

    plan = planning_agent(problem_type)

    st.session_state["problem"] = problem_type
    st.session_state["target"] = target
    st.session_state["plan"] = plan

    st.success("Analysis complete!")
    st.write(f"**Detected problem type:** {problem_type}")
    st.write(f"**Target column:** {target}")
    st.write("**Recommended plan:**")
    st.write(plan)

# ==========================================================
# ML MODEL SELECTION ENGINE
# ==========================================================
elif page == "🚀 ML Model Selection Engine":
    st.title("🚀 ML Model Selection Engine")

    if st.session_state["df"] is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    if st.session_state.get("problem") is None:
        st.warning("Please run the Statistical Decision Agent first.")
        st.stop()

    if st.session_state.get("target") is None:
        st.warning("The Decision Agent did not identify a target column. "
                   "Please run it again or check your dataset.")
        st.stop()

    if st.session_state.get("plan") is None:
        st.warning("Please run the Statistical Decision Agent first.")
        st.stop()

    df = st.session_state["df"]
    target = st.session_state["target"]
    problem_type = st.session_state["problem"]
    plan = st.session_state["plan"]

    try:
        models, X_test, y_test = execution_engine(
            df,
            target,
            problem_type,
            plan,
        )

        scores = evaluate_models(
            models,
            X_test,
            y_test,
            problem_type,
        )

        best = select_best_model(scores)

        if (
            not scores
            or best.get("best_model") is None
            or best.get("best_score") is None
        ):
            st.error(
                "No model could be evaluated for this problem type. "
                "The current ML Engine supports classification and regression; "
                "time-series forecasting and clustering are not implemented yet."
            )
            st.stop()

        st.success(
            f"🏆 Best Model: {best['best_model']} "
            f"(Score: {best['best_score']:.4f})"
        )

        st.subheader("📊 Model Performance")
        for model, score in scores.items():
            st.write(f"**{model}** : {score:.4f}")

        reflection = reflection_agent(best, problem_type)
        st.markdown(reflection)

        st.session_state["scores"] = scores
        st.session_state["best_model"] = best
        st.session_state["reflection"] = reflection

        st.success("✅ Analysis completed successfully!")

    except Exception as error:
        st.error(
            "The ML pipeline could not complete. Please check the dataset, "
            "target column, and modules/ml_engine.py."
        )
        st.exception(error)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================
elif page == "📄 Download Report":
    st.title("📄 Download Analysis Report")

    required_keys = (
        "df",
        "problem",
        "target",
        "plan",
        "scores",
        "best_model",
        "reflection",
    )
    missing_keys = [
        key for key in required_keys
        if st.session_state.get(key) is None
    ]

    if missing_keys:
        st.warning(
            "Please upload a dataset, run the Statistical Decision Agent, "
            "and successfully run the ML Model Selection Engine first."
        )
        st.stop()

    try:
        pdf = generate_pdf_report(
            df=st.session_state["df"],
            problem_type=st.session_state["problem"],
            target=st.session_state["target"],
            plan=st.session_state["plan"],
            scores=st.session_state["scores"],
            best=st.session_state["best_model"],
            reflection=st.session_state["reflection"],
        )

        st.success("✅ Report generated successfully!")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf,
            file_name="StatAgent_Report.pdf",
            mime="application/pdf",
        )

    except Exception as error:
        st.error("The PDF report could not be generated.")
        st.exception(error)
