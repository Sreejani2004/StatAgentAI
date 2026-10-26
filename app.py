


    
  
    if st.session_state.get("plan") is None:
        st.warning("Please run the Statistical Decision Agent first.")
        st.stop()

    # Retrieve information from session state
    df = st.session_state["df"]
    target = st.session_state["target"]
    problem_type = st.session_state["problem"]
    plan = st.session_state["plan"]

    try:
        # Execute ML pipeline
        models, X_test, y_test = execution_engine(
            df,
            target,
            problem_type,
            plan
        )

        # Evaluate models
        scores = evaluate_models(
            models,
            X_test,
            y_test,
            problem_type
        )

        # Select the best model
        best = select_best_model(scores)

        # Protect against empty results
        if not scores or best.get("best_model") is None or best.get("best_score") is None:
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

        # Display all model scores
        st.subheader("📊 Model Performance")

        for model, score in scores.items():
            st.write(f"**{model}** : {score:.4f}")

        # Reflection Agent
        reflection = reflection_agent(
            best,
            problem_type
        )

        st.markdown(reflection)

        # Save results for the PDF report
        st.session_state["scores"] = scores
        st.session_state["best_model"] = best
        st.session_state["reflection"] = reflection

        st.success("✅ Analysis completed successfully!")

    except Exception as error:
        st.error(
            "The ML pipeline could not complete. "
            "Please check the dataset, target column, and ML Engine."
        )
        st.exception(error)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================
elif page == "📄 Download Report":

    st.title("📄 Download Analysis Report")

    required_keys = ["df", "problem", "target", "plan", "scores", "best_model", "reflection"]

    missing_keys = [
        key for key in required_keys
        if key not in st.session_state or st.session_state[key] is None
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
            reflection=st.session_state["reflection"]
        )

        st.success("✅ Report generated successfully!")

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf,
            file_name="StatAgent_Report.pdf",
            mime="application/pdf"
        )

    except Exception as error:
        st.error("The PDF report could not be generated.")
        st.exception(error)

