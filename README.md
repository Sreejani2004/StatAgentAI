# 🤖 StatAgent AI

StatAgent AI is an AI-powered data analysis assistant built with Streamlit. It automates exploratory data analysis, dataset understanding, statistical recommendations, machine learning model selection, and report generation using an agent-based workflow.

## Features

* 📂 Upload CSV or Excel datasets
* 📊 Exploratory Data Analysis (EDA)
* 🤖 AI-powered dataset interpretation
* 🧠 Automatic dataset understanding
* 🎯 Automatic target variable detection
* 📈 Statistical test recommendations
* 🚀 Planning Agent for ML workflow
* 🤖 ML Model Selection Engine
* 🏆 Automatic best model selection
* 💡 Reflection Agent explaining model choice
* 📄 PDF report generation

## Agentic AI Workflow

```text
Upload Dataset
        ↓
EDA Dashboard
        ↓
AI Dataset Understanding
        ↓
Statistical Decision Agent
        ↓
Planning Agent
        ↓
ML Model Selection Engine
        ↓
Reflection Agent
        ↓
PDF Report Generation
```

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* ReportLab
* Ollama (Mistral)

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/StatAgentAI.git
cd StatAgentAI
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup

This project uses **Ollama Mistral** for AI-powered dataset insights.

```bash
ollama pull mistral
ollama serve
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
StatAgentAI/
│
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── data_loader.py
│   ├── eda.py
│   ├── decision_agent.py
│   ├── ml_engine.py
│   └── report_generator.py
```

## Future Improvements

* Hyperparameter tuning
* More machine learning algorithms
* Advanced visualizations
* Model explainability (e.g., SHAP)
* Time series forecasting support
* Deployment on Streamlit Community Cloud

Note:
This project uses Ollama locally for AI insights.
For deployment, LLM features may be disabled or replaced with fallback logic.

## Author

Sreejani Roy

M.Sc. Statistics and Computing, Banaras Hindu University
