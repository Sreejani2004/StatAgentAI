# 🤖 StatAgent AI

An **AI-powered data analysis assistant** built with **Streamlit** that automates exploratory data analysis, statistical recommendations, machine learning model selection, and report generation through an **Agentic AI workflow**.

---

## 🚀 Live Demo

**Try the application:**
👉 https://statagentai-x7t7dxij5vfntmicvpt9w2.streamlit.app/

> **Note**
>
> The deployed Streamlit application runs **without the local Ollama backend**. AI-powered dataset interpretation is therefore unavailable in the live demo. To experience all AI features, please run the project locally following the installation guide below.

---

## ✨ Features

* 📂 Upload CSV and Excel datasets
* 📊 Automated Exploratory Data Analysis (EDA)
* 🤖 AI-powered dataset interpretation using Ollama (Local)
* 🧠 Automatic dataset understanding
* 🎯 Intelligent target variable detection
* 📈 Statistical test recommendations
* 🚀 AI Planning Agent for workflow generation
* 🤖 Machine Learning model selection engine
* 🏆 Automatic best model selection
* 💡 Reflection Agent explaining model choice
* 📄 PDF report generation

---

## 🤖 Agentic AI Workflow

```text
Upload Dataset
        ↓
Data Quality Analysis
        ↓
Exploratory Data Analysis
        ↓
Dataset Understanding Agent
        ↓
Statistical Decision Agent
        ↓
Planning Agent
        ↓
Machine Learning Engine
        ↓
Model Evaluation
        ↓
Reflection Agent
        ↓
PDF Report Generation
```

---

## 🛠 Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* ReportLab
* Ollama (Mistral)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/roysreejani10-netizen/StatAgentAI.git
cd StatAgentAI
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🤖 Running with Ollama (Local)

Install the Mistral model:

```bash
ollama pull mistral
```

Start the Ollama server:

```bash
ollama serve
```

Launch the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
StatAgentAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── decision_agent.py
│   ├── ml_engine.py
│   └── report_generator.py
```

---

## 💡 Why StatAgent AI?

Unlike traditional AutoML tools, **StatAgent AI** follows a modular **Agentic AI architecture**, where specialized agents collaborate to automate the complete data science workflow.

### Agents

* 📊 **EDA Agent** – Performs exploratory data analysis and data quality assessment.
* 🧠 **Dataset Understanding Agent** – Interprets uploaded datasets.
* 📈 **Statistical Decision Agent** – Recommends appropriate statistical analyses.
* 🚀 **Planning Agent** – Determines the optimal machine learning workflow.
* 🤖 **Machine Learning Engine** – Trains and evaluates multiple ML models.
* 💡 **Reflection Agent** – Explains why the selected model performed best.

This modular architecture improves transparency, interpretability, and extensibility.

---

## 📊 Example Workflow

```text
Upload Titanic Dataset
        ↓
EDA Generated
        ↓
Target Variable Detected
        ↓
Classification Problem Identified
        ↓
Random Forest Trained
        ↓
Best Model Selected
        ↓
Reflection Generated
        ↓
PDF Report Exported
```

---

## 🚀 Future Improvements

* Hyperparameter tuning
* XGBoost and LightGBM integration
* SHAP-based model explainability
* Advanced visualizations
* Time-series forecasting
* Automated feature engineering
* End-to-end preprocessing pipelines
* Enhanced multi-agent collaboration

---

## 👩‍💻 Author

**Sreejani Roy**

M.Sc. Statistics & Computing
Banaras Hindu University

GitHub: https://github.com/Sreejani2004

