# 🌐 Customer Intelligence & Churn Prevention Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/Dataset%20License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Clean Architecture](https://img.shields.io/badge/architecture-modular-brightgreen.svg)](#planned-architecture)
[![Test Suite: Pytest](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)](#running-automated-tests)

An end-to-end Machine Learning and Full-Stack software engineering application designed to predict customer churn, explain individual risk drivers using Explainable AI (SHAP), and empower customer success teams with proactive retention intelligence.

---

## 📌 1. Problem Statement & Business Context

### What is Customer Churn?
**Customer Churn** occurs when a customer cancels their subscription or ceases doing business with a company over a given time horizon. 

In subscription-based industries (like telecommunications, SaaS, and banking):
- **High Acquisition Cost:** Acquiring a new customer typically costs **5x to 25x more** than retaining an existing customer.
- **Compounding Revenue Loss:** Losing high-value or long-tenure customers significantly degrades customer lifetime value (LTV).
- **The Retention Opportunity:** Even a 5% increase in customer retention can boost profitability by 25% to 95% (Bain & Company / Harvard Business Review).

### The Objective
Instead of reacting *after* a customer leaves, this platform builds a predictive early-warning system that:
1. Predicts **who** is at high risk of churning (churn probability score).
2. Explains **why** they are likely to churn (actionable local feature importance).
3. Delivers actionable intelligence through a **FastAPI backend** and an **interactive React + TypeScript dashboard** for customer success teams.

---

## 📊 2. Dataset Provenance & Licensing

To maintain complete reproducibility and data governance, we use the industry-standard Telco Customer Churn benchmark dataset:

- **Original Source Organization:** IBM Developer / IBM Watson Analytics
- **Upstream Repository:** [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Verified Raw URL:** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **License:** **Apache License 2.0** (Open for research, educational, commercial, and development use).
- **Dimensions:** 7,043 customer records across 21 attributes (demographics, services, contract terms, billing).
- **Target Distribution:**
  - `No` (Retained): 5,174 samples (73.46%)
  - `Yes` (Churned): 1,869 samples (26.54%)
  - *Data Science Note:* Class imbalance (~3:1 ratio) requires optimizing for **Precision, Recall, PR-AUC, and ROC-AUC** rather than raw accuracy.

---

## 🔍 3. Exploratory Data Analysis & Empirical Findings (Phase 3)

The complete interactive analysis is documented in [`notebooks/01_exploratory_data_analysis.ipynb`](file:///d:/@PROJECTS/ML/customer_churn_intelligence/notebooks/01_exploratory_data_analysis.ipynb).

### Key Empirical Findings:

| Driver | Group / Value | Churn Rate | Business & Behavioral Insight |
|---|---|---|---|
| **Contract Type** | Month-to-month | **42.71%** | Highest churn risk. Lack of commitment allows customers to leave at any time. |
| | One year | **11.27%** | Annual commitment cuts churn by ~74%. |
| | Two year | **2.83%** | Over 97% retained. Long-term contracts virtually eliminate churn. |
| **Tenure** | 0–12 Months | **47.44%** | Nearly half of all churn happens in year 1. Critical onboarding window. |
| | 49–72 Months | **9.51%** | Long-tenure customers demonstrate strong brand loyalty. |
| **Internet Service** | Fiber optic | **41.89%** | Premium speed but high price point ($70–$118/mo) drives price sensitivity. |
| | DSL | **18.96%** | Lower monthly cost, moderate churn. |
| | No Internet | **7.41%** | Landline-only customers are overwhelmingly retained. |
| **Payment Method** | Electronic check | **45.29%** | Manual monthly billing forces conscious reassessment of bills. |
| | Credit Card (auto) | **15.24%** | Automatic billing creates retention inertia. |
| | Bank Transfer (auto)| **16.71%** | Automated payment reduces friction. |
| **Support Services**| No TechSupport | **41.64%** | Unresolved technical issues strongly correlate with cancellation. |
| | With TechSupport | **15.17%** | Bundled tech support cuts churn by more than half. |

---

## ⚙️ 4. Feature Engineering & Leakage Prevention (Phase 3)

Located in [`src/features/`](file:///d:/@PROJECTS/ML/customer_churn_intelligence/src/features/):

1. **Identifier Exclusion:** `customerID` is stripped from predictor matrix $X$. An arbitrary alphanumeric string (`7590-VHVEG`) would otherwise lead to ID memorization and catastrophic overfitting.
2. **Target Isolation:** `Churn` is mapped to binary integers (`Yes` $\to$ 1, `No` $\to$ 0) and isolated into $y$.
3. **Data Leakage Elimination:**
   - Transformers (`StandardScaler`, `OneHotEncoder`) are fit **strictly on `X_train`** inside a Scikit-learn `ColumnTransformer`.
   - `X_test` is only transformed (`.transform()`, never `.fit()`), guaranteeing zero test-set distribution leakage.
4. **Production Resiliency:** Categorical features use `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to prevent API crashes when novel values are encountered during live inference.

---

## 🏗️ 5. Planned Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 React + TypeScript Dashboard                │
│  - Churn Risk Meter & KPI Cards                             │
│  - Interactive Customer Profile Explorer                    │
│  - SHAP Waterfall & Force Visualizations                   │
│  - "What-If" Retention Scenario Simulator                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / REST (JSON)
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Service                  │
│  - Pydantic Input Validation & Sanitization                 │
│  - /predict: Real-time churn probability scoring            │
│  - /explain: Dynamic SHAP explanation calculation           │
│  - /batch-predict: Bulk CSV processing                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ In-Memory Pipeline Execution
┌──────────────────────────────▼──────────────────────────────┐
│                   Machine Learning Core (src/)              │
│  ┌────────────────────┐  ┌──────────────────────────────┐   │
│  │ Feature Pipeline   │  │ Serialized Model (.joblib)   │   │
│  │ (OneHot/Scaling)   │  │ (Scikit-Learn / Ensemble)    │   │
│  └────────────────────┘  └──────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ SHAP Explainer (TreeExplainer / KernelExplainer)     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Training & Artifact Generation
┌──────────────────────────────▼──────────────────────────────┐
│                 Data Pipeline & Storage                     │
│  - data/raw: Untouched raw Telco Churn dataset             │
│  - data/processed: Cleaned dataset (telco_churn_cleaned.csv)│
│  - models/: Serialized pipeline and model binaries          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ 6. Project Directory Structure

```text
customer_churn_intelligence/
│
├── .gitignore                   # Excludes venv, caches, raw data, model binaries
├── requirements.txt             # Pinned core dependencies
├── README.md                    # Project roadmap, architecture & interview guide
│
├── data/
│   ├── raw/                     # Original, untouched raw Telco Churn CSV
│   └── processed/               # Cleaned data (telco_churn_cleaned.csv)
│
├── notebooks/                   # Jupyter notebooks for interactive EDA
│   └── 01_exploratory_data_analysis.ipynb   # Comprehensive executable EDA notebook
│
├── src/                         # Modular, production-ready Python package
│   ├── __init__.py              # Package marker & version definition
│   ├── data/                    # Ingestion, validation, cleaning routines
│   │   ├── __init__.py          # Package exports
│   │   ├── download.py          # Public dataset downloader with provenance
│   │   ├── loader.py            # Defensive CSV data loading
│   │   ├── validator.py         # Schema validation & integrity diagnostics
│   │   ├── cleaner.py           # Documented cleaning rules (TotalCharges)
│   │   └── pipeline.py          # Pipeline orchestrator & CLI
│   ├── features/                # Preprocessing pipelines, encoders, scalers
│   │   ├── __init__.py          # Feature subpackage exports
│   │   ├── config.py            # Column schema taxonomy and groupings
│   │   └── pipeline.py          # ColumnTransformer, leak-free preprocessing
│   ├── models/                  # Training scripts, hyperparameter tuning, metrics
│   │   └── __init__.py
│   ├── explainability/          # SHAP explainers & interpretation utilities
│   │   └── __init__.py
│   └── utils/                   # Config management, logging, plotting helpers
│       └── __init__.py
│
├── models/                      # Trained model artifacts (.joblib / .pkl)
│   └── .gitkeep
│
├── scripts/
│   ├── download_data.py         # Standalone CLI for dataset download & provenance
│   ├── run_data_pipeline.py     # Single-command entry point for the data pipeline
│   ├── generate_eda_notebook.py # Programmatically compiles & executes the EDA notebook
│   └── verify_environment.py    # Environment health check script
│
├── tests/                       # Unit and integration test suite (pytest)
│   ├── __init__.py
│   ├── test_data_loader.py      # Loader tests (existence, non-empty, error modes)
│   ├── test_data_validator.py   # Validator tests (schema, target, whitespace)
│   ├── test_data_cleaner.py     # Cleaner tests (impute_zero, drop, regression tests)
│   ├── test_features.py         # Feature engineering & leakage tests
│   └── test_pipeline.py         # End-to-end integration tests
│
├── backend/                     # (Phase 6) FastAPI backend service
└── frontend/                    # (Phase 7) React + TypeScript UI
```

---

## 🗺️ 7. Learning & Implementation Roadmap

| Phase | Milestone | Core Concepts & Deliverables | Status |
|---|---|---|---|
| **Phase 1** | **Project Foundation & Environment Setup** | Virtual environment, directory structure, requirements, `.gitignore`, verification test. | ✅ Completed |
| **Phase 2** | **Implement the Real Data Pipeline** | Verified dataset ingestion, schema validation, documented cleaning rules, automated tests, CLI entry points. | ✅ Completed |
| **Phase 3** | **Exploratory Data Analysis (EDA) & Feature Engineering** | Distribution analysis, one-hot encoding, feature scaling, preventing data leakage. | ✅ Completed |
| **Phase 4** | **Model Training, Evaluation & Benchmarking** | Baseline Logistic Regression, Random Forest, Gradient Boosting; Precision/Recall/ROC-AUC evaluation. | ⏳ Next |
| **Phase 5** | **Model Explainability with SHAP** | Global feature importance (summary plots) and local individual customer explanations (waterfall plots). | ⏳ Planned |
| **Phase 6** | **FastAPI Backend Integration** | RESTful endpoints for single and batch predictions, request validation schemas, health checks. | ⏳ Planned |
| **Phase 7** | **Interactive Web Dashboard** | Modern React + TypeScript UI for customer success teams to simulate retention strategies. | ⏳ Planned |
| **Phase 8** | **Interview Preparation & Placement Walkthrough** | Deep-dive explanations, common interviewer questions, architectural trade-offs, and deployment discussion. | ⏳ Planned |

---

## 🚀 8. Execution Commands

### Run Data Pipeline
```powershell
python scripts/run_data_pipeline.py
```

### Run EDA Notebook Generation & Verification
```powershell
python scripts/generate_eda_notebook.py
```
*(Or open [`notebooks/01_exploratory_data_analysis.ipynb`](file:///d:/@PROJECTS/ML/customer_churn_intelligence/notebooks/01_exploratory_data_analysis.ipynb) in Jupyter / VS Code / Antigravity IDE).*

### Run Full Automated Test Suite (29 Tests)
```powershell
pytest -v tests
```
