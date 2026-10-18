# 🌐 Customer Intelligence & Churn Prevention Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/Dataset%20License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Clean Architecture](https://img.shields.io/badge/architecture-modular-brightgreen.svg)](#planned-architecture)
[![Test Suite: Pytest](https://img.shields.io/badge/tests-36%20passed-brightgreen.svg)](#running-automated-tests)
[![Champion Model: Random Forest](https://img.shields.io/badge/champion-Random%20Forest%20(ROC--AUC%200.844)-blueviolet.svg)](#model-benchmarking--evaluation-phase-4)

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

## 🏆 3. Model Benchmarking & Evaluation (Phase 4)

We implemented an evaluation harness using **5-Fold Stratified Cross-Validation on `X_train`** (5,634 samples) and evaluated candidate models on a strictly held-out **Test Set** (1,409 samples). Preprocessing was encapsulated inside Scikit-learn `Pipeline` objects to guarantee zero test leakage.

### 5-Fold Stratified Cross-Validation Results (Training Set Only)

| Model Architecture | CV ROC-AUC | CV PR-AUC | CV Recall | CV Precision | CV F1 | CV Accuracy |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8482 ± 0.0126** | **0.6676 ± 0.0222** | 0.5291 ± 0.0305 | 0.6622 | 0.5880 | 80.33% |
| **Random Forest** | 0.8471 ± 0.0084 | 0.6630 ± 0.0168 | 0.5070 ± 0.0198 | **0.6730** | 0.5781 | **80.35%** |
| **Logistic Regression** | 0.8461 ± 0.0126 | 0.6615 ± 0.0194 | 0.5438 ± 0.0409 | 0.6521 | 0.5924 | 80.19% |
| **Hist Gradient Boosting** | 0.8451 ± 0.0103 | 0.6612 ± 0.0228 | 0.5258 ± 0.0250 | 0.6590 | 0.5846 | 80.17% |
| **Logistic Regression (Balanced)** | 0.8460 ± 0.0124 | 0.6601 ± 0.0196 | **0.8013 ± 0.0379** | 0.5169 | **0.6283** | 74.85% |
| **Dummy Baseline** | 0.5000 ± 0.0000 | 0.2654 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 73.46% |

### Held-Out Test Set Performance (N = 1,409)

| Model | ROC-AUC | PR-AUC | Recall (Churn) | Precision (Churn) | F1-Score | Accuracy |
|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **0.8437** | **0.6601** | 0.5053 | 0.6608 | 0.5727 | 79.99% |
| **Gradient Boosting** | 0.8433 | 0.6600 | 0.5160 | **0.6655** | 0.5813 | 80.27% |
| **Logistic Regression** | 0.8420 | 0.6337 | 0.5588 | 0.6572 | 0.6040 | **80.55%** |
| **Logistic Regression (Balanced)** | 0.8415 | 0.6325 | **0.7834** | 0.5043 | **0.6136** | 73.81% |
| **Hist Gradient Boosting** | 0.8398 | 0.6512 | 0.5134 | 0.6508 | 0.5740 | 79.77% |
| **Dummy Baseline** | 0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 73.46% |

### Business Confusion Matrix & Threshold Trade-Off Analysis

On the held-out test set at the default threshold $P \ge 0.50$, **Random Forest** produced:
- **True Negatives (TN):** 938 correctly identified retained customers.
- **False Positives (FP):** 97 loyal customers flagged as churn risks (wasted retention effort).
- **False Negatives (FN):** 185 churners missed (lost customer revenue).
- **True Positives (TP):** 189 churners caught and flagged for intervention.

#### Threshold Tuning: The Business Decision
Because acquiring a new customer is 5x–25x more expensive than retaining an existing one, missing a churner (FN) is far more damaging than reaching out to a satisfied customer (FP). 

By lowering the decision threshold, customer success teams can tune the trade-off:

| Threshold | Churners Caught (Recall) | Precision | F1-Score | Churners Missed (FN) | False Alarms (FP) | Strategic Context |
|---|---|---|---|---|---|---|
| **0.20** | **85.83%** (321 / 374) | 46.32% | 0.6017 | **53** | 372 | Aggressive early-warning: catches ~86% of all churn. |
| **0.30** | **78.07%** (292 / 374) | **53.28%** | **0.6334** | **82** | 256 | **Recommended Sweet Spot:** Highest F1 score, catches ~78% of churn with >53% precision. |
| **0.40** | 64.97% (243 / 374) | 58.27% | 0.6144 | 131 | 174 | Moderate retention intervention. |
| **0.50 (Default)**| 50.53% (189 / 374) | 66.08% | 0.5727 | 185 | 97 | Conservative: high precision, but misses half of churners. |

---

## 🏗️ 4. Planned Architecture

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
│  │ (OneHot/Scaling)   │  │ (Random Forest Champion)     │   │
│  └────────────────────┘  └──────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ SHAP Explainer (TreeExplainer)                       │   │
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

## 🗂️ 5. Project Directory Structure

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
│   │   ├── __init__.py          # Model subpackage exports
│   │   ├── registry.py          # Pipeline factory (Dummy, LogReg, RF, GBDT)
│   │   ├── evaluation.py        # Metrics, ROC-AUC, PR-AUC, threshold curves
│   │   └── trainer.py           # Cross-validation, model comparison, artifact exporter
│   ├── explainability/          # (Phase 5) SHAP explainers & interpretation utilities
│   │   └── __init__.py
│   └── utils/                   # Config management, logging, plotting helpers
│       └── __init__.py
│
├── models/                      # Serialized model artifacts (.joblib / .json)
│   ├── churn_model.joblib       # Champion Random Forest end-to-end Pipeline
│   └── model_metadata.json      # Metadata, test metrics, and threshold curves
│
├── scripts/
│   ├── download_data.py         # Standalone CLI for dataset download & provenance
│   ├── run_data_pipeline.py     # Single-command entry point for the data pipeline
│   ├── generate_eda_notebook.py # Programmatically compiles & executes the EDA notebook
│   ├── train_and_benchmark.py   # Single-command CLI for model benchmarking & export
│   └── verify_environment.py    # Environment health check script
│
├── tests/                       # Unit and integration test suite (pytest)
│   ├── __init__.py
│   ├── test_data_loader.py      # Loader tests (existence, non-empty, error modes)
│   ├── test_data_validator.py   # Validator tests (schema, target, whitespace)
│   ├── test_data_cleaner.py     # Cleaner tests (impute_zero, drop, regression tests)
│   ├── test_features.py         # Feature engineering & leakage tests
│   ├── test_models.py           # Model factory, evaluation, and pipeline tests
│   └── test_pipeline.py         # End-to-end integration tests
│
├── backend/                     # (Phase 6) FastAPI backend service
└── frontend/                    # (Phase 7) React + TypeScript UI
```

---

## 🗺️ 6. Learning & Implementation Roadmap

| Phase | Milestone | Core Concepts & Deliverables | Status |
|---|---|---|---|
| **Phase 1** | **Project Foundation & Environment Setup** | Virtual environment, directory structure, requirements, `.gitignore`, verification test. | ✅ Completed |
| **Phase 2** | **Implement the Real Data Pipeline** | Verified dataset ingestion, schema validation, documented cleaning rules, automated tests, CLI entry points. | ✅ Completed |
| **Phase 3** | **Exploratory Data Analysis (EDA) & Feature Engineering** | Distribution analysis, one-hot encoding, feature scaling, preventing data leakage. | ✅ Completed |
| **Phase 4** | **Model Training, Evaluation & Benchmarking** | Baseline Logistic Regression, Random Forest, Gradient Boosting; Precision/Recall/ROC-AUC evaluation, threshold tuning. | ✅ Completed |
| **Phase 5** | **Model Explainability with SHAP** | Global feature importance (summary plots) and local individual customer explanations (waterfall plots). | ⏳ Next |
| **Phase 6** | **FastAPI Backend Integration** | RESTful endpoints for single and batch predictions, request validation schemas, health checks. | ⏳ Planned |
| **Phase 7** | **Interactive Web Dashboard** | Modern React + TypeScript UI for customer success teams to simulate retention strategies. | ⏳ Planned |
| **Phase 8** | **Interview Preparation & Placement Walkthrough** | Deep-dive explanations, common interviewer questions, architectural trade-offs, and deployment discussion. | ⏳ Planned |

---

## 🚀 7. Execution Commands

### Run Model Training, Benchmarking & Artifact Export
```powershell
python scripts/train_and_benchmark.py
```

### Run Full Automated Test Suite (36 Tests)
```powershell
pytest -v tests
```
