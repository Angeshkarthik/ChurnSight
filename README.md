# 🌐 Customer Intelligence & Churn Prevention Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/Dataset%20License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Clean Architecture](https://img.shields.io/badge/architecture-modular-brightgreen.svg)](#planned-architecture)
[![Test Suite: Pytest](https://img.shields.io/badge/tests-41%20passed-brightgreen.svg)](#running-automated-tests)
[![Champion Model: Gradient Boosting](https://img.shields.io/badge/champion-Gradient%20Boosting%20(CV%20PR--AUC%200.672)-blueviolet.svg)](#3-model-benchmarking--rigorous-evaluation-phase-4)

An end-to-end Machine Learning and Full-Stack software engineering application designed to predict customer churn, explain individual risk drivers using Explainable AI (SHAP), and empower customer success teams with proactive retention intelligence.

---

## 📌 1. Problem Statement & Business Context

### What is Customer Churn?
**Customer Churn** occurs when a customer cancels their subscription or ceases doing business with a company over a given time horizon. 

In subscription-based industries (like telecommunications, SaaS, and banking):
- **Customer Lifetime Value (LTV):** Customer retention directly preserves recurring revenue; losing an existing subscriber incurs immediate top-line decline and replacement acquisition overhead *(Gallo, Amy. "The Value of Keeping the Right Customers", Harvard Business Review, 2014)*.
- **The Retention Opportunity:** Increasing customer retention by even 5% has been shown to produce substantial enterprise profit growth *(Reichheld, Frederick. Bain & Company / HBR)*.
- **The Predictive Advantage:** Instead of reacting *after* a customer leaves, this platform builds an early-warning classification system that flags at-risk accounts, quantifies risk drivers, and recommends retention actions before contracts expire.

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

## 🏆 3. Model Benchmarking & Rigorous Evaluation (Phase 4)

### Methodological Rigor & Data Governance
To prevent data snooping and selection leakage, our benchmarking enforces strict separation across three stages:
1. **Partitioning:** Data is split into a **Development Set** (80%, $N = 5,634$) and a **Fresh Independent Holdout** (20%, $N = 1,409$, `holdout_seed = 100`).
2. **Champion Selection:** The champion model is selected **strictly using 5-Fold Stratified Cross-Validation on Development Data (`cv_seed = 42`)**, using mean CV PR-AUC as the primary metric with a deterministic tie-breaker. *No holdout data is inspected during model selection.*
3. **Threshold Optimization:** The operational probability decision threshold is tuned **strictly on Out-Of-Fold (OOF) development predictions** by maximizing OOF F1-score, before freezing the model and threshold.
4. **Independent Final Evaluation:** The frozen champion pipeline and frozen threshold are evaluated **once on the untouched Fresh Holdout**.

---

### 5-Fold Stratified Cross-Validation Summary (Development Set Only: $N = 5,634$)

| Model Architecture | CV PR-AUC Mean | CV ROC-AUC Mean | CV F1 Mean | CV Recall Mean | CV Precision Mean | CV Accuracy Mean | Selection Status |
|---|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.6721 ± 0.0180** | **0.8484 ± 0.0060** | 0.5829 | 0.5211 | 0.6630 | 80.23% | 🏆 **Champion** |
| **Hist Gradient Boosting** | 0.6661 ± 0.0168 | 0.8443 ± 0.0057 | 0.5813 | 0.5224 | 0.6565 | 80.05% | Contender |
| **Random Forest** | 0.6632 ± 0.0181 | 0.8482 ± 0.0085 | 0.5745 | 0.4997 | **0.6768** | **80.37%** | Contender |
| **Logistic Regression** | 0.6611 ± 0.0177 | 0.8473 ± 0.0098 | 0.6027 | 0.5538 | 0.6613 | 80.64% | Linear Baseline |
| **Logistic Regression (Balanced)** | 0.6604 ± 0.0171 | 0.8473 ± 0.0098 | **0.6281** | **0.8000** | 0.5171 | 74.87% | High Recall Baseline |
| **Dummy Baseline** | 0.2654 ± 0.0001 | 0.5000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 73.46% | Sanity Floor |

---

### Operational Threshold Selection via Out-Of-Fold (OOF) Training Predictions

Evaluating decision thresholds strictly on Out-Of-Fold (OOF) training predictions on the Development set identified **$P \ge 0.35$** as the optimal operational threshold maximizing F1:

- **Optimal OOF Threshold:** **0.35**
- **OOF Metrics at 0.35:** F1 = **0.6319** | Recall = **0.7003** | Precision = **0.5756**
- *Business Rationale:* Shifting the cutoff from 0.50 to 0.35 captures **70.0% of all churning accounts** while maintaining a solid **57.6% precision**.

---

### Independent Final Evaluation: Fresh Holdout ($N = 1,409$, `holdout_seed = 100`)

Evaluated strictly after freezing the champion architecture (Gradient Boosting) and threshold (0.35):

| Evaluated Model Pipeline | ROC-AUC | PR-AUC | Recall (Churn) | Precision (Churn) | F1-Score | Accuracy |
|---|---|---|---|---|---|---|
| **Gradient Boosting (Th=0.35 Operational)** | **0.8413** | **0.6447** | **0.7139** | 0.5597 | **0.6275** | 77.50% |
| **Gradient Boosting (Th=0.50 Standard)** | **0.8413** | **0.6447** | 0.5428 | **0.6444** | 0.5893 | **79.91%** |
| **Logistic Regression** | 0.8359 | 0.6300 | 0.5508 | 0.6417 | 0.5928 | **79.91%** |
| **Logistic Regression (Balanced)** | 0.8360 | 0.6285 | **0.8316** | 0.5107 | 0.6328 | 74.38% |
| **Random Forest** | 0.8373 | 0.6492 | 0.4973 | 0.6526 | 0.5645 | 79.63% |
| **Hist Gradient Boosting** | 0.8323 | 0.6322 | 0.5535 | 0.6409 | 0.5940 | **79.91%** |
| **Dummy Baseline** | 0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 73.46% |

---

### Confusion Matrix Analysis on Fresh Holdout

At the operational threshold **$P \ge 0.35$**, the frozen **Gradient Boosting Champion** achieved:

```text
Confusion Matrix (labels=[0, 1]):
  [TN:  825 (True Retained)   | FP:  210 (False Alarms)]
  [FN:  107 (Missed Churners) | TP:  267 (Caught Churners)]
```

- **True Positives (TP = 267):** Successfully identified **71.39% of all churners** (267 out of 374).
- **False Negatives (FN = 107):** Missed churners were cut down from 171 (under standard 0.50 threshold) to 107.
- **False Positives (FP = 210):** Proactive outreach sent to customers who would have stayed anyway. Because retention check-ins or targeted discounts carry minor cost compared to lost recurring contract revenue, this operational balance yields optimal business ROI.

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
│  │ (OneHot/Scaling)   │  │ (Gradient Boosting Champion) │   │
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
│   │   ├── registry.py          # Pipeline factory (Dummy, LogReg, RF, GBDT, HistGBDT)
│   │   ├── evaluation.py        # Metrics, ROC-AUC, PR-AUC, threshold curves, input validation
│   │   └── trainer.py           # Training-only CV selection, OOF threshold tuning, holdout eval
│   ├── explainability/          # (Phase 5) SHAP explainers & interpretation utilities
│   │   └── __init__.py
│   └── utils/                   # Config management, logging, plotting helpers
│       └── __init__.py
│
├── models/                      # Serialized model artifacts (.joblib / .json)
│   ├── churn_model.joblib       # Serialized Gradient Boosting end-to-end Pipeline
│   └── model_metadata.json      # Metadata, OOF threshold curves, governance audit logs
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
│   ├── test_models.py           # Model factory, selection, OOF threshold, evaluation tests
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
| **Phase 4** | **Model Training, Evaluation & Benchmarking** | Training-only CV champion selection, OOF threshold optimization, independent holdout evaluation. | ✅ Completed |
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
Supported flags:
- `--holdout-seed 100` *(Independent holdout partition seed, default: 100)*
- `--cv-seed 42` *(Stratified cross-validation seed within development set, default: 42)*
- `--test-size 0.2` *(Holdout fraction, default: 0.2)*

### Run Full Automated Test Suite (41 Tests)
```powershell
pytest -v tests
```
All 41 unit and integration tests validate loading, schema integrity, custom exceptions, cleaning logic, feature leakage prevention, CV selection, OOF threshold tuning, explicit confusion matrix order, and end-to-end pipeline execution.
