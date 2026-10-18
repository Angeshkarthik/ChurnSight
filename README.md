# 🌐 Customer Intelligence & Churn Prevention Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/Dataset%20License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Clean Architecture](https://img.shields.io/badge/architecture-modular-brightgreen.svg)](#planned-architecture)
[![Test Suite: Pytest](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](#running-automated-tests)

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

## ⚙️ 3. Data Pipeline & Quality Engineering (Phase 2)

### The Real-World Data Anomaly: `TotalCharges` Whitespace
When loading the raw dataset, `TotalCharges` is parsed as an `object` (string) column rather than `float64`. 
- **Root Cause:** 11 customer records contain whitespace strings (`" "`) instead of numeric charges.
- **Investigation:** Cross-referencing reveals that **all 11 rows have `tenure == 0`**. These are brand new customer accounts opened in the current billing cycle who have not yet received their first invoice.
- **Domain-Accurate Decision:** We set `TotalCharges = 0.0` for `tenure == 0` accounts (`strategy="impute_zero"`). A customer with 0 months of service has accumulated $0.00 in historical charges. This preserves all 7,043 customer records without introducing statistical distortion.
- **Alternative:** The cleaner also supports `strategy="drop"` for use-cases requiring strict exclusion of newly enrolled users.

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
│   └── .gitkeep
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
│   │   └── __init__.py
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
│   └── verify_environment.py    # Environment health check script
│
├── tests/                       # Unit and integration test suite (pytest)
│   ├── __init__.py
│   ├── test_data_loader.py      # Loader tests (existence, non-empty, error modes)
│   ├── test_data_validator.py   # Validator tests (schema, target, whitespace)
│   ├── test_data_cleaner.py     # Cleaner tests (impute_zero, drop, trimming)
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
| **Phase 3** | **Exploratory Data Analysis (EDA) & Feature Engineering** | Distribution analysis, one-hot encoding, feature scaling, preventing data leakage. | ⏳ Next |
| **Phase 4** | **Model Training, Evaluation & Benchmarking** | Baseline Logistic Regression, Random Forest, Gradient Boosting; Precision/Recall/ROC-AUC evaluation. | ⏳ Planned |
| **Phase 5** | **Model Explainability with SHAP** | Global feature importance (summary plots) and local individual customer explanations (waterfall plots). | ⏳ Planned |
| **Phase 6** | **FastAPI Backend Integration** | RESTful endpoints for single and batch predictions, request validation schemas, health checks. | ⏳ Planned |
| **Phase 7** | **Interactive Web Dashboard** | Modern React + TypeScript UI for customer success teams to simulate retention strategies. | ⏳ Planned |
| **Phase 8** | **Interview Preparation & Placement Walkthrough** | Deep-dive explanations, common interviewer questions, architectural trade-offs, and deployment discussion. | ⏳ Planned |

---

## 🚀 7. Step-by-Step Execution Guide

### Step 1: Activate Virtual Environment
On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
*(On macOS / Linux: `source .venv/bin/activate`)*

### Step 2: Check Dataset Provenance & Metadata
```powershell
python scripts/download_data.py --info
```

### Step 3: Download Raw Dataset
```powershell
python scripts/download_data.py
```
*(Saved to `data/raw/Telco-Customer-Churn.csv` without modifying or mutating the raw source).*

### Step 4: Run the Complete Data Pipeline (Single Command)
```powershell
python scripts/run_data_pipeline.py
```
Optional flags:
- `--strategy impute_zero` *(default, sets TotalCharges=0.0 where tenure=0)*
- `--strategy drop` *(drops the 11 blank TotalCharges rows)*
- `--force-download` *(forces fresh download from upstream repository)*

### Step 5: Run Automated Tests
```powershell
pytest -v tests
```
All 15 unit and integration tests validate loading, schema integrity, custom exceptions, cleaning logic, and pipeline execution.
