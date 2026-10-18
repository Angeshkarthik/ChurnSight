# 🌐 Customer Intelligence & Churn Prevention Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean Architecture](https://img.shields.io/badge/architecture-modular-brightgreen.svg)](#planned-architecture)

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

## 🎯 2. Placement & Interview Value

This project is built from scratch adhering to professional engineering and data science best practices. In software engineering and machine learning placement interviews, it showcases mastery of:

- **ML Fundamentals:** Handling categorical features, numerical scaling, avoiding data leakage, addressing class imbalance, and selecting appropriate evaluation metrics (Precision, Recall, ROC-AUC, PR-AUC).
- **Explainable AI (XAI):** Utilizing **SHAP (SHapley Additive exPlanations)** based on cooperative game theory to make complex ML models fully transparent and trustworthy.
- **Clean Architecture & OOP:** Writing modular, decoupled Python packages (`src/data`, `src/features`, `src/models`, `src/explainability`) with explicit type hints and unit tests, avoiding messy single-file scripts.
- **Full-Stack ML Integration:** Serving serialized models via **FastAPI** with Pydantic validation schemas, consumed by a modern **React + TypeScript** user interface.

---

## 🏗️ 3. Planned Architecture

The platform follows a layered, scalable architecture designed for separation of concerns:

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
│  - data/processed: Train/Test splits, transformed features  │
│  - models/: Serialized pipeline and model binaries          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ 4. Project Directory Structure

```text
customer_churn_intelligence/
│
├── .gitignore                   # Ignores venv, caches, raw data, model binaries
├── requirements.txt             # Pinned core dependencies for Phase 1
├── README.md                    # Project roadmap, architecture & interview guide
│
├── data/                        # Data storage (git-ignored, kept via .gitkeep)
│   ├── raw/                     # Original, untouched raw Telco Churn CSV
│   └── processed/               # Cleaned data, train/test splits
│
├── notebooks/                   # Jupyter notebooks for interactive EDA
│   └── .gitkeep
│
├── src/                         # Modular, production-ready Python package
│   ├── __init__.py              # Package marker & version definition
│   ├── data/                    # Ingestion, validation, cleaning routines
│   │   └── __init__.py
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
├── tests/                       # Unit and integration test suite (pytest)
│   └── __init__.py
│
├── backend/                     # (Phase 6) FastAPI backend service
└── frontend/                    # (Phase 7) React + TypeScript UI
```

---

## 🗺️ 5. Learning & Implementation Roadmap

| Phase | Milestone | Core Concepts & Deliverables |
|---|---|---|
| **Phase 1** | **Project Foundation & Environment Setup** *(Current)* | Virtual environment, directory structure, requirements, `.gitignore`, verification test. |
| **Phase 2** | **Dataset Ingestion & Exploratory Data Analysis (EDA)** | Understand distributions, missing values, outliers, correlations, target class imbalance. |
| **Phase 3** | **Data Cleaning & Feature Engineering Pipeline** | Handling whitespace values (e.g. `TotalCharges`), one-hot encoding, feature scaling, preventing data leakage. |
| **Phase 4** | **Model Training, Evaluation & Benchmarking** | Baseline Logistic Regression, Random Forest, Gradient Boosting; Precision/Recall/ROC-AUC evaluation. |
| **Phase 5** | **Model Explainability with SHAP** | Global feature importance (summary plots) and local individual customer explanations (waterfall plots). |
| **Phase 6** | **FastAPI Backend Integration** | RESTful endpoints for single and batch predictions, request validation schemas, health checks. |
| **Phase 7** | **Interactive Web Dashboard** | Modern React + TypeScript UI for customer success teams to simulate retention strategies. |
| **Phase 8** | **Interview Preparation & Placement Walkthrough** | Deep-dive explanations, common interviewer questions, architectural trade-offs, and deployment discussion. |

---

## 🛠️ 6. Technology Stack & Rationale

| Tool / Library | Role | Why We Selected It |
|---|---|---|
| **Python 3.10+** | Language | Industry standard for data science and ML engineering. |
| **Pandas & NumPy** | Data Manipulation | High-performance vectorized numerical operations and tabular data cleaning. |
| **Matplotlib & Seaborn** | Visualization | Statistical visualization for deep exploratory analysis. |
| **Scikit-learn** | Machine Learning | Robust, battle-tested ML algorithms, pipelines, cross-validation, and metrics. |
| **SHAP** | Explainable AI | Grounded in cooperative game theory (Shapley values); provides mathematically consistent explanations. |
| **FastAPI** | Backend API | High-performance asynchronous REST API framework with native Pydantic validation and automatic OpenAPI docs. |
| **React + TypeScript** | Frontend UI | Component-based modern frontend with static type safety, ideal for enterprise dashboards. |
| **Pytest** | Testing | Clean testing framework to write automated tests for data integrity and model pipelines. |

---

## 🚀 7. Environment Setup & Verification

### Step 1: Clone or Navigate to the Workspace
```powershell
cd d:\@PROJECTS\ML\customer_churn_intelligence
```

### Step 2: Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(On macOS / Linux: `source .venv/bin/activate`)*

### Step 3: Upgrade Pip & Install Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 4: Verify the Installation
Run the verification script to ensure all packages import cleanly:
```powershell
.\.venv\Scripts\python.exe -c "import pandas, numpy, sklearn, shap, matplotlib, seaborn; print('Environment verified successfully!')"
```
