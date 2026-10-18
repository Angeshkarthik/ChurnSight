"""Script to generate and execute the comprehensive Phase 3 EDA Jupyter Notebook."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_exploratory_data_analysis.ipynb"

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.8"
        }
    }

    cells = []

    # 1. Header & Overview
    cells.append(nbf.v4.new_markdown_cell("""# 🔍 Phase 3: Exploratory Data Analysis & Churn Intelligence

**Project:** Customer Intelligence & Churn Prevention Platform  
**Target:** Binary Classification (`Churn`: 'Yes' vs 'No')  
**Input Data:** Cleaned Telco Customer Churn dataset (`data/processed/telco_churn_cleaned.csv`)

---

## 🎯 Learning Objectives for this Notebook
1. **Data Hygiene Audit:** Verify shape, data types, missing value absence, and uniqueness.
2. **Class Imbalance Analysis:** Quantify the ~73.5% / ~26.5% target distribution and understand why naive accuracy is misleading.
3. **Continuous Feature Dynamics:** Explore how customer `tenure`, `MonthlyCharges`, and `TotalCharges` correlate with churn.
4. **Behavioral & Contractual Drivers:** Analyze the impact of contract type, internet service tier, and payment method.
5. **Business Recommendations:** Translate statistical findings into concrete customer success retention strategies.
"""))

    # 2. Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Add project root to sys.path so we can import src modules seamlessly
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

print("Environment and visualization libraries successfully initialized!")
"""))

    # 3. Loading Cleaned Data
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Data Ingestion & Quality Audit
We load the preprocessed dataset created by our Phase 2 data pipeline.
Notice that `TotalCharges` is already properly coerced to `float64` and whitespace anomalies have been resolved.
"""))

    cells.append(nbf.v4.new_code_cell("""data_path = PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Missing Values: {df.isna().sum().sum()} total nulls")
print(f"Duplicate Customer IDs: {df['customerID'].duplicated().sum()}")

# Inspect first 3 rows
df.head(3)
"""))

    cells.append(nbf.v4.new_code_cell("""# Column data types overview
dtype_counts = df.dtypes.value_counts()
print("Data Types Breakdown:")
print(dtype_counts)
"""))

    # 4. Target Variable & Class Imbalance
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Target Variable Analysis: The Class Imbalance Challenge

In binary classification, the ratio between the majority class (retained) and minority class (churned) directly dictates our model selection and metric evaluation strategy.
"""))

    cells.append(nbf.v4.new_code_cell("""churn_counts = df['Churn'].value_counts()
churn_props = df['Churn'].value_counts(normalize=True) * 100

print("Target Class Distribution:")
for label, count in churn_counts.items():
    prop = churn_props[label]
    print(f"  - Churn = '{label}': {count:,} customers ({prop:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Bar Chart
sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
axes[0].set_title("Customer Churn Count", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Churn Status")
axes[0].set_ylabel("Number of Customers")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', fontsize=12, color='white', fontweight='bold')

# Subplot 2: Donut Chart
axes[1].pie(churn_props, labels=churn_props.index, autopct='%1.1f%%', startangle=90,
            colors=["#2ecc71", "#e74c3c"], explode=(0, 0.08), wedgeprops=dict(width=0.4, edgecolor='w'))
axes[1].set_title("Customer Churn Proportion", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Machine Learning Takeaway: The Accuracy Paradox
- **73.46%** of customers are retained (`No`), while **26.54%** churned (`Yes`).
- A naive baseline classifier that predicts `"No"` for every customer would achieve **73.46% accuracy** while detecting **0% of churners**!
- Therefore, in Phase 4 modeling, accuracy is a misleading metric. We must evaluate models using:
  1. **Recall:** What fraction of actual churners did the model flag? (Critical for business retention).
  2. **Precision:** What fraction of flagged churners were truly at risk? (Prevents spamming happy customers).
  3. **ROC-AUC & PR-AUC:** Measure discrimination ability across all classification thresholds.
"""))

    # 5. Continuous Numerical Features
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Continuous Numerical Features (`tenure`, `MonthlyCharges`, `TotalCharges`)

Let's examine how customer lifetime, monthly bill, and cumulative spend relate to churn.
"""))

    cells.append(nbf.v4.new_code_cell("""num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
num_summary = df.groupby("Churn")[num_cols].agg(["mean", "median", "std"]).round(2)
num_summary
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(num_cols):
    sns.boxplot(data=df, x="Churn", y=col, ax=axes[idx], palette=["#2ecc71", "#e74c3c"])
    axes[idx].set_title(f"{col} by Churn Status", fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Churn Status")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Tenure segmentation analysis
df_temp = df.copy()
df_temp["tenure_cohort"] = pd.cut(
    df_temp["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 Months (1 yr)", "13-24 Months (2 yrs)", "25-48 Months (3-4 yrs)", "49-72 Months (5-6 yrs)"]
)

cohort_churn = df_temp.groupby("tenure_cohort", observed=False)["Churn"].value_counts(normalize=True).unstack()["Yes"] * 100

plt.figure(figsize=(10, 4))
ax = sns.barplot(x=cohort_churn.index, y=cohort_churn.values, palette="Reds_r")
plt.title("Churn Rate by Tenure Cohort", fontsize=13, fontweight="bold")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Customer Tenure Group")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Findings on Numerical Features
1. **Tenure is the Single Strongest Churn Mitigator:**
   - Customers in their first year (`0-12 months`) experience an alarming **47.4% churn rate**.
   - Beyond 4 years (`49-72 months`), churn plummets to **9.5%**.
   - *Business Insight:* The first 90–360 days are the critical onboarding window where proactive retention offers deliver the highest ROI.
2. **Monthly Charges Drive Churn Sensitivity:**
   - Churned customers pay significantly higher monthly fees: **Median $79.65** vs. **$64.43** for retained customers (Mean: $74.44 vs $61.27).
   - Higher monthly bills correlate directly with increased price sensitivity and cancellation risk.
"""))

    # 6. Contract, Internet & Payment Drivers
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Key Categorical Drivers: Contract, Internet Service, and Payment Method
"""))

    cells.append(nbf.v4.new_code_cell("""key_cats = ["Contract", "InternetService", "PaymentMethod"]
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for idx, cat_col in enumerate(key_cats):
    # Calculate churn rate per category
    churn_rates = df.groupby(cat_col)["Churn"].value_counts(normalize=True).unstack()["Yes"] * 100
    sorted_order = churn_rates.sort_values(ascending=False).index

    sns.barplot(x=churn_rates[sorted_order].values, y=sorted_order, ax=axes[idx], palette="Blues_r")
    axes[idx].set_title(f"Churn Rate by {cat_col}", fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Churn Rate (%)")
    axes[idx].set_ylabel("")
    for p in axes[idx].patches:
        width = p.get_width()
        axes[idx].annotate(f"{width:.1f}%", (width / 2, p.get_y() + p.get_height() / 2),
                           ha='center', va='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary table of churn rates for key drivers
for cat in key_cats:
    rates = df.groupby(cat)["Churn"].value_counts(normalize=True).unstack().round(4) * 100
    rates["Total Customers"] = df.groupby(cat).size()
    print(f"=== {cat.upper()} CHURN RATES ===")
    print(rates[["No", "Yes", "Total Customers"]])
    print()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Findings on Categorical Drivers
1. **Contract Duration:**
   - **Month-to-month:** **42.71% churn rate** (3,875 customers, over half the userbase!).
   - **One year:** **11.27% churn rate**.
   - **Two year:** **2.83% churn rate** (97.17% retained!).
   - *Recommendation:* Incentivizing month-to-month customers to switch to annual commitments with modest discounts is the most effective retention lever.
2. **Internet Service:**
   - **Fiber optic users churn at 41.89%**, compared to **18.96% for DSL** and **7.41% for No Internet**.
   - *Why?* Fiber optic plans are premium-priced, leading to higher monthly charges ($70+), where service outages or unmet expectations trigger rapid cancellation.
3. **Payment Method:**
   - **Electronic check users churn at 45.29%**, compared to **~15.2% - 16.7% for automatic payments** (Bank transfer / Credit card).
   - *Behavioral Psychology:* Automatic billing removes friction and creates retention inertia, whereas manual monthly checks force customers to consciously confront and evaluate their bill every 30 days.
"""))

    # 7. Additional Services
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Value-Added Support Services (`TechSupport`, `OnlineSecurity`)
"""))

    cells.append(nbf.v4.new_code_cell("""support_cols = ["TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection"]
support_data = []

for col in support_cols:
    rate_yes = df[df[col] == "Yes"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    rate_no = df[df[col] == "No"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    support_data.append({"Service": col, "With Service (Churn %)": round(rate_yes, 2), "Without Service (Churn %)": round(rate_no, 2)})

support_df = pd.DataFrame(support_data)
print("Impact of Support Services on Churn:")
support_df
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 💡 Support Services Takeaway
Customers with **TechSupport** churn at only **15.17%**, compared to **41.64%** for customers without it! Bundling free or subsidized tech support directly bolsters retention.
"""))

    # 8. Feature Engineering Pipeline Walkthrough
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Modular Feature Engineering Pipeline (`src/features/`)

Now that our exploratory analysis has surfaced the key drivers and schema requirements, we test our production feature engineering pipeline:
1. Strip arbitrary `customerID` (prevents data leakage and overfitting).
2. Isolate target `Churn` and encode to binary `(0, 1)`.
3. Standardize numerical features using `StandardScaler`.
4. One-hot encode categorical features using `OneHotEncoder(handle_unknown='ignore')`.
5. Enforce strict train/test separation (fit ONLY on `X_train`).
"""))

    cells.append(nbf.v4.new_code_cell("""from src.features.pipeline import (
    split_features_and_target,
    encode_target,
    build_feature_preprocessor,
    create_train_test_splits,
    fit_and_transform_features,
)

# Step 1: Split predictor matrix X and target y
X, y_raw = split_features_and_target(df, drop_id=True)
y = encode_target(y_raw)

print(f"Predictor Features X Shape : {X.shape} (customerID excluded)")
print(f"Encoded Target y Shape     : {y.shape} (Values: {np.bincount(y)})")

# Step 2: Stratified Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = create_train_test_splits(
    X, y, test_size=0.2, random_state=42, stratify=True
)

print(f"X_train Shape: {X_train.shape}, y_train Class Proportions: {np.bincount(y_train) / len(y_train)}")
print(f"X_test Shape : {X_test.shape}, y_test Class Proportions : {np.bincount(y_test) / len(y_test)}")

# Step 3: Build & Fit ColumnTransformer (Strictly on X_train!)
preprocessor = build_feature_preprocessor()
X_train_trans, X_test_trans, feature_names = fit_and_transform_features(
    preprocessor, X_train, X_test
)

print(f"\\nTransformed X_train Shape: {X_train_trans.shape} (Expanded from 19 raw features to {len(feature_names)} one-hot columns)")
print(f"Transformed X_test Shape : {X_test_trans.shape}")
print(f"Sample Encoded Feature Names: {feature_names[:8]} ... (+{len(feature_names)-8} more)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""---
## 🏁 Conclusion & Readiness for Phase 4

- **EDA completed:** Verified all empirical distributions, class imbalance, and retention dynamics.
- **Feature Pipeline validated:** Clean, leak-free Scikit-learn transformer ready for model benchmarking.
- **Next Step:** Proceed to **Phase 4 — Baseline Classifier & Advanced Model Benchmarking** (Logistic Regression, Random Forest, Gradient Boosting evaluated on ROC-AUC, PR-AUC, and Recall).
"""))

    nb.cells = cells
    return nb


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = create_notebook()

    print(f"Saving notebook skeleton to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Executing notebook to verify all cells run cleanly and compute real charts...")
    client = NotebookClient(nb, timeout=60, kernel_name="python3")
    client.execute()

    print("Saving executed notebook with outputs...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated and executed {NOTEBOOK_PATH}!")


if __name__ == "__main__":
    main()
