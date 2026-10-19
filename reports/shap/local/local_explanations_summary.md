# Representative Local Customer Explanations Summary

**Analysis Date:** 2026-10-18<br />
**Model:** Gradient Boosting Classifier (`models/churn_model.joblib`)<br />
**Operational Threshold:** `0.35` (Actionable Churn Alert)<br />
**Standard Threshold:** `0.50` (Comparison Baseline)<br />
**Partition Origin:** Development Partition ($N=5,634$)<br />
**Holdout Isolation:** Independent 20% Holdout Partition ($N=1,409$) Strictly Untouched

---

## 1. Comparative Customer Cohort Overview

| Case ID | Churn Probability | Operational Status (Th=0.35) | Margin ($F(x)$) | Dominant Risk Drivers | Dominant Protective Factors |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `case_1_high_risk` | **83.99%** | **High Risk (Operational Churn Alert)** | +1.657 | Customer Tenure (4 mo) (+0.99), Contract = 'Month-to-month' (+0.75) | Tech Support != 'No' (Actual: 'Yes') (-0.23), Payment Method != 'Electronic check' (Actual: 'Mailed check') (-0.05) |
| `case_2_borderline_risk` | **47.37%** | **High Risk (Operational Churn Alert)** | -0.105 | Contract = 'Month-to-month' (+0.61), Monthly Charges ($96.35) (+0.32) | Tech Support != 'No' (Actual: 'Yes') (-0.25), Customer Tenure (34 mo) (-0.14) |
| `case_3_low_risk` | **6.98%** | **Low Risk (Retained)** | -2.590 | Monthly Charges ($106.05) (+0.71), Internet Service = 'Fiber optic' (+0.14) | Contract != 'Month-to-month' (Actual: 'Two year') (-0.78), Contract = 'Two year' (-0.42) |

---

## 2. Key Pedagogical Takeaways

1. **The Critical Value of the 0.35 Operational Threshold (Case 2):**
   - For `case_2_borderline_risk`, the model predicted a **47.37%** churn probability. Under the naive 0.50 threshold, this customer would be classified as **non-churn** (label 0) and ignored.
   - Under the optimized operational threshold (0.35), this customer is correctly flagged as actionable churn risk, allowing preemptive customer success intervention.

2. **Divergent Feature Influence Across Profiles:**
   - In **Case 1 (High Risk)**: Short tenure and month-to-month contracts dominate, pushing the margin strongly positive.
   - In **Case 2 (Borderline Risk)**: Month-to-month contract and high monthly charges push churn risk up, but established tenure (34 months) and inactive 'No Tech Support' state (customer has tech support) push risk down. The 0.35 threshold catches this critical borderline customer.
   - In **Case 3 (Low Risk)**: Long tenure and a Two-year contract commitment dominate, heavily suppressing the churn margin down to -2.59 log-odds.

3. **Mathematical Additivity Guarantee:**
   - In all three cases, the sum of SHAP values plus the baseline expected value ($-1.5893$) matched the classifier's `decision_function` margin to within machine precision ($< 10^{-14}$).
