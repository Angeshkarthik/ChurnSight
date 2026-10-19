# Local Customer Churn Explanation Report

**Analysis Case:** `case_2_borderline_risk`<br />
**Risk Classification:** **High Risk (Operational Churn Alert)**<br />
**Predicted Churn Probability:** **47.37%**<br />
**Operational Decision Threshold:** `0.35` (Actionable Churn Alert if >= 0.35)<br />
**Standard Threshold Comparison:** `0.50` (Prediction: `0`)

---

## 1. Executive Summary & Scoring Metrics

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Model Churn Probability** | **0.4737** (47.37%) | Calibrated logistic probability sigmoid(F(x)) |
| **Operational Risk Alert** | **High Risk (Operational Churn Alert)** | Flagged using frozen operational threshold 0.35 |
| **Standard Baseline Probability** | **0.1695** (16.95%) | Population prior expectation before individual features |
| **Baseline Log-Odds (F0)** | **-1.5893** | Logit intercept across development training distribution |
| **Final Model Score (F(x))** | **-0.1052** | Log-odds margin: F0 + sum(phi_i) |
| **Sum of SHAP Values (sum phi_i)** | **+1.4841** | Net shift in log-odds produced by customer features |
| **Additivity Discrepancy** | **5.83e-16** | Exact machine precision match with `decision_function` |

![SHAP Waterfall Plot](case_2_borderline_risk_waterfall.png)

---

## 2. Top Risk Factors Pushing Toward Churn (phi_i > 0)

The following features exerted the strongest mathematical push **increasing** the churn log-odds for this customer:

| Observed Feature & Status | Encoded Feature Column | SHAP Impact (Log-Odds) | Detailed Explanation |
| :--- | :--- | :--- | :--- |
| `Contract = 'Month-to-month'        ` | `Contract_Month-to-month       ` | +0.6134    | Contract is 'Month-to-month' (indicator active), contributing +0.6134 log-odds, pushing the model output toward churn by +0.6134 log-odds for this prediction. |
| `Monthly Charges ($96.35)           ` | `MonthlyCharges                ` | +0.3210    | Observed monthly bill of $96.35 contributes +0.3210 log-odds, pushing the model output toward churn by +0.3210 log-odds for this prediction. |
| `Internet Service = 'Fiber optic'   ` | `InternetService_Fiber optic   ` | +0.2591    | Internet Service is 'Fiber optic' (indicator active), contributing +0.2591 log-odds, pushing the model output toward churn by +0.2591 log-odds for this prediction. |
| `Payment Method = 'Electronic check'` | `PaymentMethod_Electronic check` | +0.2115    | Payment Method is 'Electronic check' (indicator active), contributing +0.2115 log-odds, pushing the model output toward churn by +0.2115 log-odds for this prediction. |
| `Online Security = 'No'             ` | `OnlineSecurity_No             ` | +0.2088    | Online Security is 'No' (indicator active), contributing +0.2088 log-odds, pushing the model output toward churn by +0.2088 log-odds for this prediction. |

---

## 3. Top Factors Suppressing Churn Risk (phi_i < 0)

The following features exerted the strongest mathematical push **decreasing** the churn log-odds for this customer:

| Observed Feature & Status | Encoded Feature Column | SHAP Impact (Log-Odds) | Detailed Explanation |
| :--- | :--- | :--- | :--- |
| `Tech Support != 'No' (Actual: 'Yes')` | `TechSupport_No                ` | -0.2468    | Tech Support is not 'No' (observed: 'Yes', indicator inactive), contributing -0.2468 log-odds, pushing the model output away from churn by -0.2468 log-odds for this prediction. |
| `Customer Tenure (34 mo)            ` | `tenure                        ` | -0.1447    | Observed customer tenure of 34 month(s) contributes -0.1447 log-odds, pushing the model output away from churn by -0.1447 log-odds for this prediction. |
| `Total Charges ($3190.25)           ` | `TotalCharges                  ` | -0.1123    | Observed total cumulative spend of $3190.25 contributes -0.1123 log-odds, pushing the model output away from churn by -0.1123 log-odds for this prediction. |
| `Online Backup != 'No' (Actual: 'Yes')` | `OnlineBackup_No               ` | -0.0513    | Online Backup is not 'No' (observed: 'Yes', indicator inactive), contributing -0.0513 log-odds, pushing the model output away from churn by -0.0513 log-odds for this prediction. |
| `Streaming Movies != 'Yes' (Actual: 'No')` | `StreamingMovies_Yes           ` | -0.0351    | Streaming Movies is not 'Yes' (observed: 'No', indicator inactive), contributing -0.0351 log-odds, pushing the model output away from churn by -0.0351 log-odds for this prediction. |

---

## 4. Key Customer Profile Attributes

- **Contract Commitment:** `Month-to-month`
- **Customer Tenure:** `34 months`
- **Monthly Recurring Charges:** `$96.35`
- **Total Charges to Date:** `$3190.25`
- **Internet Service:** `Fiber optic`
- **Tech Support:** `Yes`
- **Online Security:** `No`
- **Payment Method:** `Electronic check`

---

## 5. Governance and Methodology Notice

> [!WARNING]
> **Descriptive Model Behavior vs. Causal Truth:** Governance Notice: These local SHAP contributions quantify the mathematical influence of each feature on the model's log-odds output margin F(x). They explain predictive model behavior in observational data, not confirmed causal drivers or guaranteed retention outcomes. Do not interpret feature correlations as causal levers without business domain validation.

- **Scale Integrity:** SHAP values are strictly calculated and summed on the **log-odds margin scale**. They are not additive percentages and must not be added directly to probabilities.
- **Holdout Isolation Guarantee:** This customer profile was drawn exclusively from the Development partition. No final holdout evaluation data was examined or utilized.
