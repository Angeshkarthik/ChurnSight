# Global Model Explainability Report: SHAP Analysis

**Model:** Gradient Boosting Classifier<br />
**Explainer:** `shap.TreeExplainer` (Exact Tree Path Dependent Algorithm)<br />
**Output Scale:** Raw Model Margin / Log-Odds (log[p / (1 - p)])<br />
**Baseline Expected Value ($F_0$):** -1.5893<br />
**Representative Sample Size:** 1,000 development records<br />
**Partition Isolation:** Development Set Only (Holdout test set strictly preserved and untouched)

---

## 1. Top Predictive Features (Global Impact)

The table below ranks the top 15 features by their **Mean Absolute SHAP Value** (E[|phi_i|]).
This metric measures the global importance of each feature on shifting customer churn predictions away from the baseline.

| Rank | Feature Name                   | Mean |SHAP| (Log-Odds) |
| :--- | :----------------------------- | :------------------------- |
| 1    | `Contract_Month-to-month       ` | 0.6354         |
| 2    | `tenure                        ` | 0.3838         |
| 3    | `OnlineSecurity_No             ` | 0.2436         |
| 4    | `TotalCharges                  ` | 0.2219         |
| 5    | `TechSupport_No                ` | 0.2204         |
| 6    | `MonthlyCharges                ` | 0.2165         |
| 7    | `InternetService_Fiber optic   ` | 0.2145         |
| 8    | `Contract_Two year             ` | 0.1712         |
| 9    | `PaymentMethod_Electronic check` | 0.1326         |
| 10   | `MultipleLines_No              ` | 0.0716         |
| 11   | `StreamingMovies_Yes           ` | 0.0700         |
| 12   | `PaperlessBilling_Yes          ` | 0.0643         |
| 13   | `OnlineBackup_No               ` | 0.0480         |
| 14   | `Contract_One year             ` | 0.0474         |
| 15   | `PaperlessBilling_No           ` | 0.0461         |

---

## 2. Directional Behavioral Findings (Beeswarm Insights)

From the SHAP beeswarm distribution analysis, the following behavioral relationships are observed in the fitted model:

1. **Contract Structure (`Contract_Month-to-month` vs `Contract_Two year`):**
   - Customers on **Month-to-month** contracts exhibit the largest positive SHAP contributions, strongly shifting the log-odds margin toward churn.
   - Conversely, **Two-year** and **One-year** commitments exert strong negative SHAP contributions, acting as the primary protective factor against churn.

2. **Customer Tenure (`tenure`):**
   - High tenure values (red points) consistently produce negative SHAP values, significantly reducing the probability of churn.
   - Low tenure (first few months of service) exhibits positive SHAP values, representing the most vulnerable onboarding window.

3. **Support & Security Services (`OnlineSecurity_No`, `TechSupport_No`):**
   - Absence of value-added security and technical support services strongly elevates churn log-odds.
   - Customers enrolled in tech support and device security show lower churn propensities.

4. **Internet Service & Charges (`InternetService_Fiber optic`, `MonthlyCharges`, `TotalCharges`):**
   - Fiber optic service is associated with higher monthly charges and higher churn risk in this customer cohort.
   - Total charges reflect cumulative tenure multiplied by billings, interacting closely with tenure duration.

5. **Payment Method (`PaymentMethod_Electronic check`):**
   - Payment via electronic check correlates with elevated churn probabilities, whereas automated credit card and bank transfer mechanisms correlate with customer retention.

---

## 3. Methodological Framework & Output Scale

- **Scale Interpretation:** The values reported are on the **log-odds (logit)** scale, not raw probabilities. Because Scikit-learn's `GradientBoostingClassifier` models `F(x) = log(p / (1 - p))`, each feature adds linearly:
  ```
  log( P(Churn) / (1 - P(Churn)) ) = Base_Value + sum( SHAP_i )
  ```
  A positive SHAP value (`phi_i > 0`) shifts the model toward predicting Churn; a negative SHAP value (`phi_i < 0`) shifts the model toward Retained.
- **Additivity Verification:** The sum of SHAP values plus the base value was verified against the classifier's `decision_function` and matched within machine precision (<= 1e-14).

---

## 4. Analytical Limitations & Governance Notice

> [!WARNING]
> **Correlation vs. Causation:** SHAP values explain how the machine learning model combines predictive signals in the observed historical dataset. They **do not prove causal mechanisms**. For example, while electronic check payment correlates with higher churn, forcing customers to change payment methods without addressing underlying dissatisfaction will not automatically reduce churn.

- **One-Hot Encoding Fragmentation:** Categorical features (such as `Contract` and `PaymentMethod`) are one-hot encoded into individual binary indicator features. While this isolates specific contract tiers, the aggregate importance of the entire `Contract` domain is distributed across its respective one-hot columns.
- **Sample Representation:** This analysis utilized a stratified sample of 1000 records from the 80% development partition. The independent 20% final holdout partition was deliberately kept un-peeked and isolated from explanation fitting.
