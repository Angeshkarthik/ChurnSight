"""CLI script to generate local SHAP explainability artifacts for individual customers.

Provides individual customer churn explanations:
1. Local SHAP waterfall plots (log-odds margin decomposition)
2. Machine-readable JSON explanations for each customer
3. Detailed individual Markdown reports
4. Cross-profile comparative summary report

Usage Examples:
  # Generate explanations for the 3 representative development cases:
  python scripts/explain_customer.py --representative

  # Explain a specific customer row from the dataset:
  python scripts/explain_customer.py --row-index 42
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.explainability.shap_explainer import (
    DEFAULT_LOCAL_REPORTS_DIR,
    DEFAULT_OPERATIONAL_THRESHOLD,
    ChurnShapExplainer,
    explain_representative_customers,
    load_churn_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("explain_customer")


def main():
    parser = argparse.ArgumentParser(
        description="Customer Intelligence Platform - Local SHAP Customer Explainability CLI"
    )
    parser.add_argument(
        "--representative",
        action="store_true",
        default=True,
        help="Generate explanations for 3 representative development customer profiles (High, Borderline, Low Risk)",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Explain a specific row index from the processed dataset CSV",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to serialized churn model (default: models/churn_model.joblib)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to cleaned dataset CSV (default: data/processed/telco_churn_cleaned.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save local SHAP reports and plots (default: reports/shap/local/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_OPERATIONAL_THRESHOLD,
        help=f"Operational decision threshold for churn risk alert (default: {DEFAULT_OPERATIONAL_THRESHOLD})",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CUSTOMER INTELLIGENCE PLATFORM — LOCAL SHAP EXPLAINABILITY CLI")
    print("=" * 80)

    resolved_output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_LOCAL_REPORTS_DIR

    if args.row_index is not None:
        data_path = args.data_path or (PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv")
        df = pd.read_csv(data_path)
        if args.row_index < 0 or args.row_index >= len(df):
            raise IndexError(f"row-index {args.row_index} out of bounds for dataset of length {len(df)}")

        customer_row = df.iloc[args.row_index].to_dict()
        pipeline = load_churn_model(args.model_path)
        explainer = ChurnShapExplainer(pipeline)

        prefix = f"customer_row_{args.row_index}"
        explanation = explainer.explain_customer(
            customer_data=customer_row,
            operational_threshold=args.threshold,
            customer_id=str(customer_row.get("customerID", f"ROW_{args.row_index}")),
        )

        plot_path = explainer.save_customer_visualizations(
            explanation_record=explanation,
            output_dir=resolved_output_dir,
            filename_prefix=prefix,
        )
        report_path = explainer.save_customer_report(
            explanation_record=explanation,
            output_dir=resolved_output_dir,
            filename_prefix=prefix,
            waterfall_rel_path=plot_path.name,
        )

        print(f"\nCustomer Explanation for Row {args.row_index}:")
        print(f"  - Churn Probability     : {explanation.churn_probability:.2%}")
        print(f"  - Operational Status     : {explanation.risk_level} (Th={args.threshold:.2f})")
        print(f"  - Standard Status        : Label {explanation.standard_threshold_label} (Th=0.50)")
        print(f"  - Model Output (Margin)  : {explanation.model_score_log_odds:+.4f} log-odds")
        print(f"  - Baseline Log-Odds      : {explanation.baseline_log_odds:+.4f}")
        print("\nTop Risk Drivers (push toward churn):")
        for c in explanation.top_positive_contributions[:3]:
            print(f"    [+] {c.readable_feature:<30} (+{c.shap_value:.4f}) : {c.reason}")
        print("\nTop Protective Factors (push away from churn):")
        for c in explanation.top_negative_contributions[:3]:
            print(f"    [-] {c.readable_feature:<30} ({c.shap_value:.4f}) : {c.reason}")

        print(f"\nArtifacts saved to: {resolved_output_dir}")
        print(f"  - Waterfall Plot: {plot_path}")
        print(f"  - Report:         {report_path}\n")

    else:
        # Default: explain the 3 representative cases
        explanations = explain_representative_customers(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=resolved_output_dir,
            operational_threshold=args.threshold,
        )

        print(f"\nSuccessfully generated local explanations for {len(explanations)} representative cases:")
        for exp in explanations:
            print("-" * 75)
            print(f"Case ID                  : {exp.customer_id}")
            print(f"Churn Probability        : {exp.churn_probability:.2%}")
            print(f"Operational Risk Level   : {exp.risk_level} [Threshold: {exp.operational_threshold:.2f}]")
            print(f"Standard Threshold Label : {exp.standard_threshold_label} [Threshold: 0.50]")
            print(f"Model Margin (Log-Odds)  : {exp.model_score_log_odds:+.4f}")
            print(f"Baseline Log-Odds        : {exp.baseline_log_odds:+.4f}")
            if exp.top_positive_contributions:
                top_pos = exp.top_positive_contributions[0]
                print(f"Strongest Churn Driver   : {top_pos.readable_feature} (+{top_pos.shap_value:.4f})")
            if exp.top_negative_contributions:
                top_neg = exp.top_negative_contributions[0]
                print(f"Strongest Retention Factor: {top_neg.readable_feature} ({top_neg.shap_value:.4f})")
        print("-" * 75)
        print(f"\nAll artifacts saved to: {resolved_output_dir}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
