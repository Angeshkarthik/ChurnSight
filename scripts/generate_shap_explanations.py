"""CLI script to generate global SHAP explainability artifacts and reports.

Executes the global SHAP explainability pipeline on the saved Gradient Boosting
churn model and development data sample, producing:
1. SHAP global feature importance bar plot (Mean |SHAP|)
2. SHAP summary beeswarm plot (Directional impact distribution)
3. Machine-readable global importance summary (JSON and CSV)
4. Comprehensive Markdown explainability report
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.shap_explainer import run_global_explainability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_shap_explanations")


def main():
    parser = argparse.ArgumentParser(
        description="Customer Intelligence Platform - SHAP Global Explainability CLI"
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
        help="Directory to save SHAP reports and plots (default: reports/shap/)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Representative development sample size for SHAP analysis (default: 1000)",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=15,
        help="Maximum number of top features to display in plots (default: 15)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CUSTOMER INTELLIGENCE PLATFORM — SHAP GLOBAL EXPLAINABILITY PIPELINE")
    print("=" * 80)

    results = run_global_explainability(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_display=args.max_display,
    )

    importance_df = results["importance_df"]

    print("\nTop 10 Global Features by Mean |SHAP| (Log-Odds Impact):")
    print("-" * 65)
    print(importance_df.head(10).to_string(index=False))
    print("-" * 65)

    print("\nGenerated SHAP Artifacts:")
    print(f"  - Bar Plot (Mean |SHAP|) : {results['bar_plot']}")
    print(f"  - Beeswarm Summary Plot  : {results['beeswarm_plot']}")
    print(f"  - Machine-readable JSON  : {results['json_path']}")
    print(f"  - Machine-readable CSV   : {results['csv_path']}")
    print(f"  - Summary Markdown Report: {results['report_path']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
