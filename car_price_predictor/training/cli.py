"""
Command-Line Interface for CarPricePredictor ML Training (R1).

Provides CLI entrypoint for running reproducible training pipelines,
specifying datasets, hyperparameter overrides, and artifact targets.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import TrainingConfig, TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("car_price_predictor.training")


def build_parser() -> argparse.ArgumentParser:
    """Builds argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m car_price_predictor.training.train",
        description="CarPricePredictor Enterprise ML Training & Artifact Serialization Pipeline.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to input CSV dataset file.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use deterministic synthetic dataset generator for fast offline execution.",
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download Cardekho dataset from Kaggle via kagglehub.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save versioned artifacts and metadata (default: artifacts/).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Explicit version tag for serialized artifact (e.g. v1, v2). Defaults to autoincrement.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of decision trees in RandomForestRegressor (default: 100).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.80,
        help="Minimum acceptable validation R2 score quality gate (default: 0.80).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote saved model to workspace root car_price_model.joblib for production runtime.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute data loading, training, and evaluation without saving artifacts to disk.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution routine."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = TrainingConfig(
        data_path=args.data,
        use_sample=args.sample,
        download_kaggle=args.download_kaggle,
        output_dir=args.output_dir,
        version=args.version,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        min_r2=args.min_r2,
        promote=args.promote,
        dry_run=args.dry_run,
    )

    try:
        pipeline = TrainingPipeline(config)
        result = pipeline.run()

        print("\n" + "=" * 60)
        print(" TRAINING PIPELINE EXECUTION SUCCESSFUL")
        print("=" * 60)
        metrics = result["metrics"]
        print(f" Train R2:            {metrics['train']['r2']:.4f}")
        print(f" Validation R2:       {metrics['validation']['r2']:.4f}")
        print(f" Validation MAE:      INR {metrics['validation']['mae']:,.2f}")
        print(f" Validation RMSE:     INR {metrics['validation']['rmse']:,.2f}")
        print(f" Validation MAPE:     {metrics['validation']['mape']:.2f}%")
        print(f" Generalization Gap:  {metrics['generalization_gap']:.4f}")
        print(f" Quality Gates:       {'PASSED' if result['passed'] else 'FAILED'}")
        if not result["passed"] and metrics.get("gate_failures"):
            for failure in metrics["gate_failures"]:
                print(f"   - {failure}")
        if result.get("joblib_path"):
            print(f" Model Artifact:      {result['joblib_path']}")
        if result.get("metadata_path"):
            print(f" Metadata JSON:       {result['metadata_path']}")
        print("=" * 60 + "\n")
        return 0

    except Exception as exc:
        logger.exception("Training pipeline failed with error: %s", exc)
        print(f"\nERROR: Training pipeline failed: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
