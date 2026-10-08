"""
End-to-End Training Pipeline Orchestrator for CarPricePredictor (R1).

Coordinates dataset ingestion, leak-free train/validation splitting,
feature adaptation, model training, evaluation gating, and versioned serialization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .data_loader import load_dataset
from .evaluator import ModelEvaluator, ModelQualityGateError
from .preprocessor_adapter import TrainingPreprocessorAdapter
from .versioning import save_versioned_artifact

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and operational configuration for training execution."""

    data_path: str | Path | None = None
    use_sample: bool = False
    n_samples: int = 500
    download_kaggle: bool = False
    output_dir: str | Path = "artifacts"
    version: str | None = None
    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_r2: float = 0.80
    enforce_gates: bool = True
    promote: bool = False
    promote_path: str | Path | None = None
    dry_run: bool = False
    thresholds: dict[str, float] = field(default_factory=dict)


class TrainingPipeline:
    """
    Standardized, reproducible training pipeline orchestrating data ingestion,
    preprocessing, model fitting, evaluation, and artifact versioning.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def run(self, config_override: TrainingConfig | None = None) -> dict[str, Any]:
        """
        Executes end-to-end training pipeline.

        Returns:
            dict containing evaluation metrics, saved file paths, and model status.
        """
        cfg = config_override or self.config

        logger.info("Starting CarPricePredictor training pipeline...")

        # 1. Ingestion
        df = load_dataset(
            data_path=cfg.data_path,
            download_kaggle=cfg.download_kaggle,
            use_sample=cfg.use_sample,
            n_samples=cfg.n_samples,
            random_state=cfg.random_state,
        )

        if "selling_price" not in df.columns:
            raise ValueError("Dataset missing required target column 'selling_price'")

        total_records = len(df)
        logger.info("Loaded %d clean records for training pipeline.", total_records)

        # 2. Train / Holdout Validation Split (Leak-free)
        train_df, val_df = train_test_split(
            df,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
        )

        y_train = train_df["selling_price"]
        y_val = val_df["selling_price"]

        # 3. Fit preprocessing adapter on X_train only
        adapter = TrainingPreprocessorAdapter()
        adapter.fit(train_df)

        X_train = adapter.transform(train_df)
        X_val = adapter.transform(val_df)

        # 4. Fit RandomForestRegressor
        logger.info(
            "Fitting RandomForestRegressor(n_estimators=%d, random_state=%d)...",
            cfg.n_estimators,
            cfg.random_state,
        )
        model = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # 5. Model Evaluation and Quality Gates (R2)
        evaluator = ModelEvaluator()
        active_thresholds = dict(evaluator.DEFAULT_THRESHOLDS)
        active_thresholds["min_val_r2"] = cfg.min_r2
        if cfg.thresholds:
            active_thresholds.update(cfg.thresholds)

        eval_summary = evaluator.evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            thresholds=active_thresholds,
            enforce_gates=cfg.enforce_gates and not cfg.dry_run,
        )

        logger.info(
            "Evaluation complete: Train R2=%.4f, Val R2=%.4f (Gate passed=%s)",
            eval_summary["train"]["r2"],
            eval_summary["validation"]["r2"],
            eval_summary["passed"],
        )

        # 6. Artifact Serialization (R1)
        joblib_path: Path | None = None
        metadata_path: Path | None = None

        if not cfg.dry_run:
            dataset_summary = {
                "total_records": total_records,
                "train_records": len(train_df),
                "validation_records": len(val_df),
                "num_brands": len(adapter.brand_map),
                "num_models": len(adapter.model_map),
            }
            metadata_payload = {
                "dataset_summary": dataset_summary,
                "metrics": eval_summary,
                "quality_gates_passed": eval_summary["passed"],
            }

            joblib_path, metadata_path = save_versioned_artifact(
                model=model,
                columns=adapter.feature_columns,
                brand_map=adapter.brand_map,
                model_map=adapter.model_map,
                brand_to_models=adapter.brand_to_models,
                metadata=metadata_payload,
                output_dir=cfg.output_dir,
                version_tag=cfg.version,
                promote=cfg.promote,
                promote_path=cfg.promote_path,
            )
            logger.info("Saved versioned artifacts to %s and %s", joblib_path, metadata_path)
        else:
            logger.info("Dry-run requested: skipping artifact serialization.")

        return {
            "status": "success",
            "model": model,
            "metrics": eval_summary,
            "joblib_path": str(joblib_path) if joblib_path else None,
            "metadata_path": str(metadata_path) if metadata_path else None,
            "version": cfg.version,
            "passed": eval_summary["passed"],
            "dataset_rows": total_records,
        }
