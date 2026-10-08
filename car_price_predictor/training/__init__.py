"""
Enterprise ML Training and Artifact Management Package for CarPricePredictor.

Includes reproducible data ingestion, preprocessing adapter, model evaluation,
quality gating, and artifact versioning.
"""
from .data_loader import (
    clean_raw_dataframe,
    generate_synthetic_data,
    generate_synthetic_sample,
    load_dataset,
    parse_numeric_unit,
)
from .evaluator import ModelEvaluator, ModelQualityGateError
from .pipeline import TrainingConfig, TrainingPipeline
from .preprocessor_adapter import TrainingPreprocessorAdapter
from .versioning import resolve_next_version, save_versioned_artifact

__all__ = [
    "clean_raw_dataframe",
    "generate_synthetic_data",
    "generate_synthetic_sample",
    "load_dataset",
    "parse_numeric_unit",
    "ModelEvaluator",
    "ModelQualityGateError",
    "TrainingConfig",
    "TrainingPipeline",
    "TrainingPreprocessorAdapter",
    "resolve_next_version",
    "save_versioned_artifact",
]
