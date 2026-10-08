"""
ML Model Service.
Encapsulates joblib loading, model caching, metadata access, and thread-safe inference.
"""
from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from .constants import FEATURE_COLUMNS
from .preprocessor import VehiclePreprocessor

logger = logging.getLogger(__name__)


class ModelService:
    """Encapsulates car price model loading and thread-safe inference."""

    def __init__(
        self,
        model_path: str | Path = "car_price_model.joblib",
        load_on_init: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.model: Any = None
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)
        self.brand_map: dict[str, float] = {}
        self.model_map: dict[str, float] = {}
        self.brand_to_models: dict[str, list[str]] = {}
        self.preprocessor: VehiclePreprocessor | None = None
        self._is_ready: bool = False
        self._lock = threading.Lock()

        if load_on_init:
            self.load()

    def _resolve_path(self) -> Path:
        """Resolves model path whether absolute or relative to base directories."""
        if self.model_path.is_absolute() and self.model_path.exists():
            return self.model_path
        if self.model_path.exists():
            return self.model_path.resolve()

        # Traverse up from this file to project root: services -> car_price_predictor -> root
        project_root = Path(__file__).resolve().parent.parent.parent

        # 1. Preserving subdirectories relative to project root
        candidate = project_root / self.model_path
        if candidate.exists():
            return candidate.resolve()

        # 2. Directly in project root matching the filename (backward compatibility fallback)
        candidate_name = project_root / self.model_path.name
        if candidate_name.exists():
            return candidate_name.resolve()

        if not self.model_path.is_absolute():
            return candidate.resolve()
        return self.model_path.resolve()

    def load(self) -> bool:
        """Loads joblib model artifact and caches frequency maps."""
        with self._lock:
            try:
                resolved_path = self._resolve_path()
                if not resolved_path.exists():
                    logger.error("Model artifact not found at %s", resolved_path)
                    self._is_ready = False
                    return False

                logger.info("Loading model artifact from %s", resolved_path)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    data = joblib.load(resolved_path)

                if not isinstance(data, dict) or "model" not in data:
                    raise ValueError(f"Corrupt or invalid model artifact in {resolved_path}")

                self.model = data["model"]
                self.feature_columns = list(data.get("columns", FEATURE_COLUMNS))
                self.brand_map = dict(data.get("brand_map", {}))
                self.model_map = dict(data.get("model_map", {}))
                self.brand_to_models = dict(data.get("brand_to_models", {}))

                self.preprocessor = VehiclePreprocessor(
                    feature_columns=self.feature_columns,
                    brand_map=self.brand_map,
                    model_map=self.model_map,
                )
                self._is_ready = True
                logger.info(
                    "ModelService loaded successfully with %d brands and %d models.",
                    len(self.brand_map),
                    len(self.model_map),
                )
                return True
            except Exception as e:
                logger.exception("Failed to load car price model: %s", e)
                self._is_ready = False
                return False

    def is_ready(self) -> bool:
        """Returns True if the ML model is loaded and ready for predictions."""
        return self._is_ready and self.model is not None

    def predict(self, input_df: pd.DataFrame) -> float | np.ndarray:
        """Executes scikit-learn model inference thread-safely."""
        if not self.is_ready():
            raise RuntimeError("ModelService is not loaded or ready for inference")
        with self._lock:
            if hasattr(self.model, "estimators_") and len(self.model.estimators_) > 0:
                values = input_df.values
                try:
                    x_32 = values.astype(np.float32, copy=False)
                    preds = np.mean([tree.tree_.predict(x_32)[:, 0] for tree in self.model.estimators_], axis=0)
                    if len(preds) == 1:
                        return float(preds[0])
                    return preds
                except Exception:
                    pass
            prediction = self.model.predict(input_df)
            if len(prediction) == 1:
                return float(prediction[0])
            return np.asarray(prediction, dtype=float)

    def predict_batch(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Executes scikit-learn model inference for a batch of rows thread-safely.

        Args:
            input_df: Multi-row DataFrame matching the 14 model features.

        Returns:
            np.ndarray: 1D array of predicted vehicle values.
        """
        res = self.predict(input_df)
        if isinstance(res, np.ndarray):
            return res
        return np.array([res], dtype=float)

    def predict_with_interval(
        self,
        input_df: pd.DataFrame,
        confidence_level: float = 0.80,
    ) -> tuple[float, float, float]:
        """
        Calculates point prediction along with lower and upper confidence bounds
        derived from the empirical distribution across the ensemble decision trees.

        Args:
            input_df: Single-row DataFrame matching the 14 model features.
            confidence_level: Nominal confidence level (default: 0.80 for 10th-90th percentiles).

        Returns:
            tuple[float, float, float]: (point_prediction, lower_bound, upper_bound)
        """
        if not self.is_ready():
            raise RuntimeError("ModelService is not loaded or ready for inference")

        conf = float(np.clip(confidence_level, 0.01, 0.99))
        alpha = 1.0 - conf
        p_lower = 100.0 * (alpha / 2.0)
        p_upper = 100.0 * (1.0 - (alpha / 2.0))

        with self._lock:
            # If ensemble tree estimators are available, calculate empirical tree percentiles
            if hasattr(self.model, "estimators_") and len(self.model.estimators_) > 0:
                values = input_df.values
                try:
                    # Ultra-fast Cython Tree prediction path (< 1ms across 100 trees)
                    x_32 = values.astype(np.float32, copy=False)
                    tree_preds = np.array([tree.tree_.predict(x_32)[0, 0] for tree in self.model.estimators_])
                    base_prediction = float(np.mean(tree_preds))
                except Exception:
                    tree_preds = np.array([tree.predict(values)[0] for tree in self.model.estimators_])
                    base_prediction = float(self.model.predict(input_df)[0])
                lower_base = float(np.percentile(tree_preds, p_lower))
                upper_base = float(np.percentile(tree_preds, p_upper))
            else:
                base_prediction = float(self.model.predict(input_df)[0])
                # Deterministic fallback when tree estimators are not present
                spread = max(base_prediction * 0.15, 25000.0)
                lower_base = max(0.0, base_prediction - spread)
                upper_base = base_prediction + spread

        # Enforce mathematical ordering invariant: lower <= point <= upper
        lower_base = min(lower_base, base_prediction)
        upper_base = max(upper_base, base_prediction)

        return base_prediction, lower_base, upper_base

    def get_brands(self) -> list[str]:
        """Returns alphabetically sorted list of known car brands."""
        return sorted(list(self.brand_map.keys()))

    def get_hierarchy(self) -> dict[str, list[str]]:
        """Returns dictionary of brand to associated model lists."""
        return self.brand_to_models

    def get_models_for_brand(self, brand: str) -> list[str]:
        """Returns list of models for a specific brand."""
        return self.brand_to_models.get(brand, [])
