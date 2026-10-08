"""
Training Preprocessor Adapter for CarPricePredictor.

Aligns training data encoding directly with VehiclePreprocessor to guarantee
exact 14-feature column order and eliminate training/inference parity drift.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

from car_price_predictor.services.constants import FEATURE_COLUMNS
from car_price_predictor.services.preprocessor import (
    FUEL_MAPPING,
    SELLER_TYPE_MAPPING,
    TRANSMISSION_MAPPING,
    VehiclePreprocessor,
)

logger = logging.getLogger(__name__)


class TrainingPreprocessorAdapter:
    """
    Fits frequency mappings on training partition and transforms tabular
    DataFrames into the exact 14-feature numeric matrix required by RandomForestRegressor.
    """

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns: list[str] = list(feature_columns or FEATURE_COLUMNS)
        self.brand_map: dict[str, float] = {}
        self.model_map: dict[str, float] = {}
        self.brand_to_models: dict[str, list[str]] = {}
        self._fitted: bool = False
        self._inference_preprocessor: VehiclePreprocessor | None = None

    def fit(self, df: pd.DataFrame) -> TrainingPreprocessorAdapter:
        """
        Computes brand frequency map, model frequency map, and brand-to-models
        hierarchy map from the training dataset.
        """
        total_rows = float(len(df))
        if total_rows == 0:
            raise ValueError("Cannot fit preprocessor adapter on empty DataFrame")

        # 1. Compute brand frequencies
        if "brand" in df.columns:
            brand_counts = df["brand"].astype(str).str.strip().value_counts()
            self.brand_map = {str(k): float(v / total_rows) for k, v in brand_counts.items()}
        else:
            self.brand_map = {}

        # 2. Compute model frequencies
        if "model" in df.columns:
            model_counts = df["model"].astype(str).str.strip().value_counts()
            self.model_map = {str(k): float(v / total_rows) for k, v in model_counts.items()}
        else:
            self.model_map = {}

        # 3. Compute brand to models hierarchy
        self.brand_to_models = {}
        if "brand" in df.columns and "model" in df.columns:
            for brand, group in df.groupby("brand"):
                brand_str = str(brand).strip()
                unique_models = sorted(list({str(m).strip() for m in group["model"].dropna() if str(m).strip()}))
                self.brand_to_models[brand_str] = unique_models

        self._inference_preprocessor = VehiclePreprocessor(
            feature_columns=self.feature_columns,
            brand_map=self.brand_map,
            model_map=self.model_map,
        )
        self._fitted = True
        logger.info(
            "Fitted adapter with %d brands and %d models across %d rows.",
            len(self.brand_map),
            len(self.model_map),
            int(total_rows),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame into the exact 14-feature DataFrame matching FEATURE_COLUMNS.
        Ensures reference baseline dropping (CNG, Dealer, Automatic) and frequency encoding fallback (0.0).
        """
        if not self._fitted or self._inference_preprocessor is None:
            raise RuntimeError("Adapter must be fitted prior to calling transform()")

        records = df.to_dict(orient="records")
        return self._inference_preprocessor.preprocess_batch(records)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits frequency maps on df and transforms it to the 14-feature matrix."""
        return self.fit(df).transform(df)

    def get_inference_preprocessor(self) -> VehiclePreprocessor:
        """Returns configured VehiclePreprocessor matching this adapter's fitted state."""
        if not self._fitted or self._inference_preprocessor is None:
            raise RuntimeError("Adapter is not yet fitted")
        return self._inference_preprocessor
