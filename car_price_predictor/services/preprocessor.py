"""
Vehicle Preprocessor Service.

Transforms validated vehicle request payloads into exact 14-feature DataFrames
matching the scikit-learn model artifact, and extracts pricing rule parameters.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence
import pandas as pd

from .constants import (
    FEATURE_COLUMNS,
    DEFAULT_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)

# Categorical one-hot mapping tables
# Reference baselines (CNG, Dealer, Automatic) map to None (all 0.0)
FUEL_MAPPING: dict[str, str | None] = {
    "diesel": "fuel_Diesel",
    "electric": "fuel_Electric",
    "lpg": "fuel_LPG",
    "petrol": "fuel_Petrol",
    "cng": None,  # Dropped baseline
}

SELLER_TYPE_MAPPING: dict[str, str | None] = {
    "individual": "seller_type_Individual",
    "trustmark dealer": "seller_type_Trustmark Dealer",
    "dealer": None,  # Dropped baseline
}

TRANSMISSION_MAPPING: dict[str, str | None] = {
    "manual": "transmission_Manual",
    "automatic": None,  # Dropped baseline
}

# Heuristic pricing defaults
DEFAULT_YEAR: int = 2018
DEFAULT_MILEAGE: float = 18.0
DEFAULT_OWNER: str = "First Owner"


class VehiclePreprocessor:
    """
    Transforms vehicle payload dictionaries into model-ready DataFrames
    and extracts business logic parameters for pricing heuristics.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        brand_map: Mapping[str, float] | None = None,
        model_map: Mapping[str, float] | None = None,
    ) -> None:
        """
        Initialize preprocessor with feature column definition and frequency maps.

        Args:
            feature_columns: Ordered list of 14 model feature columns.
            brand_map: Dictionary mapping brand names to frequency proportions.
            model_map: Dictionary mapping model names to frequency proportions.
        """
        self.feature_columns: list[str] = (
            [str(col).strip() for col in feature_columns]
            if feature_columns is not None
            else list(DEFAULT_FEATURE_COLUMNS)
        )
        self.brand_map: dict[str, float] = dict(brand_map or {})
        self.model_map: dict[str, float] = dict(model_map or {})

        # Precompute normalized lower-cased lookup dictionaries for resilient matching.
        # Sort by frequency descending to favor higher-frequency matches on case collisions.
        self._brand_lower_map: dict[str, float] = {}
        for k, v in sorted(self.brand_map.items(), key=lambda item: item[1], reverse=True):
            lk = str(k).strip().lower()
            if lk not in self._brand_lower_map:
                self._brand_lower_map[lk] = float(v)

        self._model_lower_map: dict[str, float] = {}
        for k, v in sorted(self.model_map.items(), key=lambda item: item[1], reverse=True):
            lk = str(k).strip().lower()
            if lk not in self._model_lower_map:
                self._model_lower_map[lk] = float(v)

    @staticmethod
    def _to_dict(payload: Any) -> dict[str, Any]:
        """Convert inbound payload (dict, Pydantic model, or mapping) into a plain dict."""
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        if hasattr(payload, "dict"):
            return payload.dict()
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "__dict__"):
            return payload.__dict__
        return dict(payload)

    def encode_brand(self, brand: Any) -> float:
        """
        Encode brand name to frequency proportion.

        Performs exact lookup first, then case-insensitive and trimmed lookup,
        defaulting to 0.0 for unseen or invalid brands.
        """
        if brand is None:
            return 0.0
        cleaned = str(brand).strip()
        if not cleaned:
            return 0.0
        if cleaned in self.brand_map:
            return float(self.brand_map[cleaned])
        return float(self._brand_lower_map.get(cleaned.lower(), 0.0))

    def encode_model(self, model: Any) -> float:
        """
        Encode model name to frequency proportion.

        Performs exact lookup first, then case-insensitive and trimmed lookup,
        defaulting to 0.0 for unseen or invalid models.
        """
        if model is None:
            return 0.0
        cleaned = str(model).strip()
        if not cleaned:
            return 0.0
        if cleaned in self.model_map:
            return float(self.model_map[cleaned])
        return float(self._model_lower_map.get(cleaned.lower(), 0.0))

    def _build_row(self, data: dict[str, Any]) -> dict[str, float]:
        """Build a dictionary of the exact feature columns with numeric float values."""
        row: dict[str, float] = {col: 0.0 for col in self.feature_columns}

        # Numeric features with safe float conversion
        try:
            val = data.get("km_driven", 0.0)
            row["km_driven"] = float(val) if val is not None and val != "" else 0.0
        except (ValueError, TypeError):
            row["km_driven"] = 0.0

        try:
            val = data.get("mileage", 0.0)
            row["mileage"] = float(val) if val is not None and val != "" else 0.0
        except (ValueError, TypeError):
            row["mileage"] = 0.0

        try:
            val = data.get("engine", 0.0)
            row["engine"] = float(val) if val is not None and val != "" else 0.0
        except (ValueError, TypeError):
            row["engine"] = 0.0

        try:
            val = data.get("max_power", 0.0)
            row["max_power"] = float(val) if val is not None and val != "" else 0.0
        except (ValueError, TypeError):
            row["max_power"] = 0.0

        try:
            val = data.get("seats", 0.0)
            row["seats"] = float(val) if val is not None and val != "" else 0.0
        except (ValueError, TypeError):
            row["seats"] = 0.0

        # Frequency encoding
        row["brand_encoded"] = self.encode_brand(data.get("brand"))
        row["model_encoded"] = self.encode_model(data.get("model"))

        # One-hot encoding for fuel (CNG drops to 0.0)
        fuel_key = str(data.get("fuel", "")).strip().lower()
        fuel_col = FUEL_MAPPING.get(fuel_key)
        if fuel_col and fuel_col in row:
            row[fuel_col] = 1.0

        # One-hot encoding for seller_type (Dealer drops to 0.0)
        seller_key = str(data.get("seller_type", "")).strip().lower()
        seller_col = SELLER_TYPE_MAPPING.get(seller_key)
        if seller_col and seller_col in row:
            row[seller_col] = 1.0

        # One-hot encoding for transmission (Automatic drops to 0.0)
        trans_key = str(data.get("transmission", "")).strip().lower()
        trans_col = TRANSMISSION_MAPPING.get(trans_key)
        if trans_col and trans_col in row:
            row[trans_col] = 1.0

        return row

    def preprocess(self, payload: Mapping[str, Any] | Any) -> pd.DataFrame:
        """
        Transforms validated payload dictionary into a single-row DataFrame with exact 14 columns:
        ['km_driven', 'mileage', 'engine', 'max_power', 'seats', 'brand_encoded', 'model_encoded',
         'seller_type_Individual', 'seller_type_Trustmark Dealer',
         'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol',
         'transmission_Manual']

        Args:
            payload: Vehicle attribute dictionary or Pydantic request object.

        Returns:
            pd.DataFrame: Single-row DataFrame matching the exact 14 columns.
        """
        data = self._to_dict(payload)
        row = self._build_row(data)
        return pd.DataFrame([row], columns=self.feature_columns, dtype="float64")

    def preprocess_batch(self, payloads: Sequence[Mapping[str, Any] | Any]) -> pd.DataFrame:
        """
        Transforms a sequence of vehicle payloads into a multi-row DataFrame with exact 14 columns.

        Args:
            payloads: Sequence of vehicle attribute dictionaries.

        Returns:
            pd.DataFrame: Multi-row DataFrame matching the exact 14 columns.
        """
        rows = [self._build_row(self._to_dict(p)) for p in payloads]
        return pd.DataFrame(rows, columns=self.feature_columns, dtype="float64")

    def extract_pricing_context(self, payload: Mapping[str, Any] | Any) -> dict[str, Any]:
        """
        Extracts pricing-related parameters required by PricingService:
        - year (int)
        - mileage (float)
        - owner (str)

        Args:
            payload: Inbound vehicle attributes.

        Returns:
            dict containing typed 'year', 'mileage', and 'owner' fields.
        """
        data = self._to_dict(payload)

        try:
            year_val = int(data.get("year", DEFAULT_YEAR))
        except (ValueError, TypeError):
            year_val = DEFAULT_YEAR

        try:
            mileage_val = float(data.get("mileage", DEFAULT_MILEAGE))
        except (ValueError, TypeError):
            mileage_val = DEFAULT_MILEAGE

        owner_val = str(data.get("owner", DEFAULT_OWNER)).strip() or DEFAULT_OWNER

        return {
            "year": year_val,
            "mileage": mileage_val,
            "owner": owner_val,
        }

    def extract_pricing_inputs(self, payload: Mapping[str, Any] | Any) -> tuple[int, float, str]:
        """
        Convenience method returning (year, mileage, owner) tuple directly unpackable
        into PricingService.calculate_price(base_prediction, year, mileage, owner).
        """
        ctx = self.extract_pricing_context(payload)
        return ctx["year"], ctx["mileage"], ctx["owner"]
