"""
Pydantic v2 schemas for prediction endpoints and error payloads.
Provides strict validation, boundary constraints, and response contracts.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from car_price_predictor.services.constants import (
    DEFAULT_CURRENCY,
    ENGINE_MAX,
    ENGINE_MIN,
    KM_DRIVEN_MIN,
    MAX_POWER_MAX,
    MAX_POWER_MIN,
    MILEAGE_MAX,
    MILEAGE_MIN,
    SEATS_MAX,
    SEATS_MIN,
    VALID_FUELS,
    VALID_OWNERS,
    VALID_SELLER_TYPES,
    VALID_TRANSMISSIONS,
    YEAR_MAX,
    YEAR_MIN,
)

# Canonical case-insensitive lookup mappings
_FUEL_CANONICAL = {f.lower(): f for f in VALID_FUELS}
_TRANSMISSION_CANONICAL = {t.lower(): t for t in VALID_TRANSMISSIONS}
_SELLER_TYPE_CANONICAL = {s.lower(): s for s in VALID_SELLER_TYPES}
_OWNER_CANONICAL = {o.lower(): o for o in VALID_OWNERS}
_OWNER_CANONICAL["fourth and above owner"] = "Fourth & Above Owner"


class PredictionRequest(BaseModel):
    """
    Strict validation model for incoming vehicle prediction requests.
    Validates types, numerical boundaries, and allowable categorical values.
    Ignores extra metadata fields (e.g., 'id', 'description').
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    brand: str = Field(
        ...,
        min_length=1,
        description="Vehicle manufacturer brand name (e.g., Maruti, Hyundai, BMW)",
        examples=["Maruti"],
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Vehicle model name (e.g., Swift, City, Innova)",
        examples=["Swift"],
    )
    year: int = Field(
        ...,
        ge=YEAR_MIN,
        le=YEAR_MAX,
        description=f"Manufacturing year between {YEAR_MIN} and {YEAR_MAX}",
        examples=[2019],
    )
    km_driven: float = Field(
        ...,
        ge=KM_DRIVEN_MIN,
        description=f"Total distance driven in kilometers (>= {KM_DRIVEN_MIN})",
        examples=[45000.0],
    )
    mileage: float = Field(
        ...,
        ge=MILEAGE_MIN,
        le=MILEAGE_MAX,
        description=f"Fuel economy in kmpl between {MILEAGE_MIN} and {MILEAGE_MAX}",
        examples=[18.5],
    )
    engine: float = Field(
        ...,
        ge=ENGINE_MIN,
        le=ENGINE_MAX,
        description=f"Engine displacement in CC between {ENGINE_MIN} and {ENGINE_MAX}",
        examples=[1197.0],
    )
    max_power: float = Field(
        ...,
        ge=MAX_POWER_MIN,
        le=MAX_POWER_MAX,
        description=f"Engine maximum power in bhp between {MAX_POWER_MIN} and {MAX_POWER_MAX}",
        examples=[82.0],
    )
    seats: int = Field(
        ...,
        ge=SEATS_MIN,
        le=SEATS_MAX,
        description=f"Passenger seating capacity between {SEATS_MIN} and {SEATS_MAX}",
        examples=[5],
    )
    fuel: Literal["Petrol", "Diesel", "CNG", "LPG", "Electric"] = Field(
        ...,
        description="Fuel type: Petrol, Diesel, CNG, LPG, or Electric",
        examples=["Petrol"],
    )
    transmission: Literal["Manual", "Automatic"] = Field(
        ...,
        description="Transmission type: Manual or Automatic",
        examples=["Manual"],
    )
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"] = Field(
        ...,
        description="Seller classification: Individual, Dealer, or Trustmark Dealer",
        examples=["Individual"],
    )
    owner: Literal[
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
    ] = Field(
        ...,
        description="Vehicle ownership history tier",
        examples=["First Owner"],
    )

    # -------------------------------------------------------------------------
    # Pre-validation coercion and normalization hooks
    # -------------------------------------------------------------------------

    @field_validator("brand", "model", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("Field cannot be empty or whitespace only")
            return cleaned
        if v is None:
            raise ValueError("Field cannot be null")
        return v

    @field_validator("year", "seats", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            try:
                f = float(v)
                if f.is_integer():
                    return int(f)
                return f
            except (ValueError, TypeError):
                return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    @field_validator("km_driven", "mileage", "engine", "max_power", mode="before")
    @classmethod
    def _coerce_float(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            try:
                return float(v)
            except (ValueError, TypeError):
                return v
        return v

    @field_validator("fuel", mode="before")
    @classmethod
    def _normalize_fuel(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in _FUEL_CANONICAL:
                return _FUEL_CANONICAL[cleaned]
        return v

    @field_validator("transmission", mode="before")
    @classmethod
    def _normalize_transmission(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in _TRANSMISSION_CANONICAL:
                return _TRANSMISSION_CANONICAL[cleaned]
        return v

    @field_validator("seller_type", mode="before")
    @classmethod
    def _normalize_seller_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in _SELLER_TYPE_CANONICAL:
                return _SELLER_TYPE_CANONICAL[cleaned]
        return v

    @field_validator("owner", mode="before")
    @classmethod
    def _normalize_owner(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in _OWNER_CANONICAL:
                return _OWNER_CANONICAL[cleaned]
        return v


class ExplainRequest(PredictionRequest):
    """
    Dedicated request payload schema for POST /api/v1/explain.
    Shares validation constraints and coercion rules with PredictionRequest.
    """
    pass


class ConfidenceInterval(BaseModel):
    """
    Structured uncertainty bounds container representing empirical ensemble distribution.
    """

    model_config = ConfigDict(populate_by_name=True)

    lower_bound: float = Field(
        ...,
        ge=0.0,
        description="Estimated lower bound market price in INR",
        examples=[311298.75],
    )
    upper_bound: float = Field(
        ...,
        ge=0.0,
        description="Estimated upper bound market price in INR",
        examples=[664807.50],
    )
    formatted_lower: str = Field(
        ...,
        description="Formatted lower bound price string",
        examples=["311,298.75"],
    )
    formatted_upper: str = Field(
        ...,
        description="Formatted upper bound price string",
        examples=["664,807.50"],
    )
    confidence_level: float = Field(
        default=0.80,
        description="Nominal confidence coverage probability (e.g. 0.80 for 10th-90th percentiles)",
        examples=[0.80],
    )
    method: str = Field(
        default="ensemble_tree_percentiles",
        description="Algorithm used for interval estimation",
        examples=["ensemble_tree_percentiles"],
    )


class FeatureContribution(BaseModel):
    """
    Local attribution record for a single vehicle feature.
    """

    model_config = ConfigDict(populate_by_name=True)

    feature: str = Field(..., description="Logical feature name", examples=["max_power"])
    value: Any = Field(..., description="Submitted attribute value", examples=[82.0])
    impact_amount: float = Field(
        ...,
        description="Estimated dollar impact on final valuation in INR",
        examples=[192123.33],
    )
    percentage_impact: float = Field(
        ...,
        description="Relative percentage impact on final valuation",
        examples=[37.78],
    )
    direction: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Direction of impact on vehicle price",
        examples=["positive"],
    )
    source: Literal["ml_model", "pricing_rule"] = Field(
        ...,
        description="Subsystem generating the impact",
        examples=["ml_model"],
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of feature effect",
        examples=["High engine power increases vehicle valuation"],
    )
    importance: float | None = Field(
        default=None,
        description="Normalized relative importance score",
        examples=[0.6516],
    )


class PricingAdjustmentsBreakdown(BaseModel):
    """
    Breakdown of post-ML domain pricing heuristic multipliers.
    """

    model_config = ConfigDict(populate_by_name=True)

    owner_multiplier: float = Field(
        ...,
        description="Ownership history multiplier",
        examples=[1.0],
    )
    year_factor: float = Field(
        ...,
        description="Vehicle age adjustment factor",
        examples=[1.05],
    )
    mileage_factor: float = Field(
        ...,
        description="Fuel efficiency adjustment factor",
        examples=[1.005],
    )
    total_heuristic_multiplier: float = Field(
        ...,
        description="Combined multiplicative pricing factor",
        examples=[1.05525],
    )
    net_adjustment_amount: float = Field(
        ...,
        description="Net price change resulting from heuristic adjustments",
        examples=[26626.82],
    )


class PredictionResponse(BaseModel):
    """
    Structured response contract for POST /api/v1/predict.
    """

    model_config = ConfigDict(populate_by_name=True)

    price: str = Field(
        ...,
        description="Formatted valuation string with commas and 2 decimal places",
        examples=["508,560.15"],
    )
    valuation: float = Field(
        ...,
        description="Numerical vehicle valuation in currency units",
        examples=[508560.15],
    )
    currency: str = Field(
        default=DEFAULT_CURRENCY,
        description="Three-letter ISO currency code",
        examples=["INR"],
    )
    status: str = Field(
        default="success",
        description="Response status indicator",
        examples=["success"],
    )
    confidence_interval: ConfidenceInterval | None = Field(
        default=None,
        description="Estimated market price confidence bounds",
    )


class ExplainResponse(BaseModel):
    """
    Dedicated response contract for POST /api/v1/explain.
    """

    model_config = ConfigDict(populate_by_name=True)

    price: str = Field(
        ...,
        description="Formatted valuation string with commas and 2 decimal places",
        examples=["508,560.15"],
    )
    valuation: float = Field(
        ...,
        description="Final vehicle valuation in INR",
        examples=[508560.15],
    )
    currency: str = Field(
        default=DEFAULT_CURRENCY,
        description="Three-letter ISO currency code",
        examples=["INR"],
    )
    base_ml_valuation: float = Field(
        ...,
        description="Raw ML model prediction before heuristic adjustments",
        examples=[481933.33],
    )
    base_ml_price: float | None = Field(
        default=None,
        description="Alias for base_ml_valuation",
        examples=[481933.33],
    )
    confidence_interval: ConfidenceInterval = Field(
        ...,
        description="Uncertainty bounds on vehicle market valuation",
    )
    feature_contributions: list[FeatureContribution] = Field(
        ...,
        description="Attribution breakdown across individual vehicle features",
    )
    top_drivers: list[str] = Field(
        default_factory=list,
        description="Names of top positive and negative price driving attributes",
    )
    pricing_adjustments: PricingAdjustmentsBreakdown = Field(
        ...,
        description="Deterministic post-model pricing rule adjustments",
    )
    global_feature_importances: dict[str, float] = Field(
        default_factory=dict,
        description="Global model feature Gini importances",
    )
    status: str = Field(
        default="success",
        description="Response status indicator",
        examples=["success"],
    )


class LegacyPredictionResponse(BaseModel):
    """
    Backward-compatible response schema for legacy POST /predict alias.
    """

    model_config = ConfigDict(populate_by_name=True)

    price: str = Field(
        ...,
        description="Formatted valuation string with commas and 2 decimal places",
        examples=["508,560.15"],
    )


class ValidationErrorDetail(BaseModel):
    """
    Single validation failure detail.
    """

    field: str = Field(
        ...,
        description="Name of the payload attribute that failed validation",
        examples=["year"],
    )
    message: str = Field(
        ...,
        description="Human-readable description of the constraint failure",
        examples=["Input should be greater than or equal to 1980"],
    )
    type: str = Field(
        ...,
        description="Machine-readable error type indicator",
        examples=["greater_than_equal"],
    )


class ErrorResponse(BaseModel):
    """
    Structured 400 Bad Request error response container.
    """

    error: str = Field(
        default="Validation Error",
        description="High-level error classification",
        examples=["Validation Error"],
    )
    details: list[ValidationErrorDetail] = Field(
        default_factory=list,
        description="List of specific field validation failures",
    )

    @classmethod
    def from_validation_error(cls, exc: ValidationError) -> ErrorResponse:
        """
        Converts a Pydantic ValidationError into a structured ErrorResponse.
        """
        details: list[ValidationErrorDetail] = []
        for err in exc.errors():
            loc = err.get("loc", ())
            field_name = str(loc[-1]) if loc else "payload"
            details.append(
                ValidationErrorDetail(
                    field=field_name,
                    message=err.get("msg", "Invalid value"),
                    type=err.get("type", "value_error"),
                )
            )
        return cls(error="Validation Error", details=details)


__all__ = [
    "ConfidenceInterval",
    "ErrorResponse",
    "ExplainRequest",
    "ExplainResponse",
    "FeatureContribution",
    "LegacyPredictionResponse",
    "PredictionRequest",
    "PredictionResponse",
    "PricingAdjustmentsBreakdown",
    "ValidationErrorDetail",
]
