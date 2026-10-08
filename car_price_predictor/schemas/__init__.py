"""
Schemas module for CarPricePredictor.
Provides Pydantic v2 schemas for request validation, structured responses, and error payloads.
"""
from .prediction import (
    ConfidenceInterval,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureContribution,
    LegacyPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    PricingAdjustmentsBreakdown,
    ValidationErrorDetail,
)

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
