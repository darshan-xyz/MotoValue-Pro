"""
API v1 Blueprint Routes.
Provides versioned inference endpoint /api/v1/predict and legacy alias /predict.
"""
from __future__ import annotations

import logging
from typing import Any
from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from car_price_predictor.schemas.prediction import (
    ConfidenceInterval,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    LegacyPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from car_price_predictor.services.explainability import ExplainabilityService
from car_price_predictor.services.pricing import PricingService

logger = logging.getLogger(__name__)

api_v1_bp = Blueprint("api_v1", __name__)


def _process_prediction(is_legacy: bool = False) -> tuple[Any, int]:
    """
    Core validation and inference pipeline for prediction requests.
    Returns (response_json_or_dict, http_status_code).
    """
    # 1. Content-Type inspection
    content_type = request.content_type or ""
    if content_type and "application/json" not in content_type:
        return (
            jsonify(
                {
                    "error": "Unsupported Media Type",
                    "message": "Content-Type must be application/json.",
                }
            ),
            415,
        )

    # 2. Extract JSON payload
    raw_payload = request.get_json(silent=True)
    if not isinstance(raw_payload, dict):
        return (
            jsonify(
                {
                    "error": "Validation Error",
                    "details": [
                        {
                            "field": "body",
                            "message": "Request payload must be a JSON object",
                            "type": "type_error",
                        }
                    ],
                }
            ),
            400,
        )

    # 3. Pydantic schema validation
    try:
        req = PredictionRequest.model_validate(raw_payload)
    except ValidationError as exc:
        error_resp = ErrorResponse.from_validation_error(exc)
        return jsonify(error_resp.model_dump()), 400

    # 4. ModelService readiness check
    model_service = getattr(current_app, "model_service", None)
    if not model_service or not model_service.is_ready():
        logger.error("Inference attempted but ModelService is not ready")
        return (
            jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model service is not loaded or ready for inference",
                }
            ),
            503,
        )

    # 5. Preprocessing & ML Model Prediction
    try:
        input_df = model_service.preprocessor.preprocess(req)
        year, mileage, owner = model_service.preprocessor.extract_pricing_inputs(req)

        # 6. Formulate response contract
        if is_legacy:
            base_prediction = model_service.predict(input_df)
            final_price = PricingService.calculate_price(
                base_prediction=base_prediction,
                year=year,
                mileage=mileage,
                owner=owner,
            )
            formatted_price = PricingService.format_price(final_price)
            legacy_resp = LegacyPredictionResponse(price=formatted_price)
            return jsonify(legacy_resp.model_dump()), 200

        # Versioned v1 response with empirical confidence intervals
        base_prediction, lower_base, upper_base = model_service.predict_with_interval(
            input_df, confidence_level=0.80
        )
        final_price, lower_bound, upper_bound = PricingService.calculate_price_interval(
            base_prediction=base_prediction,
            lower_base=lower_base,
            upper_base=upper_base,
            year=year,
            mileage=mileage,
            owner=owner,
        )
        formatted_price = PricingService.format_price(final_price)
        conf_interval = ConfidenceInterval(
            lower_bound=round(lower_bound, 2),
            upper_bound=round(upper_bound, 2),
            formatted_lower=PricingService.format_price(lower_bound),
            formatted_upper=PricingService.format_price(upper_bound),
            confidence_level=0.80,
            method="ensemble_tree_percentiles",
        )

        resp = PredictionResponse(
            price=formatted_price,
            valuation=round(float(final_price), 2),
            currency="INR",
            status="success",
            confidence_interval=conf_interval,
        )
        return jsonify(resp.model_dump()), 200

    except Exception as exc:
        logger.exception("Prediction calculation error: %s", exc)
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": f"Prediction failed: {exc}",
                }
            ),
            500,
        )


@api_v1_bp.route("/api/v1/predict", methods=["POST"])
def predict_v1():
    """Versioned prediction endpoint returning full structured contract."""
    return _process_prediction(is_legacy=False)


@api_v1_bp.route("/predict", methods=["POST"])
def predict_legacy():
    """Backward-compatible legacy endpoint returning simple {'price': '...'}."""
    return _process_prediction(is_legacy=True)


@api_v1_bp.route("/api/v1/explain", methods=["POST"])
def explain_v1():
    """
    Feature importance and explainability endpoint (R3).
    Returns driver contributions, pricing adjustments breakdown, and confidence intervals.
    """
    # 1. Content-Type inspection
    content_type = request.content_type or ""
    if content_type and "application/json" not in content_type:
        return (
            jsonify(
                {
                    "error": "Unsupported Media Type",
                    "message": "Content-Type must be application/json.",
                }
            ),
            415,
        )

    # 2. Extract JSON payload
    raw_payload = request.get_json(silent=True)
    if not isinstance(raw_payload, dict):
        return (
            jsonify(
                {
                    "error": "Validation Error",
                    "details": [
                        {
                            "field": "body",
                            "message": "Request payload must be a JSON object",
                            "type": "type_error",
                        }
                    ],
                }
            ),
            400,
        )

    # 3. Pydantic schema validation
    try:
        req = ExplainRequest.model_validate(raw_payload)
    except ValidationError as exc:
        error_resp = ErrorResponse.from_validation_error(exc)
        return jsonify(error_resp.model_dump()), 400

    # 4. ModelService readiness check
    model_service = getattr(current_app, "model_service", None)
    if not model_service or not model_service.is_ready():
        logger.error("Explainability attempted but ModelService is not ready")
        return (
            jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model service is not loaded or ready for inference",
                }
            ),
            503,
        )

    # 5. Explainability analysis
    try:
        explain_service = ExplainabilityService(model_service)
        result = explain_service.explain_prediction(req)
        return jsonify(result.model_dump()), 200
    except Exception as exc:
        logger.exception("Explainability calculation error: %s", exc)
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": f"Explainability failed: {exc}",
                }
            ),
            500,
        )
