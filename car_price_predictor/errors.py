"""
Custom HTTP and validation error handlers for Flask application.
Ensures structured JSON responses for API routes and sensible defaults for HTML routes.
"""
from __future__ import annotations

import logging
from typing import Any
from flask import Flask, jsonify, request
from pydantic import ValidationError
from werkzeug.exceptions import (
    BadRequest,
    HTTPException,
    InternalServerError,
    MethodNotAllowed,
    NotFound,
    UnsupportedMediaType,
)

from car_price_predictor.schemas.prediction import ErrorResponse

logger = logging.getLogger(__name__)


def is_api_request() -> bool:
    """
    Determines if the current request expects a structured JSON response.
    """
    path = request.path
    if (
        path.startswith(("/api/", "/health", "/predict"))
        or path.endswith((".json",))
        or path in ("/predict", "/openapi.json")
    ):
        return True
    if request.is_json:
        return True
    accept = request.headers.get("Accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return True
    return False


def register_error_handlers(app: Flask) -> None:
    """
    Registers application-wide error handlers on the provided Flask app instance.
    """

    @app.errorhandler(ValidationError)
    def handle_pydantic_validation_error(exc: ValidationError) -> Any:
        """Handles uncaught Pydantic validation errors returning structured HTTP 400."""
        logger.warning("Pydantic validation error on %s: %s", request.path, exc)
        payload = ErrorResponse.from_validation_error(exc)
        return jsonify(payload.model_dump()), 400

    @app.errorhandler(400)
    @app.errorhandler(BadRequest)
    def handle_bad_request(exc: Exception) -> Any:
        """Handles HTTP 400 Bad Request."""
        if is_api_request():
            desc = getattr(exc, "description", "Bad Request")
            return (
                jsonify(
                    {
                        "error": "Validation Error",
                        "details": [
                            {
                                "field": "body",
                                "message": str(desc),
                                "type": "bad_request",
                            }
                        ],
                    }
                ),
                400,
            )
        return getattr(exc, "description", "Bad Request"), 400

    @app.errorhandler(404)
    @app.errorhandler(NotFound)
    def handle_not_found(exc: Exception) -> Any:
        """Handles HTTP 404 Not Found."""
        if is_api_request():
            return (
                jsonify(
                    {
                        "error": "Not Found",
                        "message": "The requested resource was not found on this server.",
                    }
                ),
                404,
            )
        return "Resource Not Found", 404

    @app.errorhandler(405)
    @app.errorhandler(MethodNotAllowed)
    def handle_method_not_allowed(exc: Exception) -> Any:
        """Handles HTTP 405 Method Not Allowed."""
        if is_api_request():
            return (
                jsonify(
                    {
                        "error": "Method Not Allowed",
                        "message": f"The method {request.method} is not allowed for the requested URL.",
                    }
                ),
                405,
            )
        return f"Method {request.method} Not Allowed", 405

    @app.errorhandler(415)
    @app.errorhandler(UnsupportedMediaType)
    def handle_unsupported_media_type(exc: Exception) -> Any:
        """Handles HTTP 415 Unsupported Media Type."""
        if is_api_request():
            return (
                jsonify(
                    {
                        "error": "Unsupported Media Type",
                        "message": "Content-Type must be application/json.",
                    }
                ),
                415,
            )
        return "Content-Type must be application/json", 415

    @app.errorhandler(500)
    @app.errorhandler(InternalServerError)
    def handle_internal_server_error(exc: Exception) -> Any:
        """Handles HTTP 500 Internal Server Error."""
        logger.exception("Internal server error on %s %s: %s", request.method, request.path, exc)
        if is_api_request():
            return (
                jsonify(
                    {
                        "error": "Internal Server Error",
                        "message": "An unexpected server error occurred. Please try again later.",
                    }
                ),
                500,
            )
        return "Internal Server Error", 500

    @app.errorhandler(Exception)
    def handle_unhandled_exception(exc: Exception) -> Any:
        """Fallback handler for uncaught Python exceptions."""
        if isinstance(exc, HTTPException):
            if is_api_request():
                return (
                    jsonify(
                        {
                            "error": exc.name,
                            "message": exc.description,
                        }
                    ),
                    exc.code or 500,
                )
            return exc

        logger.exception("Unhandled exception on %s %s: %s", request.method, request.path, exc)
        if is_api_request():
            return (
                jsonify(
                    {
                        "error": "Internal Server Error",
                        "message": "An unexpected server error occurred.",
                    }
                ),
                500,
            )
        return "Internal Server Error", 500
