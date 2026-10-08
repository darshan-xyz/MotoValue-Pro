"""
OpenAPI 3.0.3 specification and Swagger UI routes for CarPricePredictor.
Provides interactive API documentation without heavy external dependencies.
"""
from __future__ import annotations

from typing import Any
from flask import Blueprint, Response, jsonify

from car_price_predictor.schemas.prediction import (
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

docs_bp = Blueprint("docs", __name__)

OPENAPI_SPEC: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {
        "title": "CarPricePredictor Enterprise API",
        "version": "1.0.0",
        "description": (
            "Production-grade enterprise ML application API for vehicle valuation and price prediction. "
            "Features strict Pydantic validation, domain pricing rules, and health monitoring."
        ),
        "contact": {
            "name": "CarPricePredictor Engineering Team",
        },
        "license": {
            "name": "Proprietary / Internal",
        },
    },
    "servers": [
        {
            "url": "/",
            "description": "Current Server",
        }
    ],
    "tags": [
        {
            "name": "Prediction",
            "description": "Vehicle resale price valuation endpoints",
        },
        {
            "name": "Health",
            "description": "Application liveness and readiness monitoring probes",
        },
        {
            "name": "UI",
            "description": "Interactive web interface endpoints",
        },
        {
            "name": "Documentation",
            "description": "API contract and interactive documentation",
        },
    ],
    "paths": {
        "/api/v1/predict": {
            "post": {
                "tags": ["Prediction"],
                "summary": "Predict Car Resale Price (v1 API)",
                "description": (
                    "Primary versioned prediction endpoint. Validates payload via Pydantic, "
                    "encodes 14 model features, performs regression inference, applies pricing rules, "
                    "and returns a structured valuation response."
                ),
                "operationId": "predictPriceV1",
                "requestBody": {
                    "required": True,
                    "description": "Vehicle specifications for valuation",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/PredictionRequest"
                            },
                            "examples": {
                                "marutiSwift": {
                                    "summary": "Maruti Swift 2019 (Golden Benchmark TC-01)",
                                    "value": {
                                        "brand": "Maruti",
                                        "model": "Swift",
                                        "year": 2019,
                                        "km_driven": 45000,
                                        "mileage": 18.5,
                                        "engine": 1197.0,
                                        "max_power": 82.0,
                                        "seats": 5,
                                        "fuel": "Petrol",
                                        "transmission": "Manual",
                                        "seller_type": "Individual",
                                        "owner": "First Owner",
                                    },
                                },
                                "bmw5Series": {
                                    "summary": "BMW 5 Series 2021 Luxury Sedan (TC-02)",
                                    "value": {
                                        "brand": "BMW",
                                        "model": "5",
                                        "year": 2021,
                                        "km_driven": 25000,
                                        "mileage": 17.4,
                                        "engine": 1995.0,
                                        "max_power": 190.0,
                                        "seats": 5,
                                        "fuel": "Diesel",
                                        "transmission": "Automatic",
                                        "seller_type": "Dealer",
                                        "owner": "First Owner",
                                    },
                                },
                            },
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Successful valuation",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PredictionResponse"
                                },
                                "examples": {
                                    "swiftResponse": {
                                        "value": {
                                            "price": "508,560.15",
                                            "valuation": 508560.15,
                                            "currency": "INR",
                                            "status": "success",
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "400": {
                        "description": "Validation error or invalid request payload",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                                "examples": {
                                    "invalidYear": {
                                        "value": {
                                            "error": "Validation Error",
                                            "details": [
                                                {
                                                    "field": "year",
                                                    "message": "Input should be greater than or equal to 1980",
                                                    "type": "greater_than_equal",
                                                }
                                            ],
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "415": {
                        "description": "Unsupported media type (Content-Type must be application/json)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                            }
                        },
                    },
                    "503": {
                        "description": "Service unavailable (ModelService not ready)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                            }
                        },
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/api/v1/explain": {
            "post": {
                "tags": ["Prediction"],
                "summary": "Explain Car Resale Price Valuation (R3)",
                "description": (
                    "Decomposes vehicle price valuation into individual feature drivers "
                    "(ML model feature contributions and domain pricing heuristics), "
                    "providing uncertainty confidence intervals and percentage impacts."
                ),
                "operationId": "explainValuationV1",
                "requestBody": {
                    "required": True,
                    "description": "Vehicle specifications for valuation explanation",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ExplainRequest"
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Valuation explanation breakdown",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ExplainResponse"
                                }
                            }
                        },
                    },
                    "400": {
                        "description": "Validation error or invalid request payload",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                    "415": {
                        "description": "Unsupported media type (Content-Type must be application/json)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                            }
                        },
                    },
                    "503": {
                        "description": "Service unavailable (ModelService not ready)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                            }
                        },
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/predict": {
            "post": {
                "tags": ["Prediction"],
                "summary": "Predict Car Resale Price (Legacy Alias)",
                "description": (
                    "Backward-compatible legacy endpoint returning minimal JSON with 'price' field. "
                    "Ensures seamless operation of existing prototype clients."
                ),
                "operationId": "predictPriceLegacy",
                "requestBody": {
                    "required": True,
                    "description": "Vehicle specifications for valuation",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/PredictionRequest"
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Successful valuation (legacy format)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/LegacyPredictionResponse"
                                },
                                "examples": {
                                    "legacyResponse": {
                                        "value": {
                                            "price": "508,560.15"
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "400": {
                        "description": "Validation error or invalid request payload",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                    "415": {
                        "description": "Unsupported media type (Content-Type must be application/json)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                            }
                        },
                    },
                    "503": {
                        "description": "Service unavailable (ModelService not ready)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                            }
                        },
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/health/live": {
            "get": {
                "tags": ["Health"],
                "summary": "Process Liveness Probe",
                "description": "Verifies web process liveness for orchestrators and container engines.",
                "operationId": "healthLive",
                "responses": {
                    "200": {
                        "description": "Process is alive and responding",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthLiveResponse"
                                },
                                "examples": {
                                    "live": {
                                        "value": {
                                            "status": "alive"
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
            }
        },
        "/health/ready": {
            "get": {
                "tags": ["Health"],
                "summary": "Service Readiness Probe",
                "description": "Verifies ML model artifact and dependencies are loaded and ready to serve traffic.",
                "operationId": "healthReady",
                "responses": {
                    "200": {
                        "description": "Service is ready for inference",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthReadyResponse"
                                },
                                "examples": {
                                    "ready": {
                                        "value": {
                                            "status": "ready",
                                            "model_loaded": True,
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "503": {
                        "description": "Service is not ready (model unloaded or error)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/": {
            "get": {
                "tags": ["UI"],
                "summary": "Car Valuation Web UI",
                "description": "Serves interactive HTML web interface with brand and model selection.",
                "operationId": "renderIndexPage",
                "responses": {
                    "200": {
                        "description": "HTML web application page",
                        "content": {
                            "text/html": {
                                "schema": {
                                    "type": "string"
                                }
                            }
                        },
                    }
                },
            }
        },
        "/openapi.json": {
            "get": {
                "tags": ["Documentation"],
                "summary": "OpenAPI Specification JSON",
                "description": "Returns raw OpenAPI 3.0.3 specification in JSON format.",
                "operationId": "getOpenApiSpec",
                "responses": {
                    "200": {
                        "description": "OpenAPI 3.0.3 specification",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object"
                                }
                            }
                        },
                    }
                },
            }
        },
        "/apidocs": {
            "get": {
                "tags": ["Documentation"],
                "summary": "Interactive Swagger UI",
                "description": "Serves interactive Swagger UI HTML client documenting all API endpoints.",
                "operationId": "getSwaggerUI",
                "responses": {
                    "200": {
                        "description": "Swagger UI HTML page",
                        "content": {
                            "text/html": {
                                "schema": {
                                    "type": "string"
                                }
                            }
                        },
                    }
                },
            }
        },
    },
    "components": {
        "schemas": {
            "PredictionRequest": PredictionRequest.model_json_schema(),
            "ExplainRequest": ExplainRequest.model_json_schema(),
            "PredictionResponse": PredictionResponse.model_json_schema(),
            "LegacyPredictionResponse": LegacyPredictionResponse.model_json_schema(),
            "ConfidenceInterval": ConfidenceInterval.model_json_schema(),
            "FeatureContribution": FeatureContribution.model_json_schema(),
            "PricingAdjustmentsBreakdown": PricingAdjustmentsBreakdown.model_json_schema(),
            "ExplainResponse": ExplainResponse.model_json_schema(),
            "ValidationErrorDetail": ValidationErrorDetail.model_json_schema(),
            "ErrorResponse": ErrorResponse.model_json_schema(),
            "HealthLiveResponse": {
                "type": "object",
                "required": ["status"],
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Process liveness status",
                        "example": "alive",
                    }
                },
            },
            "HealthReadyResponse": {
                "type": "object",
                "required": ["status", "model_loaded"],
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Service readiness status",
                        "example": "ready",
                    },
                    "model_loaded": {
                        "type": "boolean",
                        "description": "True if ML model weights are loaded and ready for predictions",
                        "example": True,
                    },
                },
            },
        }
    },
}

SWAGGER_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CarPricePredictor Enterprise API Documentation</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui.min.css" />
    <link rel="icon" type="image/png" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/favicon-32x32.png" sizes="32x32" />
    <style>
        html { box-sizing: border-box; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .topbar { display: none; }
        .swagger-ui .info { margin: 25px 0; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-standalone-preset.min.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout",
                docExpansion: "list",
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1
            });
        };
    </script>
</body>
</html>
"""


@docs_bp.route("/openapi.json", methods=["GET"])
@docs_bp.route("/apidocs/openapi.json", methods=["GET"])
def openapi_json():
    """Returns the OpenAPI 3.0.3 specification in JSON format."""
    return jsonify(OPENAPI_SPEC)


@docs_bp.route("/apidocs", methods=["GET"])
@docs_bp.route("/apidocs/", methods=["GET"])
@docs_bp.route("/docs", methods=["GET"])
@docs_bp.route("/docs/", methods=["GET"])
def apidocs():
    """Serves the interactive Swagger UI page."""
    return Response(SWAGGER_UI_HTML, mimetype="text/html")
