"""
Health check blueprint exposing liveness and readiness probes.
"""
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__, url_prefix="/health")


@health_bp.route("/live", methods=["GET"])
def health_live():
    """
    Process liveness check probe.
    Returns HTTP 200 OK if process is responding.
    """
    return jsonify({"status": "alive"}), 200


@health_bp.route("/ready", methods=["GET"])
def health_ready():
    """
    Model and service readiness check probe.
    Returns HTTP 200 OK if model is loaded and ready, or HTTP 503 if not.
    """
    model_service = getattr(current_app, "model_service", None)
    is_ready = bool(model_service and model_service.is_ready())

    if is_ready:
        return jsonify({
            "status": "ready",
            "model_loaded": True
        }), 200

    return jsonify({
        "status": "not_ready",
        "model_loaded": False
    }), 503
