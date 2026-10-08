"""
UI Blueprint Routes.
Serves web interface rendering templates/index.html with brand and model hierarchy.
"""
from __future__ import annotations

import json
import logging
from flask import Blueprint, current_app, render_template

logger = logging.getLogger(__name__)

ui_bp = Blueprint("ui", __name__)


@ui_bp.route("/", methods=["GET"])
def home():
    """Renders main appraisal UI with brands and car hierarchy data."""
    model_service = getattr(current_app, "model_service", None)

    if model_service and model_service.is_ready():
        brands = model_service.get_brands()
        if hasattr(model_service, "get_brand_to_models"):
            hierarchy = model_service.get_brand_to_models()
        elif hasattr(model_service, "get_hierarchy"):
            hierarchy = model_service.get_hierarchy()
        else:
            hierarchy = getattr(model_service, "brand_to_models", {})
    else:
        logger.warning("ModelService not ready when rendering UI")
        brands = []
        hierarchy = {}

    return render_template(
        "index.html",
        brands=brands,
        car_hierarchy=json.dumps(hierarchy),
    )
