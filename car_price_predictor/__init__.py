"""
CarPricePredictor Enterprise Package.
Provides create_app application factory and package metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .config import BaseConfig, get_config
from .errors import register_error_handlers
from .services.model_service import ModelService
from .utils.logging import setup_logging
from .utils.security import init_security_headers

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


def create_app(config_name: str | type | None = None) -> Flask:
    """
    Application factory for CarPricePredictor.

    Configures template and static asset directories, binds environment
    configuration profiles, initializes logging, security headers, rate limiting,
    the ModelService singleton, registers modular blueprints (UI, API v1, Health,
    OpenAPI Docs), and attaches centralized error handlers.

    Args:
        config_name: Profile name ('development', 'production', 'testing') or BaseConfig subclass.

    Returns:
        Configured Flask application instance.
    """
    project_root = Path(__file__).resolve().parent.parent
    template_dir = project_root / "templates"
    static_dir = project_root / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    # 1. Apply Configuration
    if isinstance(config_name, type) and issubclass(config_name, BaseConfig):
        app.config.from_object(config_name)
    elif isinstance(config_name, str):
        cfg = get_config(config_name)
        app.config.from_object(cfg)
    else:
        cfg = get_config()
        app.config.from_object(cfg)

    # 2. Setup Logging
    log_level = app.config.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    # 3. Setup Security Headers
    init_security_headers(app)

    # 4. Attach ModelService Singleton
    model_path = app.config.get("MODEL_PATH", "car_price_model.joblib")
    model_svc = ModelService(model_path=model_path, load_on_init=True)
    app.model_service = model_svc
    app.extensions["model_service"] = model_svc

    # 5. Rate Limiting (Flask-Limiter)
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[app.config.get("RATELIMIT_DEFAULT", "200/day;50/hour")],
        storage_uri=app.config.get("RATELIMIT_STORAGE_URI", "memory://"),
        enabled=app.config.get("RATELIMIT_ENABLED", True),
    )
    app.extensions["limiter"] = limiter

    # 6. Register Blueprints
    from car_price_predictor.blueprints.api_v1 import api_v1_bp
    from car_price_predictor.blueprints.docs import docs_bp
    from car_price_predictor.blueprints.health import health_bp
    from car_price_predictor.blueprints.ui import ui_bp

    app.register_blueprint(ui_bp)
    app.register_blueprint(api_v1_bp)
    # Register docs_bp without url_prefix so that /openapi.json and /apidocs match exactly
    app.register_blueprint(docs_bp)
    # Register health_bp (/health/live and /health/ready)
    app.register_blueprint(health_bp)

    # 7. Custom Error Handlers
    register_error_handlers(app)

    return app


__all__ = ["create_app", "__version__"]
