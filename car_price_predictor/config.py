"""
Configuration module for CarPricePredictor.
Provides BaseConfig and environment-specific subclasses.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Type

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


class BaseConfig:
    """Base configuration with shared defaults."""
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-car-price-predictor")

    # Model Artifact Path
    _raw_model_path: str = os.getenv("MODEL_PATH", "car_price_model.joblib")
    MODEL_PATH: Path = (
        Path(_raw_model_path) if Path(_raw_model_path).is_absolute()
        else BASE_DIR / _raw_model_path
    )

    # Server Defaults
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))

    # Rate Limiting
    RATELIMIT_ENABLED: bool = True
    RATELIMIT_DEFAULT: str = os.getenv("RATELIMIT_DEFAULT", "200/day;50/hour")
    RATELIMIT_PREDICT: str = os.getenv("RATELIMIT_PREDICT", "60/minute")
    RATELIMIT_STORAGE_URI: str = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

    # Observability
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    TESTING: bool = False
    DEBUG: bool = False


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    DEBUG: bool = True
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")


class ProductionConfig(BaseConfig):
    """Production environment configuration with strict defaults."""
    DEBUG: bool = False
    TESTING: bool = False
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


class TestingConfig(BaseConfig):
    """Testing environment configuration for automated pytest runs."""
    TESTING: bool = True
    DEBUG: bool = False
    RATELIMIT_ENABLED: bool = False
    LOG_LEVEL: str = "WARNING"
    SECRET_KEY: str = "test-secret-key-testing"


CONFIG_MAP: dict[str, Type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config(config_name: str | None = None) -> Type[BaseConfig]:
    """Resolves and returns configuration class by name or environment variable."""
    if not config_name:
        config_name = os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG") or "development"
    return CONFIG_MAP.get(config_name.lower(), DevelopmentConfig)
