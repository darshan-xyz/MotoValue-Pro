"""
Unit tests for configuration management classes and environment resolution.
"""
from pathlib import Path
import pytest
from car_price_predictor.config import (
    BaseConfig,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    get_config,
)


class TestConfigInheritance:
    """Verifies class inheritance hierarchy."""

    def test_subclasses_inherit_from_base_config(self):
        assert issubclass(DevelopmentConfig, BaseConfig)
        assert issubclass(ProductionConfig, BaseConfig)
        assert issubclass(TestingConfig, BaseConfig)


class TestConfigFlags:
    """Verifies environment-specific flag settings."""

    def test_development_config_flags(self):
        assert DevelopmentConfig.DEBUG is True
        assert DevelopmentConfig.TESTING is False
        assert DevelopmentConfig.LOG_LEVEL == "DEBUG"

    def test_production_config_flags(self):
        assert ProductionConfig.DEBUG is False
        assert ProductionConfig.TESTING is False
        assert ProductionConfig.LOG_LEVEL == "INFO"

    def test_testing_config_flags(self):
        assert TestingConfig.TESTING is True
        assert TestingConfig.DEBUG is False
        assert TestingConfig.RATELIMIT_ENABLED is False
        assert TestingConfig.LOG_LEVEL == "WARNING"
        assert TestingConfig.SECRET_KEY is not None

    def test_base_config_defaults(self):
        assert BaseConfig.HOST == "0.0.0.0"
        assert BaseConfig.PORT == 5000
        assert BaseConfig.RATELIMIT_ENABLED is True
        assert isinstance(BaseConfig.MODEL_PATH, Path)
        assert BaseConfig.SECRET_KEY is not None


class TestConfigResolution:
    """Verifies get_config() function resolution logic."""

    def test_resolve_known_environments(self):
        assert get_config("development") is DevelopmentConfig
        assert get_config("production") is ProductionConfig
        assert get_config("testing") is TestingConfig

    def test_resolve_case_insensitive(self):
        assert get_config("DEVELOPMENT") is DevelopmentConfig
        assert get_config("Production") is ProductionConfig
        assert get_config("TESTING") is TestingConfig

    def test_resolve_unknown_falls_back_to_development(self):
        assert get_config("nonexistent_env") is DevelopmentConfig
        assert get_config("") is DevelopmentConfig

    def test_resolve_none_uses_env_or_default(self, monkeypatch):
        monkeypatch.delenv("FLASK_ENV", raising=False)
        monkeypatch.delenv("FLASK_CONFIG", raising=False)
        assert get_config(None) is DevelopmentConfig

        monkeypatch.setenv("FLASK_ENV", "production")
        assert get_config(None) is ProductionConfig
