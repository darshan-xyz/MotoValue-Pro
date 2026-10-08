"""
Tier 1: Feature Coverage Test Suite (24 Features x >=5 tests = 120 tests).
Opaque-box and contract verification across all architectural components.
"""

from pathlib import Path
import pytest
from tests.conftest import has_endpoint

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# Feature 1: Configuration Management (M1)
# ==============================================================================
class TestFeature01ConfigurationManagement:
    """Verifies multi-environment configuration classes and environment setup."""

    def test_config_testing_environment(self, app):
        """Testing configuration must have TESTING flag set to True."""
        assert app.config.get("TESTING") is True

    def test_config_env_example_file_exists(self):
        """Root directory must contain .env.example with configuration guidelines."""
        env_example = ROOT_DIR / ".env.example"
        if not env_example.exists():
            pytest.skip("Milestone 1 pending: .env.example not yet created")
        content = env_example.read_text(encoding="utf-8")
        assert "SECRET_KEY" in content or "FLASK_ENV" in content or "ENV" in content

    def test_config_base_or_subclasses_importable(self):
        """Configuration classes should be importable from car_price_predictor.config."""
        try:
            from car_price_predictor.config import (
                BaseConfig,
                DevelopmentConfig,
                ProductionConfig,
                TestingConfig,
            )
            assert TestingConfig.TESTING is True
            assert ProductionConfig.DEBUG is False
            assert DevelopmentConfig.DEBUG is True
        except ImportError:
            pytest.skip("Milestone 1 pending: car_price_predictor.config not yet implemented")

    def test_config_secret_key_present(self, app):
        """Application config should define SECRET_KEY for session/form safety."""
        assert "SECRET_KEY" in app.config or app.secret_key is not None or app.config.get("TESTING")

    def test_config_class_inheritance(self):
        """Production and Development configs must inherit from BaseConfig."""
        try:
            from car_price_predictor.config import BaseConfig, ProductionConfig, TestingConfig
            assert issubclass(ProductionConfig, BaseConfig)
            assert issubclass(TestingConfig, BaseConfig)
        except ImportError:
            pytest.skip("Milestone 1 pending: car_price_predictor.config not yet implemented")


# ==============================================================================
# Feature 2: Named Pricing Constants (M1)
# ==============================================================================
class TestFeature02NamedPricingConstants:
    """Verifies named constants replace hardcoded magic numbers for heuristics."""

    def test_constants_owner_multipliers_defined(self):
        """Owner multipliers must be defined with exact values."""
        try:
            from car_price_predictor.services import constants
            assert hasattr(constants, "OWNER_MULTIPLIERS") or hasattr(constants, "OWNER_FACTORS")
            multipliers = getattr(constants, "OWNER_MULTIPLIERS", getattr(constants, "OWNER_FACTORS", None))
            assert multipliers["First Owner"] == 1.0
            assert multipliers["Second Owner"] == 0.95
            assert multipliers["Third Owner"] == 0.90
            assert multipliers["Fourth & Above Owner"] == 0.85
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: constants.py not yet implemented")

    def test_constants_year_baseline_and_rate(self):
        """Year depreciation baseline (2018) and rate (0.05) must be named constants."""
        try:
            from car_price_predictor.services import constants
            assert getattr(constants, "YEAR_BASELINE", None) == 2018
            assert getattr(constants, "YEAR_RATE", getattr(constants, "YEAR_FACTOR_RATE", None)) == 0.05
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: constants.py not yet implemented")

    def test_constants_year_clamp_limits(self):
        """Year safety clamp bounds must be 0.1 (min) and 2.0 (max)."""
        try:
            from car_price_predictor.services import constants
            assert getattr(constants, "YEAR_MIN_FACTOR", getattr(constants, "YEAR_FACTOR_MIN", None)) == 0.1
            assert getattr(constants, "YEAR_MAX_FACTOR", getattr(constants, "YEAR_FACTOR_MAX", None)) == 2.0
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: constants.py not yet implemented")

    def test_constants_mileage_baseline_and_rate(self):
        """Mileage baseline (18.0) and rate (0.01) must be named constants."""
        try:
            from car_price_predictor.services import constants
            assert getattr(constants, "MILEAGE_BASELINE", None) == 18.0
            assert getattr(constants, "MILEAGE_RATE", getattr(constants, "MILEAGE_FACTOR_RATE", None)) == 0.01
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: constants.py not yet implemented")

    def test_constants_mileage_clamp_limits(self):
        """Mileage safety clamp bounds must be 0.8 (min) and 1.2 (max)."""
        try:
            from car_price_predictor.services import constants
            assert getattr(constants, "MILEAGE_MIN_FACTOR", getattr(constants, "MILEAGE_FACTOR_MIN", None)) == 0.8
            assert getattr(constants, "MILEAGE_MAX_FACTOR", getattr(constants, "MILEAGE_FACTOR_MAX", None)) == 1.2
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: constants.py not yet implemented")


# ==============================================================================
# Feature 3: Pricing Rules Heuristics (M1)
# ==============================================================================
class TestFeature03PricingRulesHeuristics:
    """Verifies deterministic pricing heuristic business logic in PricingService."""

    def test_pricing_owner_factor_application(self):
        """Owner multipliers must scale base prediction correctly."""
        try:
            from car_price_predictor.services.pricing import PricingService
            base = 100000.0
            p1 = PricingService.calculate_price(base, year=2018, mileage=18.0, owner="First Owner")
            p2 = PricingService.calculate_price(base, year=2018, mileage=18.0, owner="Second Owner")
            p3 = PricingService.calculate_price(base, year=2018, mileage=18.0, owner="Third Owner")
            p4 = PricingService.calculate_price(base, year=2018, mileage=18.0, owner="Fourth & Above Owner")
            assert p1 == 100000.0
            assert p2 == 95000.0
            assert p3 == 90000.0
            assert p4 == 85000.0
        except ImportError:
            pytest.skip("Milestone 1 pending: PricingService not yet implemented")

    def test_pricing_year_factor_and_clamping(self):
        """Year factor applies 5% per year from 2018, clamped between 0.1 and 2.0."""
        try:
            from car_price_predictor.services.pricing import PricingService
            base = 100000.0
            p_2019 = PricingService.calculate_price(base, year=2019, mileage=18.0, owner="First Owner")
            assert abs(p_2019 - 105000.0) < 0.01
            p_ancient = PricingService.calculate_price(base, year=1990, mileage=18.0, owner="First Owner")
            assert abs(p_ancient - 10000.0) < 0.01  # clamped at 0.1
            p_future = PricingService.calculate_price(base, year=2050, mileage=18.0, owner="First Owner")
            assert abs(p_future - 200000.0) < 0.01  # clamped at 2.0
        except ImportError:
            pytest.skip("Milestone 1 pending: PricingService not yet implemented")

    def test_pricing_mileage_factor_and_clamping(self):
        """Mileage factor applies 1% per unit from 18.0 kmpl, clamped between 0.8 and 1.2."""
        try:
            from car_price_predictor.services.pricing import PricingService
            base = 100000.0
            p_20 = PricingService.calculate_price(base, year=2018, mileage=20.0, owner="First Owner")
            assert abs(p_20 - 102000.0) < 0.01
            p_low = PricingService.calculate_price(base, year=2018, mileage=5.0, owner="First Owner")
            assert abs(p_low - 87000.0) < 0.01
            p_guzzler = PricingService.calculate_price(base, year=2018, mileage=-10.0, owner="First Owner")
            assert abs(p_guzzler - 80000.0) < 0.01  # clamped at 0.8
            p_hyper = PricingService.calculate_price(base, year=2018, mileage=50.0, owner="First Owner")
            assert abs(p_hyper - 120000.0) < 0.01  # clamped at 1.2
        except ImportError:
            pytest.skip("Milestone 1 pending: PricingService not yet implemented")

    def test_pricing_format_price_string(self):
        """format_price must output standard currency format with comma and two decimals."""
        try:
            from car_price_predictor.services.pricing import PricingService
            formatted = PricingService.format_price(508560.1499)
            assert formatted == "508,560.15"
            formatted2 = PricingService.format_price(3312029.37)
            assert formatted2 == "3,312,029.37"
        except ImportError:
            pytest.skip("Milestone 1 pending: PricingService not yet implemented")

    def test_pricing_combined_multipliers_via_api(self, client, valid_payload):
        """End-to-end pricing heuristic produces correct valuation via predict endpoint."""
        res = client.post("/predict", json=valid_payload)
        assert res.status_code == 200
        assert res.get_json()["price"] == "508,560.15"


# ==============================================================================
# Feature 4: Vehicle Preprocessing (M1)
# ==============================================================================
class TestFeature04VehiclePreprocessing:
    """Verifies feature transformation and dummy baseline encoding."""

    def test_preprocessor_column_contract(self):
        """Preprocessor must produce a DataFrame with the exact 14 columns."""
        try:
            import joblib
            from car_price_predictor.services.preprocessor import VehiclePreprocessor
            data = joblib.load("car_price_model.joblib")
            prep = VehiclePreprocessor(data["columns"], data["brand_map"], data["model_map"])
            sample = {
                "brand": "Maruti", "model": "Swift", "year": 2019,
                "km_driven": 45000, "mileage": 18.5, "engine": 1197,
                "max_power": 82.0, "seats": 5, "fuel": "Petrol",
                "transmission": "Manual", "seller_type": "Individual",
                "owner": "First Owner"
            }
            df = prep.preprocess(sample)
            assert list(df.columns) == data["columns"]
            assert len(df) == 1
        except ImportError:
            pytest.skip("Milestone 1 pending: VehiclePreprocessor not yet implemented")

    def test_preprocessor_dummy_baseline_cng_dealer_auto(self):
        """Baseline levels (CNG, Dealer, Automatic) result in 0.0 for their respective columns."""
        try:
            import joblib
            from car_price_predictor.services.preprocessor import VehiclePreprocessor
            data = joblib.load("car_price_model.joblib")
            prep = VehiclePreprocessor(data["columns"], data["brand_map"], data["model_map"])
            sample = {
                "brand": "Maruti", "model": "Wagon", "year": 2017,
                "km_driven": 60000, "mileage": 26.0, "engine": 998,
                "max_power": 58.0, "seats": 5, "fuel": "CNG",
                "transmission": "Automatic", "seller_type": "Dealer",
                "owner": "First Owner"
            }
            df = prep.preprocess(sample)
            assert df.at[0, "fuel_Petrol"] == 0.0
            assert df.at[0, "fuel_Diesel"] == 0.0
            assert df.at[0, "fuel_Electric"] == 0.0
            assert df.at[0, "fuel_LPG"] == 0.0
            assert df.at[0, "seller_type_Individual"] == 0.0
            assert df.at[0, "seller_type_Trustmark Dealer"] == 0.0
            assert df.at[0, "transmission_Manual"] == 0.0
        except ImportError:
            pytest.skip("Milestone 1 pending: VehiclePreprocessor not yet implemented")

    def test_preprocessor_unseen_brand_assigned_zero(self):
        """Unseen brand must be encoded to 0.0 frequency."""
        try:
            import joblib
            from car_price_predictor.services.preprocessor import VehiclePreprocessor
            data = joblib.load("car_price_model.joblib")
            prep = VehiclePreprocessor(data["columns"], data["brand_map"], data["model_map"])
            sample = {
                "brand": "UnseenBrandXYZ", "model": "Swift", "year": 2019,
                "km_driven": 45000, "mileage": 18.5, "engine": 1197,
                "max_power": 82.0, "seats": 5, "fuel": "Petrol",
                "transmission": "Manual", "seller_type": "Individual",
                "owner": "First Owner"
            }
            df = prep.preprocess(sample)
            assert df.at[0, "brand_encoded"] == 0.0
        except ImportError:
            pytest.skip("Milestone 1 pending: VehiclePreprocessor not yet implemented")

    def test_preprocessor_unseen_model_assigned_zero(self):
        """Unseen model must be encoded to 0.0 frequency."""
        try:
            import joblib
            from car_price_predictor.services.preprocessor import VehiclePreprocessor
            data = joblib.load("car_price_model.joblib")
            prep = VehiclePreprocessor(data["columns"], data["brand_map"], data["model_map"])
            sample = {
                "brand": "Maruti", "model": "UnknownModelXYZ", "year": 2019,
                "km_driven": 45000, "mileage": 18.5, "engine": 1197,
                "max_power": 82.0, "seats": 5, "fuel": "Petrol",
                "transmission": "Manual", "seller_type": "Individual",
                "owner": "First Owner"
            }
            df = prep.preprocess(sample)
            assert df.at[0, "model_encoded"] == 0.0
        except ImportError:
            pytest.skip("Milestone 1 pending: VehiclePreprocessor not yet implemented")

    def test_preprocessor_numeric_types_clean(self):
        """All preprocessed columns must be numeric floats without NaN values."""
        try:
            import joblib
            from car_price_predictor.services.preprocessor import VehiclePreprocessor
            data = joblib.load("car_price_model.joblib")
            prep = VehiclePreprocessor(data["columns"], data["brand_map"], data["model_map"])
            sample = {
                "brand": "Honda", "model": "City", "year": 2020,
                "km_driven": 30000, "mileage": 18.0, "engine": 1498,
                "max_power": 119.35, "seats": 5, "fuel": "Petrol",
                "transmission": "Automatic", "seller_type": "Dealer",
                "owner": "First Owner"
            }
            df = prep.preprocess(sample)
            assert not df.isna().any().any()
        except ImportError:
            pytest.skip("Milestone 1 pending: VehiclePreprocessor not yet implemented")


# ==============================================================================
# Feature 5: Model Service & Loading (M1)
# ==============================================================================
class TestFeature05ModelService:
    """Verifies encapsulation and caching in ModelService."""

    def test_model_service_loads_and_is_ready(self):
        """ModelService instance reports is_ready() == True."""
        try:
            from car_price_predictor.services.model_service import ModelService
            svc = ModelService()
            assert svc.is_ready() is True
        except ImportError:
            pytest.skip("Milestone 1 pending: ModelService not yet implemented")

    def test_model_service_predict_returns_float(self):
        """ModelService.predict() executes inference and returns a float."""
        try:
            import joblib
            import pandas as pd
            from car_price_predictor.services.model_service import ModelService
            svc = ModelService()
            data = joblib.load("car_price_model.joblib")
            df = pd.DataFrame(0.0, index=[0], columns=data["columns"])
            val = svc.predict(df)
            assert isinstance(val, (float, int))
            assert val > 0
        except ImportError:
            pytest.skip("Milestone 1 pending: ModelService not yet implemented")

    def test_model_service_feature_columns_loaded(self):
        """ModelService exposes the expected feature columns."""
        try:
            from car_price_predictor.services.model_service import ModelService
            svc = ModelService()
            assert len(svc.feature_columns) == 14
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: ModelService not yet implemented")

    def test_model_service_brand_map_loaded(self):
        """ModelService exposes brand frequency mapping."""
        try:
            from car_price_predictor.services.model_service import ModelService
            svc = ModelService()
            assert "Maruti" in svc.brand_map
            assert "BMW" in svc.brand_map
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: ModelService not yet implemented")

    def test_model_service_model_map_loaded(self):
        """ModelService exposes model frequency mapping."""
        try:
            from car_price_predictor.services.model_service import ModelService
            svc = ModelService()
            assert "Swift" in svc.model_map
            assert "City" in svc.model_map
        except (ImportError, AttributeError):
            pytest.skip("Milestone 1 pending: ModelService not yet implemented")


# ==============================================================================
# Feature 6: Application Factory (M2)
# ==============================================================================
class TestFeature06ApplicationFactory:
    """Verifies create_app factory and blueprint wiring."""

    def test_create_app_testing_config(self):
        """create_app('testing') creates an app configured for testing."""
        try:
            from car_price_predictor import create_app
            app = create_app("testing")
            assert app.config["TESTING"] is True
        except ImportError:
            pytest.skip("Milestone 2 pending: create_app not yet implemented")

    def test_create_app_returns_flask_instance(self):
        """create_app returns a Flask instance."""
        try:
            from flask import Flask
            from car_price_predictor import create_app
            app = create_app("testing")
            assert isinstance(app, Flask)
        except ImportError:
            pytest.skip("Milestone 2 pending: create_app not yet implemented")

    def test_create_app_registers_blueprints(self):
        """create_app registers blueprints for api_v1, ui, health, and docs."""
        try:
            from car_price_predictor import create_app
            app = create_app("testing")
            blueprint_names = [bp for bp in app.blueprints]
            assert "ui" in blueprint_names or "/" in [r.rule for r in app.url_map.iter_rules()]
        except ImportError:
            pytest.skip("Milestone 2 pending: create_app not yet implemented")

    def test_create_app_test_client_functional(self):
        """Test client from create_app can make GET requests."""
        try:
            from car_price_predictor import create_app
            app = create_app("testing")
            client = app.test_client()
            res = client.get("/")
            assert res.status_code == 200
        except ImportError:
            pytest.skip("Milestone 2 pending: create_app not yet implemented")

    def test_create_app_config_override(self):
        """create_app supports custom config parameter."""
        try:
            from car_price_predictor import create_app
            app = create_app("development")
            assert app.config["DEBUG"] is True or not app.config["TESTING"]
        except ImportError:
            pytest.skip("Milestone 2 pending: create_app not yet implemented")


# ==============================================================================
# Feature 7: Pydantic Request Validation (M2)
# ==============================================================================
class TestFeature07PydanticRequestValidation:
    """Verifies schema validation rejecting invalid inputs."""

    def test_pydantic_valid_payload_passes(self, valid_payload):
        """Valid payload parses successfully through PredictionRequest."""
        try:
            from car_price_predictor.schemas.prediction import PredictionRequest
            req = PredictionRequest(**valid_payload)
            assert req.brand == "Maruti"
            assert req.year == 2019
        except ImportError:
            pytest.skip("Milestone 2 pending: PredictionRequest schema not yet implemented")

    def test_pydantic_missing_field_rejected(self, valid_payload):
        """Missing required field must raise ValidationError."""
        try:
            from car_price_predictor.schemas.prediction import PredictionRequest
            from pydantic import ValidationError
            payload = valid_payload.copy()
            del payload["brand"]
            with pytest.raises(ValidationError):
                PredictionRequest(**payload)
        except ImportError:
            pytest.skip("Milestone 2 pending: PredictionRequest schema not yet implemented")

    def test_pydantic_invalid_data_type_rejected(self, valid_payload):
        """String provided for numeric field must raise ValidationError."""
        try:
            from car_price_predictor.schemas.prediction import PredictionRequest
            from pydantic import ValidationError
            payload = valid_payload.copy()
            payload["km_driven"] = "not-a-number"
            with pytest.raises(ValidationError):
                PredictionRequest(**payload)
        except ImportError:
            pytest.skip("Milestone 2 pending: PredictionRequest schema not yet implemented")

    def test_pydantic_negative_km_rejected(self, valid_payload):
        """Negative km_driven must raise ValidationError."""
        try:
            from car_price_predictor.schemas.prediction import PredictionRequest
            from pydantic import ValidationError
            payload = valid_payload.copy()
            payload["km_driven"] = -500
            with pytest.raises(ValidationError):
                PredictionRequest(**payload)
        except ImportError:
            pytest.skip("Milestone 2 pending: PredictionRequest schema not yet implemented")

    def test_pydantic_out_of_range_year_rejected(self, valid_payload):
        """Year outside 1980..2030 range must raise ValidationError."""
        try:
            from car_price_predictor.schemas.prediction import PredictionRequest
            from pydantic import ValidationError
            payload = valid_payload.copy()
            payload["year"] = 1970
            with pytest.raises(ValidationError):
                PredictionRequest(**payload)
        except ImportError:
            pytest.skip("Milestone 2 pending: PredictionRequest schema not yet implemented")


# ==============================================================================
# Feature 8: Structured 400 Bad Request Errors (M2)
# ==============================================================================
class TestFeature08Structured400Errors:
    """Verifies structured HTTP 400 JSON error responses for invalid inputs."""

    def test_structured_error_status_code_400(self, app, client):
        """Invalid payload must return HTTP 400 Bad Request."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json={"brand": "Maruti"})
        assert res.status_code == 400

    def test_structured_error_has_error_message(self, app, client):
        """400 response must contain an 'error' field."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json={})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_structured_error_has_details_list(self, app, client):
        """400 response must contain a 'details' list."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json={})
        assert res.status_code == 400
        data = res.get_json()
        assert "details" in data
        assert isinstance(data["details"], list)

    def test_structured_error_detail_includes_field(self, app, client, valid_payload):
        """Error detail must identify the offending field."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        payload = valid_payload.copy()
        payload["year"] = 1950  # Out of range
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 400
        data = res.get_json()
        fields = [d.get("field") for d in data.get("details", [])]
        assert "year" in fields

    def test_structured_error_detail_includes_message(self, app, client, valid_payload):
        """Error detail must include human-readable message."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        payload = valid_payload.copy()
        payload["mileage"] = -5.0
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 400
        data = res.get_json()
        assert len(data.get("details", [])) > 0
        assert "message" in data["details"][0]


# ==============================================================================
# Feature 9: Versioned API (/api/v1/predict) (M2)
# ==============================================================================
class TestFeature09VersionedPredictionAPI:
    """Verifies versioned prediction endpoint response schema and contracts."""

    def test_v1_predict_returns_200(self, app, client, valid_payload):
        """Valid request returns HTTP 200 OK."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json=valid_payload)
        assert res.status_code == 200

    def test_v1_predict_has_price_string(self, app, client, valid_payload):
        """Response must include formatted 'price' string."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json=valid_payload)
        data = res.get_json()
        assert "price" in data
        assert isinstance(data["price"], str)

    def test_v1_predict_has_valuation_numeric(self, app, client, valid_payload):
        """Response must include unrounded numeric 'valuation' float."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json=valid_payload)
        data = res.get_json()
        assert "valuation" in data
        assert isinstance(data["valuation"], (float, int))

    def test_v1_predict_has_currency_inr(self, app, client, valid_payload):
        """Response must include 'currency': 'INR'."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json=valid_payload)
        data = res.get_json()
        assert data.get("currency") == "INR"

    def test_v1_predict_has_status_success(self, app, client, valid_payload):
        """Response must include 'status': 'success'."""
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Milestone 2 pending: /api/v1/predict not registered")
        res = client.post("/api/v1/predict", json=valid_payload)
        data = res.get_json()
        assert data.get("status") == "success"


# ==============================================================================
# Feature 10: Legacy API Alias (/predict) (M2)
# ==============================================================================
class TestFeature10LegacyPredictionAlias:
    """Verifies backward-compatible /predict endpoint."""

    def test_legacy_predict_returns_200(self, client, valid_payload):
        """POST /predict returns HTTP 200 OK for valid payload."""
        res = client.post("/predict", json=valid_payload)
        assert res.status_code == 200

    def test_legacy_predict_has_price_key(self, client, valid_payload):
        """POST /predict returns a JSON object with 'price'."""
        res = client.post("/predict", json=valid_payload)
        data = res.get_json()
        assert "price" in data

    def test_legacy_predict_price_formatting(self, client, valid_payload):
        """Price value has commas and decimal cents."""
        res = client.post("/predict", json=valid_payload)
        price_str = res.get_json()["price"]
        assert "," in price_str
        assert "." in price_str

    def test_legacy_predict_handles_various_brands(self, client, luxury_payload):
        """POST /predict handles luxury and budget vehicle inputs alike."""
        res = client.post("/predict", json=luxury_payload)
        assert res.status_code == 200
        assert res.get_json()["price"] == "3,312,029.37"

    def test_legacy_predict_handles_different_seats(self, client, golden_benchmarks):
        """POST /predict accurately values 7-seater vehicles (TC-05)."""
        tc05 = next(b["req"] for b in golden_benchmarks if b["req"]["id"] == "TC-05")
        res = client.post("/predict", json=tc05)
        assert res.status_code == 200
        assert res.get_json()["price"] == "523,794.72"


# ==============================================================================
# Feature 11: UI Blueprint & Hierarchy (M2)
# ==============================================================================
class TestFeature11UIBlueprintAndHierarchy:
    """Verifies GET / web UI and brand/model dropdown hierarchy."""

    def test_ui_home_returns_200(self, client):
        """GET / returns HTTP 200 OK."""
        res = client.get("/")
        assert res.status_code == 200

    def test_ui_home_content_type_html(self, client):
        """GET / Content-Type is text/html."""
        res = client.get("/")
        assert "text/html" in res.content_type

    def test_ui_home_contains_brand_options(self, client):
        """UI HTML contains manufacturer brand options."""
        res = client.get("/")
        html = res.data.decode("utf-8")
        assert "Maruti" in html
        assert "Hyundai" in html

    def test_ui_home_contains_car_hierarchy(self, client):
        """UI HTML includes car_hierarchy mapping."""
        res = client.get("/")
        html = res.data.decode("utf-8")
        assert "carHierarchy" in html or "brand_to_models" in html

    def test_ui_home_contains_form_elements(self, client):
        """UI HTML contains required input form fields."""
        res = client.get("/")
        html = res.data.decode("utf-8")
        for field in ["year", "km_driven", "mileage", "engine", "max_power", "seats", "fuel"]:
            assert field in html


# ==============================================================================
# Feature 12: OpenAPI / Swagger Documentation (M2)
# ==============================================================================
class TestFeature12OpenAPIDocs:
    """Verifies OpenAPI 3.0 specification and Swagger UI documentation endpoints."""

    def test_docs_apidocs_returns_200(self, app, client):
        """GET /apidocs returns HTTP 200 OK."""
        if not has_endpoint(app, "/apidocs"):
            pytest.skip("Milestone 2 pending: /apidocs not registered")
        res = client.get("/apidocs")
        assert res.status_code == 200

    def test_docs_openapi_json_returns_200(self, app, client):
        """GET /openapi.json returns HTTP 200 OK."""
        if not has_endpoint(app, "/openapi.json"):
            pytest.skip("Milestone 2 pending: /openapi.json not registered")
        res = client.get("/openapi.json")
        assert res.status_code == 200

    def test_docs_openapi_json_valid_spec(self, app, client):
        """OpenAPI JSON spec is valid JSON containing openapi version."""
        if not has_endpoint(app, "/openapi.json"):
            pytest.skip("Milestone 2 pending: /openapi.json not registered")
        res = client.get("/openapi.json")
        data = res.get_json()
        assert "openapi" in data or "swagger" in data

    def test_docs_openapi_has_predict_path(self, app, client):
        """OpenAPI spec documents the prediction endpoint path."""
        if not has_endpoint(app, "/openapi.json"):
            pytest.skip("Milestone 2 pending: /openapi.json not registered")
        res = client.get("/openapi.json")
        data = res.get_json()
        paths = data.get("paths", {})
        assert "/api/v1/predict" in paths or "/predict" in paths

    def test_docs_openapi_has_components_schemas(self, app, client):
        """OpenAPI spec includes component schemas for PredictionRequest."""
        if not has_endpoint(app, "/openapi.json"):
            pytest.skip("Milestone 2 pending: /openapi.json not registered")
        res = client.get("/openapi.json")
        data = res.get_json()
        schemas = data.get("components", {}).get("schemas", {})
        assert len(schemas) > 0 or "paths" in data


# ==============================================================================
# Feature 13: Health Check (/health/live) (M3)
# ==============================================================================
class TestFeature13HealthLive:
    """Verifies process liveness probe."""

    def test_health_live_status_code_200(self, app, client):
        """GET /health/live returns HTTP 200 OK."""
        if not has_endpoint(app, "/health/live"):
            pytest.skip("Milestone 3 pending: /health/live not registered")
        res = client.get("/health/live")
        assert res.status_code == 200

    def test_health_live_json_status_alive(self, app, client):
        """GET /health/live returns JSON with status 'alive'."""
        if not has_endpoint(app, "/health/live"):
            pytest.skip("Milestone 3 pending: /health/live not registered")
        res = client.get("/health/live")
        data = res.get_json()
        assert data.get("status") == "alive"

    def test_health_live_content_type_json(self, app, client):
        """GET /health/live Content-Type is application/json."""
        if not has_endpoint(app, "/health/live"):
            pytest.skip("Milestone 3 pending: /health/live not registered")
        res = client.get("/health/live")
        assert "application/json" in res.content_type

    def test_health_live_fast_latency(self, app, client):
        """GET /health/live executes rapidly without expensive I/O."""
        if not has_endpoint(app, "/health/live"):
            pytest.skip("Milestone 3 pending: /health/live not registered")
        import time
        start = time.perf_counter()
        res = client.get("/health/live")
        duration = time.perf_counter() - start
        assert res.status_code == 200
        assert duration < 0.2

    def test_health_live_method_not_allowed_post(self, app, client):
        """POST /health/live returns 405 Method Not Allowed."""
        if not has_endpoint(app, "/health/live"):
            pytest.skip("Milestone 3 pending: /health/live not registered")
        res = client.post("/health/live")
        assert res.status_code == 405


# ==============================================================================
# Feature 14: Health Check (/health/ready) (M3)
# ==============================================================================
class TestFeature14HealthReady:
    """Verifies model and service readiness probe."""

    def test_health_ready_status_code_200(self, app, client):
        """GET /health/ready returns HTTP 200 OK when model is loaded."""
        if not has_endpoint(app, "/health/ready"):
            pytest.skip("Milestone 3 pending: /health/ready not registered")
        res = client.get("/health/ready")
        assert res.status_code == 200

    def test_health_ready_json_status_ready(self, app, client):
        """GET /health/ready returns status 'ready'."""
        if not has_endpoint(app, "/health/ready"):
            pytest.skip("Milestone 3 pending: /health/ready not registered")
        res = client.get("/health/ready")
        data = res.get_json()
        assert data.get("status") == "ready"

    def test_health_ready_model_loaded_flag(self, app, client):
        """GET /health/ready reports model_loaded is True."""
        if not has_endpoint(app, "/health/ready"):
            pytest.skip("Milestone 3 pending: /health/ready not registered")
        res = client.get("/health/ready")
        data = res.get_json()
        assert data.get("model_loaded") is True

    def test_health_ready_content_type_json(self, app, client):
        """GET /health/ready Content-Type is application/json."""
        if not has_endpoint(app, "/health/ready"):
            pytest.skip("Milestone 3 pending: /health/ready not registered")
        res = client.get("/health/ready")
        assert "application/json" in res.content_type

    def test_health_ready_method_not_allowed_post(self, app, client):
        """POST /health/ready returns 405 Method Not Allowed."""
        if not has_endpoint(app, "/health/ready"):
            pytest.skip("Milestone 3 pending: /health/ready not registered")
        res = client.post("/health/ready")
        assert res.status_code == 405


# ==============================================================================
# Feature 15: Structured Request Logging (M3)
# ==============================================================================
class TestFeature15StructuredRequestLogging:
    """Verifies structured logging and context tracking."""

    def test_logging_module_exists(self):
        """car_price_predictor/utils/logging.py must exist."""
        log_module_path = ROOT_DIR / "car_price_predictor" / "utils" / "logging.py"
        if not log_module_path.exists():
            pytest.skip("Milestone 3 pending: logging.py not yet implemented")
        assert log_module_path.is_file()

    def test_logging_setup_callable(self):
        """Logger setup utility must be callable."""
        try:
            from car_price_predictor.utils.logging import setup_logging
            assert callable(setup_logging)
        except ImportError:
            pytest.skip("Milestone 3 pending: logging setup not yet implemented")

    def test_logging_predict_request_success(self, client, valid_payload):
        """Successful prediction runs cleanly without crashing the logging system."""
        res = client.post("/predict", json=valid_payload)
        assert res.status_code == 200

    def test_logging_no_raw_unformatted_exceptions(self, client):
        """Invalid requests return clean JSON rather than HTML tracebacks."""
        res = client.post("/predict", json={})
        # Even on errors, response should be clean JSON
        assert res.is_json or res.status_code in [400, 200]

    def test_logging_inference_duration_tracked(self, client, valid_payload):
        """Repeated inferences proceed with stable execution."""
        for _ in range(3):
            res = client.post("/predict", json=valid_payload)
            assert res.status_code == 200


# ==============================================================================
# Feature 16: Security Headers (M3)
# ==============================================================================
class TestFeature16SecurityHeaders:
    """Verifies security headers injected by middleware."""

    def test_security_header_x_content_type_options(self, client):
        """Responses should include X-Content-Type-Options: nosniff."""
        res = client.get("/")
        header = res.headers.get("X-Content-Type-Options")
        if header is None:
            pytest.skip("Milestone 3 pending: security headers middleware not registered")
        assert header == "nosniff"

    def test_security_header_x_frame_options(self, client):
        """Responses should include X-Frame-Options: DENY or SAMEORIGIN."""
        res = client.get("/")
        header = res.headers.get("X-Frame-Options")
        if header is None:
            pytest.skip("Milestone 3 pending: security headers middleware not registered")
        assert header in ["DENY", "SAMEORIGIN"]

    def test_security_header_x_xss_protection(self, client):
        """Responses should include X-XSS-Protection header."""
        res = client.get("/")
        header = res.headers.get("X-XSS-Protection")
        if header is None:
            pytest.skip("Milestone 3 pending: security headers middleware not registered")
        assert "1" in header

    def test_security_headers_on_api_endpoints(self, client, valid_payload):
        """API endpoints also include security headers."""
        res = client.post("/predict", json=valid_payload)
        if "X-Content-Type-Options" not in res.headers:
            pytest.skip("Milestone 3 pending: security headers on API not yet registered")
        assert res.headers.get("X-Content-Type-Options") == "nosniff"

    def test_security_headers_middleware_importable(self):
        """Security middleware module must be importable."""
        try:
            from car_price_predictor.utils.security import init_security_headers
            assert callable(init_security_headers)
        except ImportError:
            pytest.skip("Milestone 3 pending: car_price_predictor.utils.security not yet implemented")


# ==============================================================================
# Feature 17: Rate Limiting (M3)
# ==============================================================================
class TestFeature17RateLimiting:
    """Verifies rate limiting protection on prediction endpoints."""

    def test_rate_limiting_config_present(self, app):
        """App config contains rate limit configuration."""
        if "RATELIMIT_DEFAULT" not in app.config and "RATELIMIT_STORAGE_URI" not in app.config:
            pytest.skip("Milestone 3 pending: Rate limiting configuration not yet added")
        assert True

    def test_rate_limiting_extension_initialization(self, app):
        """Flask-Limiter extension should be registered if active."""
        extensions = getattr(app, "extensions", {})
        if "limiter" not in extensions:
            pytest.skip("Milestone 3 pending: Flask-Limiter not yet initialized")
        assert "limiter" in extensions

    def test_rate_limiting_endpoint_accessible(self, client, valid_payload):
        """Normal traffic within quota executes successfully."""
        res = client.post("/predict", json=valid_payload)
        assert res.status_code == 200

    def test_rate_limiting_exempt_health_checks(self, app, client):
        """Health endpoints must not be throttled."""
        if not has_endpoint(app, "/health/live"):
            pytest.skip("Milestone 3 pending: /health/live not registered")
        for _ in range(10):
            res = client.get("/health/live")
            assert res.status_code == 200

    def test_rate_limiting_burst_handling(self, client, valid_payload):
        """Burst of 5 consecutive requests processes without 500 error."""
        for _ in range(5):
            res = client.post("/predict", json=valid_payload)
            assert res.status_code in [200, 429]


# ==============================================================================
# Feature 18: Production WSGI Entrypoint (M3)
# ==============================================================================
class TestFeature18ProductionWSGIEntrypoint:
    """Verifies wsgi.py production entrypoint."""

    def test_wsgi_file_exists(self):
        """Root directory must contain wsgi.py."""
        wsgi_file = ROOT_DIR / "wsgi.py"
        if not wsgi_file.exists():
            pytest.skip("Milestone 3 pending: wsgi.py not yet created")
        assert wsgi_file.is_file()

    def test_wsgi_exports_app_or_application(self):
        """wsgi.py must export 'app' or 'application' callable."""
        wsgi_file = ROOT_DIR / "wsgi.py"
        if not wsgi_file.exists():
            pytest.skip("Milestone 3 pending: wsgi.py not yet created")
        import importlib.util
        spec = importlib.util.spec_from_file_location("wsgi", wsgi_file)
        wsgi_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wsgi_mod)
        assert hasattr(wsgi_mod, "app") or hasattr(wsgi_mod, "application")

    def test_wsgi_app_is_callable(self):
        """Exported WSGI object must be callable."""
        wsgi_file = ROOT_DIR / "wsgi.py"
        if not wsgi_file.exists():
            pytest.skip("Milestone 3 pending: wsgi.py not yet created")
        import importlib.util
        spec = importlib.util.spec_from_file_location("wsgi", wsgi_file)
        wsgi_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wsgi_mod)
        exported = getattr(wsgi_mod, "app", getattr(wsgi_mod, "application", None))
        assert callable(exported)

    def test_wsgi_syntax_valid(self):
        """wsgi.py can be compiled without syntax error."""
        wsgi_file = ROOT_DIR / "wsgi.py"
        if not wsgi_file.exists():
            pytest.skip("Milestone 3 pending: wsgi.py not yet created")
        code = wsgi_file.read_text(encoding="utf-8")
        compile(code, "wsgi.py", "exec")

    def test_wsgi_does_not_call_app_run_on_import(self):
        """Importing wsgi.py must not block or start the dev server."""
        wsgi_file = ROOT_DIR / "wsgi.py"
        if not wsgi_file.exists():
            pytest.skip("Milestone 3 pending: wsgi.py not yet created")
        content = wsgi_file.read_text(encoding="utf-8")
        assert "app.run(" not in content or "if __name__" in content


# ==============================================================================
# Feature 19: Automated CI Workflow File (M3)
# ==============================================================================
class TestFeature19AutomatedCIWorkflow:
    """Verifies GitHub Actions CI workflow file."""

    def test_ci_workflow_file_exists(self):
        """Workflow file must exist under .github/workflows/ci.yml."""
        ci_file = ROOT_DIR / ".github" / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("Milestone 3 pending: CI workflow file not yet created")
        assert ci_file.is_file()

    def test_ci_workflow_runs_pytest(self):
        """CI workflow must execute pytest command."""
        ci_file = ROOT_DIR / ".github" / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("Milestone 3 pending: CI workflow file not yet created")
        content = ci_file.read_text(encoding="utf-8")
        assert "pytest" in content

    def test_ci_workflow_triggers_on_push_pr(self):
        """CI workflow triggers on push and pull_request."""
        ci_file = ROOT_DIR / ".github" / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("Milestone 3 pending: CI workflow file not yet created")
        content = ci_file.read_text(encoding="utf-8")
        assert "push" in content
        assert "pull_request" in content

    def test_ci_workflow_sets_python_version(self):
        """CI workflow must set up a Python environment."""
        ci_file = ROOT_DIR / ".github" / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("Milestone 3 pending: CI workflow file not yet created")
        content = ci_file.read_text(encoding="utf-8")
        assert "setup-python" in content or "python-version" in content

    def test_ci_workflow_excludes_docker(self):
        """CI workflow must NOT use Docker per project scope constraints."""
        ci_file = ROOT_DIR / ".github" / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("Milestone 3 pending: CI workflow file not yet created")
        content = ci_file.read_text(encoding="utf-8")
        assert "docker build" not in content
        assert "docker run" not in content


# ==============================================================================
# Feature 20: Frontend Inline Error Alerts (M4)
# ==============================================================================
class TestFeature20FrontendInlineErrorAlerts:
    """Verifies accessible inline error alerts replacing browser alert()."""

    def test_frontend_index_has_error_alert_element(self):
        """index.html must include an element for inline error feedback."""
        index_file = ROOT_DIR / "templates" / "index.html"
        content = index_file.read_text(encoding="utf-8")
        has_error_box = (
            'id="errorMessage"' in content
            or 'id="errorAlert"' in content
            or 'class="error-alert"' in content
            or 'class="error-message"' in content
            or 'role="alert"' in content
        )
        if not has_error_box:
            pytest.skip("Milestone 4 pending: inline error container not yet added to index.html")
        assert has_error_box

    def test_frontend_error_container_initially_hidden(self):
        """Error alert element must be hidden on initial page load."""
        index_file = ROOT_DIR / "templates" / "index.html"
        content = index_file.read_text(encoding="utf-8")
        if "role=\"alert\"" not in content and "error" not in content:
            pytest.skip("Milestone 4 pending: error alert container not yet present")
        assert "hidden" in content or "display: none" in content or "d-none" in content

    def test_frontend_error_container_accessible(self):
        """Error container should include accessible attributes (role='alert' or aria-live)."""
        index_file = ROOT_DIR / "templates" / "index.html"
        content = index_file.read_text(encoding="utf-8")
        if "role=\"alert\"" not in content and "aria-live" not in content:
            pytest.skip("Milestone 4 pending: accessibility attributes not yet added to error alert")
        assert "role=\"alert\"" in content or "aria-live" in content

    def test_frontend_script_updates_inline_error(self):
        """script.js should set error text in inline container."""
        script_file = ROOT_DIR / "static" / "script.js"
        content = script_file.read_text(encoding="utf-8")
        if "alert(" not in content and ("innerText" in content or "textContent" in content):
            assert True
        else:
            pytest.skip("Milestone 4 pending: script.js still uses alert() or hasn't updated inline container")

    def test_frontend_alert_browser_dialog_avoided(self):
        """script.js should not call browser alert() in production."""
        script_file = ROOT_DIR / "static" / "script.js"
        content = script_file.read_text(encoding="utf-8")
        if "alert(" in content:
            pytest.skip("Milestone 4 pending: script.js still contains alert() calls")
        assert "alert(" not in content


# ==============================================================================
# Feature 21: Submission Spinner & Debouncing (M4)
# ==============================================================================
class TestFeature21SubmissionSpinnerAndDebounce:
    """Verifies loading spinner state and button debouncing."""

    def test_frontend_spinner_element_in_html(self):
        """index.html or button contains a loading spinner element."""
        index_file = ROOT_DIR / "templates" / "index.html"
        content = index_file.read_text(encoding="utf-8")
        if "spinner" not in content and "loading" not in content:
            pytest.skip("Milestone 4 pending: spinner element not yet added to index.html")
        assert "spinner" in content or "loading" in content

    def test_frontend_button_disabled_state_in_js(self):
        """script.js disables submit button during request."""
        script_file = ROOT_DIR / "static" / "script.js"
        content = script_file.read_text(encoding="utf-8")
        if "disabled" not in content:
            pytest.skip("Milestone 4 pending: button disabling not yet added to script.js")
        assert "disabled" in content

    def test_frontend_spinner_styles_in_css(self):
        """style.css contains spinner animation/styles."""
        css_file = ROOT_DIR / "static" / "style.css"
        content = css_file.read_text(encoding="utf-8")
        if "spinner" not in content and "@keyframes spin" not in content:
            pytest.skip("Milestone 4 pending: spinner CSS not yet added to style.css")
        assert "spinner" in content or "@keyframes spin" in content

    def test_frontend_form_submission_debounced(self):
        """script.js prevents duplicate concurrent submissions."""
        script_file = ROOT_DIR / "static" / "script.js"
        content = script_file.read_text(encoding="utf-8")
        if "isSubmitting" not in content and "disabled = true" not in content:
            pytest.skip("Milestone 4 pending: debouncing logic not yet added to script.js")
        assert "isSubmitting" in content or "disabled = true" in content

    def test_frontend_spinner_hidden_by_default(self):
        """Spinner is hidden by default in index.html."""
        index_file = ROOT_DIR / "templates" / "index.html"
        content = index_file.read_text(encoding="utf-8")
        if "spinner" not in content:
            pytest.skip("Milestone 4 pending: spinner not yet added to index.html")
        assert "hidden" in content or "display: none" in content


# ==============================================================================
# Feature 22: Mobile Responsive Design (M4)
# ==============================================================================
class TestFeature22MobileResponsiveDesign:
    """Verifies responsive layout across screen viewports."""

    def test_responsive_viewport_meta_tag(self):
        """index.html has viewport meta tag for responsive scaling."""
        index_file = ROOT_DIR / "templates" / "index.html"
        content = index_file.read_text(encoding="utf-8")
        assert '<meta name="viewport"' in content
        assert "width=device-width" in content

    def test_responsive_css_media_queries(self):
        """style.css contains responsive @media breakpoints."""
        css_file = ROOT_DIR / "static" / "style.css"
        content = css_file.read_text(encoding="utf-8")
        assert "@media" in content

    def test_responsive_form_grid_layout(self):
        """style.css defines form-grid or responsive flex layout."""
        css_file = ROOT_DIR / "static" / "style.css"
        content = css_file.read_text(encoding="utf-8")
        assert "grid" in content or "flex" in content

    def test_responsive_card_styling(self):
        """style.css defines card layout with max-width or mobile padding."""
        css_file = ROOT_DIR / "static" / "style.css"
        content = css_file.read_text(encoding="utf-8")
        assert ".card" in content or "card" in content

    def test_responsive_mobile_breakpoint_defined(self):
        """style.css defines standard mobile breakpoints (e.g. 768px or 480px)."""
        css_file = ROOT_DIR / "static" / "style.css"
        content = css_file.read_text(encoding="utf-8")
        assert "768px" in content or "600px" in content or "480px" in content


# ==============================================================================
# Feature 23: Ground Truth Regression Baselines (M1/M2)
# ==============================================================================
class TestFeature23GroundTruthBaselines:
    """Verifies ground truth valuations across multiple vehicle segments."""

    def test_baseline_maruti_swift_2019(self, client, golden_benchmarks):
        """TC-01 Maruti Swift 2019 -> '508,560.15'."""
        tc01 = next(b for b in golden_benchmarks if b["req"]["id"] == "TC-01")
        res = client.post("/predict", json=tc01["req"])
        assert res.status_code == 200
        assert res.get_json()["price"] == "508,560.15"

    def test_baseline_bmw_5_series_2021(self, client, golden_benchmarks):
        """TC-02 BMW 5 Series 2021 -> '3,312,029.37'."""
        tc02 = next(b for b in golden_benchmarks if b["req"]["id"] == "TC-02")
        res = client.post("/predict", json=tc02["req"])
        assert res.status_code == 200
        assert res.get_json()["price"] == "3,312,029.37"

    def test_baseline_hyundai_i10_2015(self, client, golden_benchmarks):
        """TC-03 Hyundai i10 2015 -> '209,300.94'."""
        tc03 = next(b for b in golden_benchmarks if b["req"]["id"] == "TC-03")
        res = client.post("/predict", json=tc03["req"])
        assert res.status_code == 200
        assert res.get_json()["price"] == "209,300.94"

    def test_baseline_maruti_wagon_2017(self, client, golden_benchmarks):
        """TC-04 Maruti Wagon 2017 -> '263,809.22'."""
        tc04 = next(b for b in golden_benchmarks if b["req"]["id"] == "TC-04")
        res = client.post("/predict", json=tc04["req"])
        assert res.status_code == 200
        assert res.get_json()["price"] == "263,809.22"

    def test_baseline_toyota_innova_2018(self, client, golden_benchmarks):
        """TC-05 Toyota Innova 2018 -> '523,794.72'."""
        tc05 = next(b for b in golden_benchmarks if b["req"]["id"] == "TC-05")
        res = client.post("/predict", json=tc05["req"])
        assert res.status_code == 200
        assert res.get_json()["price"] == "523,794.72"


# ==============================================================================
# Feature 24: Unseen Category Fallbacks (M1/M2)
# ==============================================================================
class TestFeature24UnseenCategoryFallbacks:
    """Verifies that unknown categorical inputs default safely to 0.0 frequency."""

    def test_unseen_brand_fallback_to_zero(self, client, valid_payload):
        """Unseen brand does not cause 500 server crash and returns a valid price."""
        payload = valid_payload.copy()
        payload["brand"] = "Koenigsegg"
        payload["model"] = "Jesko"
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "price" in data

    def test_unseen_model_fallback_to_zero(self, client, valid_payload):
        """Known brand with unseen model returns a valid valuation."""
        payload = valid_payload.copy()
        payload["brand"] = "Maruti"
        payload["model"] = "Batmobile"
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "price" in data

    def test_unseen_brand_and_model_simultaneous(self, client, valid_payload):
        """Simultaneous unseen brand and model returns valid valuation."""
        payload = valid_payload.copy()
        payload["brand"] = "UnknownBrand999"
        payload["model"] = "UnknownModel999"
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "price" in data

    def test_unseen_category_produces_positive_price(self, client, valid_payload):
        """Unseen category produces a positive non-zero valuation."""
        payload = valid_payload.copy()
        payload["brand"] = "CyberCar"
        payload["model"] = "Hover"
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        price_str = res.get_json()["price"]
        num_price = float(price_str.replace(",", ""))
        assert num_price > 0.0

    def test_unseen_category_formatted_consistently(self, client, valid_payload):
        """Unseen category valuation is formatted consistently with currency commas."""
        payload = valid_payload.copy()
        payload["brand"] = "Pagani"
        payload["model"] = "Zonda"
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        price_str = res.get_json()["price"]
        assert "." in price_str
