"""
Global Pytest fixtures for CarPricePredictor test suite.
Provides app, client, valid payloads, and 8 golden benchmark vectors.
"""

import json
import os
from pathlib import Path
import pytest
import pandas as pd

# Support list == pd.Index assertions without ambiguous array truth-value error
_orig_index_eq = pd.Index.__eq__


def _patched_index_eq(self, other):
    if isinstance(other, list):
        if len(self) != len(other):
            return False
        return list(self) == other
    return _orig_index_eq(self, other)


pd.Index.__eq__ = _patched_index_eq


BENCHMARKS_PATH = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "teamwork"
    / "explorer_survey_2"
    / "baseline_benchmarks.json"
)

# Canonical fallback benchmark dataset (8 ground truth records)
FALLBACK_BENCHMARKS = [
    {
        "req": {
            "id": "TC-01",
            "description": "Default UI Form Values (Maruti Swift 2019)",
            "brand": "Maruti",
            "model": "Swift",
            "year": 2019,
            "km_driven": 45000,
            "mileage": 18.5,
            "engine": 1197,
            "max_power": 82.0,
            "seats": 5,
            "fuel": "Petrol",
            "transmission": "Manual",
            "seller_type": "Individual",
            "owner": "First Owner",
        },
        "base_prediction": 481933.33333333326,
        "owner_factor": 1.0,
        "year_factor": 1.05,
        "mileage_factor": 1.005,
        "final_price": 508560.1499999999,
        "formatted_price": "508,560.15",
    },
    {
        "req": {
            "id": "TC-02",
            "description": "Luxury Sedan Diesel Automatic (BMW 5 Series 2021)",
            "brand": "BMW",
            "model": "5",
            "year": 2021,
            "km_driven": 25000,
            "mileage": 17.4,
            "engine": 1995,
            "max_power": 190.0,
            "seats": 5,
            "fuel": "Diesel",
            "transmission": "Automatic",
            "seller_type": "Dealer",
            "owner": "First Owner",
        },
        "base_prediction": 2897410.0,
        "owner_factor": 1.0,
        "year_factor": 1.15,
        "mileage_factor": 0.994,
        "final_price": 3312029.3709999993,
        "formatted_price": "3,312,029.37",
    },
    {
        "req": {
            "id": "TC-03",
            "description": "Budget Hatchback Second Owner (Hyundai i10 2015)",
            "brand": "Hyundai",
            "model": "i10",
            "year": 2015,
            "km_driven": 75000,
            "mileage": 19.8,
            "engine": 1086,
            "max_power": 68.0,
            "seats": 5,
            "fuel": "Petrol",
            "transmission": "Manual",
            "seller_type": "Individual",
            "owner": "Second Owner",
        },
        "base_prediction": 254613.1787656788,
        "owner_factor": 0.95,
        "year_factor": 0.85,
        "mileage_factor": 1.018,
        "final_price": 209300.9444066448,
        "formatted_price": "209,300.94",
    },
    {
        "req": {
            "id": "TC-04",
            "description": "CNG WagonR Third Owner (Maruti Wagon 2017)",
            "brand": "Maruti",
            "model": "Wagon",
            "year": 2017,
            "km_driven": 60000,
            "mileage": 26.0,
            "engine": 998,
            "max_power": 58.0,
            "seats": 5,
            "fuel": "CNG",
            "transmission": "Manual",
            "seller_type": "Trustmark Dealer",
            "owner": "Third Owner",
        },
        "base_prediction": 285693.3333333333,
        "owner_factor": 0.9,
        "year_factor": 0.95,
        "mileage_factor": 1.08,
        "final_price": 263809.224,
        "formatted_price": "263,809.22",
    },
    {
        "req": {
            "id": "TC-05",
            "description": "7-Seater Diesel SUV Fourth Owner (Toyota Innova 2018)",
            "brand": "Toyota",
            "model": "Innova",
            "year": 2018,
            "km_driven": 120000,
            "mileage": 12.0,
            "engine": 2494,
            "max_power": 102.0,
            "seats": 7,
            "fuel": "Diesel",
            "transmission": "Manual",
            "seller_type": "Individual",
            "owner": "Fourth & Above Owner",
        },
        "base_prediction": 655562.8571428572,
        "owner_factor": 0.85,
        "year_factor": 1.0,
        "mileage_factor": 0.94,
        "final_price": 523794.7228571428,
        "formatted_price": "523,794.72",
    },
    {
        "req": {
            "id": "TC-06",
            "description": "Mid-tier Sedan Automatic (Honda City 2020)",
            "brand": "Honda",
            "model": "City",
            "year": 2020,
            "km_driven": 30000,
            "mileage": 18.0,
            "engine": 1498,
            "max_power": 119.35,
            "seats": 5,
            "fuel": "Petrol",
            "transmission": "Automatic",
            "seller_type": "Dealer",
            "owner": "First Owner",
        },
        "base_prediction": 1059360.0,
        "owner_factor": 1.0,
        "year_factor": 1.1,
        "mileage_factor": 1.0,
        "final_price": 1165296.0,
        "formatted_price": "1,165,296.00",
    },
    {
        "req": {
            "id": "TC-07",
            "description": "Older Hatchback Low Year Factor (Maruti Alto 2005)",
            "brand": "Maruti",
            "model": "Alto",
            "year": 2005,
            "km_driven": 90000,
            "mileage": 19.7,
            "engine": 796,
            "max_power": 46.3,
            "seats": 5,
            "fuel": "Petrol",
            "transmission": "Manual",
            "seller_type": "Individual",
            "owner": "Second Owner",
        },
        "base_prediction": 92500.19841269842,
        "owner_factor": 0.95,
        "year_factor": 0.35,
        "mileage_factor": 1.017,
        "final_price": 31279.173343749993,
        "formatted_price": "31,279.17",
    },
    {
        "req": {
            "id": "TC-08",
            "description": "High Mileage Vehicle (Renault KWID 2022, 45 kmpl clamped)",
            "brand": "Renault",
            "model": "KWID",
            "year": 2022,
            "km_driven": 15000,
            "mileage": 45.0,
            "engine": 999,
            "max_power": 67.0,
            "seats": 5,
            "fuel": "Petrol",
            "transmission": "Manual",
            "seller_type": "Individual",
            "owner": "First Owner",
        },
        "base_prediction": 375535.7261904762,
        "owner_factor": 1.0,
        "year_factor": 1.2,
        "mileage_factor": 1.2,
        "final_price": 540771.4457142857,
        "formatted_price": "540,771.45",
    },
]


@pytest.fixture(scope="session")
def app():
    """
    Session-scoped Flask application fixture.
    Loads create_app('testing') from car_price_predictor if available,
    otherwise falls back to app.py.
    """
    try:
        from car_price_predictor import create_app
        test_app = create_app("testing")
    except (ImportError, ModuleNotFoundError):
        import app as legacy_app_module
        test_app = legacy_app_module.app
        test_app.config["TESTING"] = True
    return test_app


@pytest.fixture
def client(app):
    """Function-scoped Flask test client fixture."""
    return app.test_client()


@pytest.fixture(scope="session")
def golden_benchmarks():
    """Provides the 8 golden benchmark records from baseline_benchmarks.json."""
    if BENCHMARKS_PATH.exists():
        try:
            with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return FALLBACK_BENCHMARKS


@pytest.fixture
def valid_payload():
    """
    Canonical valid prediction request payload (TC-01 Maruti Swift 2019).
    Matches both legacy prototype schema and Pydantic PredictionRequest schema.
    """
    return {
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
    }


@pytest.fixture
def luxury_payload():
    """
    Valid payload for a luxury sedan (TC-02 BMW 5 Series 2021).
    """
    return {
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
    }


@pytest.fixture
def sample_payloads(golden_benchmarks):
    """
    Dictionary of diverse valid payloads indexed by benchmark ID.
    """
    return {b["req"]["id"]: b["req"] for b in golden_benchmarks}


def has_endpoint(app_instance, rule_path):
    """Helper to check if a specific endpoint path is registered in Flask url_map."""
    return any(rule.rule == rule_path for rule in app_instance.url_map.iter_rules())


# ==============================================================================
# Milestone 1 Unit Test Fixtures
# ==============================================================================

@pytest.fixture
def feature_columns() -> list[str]:
    """Expected 14 model feature columns in exact order."""
    from car_price_predictor.services.constants import FEATURE_COLUMNS
    return list(FEATURE_COLUMNS)


@pytest.fixture
def brand_map() -> dict[str, float]:
    """Sample brand frequency map for fast isolated unit testing."""
    return {
        "Maruti": 0.3239244695347479,
        "Hyundai": 0.1934981506715982,
        "Honda": 0.09635974304068523,
        "BMW": 0.015,
        "Toyota": 0.08,
        "Renault": 0.04,
    }


@pytest.fixture
def model_map() -> dict[str, float]:
    """Sample model frequency map for fast isolated unit testing."""
    return {
        "Swift": 0.10842904418921549,
        "i10": 0.045,
        "i20": 0.05878917656219584,
        "Alto": 0.05048342093309973,
        "5": 0.005,
        "Wagon": 0.035,
        "Innova": 0.025,
        "City": 0.040,
        "KWID": 0.020,
    }


@pytest.fixture
def preprocessor(feature_columns, brand_map, model_map):
    """Pre-configured isolated VehiclePreprocessor instance for unit testing."""
    from car_price_predictor.services.preprocessor import VehiclePreprocessor
    return VehiclePreprocessor(
        feature_columns=feature_columns,
        brand_map=brand_map,
        model_map=model_map,
    )


@pytest.fixture(scope="session")
def model_service():
    """Provides a loaded ModelService instance for the test session."""
    from car_price_predictor.services.model_service import ModelService
    service = ModelService(load_on_init=True)
    assert service.is_ready(), "ModelService failed to initialize in test fixture"
    return service


@pytest.fixture
def sample_payload(valid_payload):
    """Alias for canonical valid payload (Maruti Swift 2019)."""
    return dict(valid_payload)


@pytest.fixture
def swift_payload(valid_payload):
    """Alias for Swift payload."""
    return dict(valid_payload)

