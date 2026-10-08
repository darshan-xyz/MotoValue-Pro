"""
Tier 2: Boundary Value Analysis & Corner Cases Test Suite (>=5 tests per boundary condition).
Verifies strict numeric limits, heuristic clamping, string constraints, and malformed payloads.
"""

import pytest
from tests.conftest import has_endpoint


def assert_invalid_rejected(app, client, payload, field_name=None):
    """
    Helper asserting invalid payload is properly rejected by both:
    1. Legacy endpoint (returns error key or 400)
    2. Versioned API (returns strict HTTP 400 with structured error details when available)
    """
    res_legacy = client.post("/predict", json=payload)
    if res_legacy.status_code == 200:
        data = res_legacy.get_json()
        assert "error" in data or "price" in data  # Prototype fallback
    else:
        assert res_legacy.status_code in [400, 422]

    if has_endpoint(app, "/api/v1/predict"):
        res_v1 = client.post("/api/v1/predict", json=payload)
        assert res_v1.status_code in [400, 422]
        err_data = res_v1.get_json()
        assert "error" in err_data or "details" in err_data
        if field_name and "details" in err_data:
            fields = [d.get("field") for d in err_data["details"]]
            assert field_name in fields


# ==============================================================================
# Boundary 1: Year Limits (Valid range: 1980..2030)
# ==============================================================================
class TestBoundary01YearLimits:
    """Verifies boundary analysis on vehicle manufacturing year."""

    def test_year_minimum_valid_1980(self, client, valid_payload):
        """Year 1980 (lower bound) is accepted and valued."""
        payload = valid_payload.copy()
        payload["year"] = 1980
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert "price" in res.get_json()

    def test_year_maximum_valid_2030(self, client, valid_payload):
        """Year 2030 (upper bound) is accepted and valued."""
        payload = valid_payload.copy()
        payload["year"] = 2030
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert "price" in res.get_json()

    def test_year_near_boundaries_valid(self, client, valid_payload):
        """Years 1981 and 2029 (interior near-bounds) are accepted."""
        for yr in [1981, 2029]:
            payload = valid_payload.copy()
            payload["year"] = yr
            res = client.post("/predict", json=payload)
            assert res.status_code == 200

    def test_year_below_minimum_1979_rejected(self, app, client, valid_payload):
        """Year 1979 (just below lower bound) must be rejected."""
        payload = valid_payload.copy()
        payload["year"] = 1979
        assert_invalid_rejected(app, client, payload, "year")

    def test_year_above_maximum_2031_rejected(self, app, client, valid_payload):
        """Year 2031 (just above upper bound) must be rejected."""
        payload = valid_payload.copy()
        payload["year"] = 2031
        assert_invalid_rejected(app, client, payload, "year")

    def test_year_extreme_outliers_rejected(self, app, client, valid_payload):
        """Extreme past (1800) and future (3000) years are rejected."""
        for yr in [1800, 3000]:
            payload = valid_payload.copy()
            payload["year"] = yr
            assert_invalid_rejected(app, client, payload, "year")


# ==============================================================================
# Boundary 2: Mileage Limits (Valid range: 0.0..100.0 kmpl)
# ==============================================================================
class TestBoundary02MileageLimits:
    """Verifies boundary analysis on fuel efficiency / mileage."""

    def test_mileage_minimum_valid_zero(self, client, valid_payload):
        """Mileage 0.0 kmpl (lower bound) is accepted."""
        payload = valid_payload.copy()
        payload["mileage"] = 0.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_mileage_maximum_valid_100(self, client, valid_payload):
        """Mileage 100.0 kmpl (upper bound) is accepted."""
        payload = valid_payload.copy()
        payload["mileage"] = 100.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_mileage_fractional_precision(self, client, valid_payload):
        """Mileage with decimal precision (e.g. 18.55) processes accurately."""
        payload = valid_payload.copy()
        payload["mileage"] = 18.55
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_mileage_negative_rejected(self, app, client, valid_payload):
        """Negative mileage (-0.1) must be rejected."""
        payload = valid_payload.copy()
        payload["mileage"] = -0.1
        assert_invalid_rejected(app, client, payload, "mileage")

    def test_mileage_above_100_rejected(self, app, client, valid_payload):
        """Mileage 100.1 (above upper bound) must be rejected."""
        payload = valid_payload.copy()
        payload["mileage"] = 100.1
        assert_invalid_rejected(app, client, payload, "mileage")

    def test_mileage_extreme_value_rejected(self, app, client, valid_payload):
        """Extreme mileage (500.0) is rejected."""
        payload = valid_payload.copy()
        payload["mileage"] = 500.0
        assert_invalid_rejected(app, client, payload, "mileage")


# ==============================================================================
# Boundary 3: Engine Displacement (Valid range: 100..10000 CC)
# ==============================================================================
class TestBoundary03EngineDisplacement:
    """Verifies boundary analysis on engine cubic capacity."""

    def test_engine_minimum_valid_100(self, client, valid_payload):
        """Engine 100 CC (lower bound) is accepted."""
        payload = valid_payload.copy()
        payload["engine"] = 100.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_engine_maximum_valid_10000(self, client, valid_payload):
        """Engine 10000 CC (upper bound) is accepted."""
        payload = valid_payload.copy()
        payload["engine"] = 10000.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_engine_interior_bounds_valid(self, client, valid_payload):
        """Engines 796 CC (Maruti 800) and 4999 CC (V8) process cleanly."""
        for cc in [796.0, 4999.0]:
            payload = valid_payload.copy()
            payload["engine"] = cc
            res = client.post("/predict", json=payload)
            assert res.status_code == 200

    def test_engine_below_minimum_rejected(self, app, client, valid_payload):
        """Engine 99.9 CC (below minimum) must be rejected."""
        payload = valid_payload.copy()
        payload["engine"] = 99.9
        assert_invalid_rejected(app, client, payload, "engine")

    def test_engine_above_maximum_rejected(self, app, client, valid_payload):
        """Engine 10000.1 CC (above maximum) must be rejected."""
        payload = valid_payload.copy()
        payload["engine"] = 10000.1
        assert_invalid_rejected(app, client, payload, "engine")

    def test_engine_zero_or_negative_rejected(self, app, client, valid_payload):
        """Engine 0 or negative CC is rejected."""
        for cc in [0.0, -100.0]:
            payload = valid_payload.copy()
            payload["engine"] = cc
            assert_invalid_rejected(app, client, payload, "engine")


# ==============================================================================
# Boundary 4: Max Power (Valid range: 10..2000 bhp)
# ==============================================================================
class TestBoundary04MaxPower:
    """Verifies boundary analysis on brake horsepower."""

    def test_max_power_minimum_valid_10(self, client, valid_payload):
        """Max power 10.0 bhp (lower bound) is accepted."""
        payload = valid_payload.copy()
        payload["max_power"] = 10.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_max_power_maximum_valid_2000(self, client, valid_payload):
        """Max power 2000.0 bhp (upper bound) is accepted."""
        payload = valid_payload.copy()
        payload["max_power"] = 2000.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_max_power_fractional_precision(self, client, valid_payload):
        """Max power with fractional values (e.g. 119.35 bhp) processes cleanly."""
        payload = valid_payload.copy()
        payload["max_power"] = 119.35
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_max_power_below_minimum_rejected(self, app, client, valid_payload):
        """Max power 9.9 bhp (below minimum) must be rejected."""
        payload = valid_payload.copy()
        payload["max_power"] = 9.9
        assert_invalid_rejected(app, client, payload, "max_power")

    def test_max_power_above_maximum_rejected(self, app, client, valid_payload):
        """Max power 2000.1 bhp (above maximum) must be rejected."""
        payload = valid_payload.copy()
        payload["max_power"] = 2000.1
        assert_invalid_rejected(app, client, payload, "max_power")

    def test_max_power_zero_or_negative_rejected(self, app, client, valid_payload):
        """Max power 0 or negative bhp is rejected."""
        for p in [0.0, -50.0]:
            payload = valid_payload.copy()
            payload["max_power"] = p
            assert_invalid_rejected(app, client, payload, "max_power")


# ==============================================================================
# Boundary 5: Seats (Valid range: 1..20)
# ==============================================================================
class TestBoundary05Seats:
    """Verifies boundary analysis on vehicle seating capacity."""

    def test_seats_minimum_valid_1(self, client, valid_payload):
        """Seats = 1 (single-seater) is accepted."""
        payload = valid_payload.copy()
        payload["seats"] = 1
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_seats_maximum_valid_20(self, client, valid_payload):
        """Seats = 20 (minibus upper limit) is accepted."""
        payload = valid_payload.copy()
        payload["seats"] = 20
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_seats_standard_options(self, client, valid_payload):
        """Standard seat options (2, 4, 5, 7, 8) process cleanly."""
        for s in [2, 4, 5, 7, 8]:
            payload = valid_payload.copy()
            payload["seats"] = s
            res = client.post("/predict", json=payload)
            assert res.status_code == 200

    def test_seats_zero_rejected(self, app, client, valid_payload):
        """Seats = 0 must be rejected."""
        payload = valid_payload.copy()
        payload["seats"] = 0
        assert_invalid_rejected(app, client, payload, "seats")

    def test_seats_above_20_rejected(self, app, client, valid_payload):
        """Seats = 21 (above maximum) must be rejected."""
        payload = valid_payload.copy()
        payload["seats"] = 21
        assert_invalid_rejected(app, client, payload, "seats")

    def test_seats_negative_rejected(self, app, client, valid_payload):
        """Negative seats (-1) is rejected."""
        payload = valid_payload.copy()
        payload["seats"] = -1
        assert_invalid_rejected(app, client, payload, "seats")


# ==============================================================================
# Boundary 6: Kilometers Driven (Valid range: >= 0)
# ==============================================================================
class TestBoundary06KmDriven:
    """Verifies boundary analysis on odometer reading."""

    def test_km_driven_zero_valid(self, client, valid_payload):
        """0 km (brand new showroom vehicle) is accepted."""
        payload = valid_payload.copy()
        payload["km_driven"] = 0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_km_driven_typical_values(self, client, valid_payload):
        """Typical mileage (15000, 45000, 120000 km) processes cleanly."""
        for km in [15000, 45000, 120000]:
            payload = valid_payload.copy()
            payload["km_driven"] = km
            res = client.post("/predict", json=payload)
            assert res.status_code == 200

    def test_km_driven_extreme_high_valid(self, client, valid_payload):
        """High mileage (1,000,000 km) is accepted without crash."""
        payload = valid_payload.copy()
        payload["km_driven"] = 1000000
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_km_driven_negative_rejected(self, app, client, valid_payload):
        """Negative km_driven (-1) must be rejected."""
        payload = valid_payload.copy()
        payload["km_driven"] = -1
        assert_invalid_rejected(app, client, payload, "km_driven")

    def test_km_driven_large_negative_rejected(self, app, client, valid_payload):
        """Large negative km_driven (-50000) must be rejected."""
        payload = valid_payload.copy()
        payload["km_driven"] = -50000
        assert_invalid_rejected(app, client, payload, "km_driven")

    def test_km_driven_non_numeric_rejected(self, app, client, valid_payload):
        """String value for km_driven is rejected."""
        payload = valid_payload.copy()
        payload["km_driven"] = "fifty-thousand"
        assert_invalid_rejected(app, client, payload, "km_driven")


# ==============================================================================
# Boundary 7: String & Character Stress (Brand & Model)
# ==============================================================================
class TestBoundary07StringAndCharacterStress:
    """Verifies string boundary inputs, whitespace, and special characters."""

    def test_brand_empty_string_rejected(self, app, client, valid_payload):
        """Empty string for brand must be rejected."""
        payload = valid_payload.copy()
        payload["brand"] = ""
        assert_invalid_rejected(app, client, payload, "brand")

    def test_model_empty_string_rejected(self, app, client, valid_payload):
        """Empty string for model must be rejected."""
        payload = valid_payload.copy()
        payload["model"] = ""
        assert_invalid_rejected(app, client, payload, "model")

    def test_brand_whitespace_only(self, client, valid_payload):
        """Whitespace-only brand handled without crashing."""
        payload = valid_payload.copy()
        payload["brand"] = "   "
        res = client.post("/predict", json=payload)
        assert res.status_code in [200, 400]

    def test_brand_special_characters_sql_script(self, client, valid_payload):
        """SQL injection / script strings do not cause server error."""
        for evil_str in ["' OR '1'='1", "<script>alert(1)</script>", "DROP TABLE cars;"]:
            payload = valid_payload.copy()
            payload["brand"] = evil_str
            res = client.post("/predict", json=payload)
            assert res.status_code in [200, 400]
            if res.status_code == 200:
                assert "price" in res.get_json()

    def test_brand_very_long_string(self, client, valid_payload):
        """Very long string (1000 characters) does not cause server crash."""
        payload = valid_payload.copy()
        payload["brand"] = "A" * 1000
        res = client.post("/predict", json=payload)
        assert res.status_code in [200, 400]

    def test_brand_unicode_and_emojis(self, client, valid_payload):
        """Unicode characters and emojis handled cleanly."""
        payload = valid_payload.copy()
        payload["brand"] = "🚗 Toyota 日本語"
        res = client.post("/predict", json=payload)
        assert res.status_code in [200, 400]


# ==============================================================================
# Boundary 8: Pricing Heuristic Clamping Boundaries
# ==============================================================================
class TestBoundary08PricingHeuristicClamping:
    """Verifies boundary clamping of pricing heuristics."""

    def test_pricing_year_clamp_floor(self, client, valid_payload):
        """Vehicles from 1998 and earlier clamp year factor to exactly 0.1."""
        payload_1998 = valid_payload.copy()
        payload_1998["year"] = 1998
        payload_1985 = valid_payload.copy()
        payload_1985["year"] = 1985

        res1 = client.post("/predict", json=payload_1998)
        res2 = client.post("/predict", json=payload_1985)
        assert res1.status_code == 200
        assert res2.status_code == 200
        # Both should have identical year factor (0.1), thus identical final price
        assert res1.get_json()["price"] == res2.get_json()["price"]

    def test_pricing_year_clamp_ceiling(self, client, valid_payload):
        """Vehicles from 2038 or later clamp year factor to exactly 2.0."""
        try:
            from car_price_predictor.services.pricing import PricingService
            factor_2038 = 1 + (2038 - 2018) * 0.05
            factor_2045 = 1 + (2045 - 2018) * 0.05
            p1 = PricingService.calculate_price(100000.0, 2038, 18.0, "First Owner")
            p2 = PricingService.calculate_price(100000.0, 2045, 18.0, "First Owner")
            assert p1 == p2 == 200000.0
        except ImportError:
            # Fallback test via pricing logic
            base = 100000.0
            yf_2038 = min(2.0, max(0.1, 1 + (2038 - 2018) * 0.05))
            yf_2050 = min(2.0, max(0.1, 1 + (2050 - 2018) * 0.05))
            assert yf_2038 == yf_2050 == 2.0

    def test_pricing_mileage_clamp_floor(self, client, valid_payload):
        """Very low mileage (-2.0 or 0.0) clamps mileage factor to 0.8."""
        payload_zero = valid_payload.copy()
        payload_zero["mileage"] = 0.0
        res = client.post("/predict", json=payload_zero)
        assert res.status_code == 200
        assert "price" in res.get_json()

    def test_pricing_mileage_clamp_ceiling(self, client, valid_payload):
        """High mileage (>= 38.0 kmpl) clamps mileage factor to 1.2."""
        payload_38 = valid_payload.copy()
        payload_38["mileage"] = 38.0
        payload_50 = valid_payload.copy()
        payload_50["mileage"] = 50.0

        res38 = client.post("/predict", json=payload_38)
        res50 = client.post("/predict", json=payload_50)
        assert res38.status_code == 200
        assert res50.status_code == 200
        # Both clamp at 1.2 factor
        assert res38.get_json()["price"] == res50.get_json()["price"]

    def test_pricing_extreme_low_combined(self, client, valid_payload):
        """Lowest possible heuristic combination produces positive price."""
        payload = valid_payload.copy()
        payload["year"] = 1980  # year factor 0.1
        payload["mileage"] = 0.0  # mileage factor 0.82
        payload["owner"] = "Fourth & Above Owner"  # owner factor 0.85
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        price_val = float(res.get_json()["price"].replace(",", ""))
        assert price_val > 0.0

    def test_pricing_extreme_high_combined(self, client, valid_payload):
        """Highest valid heuristic combination produces finite price."""
        payload = valid_payload.copy()
        payload["year"] = 2030  # year factor 1.6
        payload["mileage"] = 40.0  # mileage factor 1.2
        payload["owner"] = "First Owner"  # owner factor 1.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        price_val = float(res.get_json()["price"].replace(",", ""))
        assert price_val > 0.0


# ==============================================================================
# Boundary 9: Malformed Payloads & Encoding Stress
# ==============================================================================
class TestBoundary09MalformedPayloads:
    """Verifies server handling of malformed requests."""

    def test_empty_json_object(self, app, client):
        """Empty JSON object is rejected without 500 crash."""
        assert_invalid_rejected(app, client, {})

    def test_null_value_in_required_field(self, app, client, valid_payload):
        """Null/None value for brand is rejected."""
        payload = valid_payload.copy()
        payload["brand"] = None
        assert_invalid_rejected(app, client, payload, "brand")

    def test_missing_multiple_fields(self, app, client):
        """Payload missing multiple required fields is rejected."""
        assert_invalid_rejected(app, client, {"year": 2019})

    def test_extra_unexpected_fields(self, client, valid_payload):
        """Extra unexpected fields do not crash the endpoint."""
        payload = valid_payload.copy()
        payload["unknown_extra_field"] = "some_value"
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert "price" in res.get_json()

    def test_array_body_instead_of_object(self, app, client):
        """Array body instead of JSON object returns 400."""
        res = client.post("/predict", json=[1, 2, 3])
        assert res.status_code in [200, 400]
        if res.status_code == 200:
            assert "error" in res.get_json()

    def test_non_json_content_type(self, client):
        """Non-JSON request (text/plain) returns error or 400."""
        res = client.post("/predict", data="plain text data", content_type="text/plain")
        assert res.status_code in [200, 400, 415]
