"""
Integration tests for prediction endpoints:
- POST /api/v1/predict (versioned API)
- POST /predict (backward-compatible legacy alias)
"""
import pytest


class TestPredictionApiIntegration:
    """Verifies end-to-end inference workflows and contract compliance."""

    def test_predict_v1_valid_payload_success(self, client, valid_payload):
        """POST /api/v1/predict returns 200 OK and matches structured contract."""
        res = client.post("/api/v1/predict", json=valid_payload)
        assert res.status_code == 200
        data = res.get_json()

        assert "price" in data
        assert isinstance(data["price"], str)
        assert data["price"] == "508,560.15"

        assert "valuation" in data
        assert isinstance(data["valuation"], (float, int))
        assert abs(data["valuation"] - 508560.15) < 0.1

        assert data.get("currency") == "INR"
        assert data.get("status") == "success"

    def test_predict_legacy_valid_payload_success(self, client, valid_payload):
        """POST /predict returns 200 OK with legacy {'price': '...'} format."""
        res = client.post("/predict", json=valid_payload)
        assert res.status_code == 200
        data = res.get_json()

        assert "price" in data
        assert data["price"] == "508,560.15"
        assert "valuation" not in data

    def test_v1_and_legacy_return_identical_price_formatting(self, client, valid_payload):
        """Both endpoints compute the identical formatted price for the same input."""
        res_v1 = client.post("/api/v1/predict", json=valid_payload)
        res_leg = client.post("/predict", json=valid_payload)

        assert res_v1.status_code == 200
        assert res_leg.status_code == 200
        assert res_v1.get_json()["price"] == res_leg.get_json()["price"]

    def test_predict_luxury_diesel_automatic(self, client, luxury_payload):
        """POST /api/v1/predict correctly values luxury diesel automatic vehicle (TC-02)."""
        res = client.post("/api/v1/predict", json=luxury_payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["price"] == "3,312,029.37"
        assert abs(data["valuation"] - 3312029.37) < 0.1

    def test_predict_cng_vehicle(self, client, golden_benchmarks):
        """POST /api/v1/predict handles CNG vehicle with reference baseline encoding (TC-04)."""
        tc04 = next(b["req"] for b in golden_benchmarks if b["req"]["id"] == "TC-04")
        res = client.post("/api/v1/predict", json=tc04)
        assert res.status_code == 200
        assert res.get_json()["price"] == "263,809.22"

    def test_predict_seven_seater_diesel(self, client, golden_benchmarks):
        """POST /api/v1/predict handles 7-seater diesel vehicle (TC-05)."""
        tc05 = next(b["req"] for b in golden_benchmarks if b["req"]["id"] == "TC-05")
        res = client.post("/api/v1/predict", json=tc05)
        assert res.status_code == 200
        assert res.get_json()["price"] == "523,794.72"

    def test_predict_unseen_brand_and_model_graceful_fallback(self, client, valid_payload):
        """Unseen brand/model falls back to 0.0 frequency without crashing."""
        payload = valid_payload.copy()
        payload["brand"] = "NonExistentSuperBrand"
        payload["model"] = "ImaginaryHyperCar"

        res_v1 = client.post("/api/v1/predict", json=payload)
        assert res_v1.status_code == 200
        data = res_v1.get_json()
        assert data["status"] == "success"
        assert data["valuation"] > 0

    def test_predict_case_insensitive_categories(self, client, valid_payload):
        """Case variations in fuel, transmission, seller, and owner normalize properly."""
        payload = valid_payload.copy()
        payload["fuel"] = "petrol"
        payload["transmission"] = "MANUAL"
        payload["seller_type"] = "individual"
        payload["owner"] = "first owner"

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        assert res.get_json()["price"] == "508,560.15"

    def test_predict_extra_metadata_fields_ignored(self, client, valid_payload):
        """Extra unexpected fields do not trigger validation error."""
        payload = valid_payload.copy()
        payload["id"] = "BENCHMARK-99"
        payload["notes"] = "Test vehicle metadata"
        payload["vin"] = "ABC123456789"

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        assert res.get_json()["status"] == "success"

    def test_predict_numeric_string_coercion(self, client, valid_payload):
        """Numeric strings are safely coerced to int/float."""
        payload = valid_payload.copy()
        payload["year"] = "2019"
        payload["km_driven"] = "45000"
        payload["mileage"] = "18.5"
        payload["engine"] = "1197.0"
        payload["max_power"] = "82.0"
        payload["seats"] = "5"

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        assert res.get_json()["price"] == "508,560.15"

    def test_predict_multiple_sequential_inferences(self, client, golden_benchmarks):
        """Multiple sequential predictions maintain accuracy without cross-talk."""
        for record in golden_benchmarks:
            res = client.post("/api/v1/predict", json=record["req"])
            assert res.status_code == 200
            assert res.get_json()["price"] == record["formatted_price"]
