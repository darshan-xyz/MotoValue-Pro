"""
Integration tests for schema validation and error responses:
- 400 Bad Request structured errors
- Type validation, range checks, and categorical validation
- Malformed bodies and content type enforcement
"""
import pytest


class TestValidationIntegration:
    """Verifies that invalid requests are cleanly rejected with structured 400 errors."""

    def test_missing_required_brand_rejected(self, client, valid_payload):
        """Missing 'brand' returns 400 with field detail."""
        payload = valid_payload.copy()
        del payload["brand"]
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        assert data.get("error") == "Validation Error"
        fields = [d["field"] for d in data.get("details", [])]
        assert "brand" in fields

    def test_missing_required_model_rejected(self, client, valid_payload):
        """Missing 'model' returns 400 with field detail."""
        payload = valid_payload.copy()
        del payload["model"]
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "model" in fields

    def test_year_below_minimum_rejected(self, client, valid_payload):
        """Year 1979 (below 1980 minimum) returns 400 Bad Request."""
        payload = valid_payload.copy()
        payload["year"] = 1979
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "year" in fields
        detail = next(d for d in data["details"] if d["field"] == "year")
        assert "1980" in detail["message"]

    def test_year_above_maximum_rejected(self, client, valid_payload):
        """Year 2031 (above 2030 maximum) returns 400 Bad Request."""
        payload = valid_payload.copy()
        payload["year"] = 2031
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "year" in fields

    def test_negative_km_driven_rejected(self, client, valid_payload):
        """Negative odometer reading returns 400 Bad Request."""
        payload = valid_payload.copy()
        payload["km_driven"] = -500.0
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "km_driven" in fields

    def test_out_of_range_mileage_rejected(self, client, valid_payload):
        """Mileage > 100 kmpl returns 400 Bad Request."""
        payload = valid_payload.copy()
        payload["mileage"] = 100.5
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "mileage" in fields

    def test_out_of_range_engine_rejected(self, client, valid_payload):
        """Engine displacement < 100 CC returns 400 Bad Request."""
        payload = valid_payload.copy()
        payload["engine"] = 50.0
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "engine" in fields

    def test_out_of_range_seats_rejected(self, client, valid_payload):
        """Seats = 0 or Seats > 20 returns 400 Bad Request."""
        for bad_seats in [0, 21]:
            payload = valid_payload.copy()
            payload["seats"] = bad_seats
            res = client.post("/api/v1/predict", json=payload)
            assert res.status_code == 400
            data = res.get_json()
            fields = [d["field"] for d in data.get("details", [])]
            assert "seats" in fields

    def test_invalid_fuel_type_rejected(self, client, valid_payload):
        """Unrecognized fuel type returns 400 Bad Request."""
        payload = valid_payload.copy()
        payload["fuel"] = "Hydrogen"
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "fuel" in fields

    def test_invalid_transmission_type_rejected(self, client, valid_payload):
        """Unrecognized transmission returns 400 Bad Request."""
        payload = valid_payload.copy()
        payload["transmission"] = "CVT-Semi"
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "transmission" in fields

    def test_empty_string_brand_rejected(self, client, valid_payload):
        """Empty string or whitespace-only brand is rejected with 400."""
        for bad_brand in ["", "   "]:
            payload = valid_payload.copy()
            payload["brand"] = bad_brand
            res = client.post("/api/v1/predict", json=payload)
            assert res.status_code == 400

    def test_non_numeric_type_for_numeric_field(self, client, valid_payload):
        """Alphabetic string in numeric field returns 400."""
        payload = valid_payload.copy()
        payload["km_driven"] = "fifty-thousand"
        res = client.post("/api/v1/predict", json=payload)

        assert res.status_code == 400
        data = res.get_json()
        fields = [d["field"] for d in data.get("details", [])]
        assert "km_driven" in fields

    def test_empty_payload_rejected(self, client):
        """Empty JSON object returns 400 with multiple missing field details."""
        res = client.post("/api/v1/predict", json={})
        assert res.status_code == 400
        data = res.get_json()
        assert data.get("error") == "Validation Error"
        assert len(data.get("details", [])) >= 10

    def test_array_payload_rejected(self, client):
        """Array body returns 400 with type error detail."""
        res = client.post("/api/v1/predict", json=[{"brand": "Maruti"}])
        assert res.status_code == 400
        data = res.get_json()
        assert data.get("error") == "Validation Error"

    def test_unsupported_media_type_rejected(self, client):
        """Non-JSON content type returns 415 Unsupported Media Type."""
        res = client.post("/api/v1/predict", data="hello=world", content_type="text/plain")
        assert res.status_code == 415
        data = res.get_json()
        assert "Unsupported Media Type" in data.get("error", "")

    def test_legacy_endpoint_rejects_invalid_payload(self, client, valid_payload):
        """Legacy /predict endpoint also returns 400 on invalid payload."""
        payload = valid_payload.copy()
        payload["year"] = 1970
        res = client.post("/predict", json=payload)
        assert res.status_code == 400
        data = res.get_json()
        assert data.get("error") == "Validation Error"
