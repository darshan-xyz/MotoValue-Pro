"""
Integration tests for Explainability Endpoint POST /api/v1/explain (R3).
"""
import concurrent.futures
import pytest


class TestExplainApiIntegration:
    """Verifies POST /api/v1/explain contract, golden benchmarks, and edge cases."""

    def test_explain_endpoint_valid_payload_200(self, client, valid_payload):
        """POST /api/v1/explain returns 200 OK and conforms to ExplainResponse schema."""
        res = client.post("/api/v1/explain", json=valid_payload)
        assert res.status_code == 200

        data = res.get_json()
        assert data.get("status") == "success"
        assert data.get("currency") == "INR"
        assert data.get("price") == "508,560.15"
        assert abs(data.get("valuation") - 508560.15) < 0.1
        assert "base_ml_valuation" in data

        # Confidence interval
        ci = data.get("confidence_interval")
        assert ci is not None
        assert ci["lower_bound"] <= data["valuation"] <= ci["upper_bound"]

        # Pricing adjustments
        adj = data.get("pricing_adjustments")
        assert adj["owner_multiplier"] == 1.0
        assert adj["year_factor"] == 1.05
        assert adj["mileage_factor"] == 1.005

        # Feature contributions
        fcs = data.get("feature_contributions")
        assert isinstance(fcs, list)
        assert len(fcs) > 5

        # Top drivers
        drivers = data.get("top_drivers")
        assert isinstance(drivers, list)
        assert len(drivers) > 0

    def test_explain_endpoint_luxury_benchmark_tc02(self, client, luxury_payload):
        """POST /api/v1/explain works seamlessly for luxury vehicle TC-02."""
        res = client.post("/api/v1/explain", json=luxury_payload)
        assert res.status_code == 200

        data = res.get_json()
        assert data["price"] == "3,312,029.37"
        assert abs(data["valuation"] - 3312029.37) < 0.1

        # max_power should be a dominant driver for luxury car
        power_contrib = next(c for c in data["feature_contributions"] if c["feature"] == "max_power")
        assert power_contrib["impact_amount"] > 500000.0

    def test_explain_endpoint_all_8_golden_benchmarks(self, client, golden_benchmarks):
        """All 8 golden benchmarks return valid explanations with correct prices."""
        for record in golden_benchmarks:
            res = client.post("/api/v1/explain", json=record["req"])
            assert res.status_code == 200, f"Failed for {record['req'].get('id')}"
            data = res.get_json()
            assert data["price"] == record["formatted_price"]
            assert data["confidence_interval"]["lower_bound"] <= data["valuation"] <= data["confidence_interval"]["upper_bound"]

    def test_explain_endpoint_invalid_payload_400(self, client, valid_payload):
        """Payload violating bounds returns HTTP 400 with structured validation error."""
        bad_payload = valid_payload.copy()
        bad_payload["year"] = 1970  # Min is 1980

        res = client.post("/api/v1/explain", json=bad_payload)
        assert res.status_code == 400
        data = res.get_json()
        assert data.get("error") == "Validation Error"
        assert any(d["field"] == "year" for d in data.get("details", []))

    def test_explain_endpoint_content_type_415(self, client, valid_payload):
        """Non-JSON content type returns HTTP 415 Unsupported Media Type."""
        res = client.post("/api/v1/explain", data="plain text", content_type="text/plain")
        assert res.status_code == 415

    def test_explain_endpoint_concurrency_thread_safety(self, client, valid_payload):
        """Concurrent requests to /api/v1/explain produce consistent results without data races."""
        def make_call():
            return client.post("/api/v1/explain", json=valid_payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(make_call) for _ in range(16)]
            results = [f.result() for f in futures]

        for r in results:
            assert r.status_code == 200
            assert r.get_json()["price"] == "508,560.15"

    def test_openapi_spec_documents_explain_endpoint(self, client):
        """GET /openapi.json documents /api/v1/explain path and schemas."""
        res = client.get("/openapi.json")
        assert res.status_code == 200
        spec = res.get_json()

        assert "/api/v1/explain" in spec["paths"]
        assert "post" in spec["paths"]["/api/v1/explain"]
        assert "ExplainResponse" in spec["components"]["schemas"]
        assert "ConfidenceInterval" in spec["components"]["schemas"]
        assert "FeatureContribution" in spec["components"]["schemas"]

    def test_explain_endpoint_non_dict_payload_400(self, client):
        """Non-dict JSON payload (e.g. array) returns HTTP 400 Bad Request."""
        res = client.post("/api/v1/explain", json=["not", "a", "dict"])
        assert res.status_code == 400
        data = res.get_json()
        assert data.get("error") == "Validation Error"

    def test_explain_endpoint_extra_fields_ignored(self, client, valid_payload):
        """Extra metadata fields in payload are ignored gracefully."""
        payload = valid_payload.copy()
        payload["vin"] = "1HGCR2F83HA000000"
        payload["dealer_notes"] = "Inspected and verified"
        res = client.post("/api/v1/explain", json=payload)
        assert res.status_code == 200
        assert res.get_json()["price"] == "508,560.15"

    def test_explain_endpoint_numeric_string_coercion(self, client, valid_payload):
        """Numeric strings in fields are coerced to floats/ints seamlessly."""
        payload = valid_payload.copy()
        payload["year"] = "2019"
        payload["km_driven"] = "45000"
        payload["mileage"] = "18.5"
        payload["engine"] = "1197.0"
        payload["max_power"] = "82.0"
        res = client.post("/api/v1/explain", json=payload)
        assert res.status_code == 200
        assert res.get_json()["price"] == "508,560.15"

    def test_explain_endpoint_http_methods_not_allowed(self, client):
        """GET, PUT, DELETE, PATCH on /api/v1/explain return HTTP 405 Method Not Allowed."""
        for method in ["get", "put", "delete", "patch"]:
            caller = getattr(client, method)
            res = caller("/api/v1/explain")
            assert res.status_code == 405
