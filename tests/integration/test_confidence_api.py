"""
Integration tests for Valuation Confidence Intervals in Prediction APIs (R4).
"""
import pytest


class TestConfidenceApiIntegration:
    """Verifies confidence interval integration into /api/v1/predict and legacy /predict decoupling."""

    def test_predict_v1_includes_confidence_interval(self, client, valid_payload):
        """POST /api/v1/predict response payload enriches valuation with confidence_interval."""
        res = client.post("/api/v1/predict", json=valid_payload)
        assert res.status_code == 200
        data = res.get_json()

        assert "confidence_interval" in data
        ci = data["confidence_interval"]
        assert isinstance(ci, dict)

        assert "lower_bound" in ci
        assert "upper_bound" in ci
        assert "formatted_lower" in ci
        assert "formatted_upper" in ci
        assert ci["confidence_level"] == 0.80

        # Boundary invariants
        assert ci["lower_bound"] >= 0.0
        assert ci["lower_bound"] <= data["valuation"] <= ci["upper_bound"]

    def test_all_8_golden_benchmarks_have_valid_intervals(self, client, golden_benchmarks):
        """Every golden benchmark in /api/v1/predict has valid bounds enclosing its valuation."""
        for record in golden_benchmarks:
            res = client.post("/api/v1/predict", json=record["req"])
            assert res.status_code == 200
            data = res.get_json()

            ci = data["confidence_interval"]
            assert ci["lower_bound"] <= data["valuation"] <= ci["upper_bound"], (
                f"Interval [{ci['lower_bound']}, {ci['upper_bound']}] failed to enclose "
                f"valuation {data['valuation']} for {record['req'].get('id')}"
            )
            assert ci["formatted_lower"] is not None
            assert ci["formatted_upper"] is not None

    def test_legacy_predict_strictly_omits_confidence_interval(self, client, valid_payload):
        """Legacy POST /predict must NEVER contain 'confidence_interval' or 'valuation'."""
        res = client.post("/predict", json=valid_payload)
        assert res.status_code == 200
        data = res.get_json()

        assert "price" in data
        assert data["price"] == "508,560.15"
        assert "valuation" not in data
        assert "confidence_interval" not in data
        assert list(data.keys()) == ["price"]

    def test_price_string_equality_between_v1_and_legacy(self, client, valid_payload):
        """Both endpoints yield bit-for-bit identical formatted price string."""
        res_v1 = client.post("/api/v1/predict", json=valid_payload)
        res_leg = client.post("/predict", json=valid_payload)

        assert res_v1.status_code == 200
        assert res_leg.status_code == 200
        assert res_v1.get_json()["price"] == res_leg.get_json()["price"]

    def test_confidence_interval_luxury_vehicle(self, client, luxury_payload):
        """TC-02 Luxury vehicle has valid wide confidence bounds containing valuation."""
        res = client.post("/api/v1/predict", json=luxury_payload)
        assert res.status_code == 200
        data = res.get_json()
        ci = data["confidence_interval"]
        assert ci["lower_bound"] <= data["valuation"] <= ci["upper_bound"]
        assert ci["lower_bound"] > 1000000.0

    def test_confidence_interval_unseen_category(self, client, valid_payload):
        """Unseen brand and model request receives valid confidence interval."""
        payload = valid_payload.copy()
        payload["brand"] = "FuturisticBrand"
        payload["model"] = "CyberSpeed"
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        ci = data["confidence_interval"]
        assert ci["lower_bound"] <= data["valuation"] <= ci["upper_bound"]

    def test_legacy_predict_preserves_single_key_on_all_benchmarks(self, client, golden_benchmarks):
        """Legacy /predict returns ONLY {'price': '...'} across all 8 golden benchmarks."""
        for record in golden_benchmarks:
            res = client.post("/predict", json=record["req"])
            assert res.status_code == 200
            data = res.get_json()
            assert list(data.keys()) == ["price"]
            assert data["price"] == record["formatted_price"]
