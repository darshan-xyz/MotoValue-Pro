"""
Regression tests verifying backward compatibility against 8 golden benchmarks.
Ensures zero valuation regressions from baseline model.
"""

import pytest
from tests.conftest import has_endpoint


class TestGoldenBenchmarkValuations:
    """Verifies that the 8 golden benchmark records yield exact price valuations."""

    @pytest.mark.parametrize(
        "benchmark_id",
        ["TC-01", "TC-02", "TC-03", "TC-04", "TC-05", "TC-06", "TC-07", "TC-08"],
    )
    def test_golden_benchmark_predict_endpoint(self, client, golden_benchmarks, benchmark_id):
        """
        Verify each golden benchmark record yields exact formatted price on POST /predict.
        """
        record = next((b for b in golden_benchmarks if b["req"]["id"] == benchmark_id), None)
        assert record is not None, f"Benchmark {benchmark_id} not found"

        req_payload = record["req"]
        expected_price = record["formatted_price"]

        response = client.post("/predict", json=req_payload)
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.data}"

        data = response.get_json()
        assert data is not None, "Response body was not valid JSON"
        assert "price" in data, f"Response missing 'price' key: {data}"
        assert data["price"] == expected_price, (
            f"Valuation regression on {benchmark_id} ({record['req']['brand']} {record['req']['model']}): "
            f"expected {expected_price}, got {data['price']}"
        )

    def test_golden_benchmarks_sequential_execution(self, client, golden_benchmarks):
        """
        Execute all 8 golden benchmarks in sequential succession.
        Ensures model state and caches do not corrupt sequential inferences.
        """
        for record in golden_benchmarks:
            req_payload = record["req"]
            expected_price = record["formatted_price"]
            b_id = req_payload["id"]

            response = client.post("/predict", json=req_payload)
            assert response.status_code == 200, f"Failed on {b_id}"
            data = response.get_json()
            assert data["price"] == expected_price, f"Failed on {b_id}: {data['price']} != {expected_price}"

    def test_golden_benchmark_idempotency(self, client, golden_benchmarks):
        """
        Repeated calls with the same payload must yield the exact same valuation.
        """
        tc01 = golden_benchmarks[0]["req"]
        expected = golden_benchmarks[0]["formatted_price"]

        for iteration in range(5):
            res = client.post("/predict", json=tc01)
            assert res.status_code == 200
            assert res.get_json()["price"] == expected, f"Iteration {iteration} drifted from expected price"

    def test_versioned_api_golden_benchmarks(self, app, client, golden_benchmarks):
        """
        When /api/v1/predict is available, verify it yields identical valuations to golden benchmarks.
        """
        if not has_endpoint(app, "/api/v1/predict"):
            pytest.skip("Endpoint /api/v1/predict not registered (Milestone 2 pending)")

        for record in golden_benchmarks:
            req_payload = record["req"]
            expected_formatted = record["formatted_price"]
            expected_valuation = record["final_price"]

            response = client.post("/api/v1/predict", json=req_payload)
            assert response.status_code == 200
            data = response.get_json()

            assert data.get("status") == "success"
            assert data.get("currency") == "INR"
            assert data.get("price") == expected_formatted
            assert abs(data.get("valuation") - expected_valuation) < 0.05

    def test_golden_benchmark_numerical_precision(self, client, golden_benchmarks):
        """
        Verify that numerical values parsed from the price string match the floating point ground truth.
        """
        for record in golden_benchmarks:
            response = client.post("/predict", json=record["req"])
            assert response.status_code == 200
            price_str = response.get_json()["price"]
            parsed_price = float(price_str.replace(",", ""))
            assert abs(parsed_price - record["final_price"]) < 0.05, (
                f"Numerical disparity for {record['req']['id']}: {parsed_price} vs {record['final_price']}"
            )
