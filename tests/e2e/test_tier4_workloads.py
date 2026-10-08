"""
Tier 4: Real-World Workload Scenarios Test Suite.
Verifies complete end-to-end user flows, enterprise fleet appraisals,
faulty client recovery, platform monitoring, and high-rate submission.
"""

import json
import time
import pytest
from tests.conftest import has_endpoint


# ==============================================================================
# Scenario 1: Standard Car Buyer Flow
# ==============================================================================
class TestWorkloadScenario1CarBuyerFlow:
    """Simulates a typical retail car buyer using the application."""

    def test_full_buyer_journey(self, client):
        # Step 1: User visits homepage
        res_home = client.get("/")
        assert res_home.status_code == 200
        html = res_home.data.decode("utf-8")

        # Step 2: Homepage loads dropdown options and car hierarchy
        assert "Maruti" in html
        assert "Swift" in html or "brand_to_models" in html or "carHierarchy" in html

        # Step 3: User fills out form with Maruti Swift 2019 details
        user_form = {
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

        # Step 4: User clicks 'Calculate Value' button, triggering POST /predict
        res_predict = client.post("/predict", json=user_form)
        assert res_predict.status_code == 200
        predict_data = res_predict.get_json()

        # Step 5: User verifies valuation estimate
        assert "price" in predict_data
        assert predict_data["price"] == "508,560.15"

        # Step 6: User repeats with updated kilometers (e.g. 60000 km)
        user_form["km_driven"] = 60000
        res_predict_updated = client.post("/predict", json=user_form)
        assert res_predict_updated.status_code == 200
        assert "price" in res_predict_updated.get_json()


# ==============================================================================
# Scenario 2: Fleet Luxury Appraisal
# ==============================================================================
class TestWorkloadScenario2FleetLuxuryAppraisal:
    """Simulates enterprise fleet management appraising high-end inventory."""

    def test_batch_luxury_fleet_appraisal(self, client, golden_benchmarks):
        luxury_fleet = [
            # TC-02 Golden Benchmark
            next(b["req"] for b in golden_benchmarks if b["req"]["id"] == "TC-02"),
            {
                "brand": "BMW",
                "model": "X5",
                "year": 2020,
                "km_driven": 35000,
                "mileage": 13.38,
                "engine": 2993.0,
                "max_power": 261.5,
                "seats": 5,
                "fuel": "Diesel",
                "transmission": "Automatic",
                "seller_type": "Dealer",
                "owner": "First Owner",
            },
            {
                "brand": "Audi",
                "model": "A4",
                "year": 2021,
                "km_driven": 20000,
                "mileage": 17.42,
                "engine": 1984.0,
                "max_power": 187.74,
                "seats": 5,
                "fuel": "Petrol",
                "transmission": "Automatic",
                "seller_type": "Dealer",
                "owner": "First Owner",
            },
            {
                "brand": "Mercedes-Benz",
                "model": "E-Class",
                "year": 2019,
                "km_driven": 40000,
                "mileage": 16.1,
                "engine": 1950.0,
                "max_power": 194.0,
                "seats": 5,
                "fuel": "Diesel",
                "transmission": "Automatic",
                "seller_type": "Dealer",
                "owner": "Second Owner",
            },
            {
                "brand": "Volvo",
                "model": "XC60",
                "year": 2020,
                "km_driven": 30000,
                "mileage": 11.2,
                "engine": 1969.0,
                "max_power": 235.0,
                "seats": 5,
                "fuel": "Diesel",
                "transmission": "Automatic",
                "seller_type": "Trustmark Dealer",
                "owner": "First Owner",
            },
        ]

        valuations = []
        for vehicle in luxury_fleet:
            res = client.post("/predict", json=vehicle)
            assert res.status_code == 200, f"Appraisal failed for {vehicle['brand']} {vehicle['model']}"
            data = res.get_json()
            assert "price" in data
            price_val = float(data["price"].replace(",", ""))
            assert price_val > 500000.0, f"Unreasonably low luxury appraisal: {price_val}"
            valuations.append(price_val)

        # First vehicle (TC-02 BMW 5 Series) must match exact ground truth
        assert luxury_fleet[0]["id"] == "TC-02"
        assert f"{valuations[0]:,.2f}" == "3,312,029.37"


# ==============================================================================
# Scenario 3: Faulty API Client Robustness & Recovery
# ==============================================================================
class TestWorkloadScenario3FaultyClientRobustness:
    """Simulates a buggy API client sending consecutive malformed payloads followed by recovery."""

    def test_faulty_client_handling_and_recovery(self, app, client, valid_payload):
        malformed_inputs = [
            # 1. Missing brand and model
            {"year": 2019, "km_driven": 45000, "mileage": 18.5, "engine": 1197, "max_power": 82.0, "seats": 5},
            # 2. String instead of int for year
            {**valid_payload, "year": "nineteen-ninety-five"},
            # 3. String instead of float for km_driven
            {**valid_payload, "km_driven": "lots-of-km"},
            # 4. Negative kilometers driven
            {**valid_payload, "km_driven": -25000},
            # 5. Out of range mileage (150 kmpl)
            {**valid_payload, "mileage": 150.0},
            # 6. Completely empty object
            {},
        ]

        # Verify none of the bad payloads cause unhandled 500 server crash
        for idx, bad_payload in enumerate(malformed_inputs):
            res = client.post("/predict", json=bad_payload)
            assert res.status_code != 500, f"Server crashed with 500 on bad payload #{idx}"
            assert res.status_code in [200, 400, 422]
            if res.status_code == 200 and idx in [0, 1, 2, 5]:
                # Missing fields or type errors trigger error response in prototype
                assert "error" in res.get_json()

        # When M2 versioned API is active, all malformed payloads must strictly return HTTP 400
        if has_endpoint(app, "/api/v1/predict"):
            for bad_payload in malformed_inputs:
                res_v1 = client.post("/api/v1/predict", json=bad_payload)
                assert res_v1.status_code == 400
                assert "error" in res_v1.get_json()

        # Step 7: System recovery check - send valid payload immediately after errors
        res_valid = client.post("/predict", json=valid_payload)
        assert res_valid.status_code == 200, "Server failed to process valid payload after bad requests"
        assert res_valid.get_json()["price"] == "508,560.15"


# ==============================================================================
# Scenario 4: Platform Health & Monitoring Flow
# ==============================================================================
class TestWorkloadScenario4HealthAndMonitoring:
    """Simulates synthetic monitoring / APM uptime probes."""

    def test_synthetic_health_monitoring(self, app, client):
        if not has_endpoint(app, "/health/live"):
            pytest.skip("Milestone 3 pending: /health/live not registered")

        # Test liveness probe
        res_live = client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.get_json()["status"] == "alive"

        # Test readiness probe
        res_ready = client.get("/health/ready")
        assert res_ready.status_code == 200
        ready_json = res_ready.get_json()
        assert ready_json["status"] == "ready"
        assert ready_json["model_loaded"] is True

        # Simulate repeated 5-second polling interval
        for _ in range(5):
            t0 = time.perf_counter()
            r = client.get("/health/live")
            dt = time.perf_counter() - t0
            assert r.status_code == 200
            assert dt < 0.1  # Sub-100ms response time


# ==============================================================================
# Scenario 5: Edge Fleet Evaluation (Outliers)
# ==============================================================================
class TestWorkloadScenario5EdgeFleetOutliers:
    """Evaluates edge-case and extreme outlier vehicles."""

    def test_edge_case_fleet(self, client, valid_payload):
        outliers = [
            # High mileage clamped (TC-08)
            {
                "brand": "Renault", "model": "KWID", "year": 2022,
                "km_driven": 15000, "mileage": 45.0, "engine": 999,
                "max_power": 67.0, "seats": 5, "fuel": "Petrol",
                "transmission": "Manual", "seller_type": "Individual",
                "owner": "First Owner"
            },
            # Older vehicle (TC-07)
            {
                "brand": "Maruti", "model": "Alto", "year": 2005,
                "km_driven": 90000, "mileage": 19.7, "engine": 796,
                "max_power": 46.3, "seats": 5, "fuel": "Petrol",
                "transmission": "Manual", "seller_type": "Individual",
                "owner": "Second Owner"
            },
            # Rare category: Electric automatic
            {
                "brand": "Tata", "model": "Nexon", "year": 2022,
                "km_driven": 12000, "mileage": 0.0, "engine": 1000,
                "max_power": 127.0, "seats": 5, "fuel": "Electric",
                "transmission": "Automatic", "seller_type": "Dealer",
                "owner": "First Owner"
            },
            # Completely unseen brand and model
            {
                "brand": "Rimac", "model": "Nevera", "year": 2023,
                "km_driven": 2000, "mileage": 0.0, "engine": 2000,
                "max_power": 1914.0, "seats": 2, "fuel": "Electric",
                "transmission": "Automatic", "seller_type": "Dealer",
                "owner": "First Owner"
            },
        ]

        for idx, vehicle in enumerate(outliers):
            res = client.post("/predict", json=vehicle)
            assert res.status_code == 200, f"Outlier #{idx} failed: {vehicle['brand']}"
            price_str = res.get_json()["price"]
            num_val = float(price_str.replace(",", ""))
            assert num_val > 0.0, f"Outlier #{idx} produced non-positive price"


# ==============================================================================
# Scenario 6: High-Rate Submission & Stability
# ==============================================================================
class TestWorkloadScenario6HighRateSubmission:
    """Verifies server responsiveness under rapid burst submissions."""

    def test_burst_prediction_submissions(self, client, valid_payload):
        burst_count = 15
        start_time = time.perf_counter()

        for i in range(burst_count):
            res = client.post("/predict", json=valid_payload)
            # Accept either 200 OK or 429 Too Many Requests (if rate limiting triggers)
            assert res.status_code in [200, 429], f"Unexpected status {res.status_code} on request #{i}"
            if res.status_code == 200:
                assert "price" in res.get_json()

        total_duration = time.perf_counter() - start_time
        avg_latency = total_duration / burst_count
        assert avg_latency < 0.5, f"Excessive average request latency: {avg_latency:.3f}s"
