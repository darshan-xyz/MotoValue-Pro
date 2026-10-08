"""
Milestone 2 Adversarial Stress & Empirical Challenge Test Battery.
Authored by Empirical Challenger 1 to stress-test:
1. Golden benchmark valuations and numerical precision on /api/v1/predict and /predict.
2. Web UI at GET / serving HTML with brand and hierarchy data.
3. Interactive OpenAPI 3.0.3 docs and JSON spec at /openapi.json and /apidocs.
4. Schema validation robustness against malicious, boundary, fuzzing, and corrupted inputs.
5. HTTP protocol conformance, method guards, and structured error responses.
6. Multi-threaded concurrent inference load.
"""
from __future__ import annotations

import concurrent.futures
import json
import random
import re
import pytest
from flask import Flask

from car_price_predictor import create_app
from car_price_predictor.services.pricing import PricingService


# ==============================================================================
# 1. Golden Benchmark Exactness Oracle
# ==============================================================================
class TestAdversarialGoldenBenchmarks:
    """Verifies that all 8 golden benchmarks yield exact valuations matching baseline_benchmarks.json."""

    def test_all_8_golden_benchmarks_on_v1_predict(self, client, golden_benchmarks):
        """Every golden benchmark record must return HTTP 200 and exact formatted price & valuation on /api/v1/predict."""
        assert len(golden_benchmarks) == 8, f"Expected 8 golden benchmarks, found {len(golden_benchmarks)}"

        for record in golden_benchmarks:
            req_payload = record["req"]
            expected_formatted = record["formatted_price"]
            expected_valuation = round(float(record["final_price"]), 2)

            res = client.post("/api/v1/predict", json=req_payload)
            assert res.status_code == 200, f"Benchmark {req_payload.get('id')} failed with status {res.status_code}: {res.get_data(as_text=True)}"
            
            data = res.get_json()
            assert isinstance(data, dict), f"Response must be JSON dict for {req_payload.get('id')}"
            assert data.get("status") == "success"
            assert data.get("currency") == "INR"
            assert data.get("price") == expected_formatted, f"Price mismatch for {req_payload.get('id')}: got {data.get('price')}, expected {expected_formatted}"
            assert data.get("valuation") == expected_valuation, f"Valuation mismatch for {req_payload.get('id')}: got {data.get('valuation')}, expected {expected_valuation}"

    def test_all_8_golden_benchmarks_on_legacy_predict(self, client, golden_benchmarks):
        """Every golden benchmark record must return HTTP 200 and exact formatted price on legacy /predict."""
        for record in golden_benchmarks:
            req_payload = record["req"]
            expected_formatted = record["formatted_price"]

            res = client.post("/predict", json=req_payload)
            assert res.status_code == 200, f"Benchmark {req_payload.get('id')} failed on /predict: {res.get_data(as_text=True)}"
            
            data = res.get_json()
            assert isinstance(data, dict)
            assert "price" in data
            assert data["price"] == expected_formatted, f"Legacy price mismatch for {req_payload.get('id')}: got {data['price']}, expected {expected_formatted}"

    def test_v1_and_legacy_endpoints_yield_identical_price_strings(self, client, golden_benchmarks):
        """Ensure bit-for-bit identical price string formatting between /api/v1/predict and /predict."""
        for record in golden_benchmarks:
            req = record["req"]
            res_v1 = client.post("/api/v1/predict", json=req)
            res_leg = client.post("/predict", json=req)

            assert res_v1.status_code == 200
            assert res_leg.status_code == 200
            assert res_v1.get_json()["price"] == res_leg.get_json()["price"]

    def test_pricing_formula_direct_oracle(self, golden_benchmarks):
        """Directly verify PricingService mathematical formula against golden benchmark records."""
        for record in golden_benchmarks:
            base_pred = record["base_prediction"]
            req = record["req"]
            year = int(req["year"])
            mileage = float(req["mileage"])
            owner = req["owner"]

            calc_price = PricingService.calculate_price(base_pred, year, mileage, owner)
            expected_price = record["final_price"]

            diff = abs(calc_price - expected_price)
            assert diff < 1e-5, f"Pricing calculation divergence for {req.get('id')}: diff={diff}"
            assert PricingService.format_price(calc_price) == record["formatted_price"]

    def test_randomized_order_idempotency_stress(self, client, golden_benchmarks):
        """Submit benchmarks in randomized sequence 40 times to verify absence of state leakage or drift."""
        rng = random.Random(42)
        indices = list(range(len(golden_benchmarks)))

        for _ in range(40):
            idx = rng.choice(indices)
            record = golden_benchmarks[idx]
            res = client.post("/api/v1/predict", json=record["req"])
            assert res.status_code == 200
            assert res.get_json()["price"] == record["formatted_price"]


# ==============================================================================
# 2. UI Blueprint Verification
# ==============================================================================
class TestAdversarialUI:
    """Verifies that GET / serves HTML with complete brand and hierarchy data."""

    def test_get_root_serves_html_with_brand_options(self, client):
        """GET / must respond with HTTP 200, text/html content-type, and brand select options."""
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.content_type

        html = res.get_data(as_text=True)
        assert "<!DOCTYPE html>" in html
        assert '<select id="brand"' in html
        assert '<select id="model"' in html
        assert '<button type="button" onclick="predictPrice()"' in html

    def test_car_hierarchy_json_embedded_and_valid(self, client):
        """Embedded carHierarchy JavaScript constant must contain valid JSON representing brand-to-model mapping."""
        res = client.get("/")
        assert res.status_code == 200
        html = res.get_data(as_text=True)

        # Extract 'const carHierarchy = { ... };'
        match = re.search(r"const\s+carHierarchy\s*=\s*(\{.*?\});", html, re.DOTALL)
        assert match is not None, "carHierarchy JSON assignment not found in HTML response"

        hierarchy_raw = match.group(1)
        hierarchy = json.loads(hierarchy_raw)
        assert isinstance(hierarchy, dict)
        assert len(hierarchy) > 0, "carHierarchy dictionary must not be empty"

        # Check major known brands exist with list of models
        for major_brand in ["Maruti", "Hyundai", "Honda", "Toyota"]:
            assert major_brand in hierarchy, f"Brand {major_brand} missing from embedded hierarchy"
            models = hierarchy[major_brand]
            assert isinstance(models, list)
            assert len(models) > 0, f"Models list for {major_brand} must not be empty"

    def test_ui_query_string_injection_resilience(self, client):
        """UI route must safely handle strange or malicious query strings without crashing."""
        for qs in [
            "?brand=<script>alert(1)</script>",
            "?model=../../etc/passwd",
            "?year=9999999999999999999999999999",
            "?foo=bar&baz=1&quote=' OR '1'='1",
        ]:
            res = client.get(f"/{qs}")
            assert res.status_code == 200
            assert "text/html" in res.content_type


# ==============================================================================
# 3. OpenAPI Documentation Verification
# ==============================================================================
class TestAdversarialOpenAPIDocs:
    """Verifies that GET /openapi.json and GET /apidocs respond with HTTP 200 and accurate schema specifications."""

    def test_openapi_json_returns_200_and_valid_spec(self, client):
        """GET /openapi.json returns HTTP 200, application/json, and OpenAPI 3.0 specification."""
        res = client.get("/openapi.json")
        assert res.status_code == 200
        assert "application/json" in res.content_type

        spec = res.get_json()
        assert isinstance(spec, dict)
        assert spec.get("openapi", "").startswith("3.0")
        assert "info" in spec
        assert "paths" in spec
        assert "components" in spec

        # Verify essential paths exist
        paths = spec["paths"]
        assert "/api/v1/predict" in paths
        assert "post" in paths["/api/v1/predict"]
        assert "/predict" in paths
        assert "post" in paths["/predict"]
        assert "/health/live" in paths
        assert "/health/ready" in paths
        assert "/" in paths
        assert "/openapi.json" in paths
        assert "/apidocs" in paths

        # Verify component schemas
        schemas = spec["components"]["schemas"]
        for schema_name in [
            "PredictionRequest",
            "PredictionResponse",
            "LegacyPredictionResponse",
            "ErrorResponse",
            "ValidationErrorDetail",
        ]:
            assert schema_name in schemas, f"Schema {schema_name} missing from components"

    def test_apidocs_swagger_ui_returns_200_html(self, client):
        """GET /apidocs serves HTML with Swagger UI container and script."""
        res = client.get("/apidocs")
        assert res.status_code == 200
        assert "text/html" in res.content_type

        html = res.get_data(as_text=True)
        assert "swagger-ui" in html
        assert "/openapi.json" in html

    def test_documentation_route_aliases(self, client):
        """Verify documentation aliases /apidocs/, /docs, /docs/ all return HTTP 200."""
        for alias in ["/apidocs/", "/docs", "/docs/"]:
            res = client.get(alias)
            assert res.status_code == 200, f"Alias {alias} returned status {res.status_code}"
            assert "text/html" in res.content_type


# ==============================================================================
# 4. Adversarial Payload & Schema Validation Stress
# ==============================================================================
class TestAdversarialPayloadValidation:
    """Adversarial stress-testing of Pydantic validation on /api/v1/predict and /predict."""

    def test_non_json_content_types_rejected(self, client, valid_payload):
        """Submitting non-JSON Content-Type must return HTTP 415 or 400 with structured JSON."""
        for ct in ["text/plain", "application/xml", "multipart/form-data"]:
            res = client.post("/api/v1/predict", data=json.dumps(valid_payload), content_type=ct)
            assert res.status_code in [400, 415]
            data = res.get_json()
            assert isinstance(data, dict)
            assert "error" in data

    def test_corrupted_json_syntax_rejected(self, client):
        """Submitting broken JSON syntax must return HTTP 400 with structured JSON."""
        res = client.post(
            "/api/v1/predict",
            data='{"brand": "Maruti", "model": broken_syntax',
            content_type="application/json",
        )
        assert res.status_code == 400
        data = res.get_json()
        assert isinstance(data, dict)
        assert "error" in data or "details" in data

    def test_empty_or_non_object_json_bodies(self, client):
        """Empty JSON, arrays, primitives, and null must return HTTP 400."""
        for invalid_body in ["{}", "[]", "[1, 2, 3]", '"just a string"', "12345", "null", "true"]:
            res = client.post(
                "/api/v1/predict",
                data=invalid_body,
                content_type="application/json",
            )
            assert res.status_code == 400
            data = res.get_json()
            assert isinstance(data, dict)
            assert "error" in data

    @pytest.mark.parametrize(
        "field_name,bad_val",
        [
            ("year", 1979),
            ("year", 2031),
            ("year", -1),
            ("year", "nineteen-ninety"),
            ("year", 2019.5),  # Non-integer float for integer year
            ("km_driven", -0.01),
            ("km_driven", -50000),
            ("km_driven", "abc"),
            ("mileage", -0.1),
            ("mileage", 100.1),
            ("mileage", "invalid"),
            ("engine", 99.9),
            ("engine", 10000.1),
            ("engine", 0),
            ("max_power", 9.9),
            ("max_power", 2000.1),
            ("max_power", -10),
            ("seats", 0),
            ("seats", 21),
            ("seats", -1),
            ("seats", 5.5),  # Non-integer float for integer seats
            ("brand", ""),
            ("brand", "   "),
            ("model", ""),
            ("model", "   "),
            ("fuel", "Nuclear"),
            ("fuel", "Hydrogen"),
            ("transmission", "CVT"),
            ("transmission", "Dual-Clutch"),
            ("seller_type", "Auction"),
            ("owner", "Fifth Owner"),
            ("owner", "Zero Owner"),
        ],
    )
    def test_field_boundary_and_domain_violations_rejected(self, client, valid_payload, field_name, bad_val):
        """Each field boundary and enum domain violation must be rejected with HTTP 400 and structured detail."""
        payload = valid_payload.copy()
        payload[field_name] = bad_val

        # Test on /api/v1/predict
        res_v1 = client.post("/api/v1/predict", json=payload)
        assert res_v1.status_code == 400, f"Expected 400 for {field_name}={bad_val}, got {res_v1.status_code}"
        data_v1 = res_v1.get_json()
        assert "error" in data_v1
        assert "details" in data_v1
        fields_reported = [d.get("field") for d in data_v1.get("details", [])]
        assert field_name in fields_reported, f"Field '{field_name}' not identified in error details: {fields_reported}"

        # Test on legacy /predict
        res_leg = client.post("/predict", json=payload)
        assert res_leg.status_code == 400, f"Expected 400 on /predict for {field_name}={bad_val}, got {res_leg.status_code}"

    @pytest.mark.parametrize(
        "field_name",
        [
            "brand",
            "model",
            "year",
            "km_driven",
            "mileage",
            "engine",
            "max_power",
            "seats",
            "fuel",
            "transmission",
            "seller_type",
            "owner",
        ],
    )
    def test_missing_each_required_field_rejected(self, client, valid_payload, field_name):
        """Omitting any single required field must result in HTTP 400."""
        payload = valid_payload.copy()
        del payload[field_name]

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 400
        data = res.get_json()
        fields_reported = [d.get("field") for d in data.get("details", [])]
        assert field_name in fields_reported

    @pytest.mark.parametrize(
        "field_name",
        [
            "brand",
            "model",
            "year",
            "km_driven",
            "mileage",
            "engine",
            "max_power",
            "seats",
            "fuel",
            "transmission",
            "seller_type",
            "owner",
        ],
    )
    def test_null_value_for_each_required_field_rejected(self, client, valid_payload, field_name):
        """Sending null/None for any required field must result in HTTP 400."""
        payload = valid_payload.copy()
        payload[field_name] = None

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 400
        data = res.get_json()
        fields_reported = [d.get("field") for d in data.get("details", [])]
        assert field_name in fields_reported

    def test_exact_numeric_boundary_acceptances(self, client, valid_payload):
        """Exact valid boundary limits must succeed and return HTTP 200."""
        test_cases = [
            ("year", 1980),
            ("year", 2030),
            ("km_driven", 0.0),
            ("mileage", 0.0),
            ("mileage", 100.0),
            ("engine", 100.0),
            ("engine", 10000.0),
            ("max_power", 10.0),
            ("max_power", 2000.0),
            ("seats", 1),
            ("seats", 20),
        ]
        for field, bound_val in test_cases:
            payload = valid_payload.copy()
            payload[field] = bound_val
            res = client.post("/api/v1/predict", json=payload)
            assert res.status_code == 200, f"Valid boundary {field}={bound_val} was unexpectedly rejected: {res.get_data(as_text=True)}"

    def test_case_and_whitespace_normalization_in_enums(self, client, valid_payload):
        """Mixed casing and surrounding whitespace in enums must be canonicalized and accepted."""
        payload = valid_payload.copy()
        payload["fuel"] = "  pEtRoL  "
        payload["transmission"] = "  mAnUaL "
        payload["seller_type"] = " inDiViDuAl "
        payload["owner"] = "  fIrSt OwNeR  "

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        assert res.get_json()["status"] == "success"

    def test_fourth_and_above_owner_canonical_variations(self, client, valid_payload):
        """Owner tier 'fourth and above owner' must be normalized and accepted."""
        for var in ["fourth and above owner", "FOURTH & ABOVE OWNER", "Fourth & Above Owner"]:
            payload = valid_payload.copy()
            payload["owner"] = var
            res = client.post("/api/v1/predict", json=payload)
            assert res.status_code == 200

    def test_unseen_brand_and_model_graceful_handling(self, client, valid_payload):
        """Completely novel/unseen brand and model must not crash inference (encoded as 0.0)."""
        payload = valid_payload.copy()
        payload["brand"] = "NonExistentBrandX"
        payload["model"] = "SuperHyperCarY"

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "success"
        assert isinstance(data["valuation"], float)
        assert data["valuation"] > 0

    def test_hostile_strings_sql_xss_unicode(self, client, valid_payload):
        """SQL injection, XSS, and Unicode emoji payloads in string fields must be safely sanitized/handled."""
        hostile_brands = [
            "'; DROP TABLE vehicles; --",
            "<script>alert(document.cookie)</script>",
            "🏎️ 🚗 🚘 🚙 🏍️",
            "A" * 5000,  # 5,000 character string
        ]
        for hostile_str in hostile_brands:
            payload = valid_payload.copy()
            payload["brand"] = hostile_str
            res = client.post("/api/v1/predict", json=payload)
            # Should safely return 200 (since it's a non-empty string and handled as unseen brand)
            assert res.status_code == 200
            assert "price" in res.get_json()

    def test_extra_arbitrary_fields_ignored(self, client, valid_payload):
        """Extra unexpected payload fields (e.g. metadata, tracking) must be ignored without raising errors."""
        payload = valid_payload.copy()
        payload["id"] = "TC-EXT-99"
        payload["meta"] = {"source": "mobile_app", "version": "2.4"}
        payload["arbitrary_list"] = [1, 2, 3]

        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        assert res.get_json()["status"] == "success"


# ==============================================================================
# 5. HTTP Protocol & Method Guard Tests
# ==============================================================================
class TestAdversarialHttpProtocol:
    """Verifies HTTP protocol edge cases, method restrictions, and error structures."""

    def test_method_not_allowed_on_prediction_endpoints(self, client):
        """GET, PUT, DELETE, PATCH on /api/v1/predict and /predict must return HTTP 405 with structured error."""
        disallowed_methods = ["get", "put", "delete", "patch"]
        for method in disallowed_methods:
            caller = getattr(client, method)
            
            # Test /api/v1/predict
            res_v1 = caller("/api/v1/predict")
            assert res_v1.status_code == 405, f"Expected 405 for {method.upper()} on /api/v1/predict, got {res_v1.status_code}"
            assert res_v1.is_json
            data_v1 = res_v1.get_json()
            assert data_v1.get("error") == "Method Not Allowed"

            # Test /predict
            res_leg = caller("/predict")
            assert res_leg.status_code == 405, f"Expected 405 for {method.upper()} on /predict, got {res_leg.status_code}"
            assert res_leg.is_json
            data_leg = res_leg.get_json()
            assert data_leg.get("error") == "Method Not Allowed"

    def test_not_found_on_api_route_returns_structured_json(self, client):
        """404 on API subpaths must return structured JSON, not raw HTML."""
        for path in ["/api/v1/non_existent_endpoint", "/api/v2/predict"]:
            res = client.get(path)
            assert res.status_code == 404
            assert res.is_json
            data = res.get_json()
            assert data.get("error") == "Not Found"


# ==============================================================================
# 6. Concurrency & Multi-threaded Stress Test
# ==============================================================================
class TestAdversarialConcurrency:
    """Verifies that ModelService and application factory endpoints survive concurrent multi-threaded requests."""

    def test_concurrent_prediction_requests(self, client, golden_benchmarks):
        """Execute 64 concurrent prediction requests across 8 worker threads without failure or race conditions."""
        tc01_req = golden_benchmarks[0]["req"]
        tc02_req = golden_benchmarks[1]["req"]
        tc01_expected = golden_benchmarks[0]["formatted_price"]
        tc02_expected = golden_benchmarks[1]["formatted_price"]

        def make_request(req_and_expected):
            req, expected = req_and_expected
            res = client.post("/api/v1/predict", json=req)
            return res.status_code, res.get_json().get("price") == expected

        tasks = []
        for i in range(64):
            if i % 2 == 0:
                tasks.append((tc01_req, tc01_expected))
            else:
                tasks.append((tc02_req, tc02_expected))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(make_request, tasks))

        for status_code, match in results:
            assert status_code == 200
            assert match is True
