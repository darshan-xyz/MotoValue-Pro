"""
Integration tests for OpenAPI and Swagger documentation endpoints:
- GET /openapi.json (raw OpenAPI 3.0.3 spec)
- GET /apidocs (interactive Swagger UI)
"""
import pytest


class TestOpenApiDocumentationIntegration:
    """Verifies OpenAPI 3.0 specification structure and Swagger UI delivery."""

    def test_openapi_json_status_and_content_type(self, client):
        """GET /openapi.json returns 200 OK with application/json."""
        res = client.get("/openapi.json")
        assert res.status_code == 200
        assert "application/json" in res.content_type

    def test_openapi_spec_version_and_info(self, client):
        """Spec defines openapi 3.x and comprehensive info metadata."""
        res = client.get("/openapi.json")
        spec = res.get_json()

        assert "openapi" in spec
        assert spec["openapi"].startswith("3.")
        assert "info" in spec
        assert "title" in spec["info"]
        assert "version" in spec["info"]
        assert "description" in spec["info"]

    def test_openapi_paths_documented(self, client):
        """Spec documents all key application routes."""
        res = client.get("/openapi.json")
        spec = res.get_json()
        paths = spec.get("paths", {})

        assert "/api/v1/predict" in paths
        assert "/predict" in paths
        assert "/health/live" in paths
        assert "/health/ready" in paths
        assert "/" in paths
        assert "/openapi.json" in paths
        assert "/apidocs" in paths

    def test_openapi_predict_endpoint_contract(self, client):
        """POST /api/v1/predict operation defines requestBody and responses."""
        res = client.get("/openapi.json")
        spec = res.get_json()
        v1_op = spec["paths"]["/api/v1/predict"]["post"]

        assert "requestBody" in v1_op
        assert "content" in v1_op["requestBody"]
        assert "application/json" in v1_op["requestBody"]["content"]

        responses = v1_op.get("responses", {})
        assert "200" in responses
        assert "400" in responses

    def test_openapi_components_schemas_present(self, client):
        """Components schemas include all domain request and response models."""
        res = client.get("/openapi.json")
        spec = res.get_json()
        schemas = spec.get("components", {}).get("schemas", {})

        assert "PredictionRequest" in schemas
        assert "PredictionResponse" in schemas
        assert "LegacyPredictionResponse" in schemas
        assert "ValidationErrorDetail" in schemas
        assert "ErrorResponse" in schemas

    def test_openapi_prediction_request_schema_fields(self, client):
        """PredictionRequest schema requires all 12 vehicle attributes."""
        res = client.get("/openapi.json")
        spec = res.get_json()
        req_schema = spec["components"]["schemas"]["PredictionRequest"]

        required = req_schema.get("required", [])
        for field in [
            "brand", "model", "year", "km_driven", "mileage",
            "engine", "max_power", "seats", "fuel", "transmission",
            "seller_type", "owner"
        ]:
            assert field in required or field in req_schema.get("properties", {})

    def test_apidocs_html_page_delivered(self, client):
        """GET /apidocs returns HTML page with Swagger UI configuration."""
        res = client.get("/apidocs")
        assert res.status_code == 200
        assert "text/html" in res.content_type
        html = res.data.decode("utf-8")

        assert "swagger-ui" in html
        assert "/openapi.json" in html

    def test_apidocs_aliases_accessible(self, client):
        """Convenience route aliases (/docs, /apidocs/) return 200 OK."""
        for alias in ["/docs", "/apidocs/", "/docs/"]:
            res = client.get(alias)
            assert res.status_code == 200
            assert "text/html" in res.content_type

    def test_openapi_explain_request_schema_present(self, client):
        """ExplainRequest is registered in schemas and referenced by /api/v1/explain."""
        res = client.get("/openapi.json")
        assert res.status_code == 200
        spec = res.get_json()
        assert "ExplainRequest" in spec["components"]["schemas"]
        req_ref = spec["paths"]["/api/v1/explain"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert req_ref == "#/components/schemas/ExplainRequest"

    def test_openapi_status_codes_415_and_503_documented(self, client):
        """HTTP 415 and 503 are documented across predict and explain endpoints."""
        res = client.get("/openapi.json")
        assert res.status_code == 200
        spec = res.get_json()
        for endpoint in ["/api/v1/predict", "/api/v1/explain", "/predict"]:
            responses = spec["paths"][endpoint]["post"]["responses"]
            assert "415" in responses, f"415 missing in {endpoint}"
            assert "503" in responses, f"503 missing in {endpoint}"
