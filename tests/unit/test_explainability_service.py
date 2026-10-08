"""
Unit tests for Feature Importance and Explainability Service (R3).
"""
import pytest

from car_price_predictor.schemas.prediction import ExplainResponse, PredictionRequest
from car_price_predictor.services.explainability import ExplainabilityService
from car_price_predictor.services.model_service import ModelService


@pytest.fixture(scope="module")
def model_service():
    """Shared ModelService fixture."""
    service = ModelService(load_on_init=True)
    assert service.is_ready()
    return service


@pytest.fixture
def explain_service(model_service):
    """ExplainabilityService instance fixture."""
    return ExplainabilityService(model_service)


class TestExplainabilityServiceUnit:
    """Tests feature attribution decomposition and pricing rule reconciliation."""

    def test_service_initialization(self, model_service):
        """Service initializes cleanly with ModelService instance."""
        service = ExplainabilityService(model_service)
        assert service.model_service is model_service
        assert "max_power" in service.BASELINE_VEHICLE

    def test_get_global_feature_importances(self, explain_service):
        """Extracts 14 normalized Gini importances summing to 1.0."""
        importances = explain_service.get_global_feature_importances()
        assert isinstance(importances, dict)
        assert len(importances) == 14
        assert pytest.approx(sum(importances.values()), rel=1e-3) == 1.0

        # Dominant drivers
        assert importances["max_power"] > 0.40
        assert importances["km_driven"] > 0.05

    def test_explain_prediction_valid_payload_contract(self, explain_service, valid_payload):
        """explain_prediction returns structured ExplainResponse."""
        resp = explain_service.explain_prediction(valid_payload)
        assert isinstance(resp, ExplainResponse)
        assert resp.status == "success"
        assert resp.currency == "INR"
        assert resp.price == "508,560.15"
        assert abs(resp.valuation - 508560.15) < 0.1
        assert resp.base_ml_valuation > 0

        # Confidence interval
        assert resp.confidence_interval is not None
        assert resp.confidence_interval.lower_bound <= resp.valuation <= resp.confidence_interval.upper_bound

        # Pricing adjustments
        adj = resp.pricing_adjustments
        assert adj.owner_multiplier == 1.0
        assert adj.year_factor == 1.05
        assert adj.mileage_factor == 1.005
        assert adj.total_heuristic_multiplier == pytest.approx(1.05525, abs=1e-4)

        # Feature contributions
        assert len(resp.feature_contributions) > 5
        assert len(resp.top_drivers) > 0

    def test_explain_prediction_feature_contribution_fields(self, explain_service, valid_payload):
        """Each feature contribution contains required metadata, direction, and description."""
        resp = explain_service.explain_prediction(valid_payload)
        for fc in resp.feature_contributions:
            assert fc.feature is not None
            assert fc.direction in ("positive", "negative", "neutral")
            assert fc.source in ("ml_model", "pricing_rule")
            assert isinstance(fc.impact_amount, float)
            assert isinstance(fc.percentage_impact, float)
            assert len(fc.description) > 0

    def test_explain_prediction_second_owner_depreciation(self, explain_service, valid_payload):
        """Second owner vehicle reflects negative owner contribution."""
        payload = valid_payload.copy()
        payload["owner"] = "Second Owner"

        resp = explain_service.explain_prediction(payload)
        assert resp.pricing_adjustments.owner_multiplier == 0.95

        owner_contrib = next(c for c in resp.feature_contributions if c.feature == "owner")
        assert owner_contrib.impact_amount < 0
        assert owner_contrib.direction == "negative"

    def test_explain_prediction_unseen_brand_safety(self, explain_service, valid_payload):
        """Unseen brand/model explanation executes cleanly without crashing."""
        payload = valid_payload.copy()
        payload["brand"] = "AeroSpaceCar"
        payload["model"] = "Orbit9"

        resp = explain_service.explain_prediction(payload)
        assert resp.valuation > 0
        assert resp.confidence_interval.lower_bound <= resp.valuation <= resp.confidence_interval.upper_bound

    def test_explain_prediction_execution_latency(self, explain_service, valid_payload):
        """Explanation runs in under 100ms fulfilling latency constraint and API SLA."""
        import time

        # Warmup invocation to initialize JIT and memory structures
        explain_service.explain_prediction(valid_payload)

        iterations = 3
        start = time.perf_counter()
        for _ in range(iterations):
            resp = explain_service.explain_prediction(valid_payload)
        elapsed_per_call = ((time.perf_counter() - start) / iterations) * 1000.0

        assert resp.valuation > 0
        assert elapsed_per_call < 100.0

    def test_pricing_rule_contributions_exact_additive_reconciliation(self, explain_service, valid_payload):
        """Sum of pricing rule feature contributions identically matches net_adjustment_amount."""
        for p in [
            valid_payload,
            dict(valid_payload, owner="Second Owner", year=2021, mileage=22.0),
            dict(valid_payload, owner="Fourth & Above Owner", year=2012, mileage=12.0),
        ]:
            resp = explain_service.explain_prediction(p)
            pricing_rule_sum = sum(
                fc.impact_amount for fc in resp.feature_contributions if fc.source == "pricing_rule"
            )
            net_adj = resp.pricing_adjustments.net_adjustment_amount
            assert pytest.approx(pricing_rule_sum, abs=0.01) == net_adj

    def test_pricing_rule_mileage_contribution_present(self, explain_service, valid_payload):
        """Mileage pricing rule factor is explicitly included with source='pricing_rule'."""
        resp = explain_service.explain_prediction(valid_payload)
        mileage_rules = [
            fc for fc in resp.feature_contributions
            if fc.feature == "mileage" and fc.source == "pricing_rule"
        ]
        assert len(mileage_rules) == 1
        assert mileage_rules[0].description != ""

    def test_explain_prediction_third_and_fourth_owner_multipliers(self, explain_service, valid_payload):
        """Third Owner (0.90) and Fourth & Above Owner (0.85) multipliers reflect properly."""
        p3 = valid_payload.copy()
        p3["owner"] = "Third Owner"
        r3 = explain_service.explain_prediction(p3)
        assert r3.pricing_adjustments.owner_multiplier == 0.90

        p4 = valid_payload.copy()
        p4["owner"] = "Fourth & Above Owner"
        r4 = explain_service.explain_prediction(p4)
        assert r4.pricing_adjustments.owner_multiplier == 0.85

    def test_explain_prediction_year_gradient_clamping(self, explain_service, valid_payload):
        """Extreme years clamp between 0.1 and 2.0."""
        # 1980 -> clamped to 0.1
        p_old = valid_payload.copy()
        p_old["year"] = 1980
        r_old = explain_service.explain_prediction(p_old)
        assert r_old.pricing_adjustments.year_factor == 0.1

        # 2030 -> clamped to 1.6 (rate: 1.0 + (2030-2018)*0.05 = 1.6)
        p_new = valid_payload.copy()
        p_new["year"] = 2030
        r_new = explain_service.explain_prediction(p_new)
        assert r_new.pricing_adjustments.year_factor == 1.6

    def test_explain_service_unready_model_raises(self):
        """ExplainabilityService raises RuntimeError if ModelService is not ready."""
        unready_service = ModelService(model_path="non_existent.joblib", load_on_init=False)
        service = ExplainabilityService(unready_service)
        with pytest.raises(RuntimeError, match="not ready"):
            service.explain_prediction({"brand": "Maruti"})

    def test_explain_prediction_base_ml_price_alias_matches(self, explain_service, valid_payload):
        """Both base_ml_valuation and base_ml_price alias are populated with equal values."""
        resp = explain_service.explain_prediction(valid_payload)
        assert resp.base_ml_price is not None
        assert resp.base_ml_price == resp.base_ml_valuation
