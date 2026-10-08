"""
Unit tests for Valuation Confidence Intervals (R4).
"""
import time
import pytest

from car_price_predictor.services.model_service import ModelService
from car_price_predictor.services.pricing import PricingService


@pytest.fixture(scope="module")
def model_service():
    """ModelService fixture with active trained model."""
    service = ModelService(load_on_init=True)
    assert service.is_ready()
    return service


class TestConfidenceIntervalsUnit:
    """Tests empirical ensemble confidence intervals and heuristic scaling."""

    def test_predict_with_interval_ordering_invariant(self, model_service, valid_payload):
        """Strictly asserts lower_base <= base_prediction <= upper_base."""
        input_df = model_service.preprocessor.preprocess(valid_payload)
        base_pred, lower_base, upper_base = model_service.predict_with_interval(
            input_df, confidence_level=0.80
        )

        assert lower_base <= base_pred <= upper_base
        assert lower_base >= 0.0
        assert upper_base > lower_base

    def test_pricing_service_calculate_price_interval_ordering(self):
        """PricingService scales bounds preserving lower_bound <= valuation <= upper_bound."""
        base_pred = 500000.0
        lower_base = 350000.0
        upper_base = 650000.0

        val, lower, upper = PricingService.calculate_price_interval(
            base_prediction=base_pred,
            lower_base=lower_base,
            upper_base=upper_base,
            year=2019,
            mileage=18.5,
            owner="First Owner",
        )

        assert lower <= val <= upper
        assert lower >= 0.0

    def test_confidence_level_parameter_expansion(self, model_service, valid_payload):
        """Higher confidence level (e.g. 95% vs 80%) yields wider or equal intervals."""
        input_df = model_service.preprocessor.preprocess(valid_payload)

        _, lower_80, upper_80 = model_service.predict_with_interval(input_df, confidence_level=0.80)
        _, lower_95, upper_95 = model_service.predict_with_interval(input_df, confidence_level=0.95)

        spread_80 = upper_80 - lower_80
        spread_95 = upper_95 - lower_95

        assert spread_95 >= spread_80
        assert lower_95 <= lower_80
        assert upper_95 >= upper_80

    def test_heuristic_scaling_proportionality(self):
        """Pricing multiplier M scales both lower and upper bounds proportionally."""
        base_pred = 400000.0
        lower_base = 300000.0
        upper_base = 500000.0

        # Multiplier: 1.0 (owner) * 1.0 (2018) * 1.0 (18.0) = 1.0
        v1, l1, u1 = PricingService.calculate_price_interval(
            base_pred, lower_base, upper_base, year=2018, mileage=18.0, owner="First Owner"
        )
        assert v1 == pytest.approx(400000.0)
        assert l1 == pytest.approx(300000.0)
        assert u1 == pytest.approx(500000.0)

        # Multiplier with depreciation: 0.90 (Third Owner)
        v2, l2, u2 = PricingService.calculate_price_interval(
            base_pred, lower_base, upper_base, year=2018, mileage=18.0, owner="Third Owner"
        )
        assert v2 == pytest.approx(400000.0 * 0.90)
        assert l2 == pytest.approx(300000.0 * 0.90)
        assert u2 == pytest.approx(500000.0 * 0.90)

    def test_non_negative_lower_bound_boundary(self):
        """Lower bound is never negative even with extreme depreciation."""
        val, lower, upper = PricingService.calculate_price_interval(
            base_prediction=10000.0,
            lower_base=0.0,
            upper_base=20000.0,
            year=1980,
            mileage=5.0,
            owner="Fourth & Above Owner",
        )
        assert lower >= 0.0
        assert lower <= val <= upper

    def test_confidence_interval_inference_latency(self, model_service, valid_payload):
        """predict_with_interval executes in under 25ms."""
        input_df = model_service.preprocessor.preprocess(valid_payload)

        # Warmup
        model_service.predict_with_interval(input_df)

        start = time.perf_counter()
        for _ in range(5):
            model_service.predict_with_interval(input_df)
        elapsed_per_call = (time.perf_counter() - start) / 5.0 * 1000.0

        assert elapsed_per_call < 25.0

    def test_predict_with_interval_unready_service_raises(self):
        """predict_with_interval raises RuntimeError if ModelService is not ready."""
        unready = ModelService(model_path="non_existent.joblib", load_on_init=False)
        import pandas as pd
        with pytest.raises(RuntimeError, match="not loaded or ready"):
            unready.predict_with_interval(pd.DataFrame())

    def test_confidence_level_extreme_bounds_clamped(self, model_service, valid_payload):
        """Extreme confidence levels (0.0001, 1.0) clamp gracefully without math crashes."""
        input_df = model_service.preprocessor.preprocess(valid_payload)
        _, l_low, u_low = model_service.predict_with_interval(input_df, confidence_level=0.0001)
        _, l_high, u_high = model_service.predict_with_interval(input_df, confidence_level=1.0)

        assert l_low <= u_low
        assert l_high <= u_high

    def test_all_8_golden_benchmarks_interval_containment(self, model_service, golden_benchmarks):
        """All 8 golden benchmarks strictly satisfy lower_bound <= valuation <= upper_bound."""
        for record in golden_benchmarks:
            payload = record["req"]
            input_df = model_service.preprocessor.preprocess(payload)
            base_pred, lower_base, upper_base = model_service.predict_with_interval(input_df)

            year, mileage, owner = model_service.preprocessor.extract_pricing_inputs(payload)
            val, lower, upper = PricingService.calculate_price_interval(
                base_pred, lower_base, upper_base, year, mileage, owner
            )

            assert lower <= val <= upper
            assert abs(val - float(record["final_price"])) < 0.1
