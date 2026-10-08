"""Unit tests for PricingService heuristics and constants."""

import pytest
from car_price_predictor.services.constants import (
    OWNER_MULTIPLIERS,
    DEFAULT_OWNER_MULTIPLIER,
    YEAR_BASELINE,
    YEAR_FACTOR_RATE,
    YEAR_FACTOR_MIN,
    YEAR_FACTOR_MAX,
    MILEAGE_BASELINE,
    MILEAGE_FACTOR_RATE,
    MILEAGE_FACTOR_MIN,
    MILEAGE_FACTOR_MAX,
)
from car_price_predictor.services.pricing import PricingService


class TestPricingConstants:
    """Verify named constants are correctly defined and match domain specifications."""

    def test_owner_multipliers_constants(self):
        assert OWNER_MULTIPLIERS == {
            "First Owner": 1.0,
            "Second Owner": 0.95,
            "Third Owner": 0.90,
            "Fourth & Above Owner": 0.85,
        }
        assert DEFAULT_OWNER_MULTIPLIER == 1.0

    def test_year_heuristic_constants(self):
        assert YEAR_BASELINE == 2018
        assert YEAR_FACTOR_RATE == 0.05
        assert YEAR_FACTOR_MIN == 0.1
        assert YEAR_FACTOR_MAX == 2.0

    def test_mileage_heuristic_constants(self):
        assert MILEAGE_BASELINE == 18.0
        assert MILEAGE_FACTOR_RATE == 0.01
        assert MILEAGE_FACTOR_MIN == 0.8
        assert MILEAGE_FACTOR_MAX == 1.2


class TestOwnerMultipliers:
    """Verify owner depreciation multipliers."""

    @pytest.mark.parametrize(
        "owner,expected_multiplier",
        [
            ("First Owner", 1.0),
            ("Second Owner", 0.95),
            ("Third Owner", 0.90),
            ("Fourth & Above Owner", 0.85),
        ],
    )
    def test_known_owner_multipliers(self, owner, expected_multiplier):
        price = PricingService.calculate_price(100_000.0, 2018, 18.0, owner)
        assert price == pytest.approx(100_000.0 * expected_multiplier)

    @pytest.mark.parametrize(
        "unknown_owner",
        [
            "Fifth Owner",
            "Unknown",
            "",
            "Random string",
        ],
    )
    def test_unknown_owner_fallback(self, unknown_owner):
        """Unrecognized owner tiers default to 1.0 multiplier without raising errors."""
        price = PricingService.calculate_price(100_000.0, 2018, 18.0, unknown_owner)
        assert price == pytest.approx(100_000.0 * DEFAULT_OWNER_MULTIPLIER)


class TestYearAdjustmentAndClamping:
    """Verify year depreciation gradient and clamping limits."""

    def test_year_baseline_2018(self):
        price = PricingService.calculate_price(100_000.0, 2018, 18.0, "First Owner")
        assert price == pytest.approx(100_000.0)

    @pytest.mark.parametrize(
        "year,expected_factor",
        [
            (2019, 1.05),
            (2020, 1.10),
            (2022, 1.20),
            (2017, 0.95),
            (2015, 0.85),
            (2010, 0.60),
        ],
    )
    def test_year_linear_gradient_unclamped(self, year, expected_factor):
        price = PricingService.calculate_price(100_000.0, year, 18.0, "First Owner")
        assert price == pytest.approx(100_000.0 * expected_factor)

    def test_year_upper_boundary_exact(self):
        price = PricingService.calculate_price(100_000.0, 2038, 18.0, "First Owner")
        assert price == pytest.approx(200_000.0)

    @pytest.mark.parametrize("future_year", [2039, 2045, 2050, 2100])
    def test_year_upper_clamping_enforced(self, future_year):
        price = PricingService.calculate_price(100_000.0, future_year, 18.0, "First Owner")
        assert price == pytest.approx(200_000.0)

    def test_year_lower_boundary_exact(self):
        price = PricingService.calculate_price(100_000.0, 2000, 18.0, "First Owner")
        assert price == pytest.approx(10_000.0)

    @pytest.mark.parametrize("old_year", [1999, 1990, 1980, 1950])
    def test_year_lower_clamping_enforced(self, old_year):
        price = PricingService.calculate_price(100_000.0, old_year, 18.0, "First Owner")
        assert price == pytest.approx(10_000.0)


class TestMileageAdjustmentAndClamping:
    """Verify fuel efficiency mileage gradient and clamping limits."""

    def test_mileage_baseline_18(self):
        price = PricingService.calculate_price(100_000.0, 2018, 18.0, "First Owner")
        assert price == pytest.approx(100_000.0)

    @pytest.mark.parametrize(
        "mileage,expected_factor",
        [
            (19.0, 1.01),
            (23.0, 1.05),
            (28.0, 1.10),
            (17.0, 0.99),
            (15.0, 0.97),
            (10.0, 0.92),
            (0.0, 0.82),
        ],
    )
    def test_mileage_linear_gradient_unclamped(self, mileage, expected_factor):
        price = PricingService.calculate_price(100_000.0, 2018, mileage, "First Owner")
        assert price == pytest.approx(100_000.0 * expected_factor)

    def test_mileage_upper_boundary_exact(self):
        price = PricingService.calculate_price(100_000.0, 2018, 38.0, "First Owner")
        assert price == pytest.approx(120_000.0)

    @pytest.mark.parametrize("high_mileage", [40.0, 45.0, 50.0, 100.0])
    def test_mileage_upper_clamping_enforced(self, high_mileage):
        price = PricingService.calculate_price(100_000.0, 2018, high_mileage, "First Owner")
        assert price == pytest.approx(120_000.0)

    def test_mileage_lower_boundary_exact(self):
        price = PricingService.calculate_price(100_000.0, 2018, -2.0, "First Owner")
        assert price == pytest.approx(80_000.0)

    @pytest.mark.parametrize("low_mileage", [-5.0, -10.0, -50.0])
    def test_mileage_lower_clamping_enforced(self, low_mileage):
        price = PricingService.calculate_price(100_000.0, 2018, low_mileage, "First Owner")
        assert price == pytest.approx(80_000.0)


class TestCombinedPricingInteractions:
    """Verify simultaneous multi-factor compounding matches golden benchmarks."""

    def test_golden_benchmark_vectors(self, golden_benchmarks):
        for case in golden_benchmarks:
            req = case.get("req", case)
            b_id = req.get("id", case.get("id", "UNKNOWN"))
            computed_price = PricingService.calculate_price(
                base_prediction=case["base_prediction"],
                year=req["year"],
                mileage=req["mileage"],
                owner=req["owner"],
            )
            expected_final = case.get("expected_final_price", case.get("final_price"))
            assert computed_price == pytest.approx(expected_final, abs=1e-2), (
                f"Failed for benchmark {b_id}"
            )
            expected_formatted = case.get("expected_formatted_price", case.get("formatted_price"))
            formatted = PricingService.format_price(computed_price)
            assert formatted == expected_formatted, (
                f"Format mismatch for benchmark {b_id}"
            )


class TestFormatPrice:
    """Verify string formatting matches standard comma-separated currency representation."""

    @pytest.mark.parametrize(
        "raw_value,expected_formatted",
        [
            (508560.1499999999, "508,560.15"),
            (3312029.3709999993, "3,312,029.37"),
            (209300.9444066448, "209,300.94"),
            (1165296.0, "1,165,296.00"),
            (31279.173343749993, "31,279.17"),
            (1000.0, "1,000.00"),
            (100.5, "100.50"),
            (0.0, "0.00"),
            (123456789.987, "123,456,789.99"),
        ],
    )
    def test_format_price_values(self, raw_value, expected_formatted):
        assert PricingService.format_price(raw_value) == expected_formatted
