"""
Pricing heuristic service module.
Applies post-prediction business rules for owner history, vehicle age, and mileage efficiency.
"""
from __future__ import annotations
from .constants import (
    OWNER_MULTIPLIERS,
    DEFAULT_OWNER_MULTIPLIER,
    YEAR_BASELINE,
    YEAR_RATE,
    YEAR_FACTOR_MIN,
    YEAR_FACTOR_MAX,
    MILEAGE_BASELINE,
    MILEAGE_RATE,
    MILEAGE_FACTOR_MIN,
    MILEAGE_FACTOR_MAX,
    PRICE_FORMAT,
)


class PricingService:
    """Calculates post-model pricing adjustments deterministically."""

    @staticmethod
    def get_owner_multiplier(owner: str) -> float:
        """Returns depreciation multiplier for vehicle ownership history."""
        return float(OWNER_MULTIPLIERS.get(owner, DEFAULT_OWNER_MULTIPLIER))

    @staticmethod
    def get_year_factor(year: int) -> float:
        """Calculates clamped age factor relative to 2018 baseline (0.1 to 2.0)."""
        year_diff = int(year) - YEAR_BASELINE
        year_factor = 1.0 + (year_diff * YEAR_RATE)
        if year_factor < YEAR_FACTOR_MIN:
            return float(YEAR_FACTOR_MIN)
        if year_factor > YEAR_FACTOR_MAX:
            return float(YEAR_FACTOR_MAX)
        return float(year_factor)

    @staticmethod
    def get_mileage_factor(mileage: float) -> float:
        """Calculates clamped fuel efficiency factor relative to 18.0 kmpl baseline (0.8 to 1.2)."""
        mileage_diff = float(mileage) - MILEAGE_BASELINE
        mileage_factor = 1.0 + (mileage_diff * MILEAGE_RATE)
        if mileage_factor < MILEAGE_FACTOR_MIN:
            return float(MILEAGE_FACTOR_MIN)
        if mileage_factor > MILEAGE_FACTOR_MAX:
            return float(MILEAGE_FACTOR_MAX)
        return float(mileage_factor)

    @classmethod
    def calculate_price(
        cls,
        base_prediction: float,
        year: int,
        mileage: float,
        owner: str,
    ) -> float:
        """
        Applies owner, year, and mileage heuristic multipliers to base_prediction sequentially.
        Matches the legacy app.py calculation with 100% precision.
        """
        final_price = float(base_prediction)
        final_price *= cls.get_owner_multiplier(owner)
        final_price *= cls.get_year_factor(year)
        final_price *= cls.get_mileage_factor(mileage)
        return float(final_price)

    @classmethod
    def calculate_price_interval(
        cls,
        base_prediction: float,
        lower_base: float,
        upper_base: float,
        year: int,
        mileage: float,
        owner: str,
    ) -> tuple[float, float, float]:
        """
        Applies owner, year, and mileage heuristic multipliers to base prediction
        and uncertainty bounds, guaranteeing lower_bound <= valuation <= upper_bound.

        Returns:
            tuple[float, float, float]: (valuation, lower_bound, upper_bound)
        """
        owner_mult = cls.get_owner_multiplier(owner)
        year_factor = cls.get_year_factor(year)
        mileage_factor = cls.get_mileage_factor(mileage)
        total_multiplier = owner_mult * year_factor * mileage_factor

        valuation = float(base_prediction * total_multiplier)
        lower_bound = float(max(0.0, lower_base * total_multiplier))
        upper_bound = float(max(valuation, upper_base * total_multiplier))
        lower_bound = float(min(lower_bound, valuation))

        return valuation, lower_bound, upper_bound

    @staticmethod
    def format_price(final_price: float) -> str:
        """Returns comma-separated currency representation with 2 decimal places."""
        return PRICE_FORMAT.format(float(final_price))
