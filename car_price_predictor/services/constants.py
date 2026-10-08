"""
Domain constants for CarPricePredictor.
Replaces magic numbers in business rules, model columns, and validation limits.
"""
from typing import Final

# Owner Multipliers
OWNER_FIRST: Final[str] = "First Owner"
OWNER_SECOND: Final[str] = "Second Owner"
OWNER_THIRD: Final[str] = "Third Owner"
OWNER_FOURTH_ABOVE: Final[str] = "Fourth & Above Owner"

OWNER_MULTIPLIERS: Final[dict[str, float]] = {
    OWNER_FIRST: 1.0,
    OWNER_SECOND: 0.95,
    OWNER_THIRD: 0.90,
    OWNER_FOURTH_ABOVE: 0.85,
}
OWNER_FACTORS: Final[dict[str, float]] = OWNER_MULTIPLIERS
DEFAULT_OWNER_MULTIPLIER: Final[float] = 1.0

# Year Heuristics
YEAR_BASELINE: Final[int] = 2018
YEAR_RATE: Final[float] = 0.05
YEAR_FACTOR_RATE: Final[float] = 0.05
YEAR_FACTOR_MIN: Final[float] = 0.1
YEAR_MIN_FACTOR: Final[float] = 0.1
YEAR_FACTOR_MAX: Final[float] = 2.0
YEAR_MAX_FACTOR: Final[float] = 2.0

# Mileage Heuristics
MILEAGE_BASELINE: Final[float] = 18.0
MILEAGE_RATE: Final[float] = 0.01
MILEAGE_FACTOR_RATE: Final[float] = 0.01
MILEAGE_FACTOR_MIN: Final[float] = 0.8
MILEAGE_MIN_FACTOR: Final[float] = 0.8
MILEAGE_FACTOR_MAX: Final[float] = 1.2
MILEAGE_MAX_FACTOR: Final[float] = 1.2

# Exact 14 Model Feature Columns
FEATURE_COLUMNS: Final[list[str]] = [
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "brand_encoded",
    "model_encoded",
    "seller_type_Individual",
    "seller_type_Trustmark Dealer",
    "fuel_Diesel",
    "fuel_Electric",
    "fuel_LPG",
    "fuel_Petrol",
    "transmission_Manual",
]
DEFAULT_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(FEATURE_COLUMNS)

# Baseline Reference Categories (dropped in get_dummies)
FUEL_BASELINE: Final[str] = "CNG"
SELLER_TYPE_BASELINE: Final[str] = "Dealer"
TRANSMISSION_BASELINE: Final[str] = "Automatic"

# Valid Categories for Validation and Selection
VALID_FUELS: Final[tuple[str, ...]] = ("Petrol", "Diesel", "CNG", "LPG", "Electric")
VALID_TRANSMISSIONS: Final[tuple[str, ...]] = ("Manual", "Automatic")
VALID_SELLER_TYPES: Final[tuple[str, ...]] = ("Individual", "Dealer", "Trustmark Dealer")
VALID_OWNERS: Final[tuple[str, ...]] = (
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
)

# Numeric Boundaries
YEAR_MIN: Final[int] = 1980
YEAR_MAX: Final[int] = 2030
KM_DRIVEN_MIN: Final[float] = 0.0
MILEAGE_MIN: Final[float] = 0.0
MILEAGE_MAX: Final[float] = 100.0
ENGINE_MIN: Final[float] = 100.0
ENGINE_MAX: Final[float] = 10000.0
MAX_POWER_MIN: Final[float] = 10.0
MAX_POWER_MAX: Final[float] = 2000.0
SEATS_MIN: Final[int] = 1
SEATS_MAX: Final[int] = 20

# Presentation Formats
DEFAULT_CURRENCY: Final[str] = "INR"
PRICE_FORMAT: Final[str] = "{:,.2f}"
