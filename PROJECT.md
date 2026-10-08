# Project: CarPricePredictor Enterprise ML Upgrade

## Architecture
The CarPricePredictor upgrade transitions a monolithic prototype into a modular, production-grade enterprise ML application using:
- **Application Factory Pattern** (`create_app` in `car_price_predictor/__init__.py`) supporting multiple environments (`DevelopmentConfig`, `ProductionConfig`, `TestingConfig`).
- **Domain Service Layer**:
  - `ModelService`: Encapsulates joblib loading of `car_price_model.joblib`, brand/model frequency maps, and model prediction.
  - `VehiclePreprocessor`: Validates and transforms raw input dictionaries into exact 14-feature model DataFrames, handling unseen categories and dummy variables (CNG, Dealer, Automatic as references) gracefully.
  - `PricingService`: Implements post-prediction heuristic pricing adjustments (owner, year, mileage) using named constants in `constants.py`.
- **Schema Validation & REST API**:
  - Strict Pydantic v2 schemas (`PredictionRequest`, `PredictionResponse`, `ErrorResponse`).
  - Blueprint `api_v1` exposing `POST /api/v1/predict` and legacy alias `POST /predict`.
  - Structured 400 Bad Request error responses on invalid/malformed payloads.
  - OpenAPI 3.0 / Swagger UI documentation at `/apidocs` and `/openapi.json`.
- **Observability & Production Readiness**:
  - Blueprint `health` exposing `GET /health/live` (process health) and `GET /health/ready` (model & service readiness).
  - Structured request logging with inference duration tracking.
  - Security headers middleware (`X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`).
  - Rate limiting (Flask-Limiter) protecting `/predict` endpoints.
  - Production WSGI entrypoint (`wsgi.py`).
  - CI workflow (`.github/workflows/ci.yml`). Docker containerization excluded.
- **Frontend Reliability**:
  - Enhanced `templates/index.html` and `static/script.js` / `static/style.css`.
  - Accessible inline error feedback container replacing browser `alert()`.
  - Visual loading spinner state and button debouncing to prevent duplicate submissions.
  - Responsive layout for mobile devices.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Configuration Management | Multi-environment config classes (`DevelopmentConfig`, `ProductionConfig`, `TestingConfig`) with `.env.example` | M1 | ORIGINAL_REQUEST R1, R5 |
| 2 | Named Pricing Constants | Extract magic numbers from pricing heuristics into `services/constants.py` | M1 | ORIGINAL_REQUEST R1 |
| 3 | Pricing Rules Service | Isolated, deterministic business logic in `services/pricing.py` | M1 | ORIGINAL_REQUEST R1 |
| 4 | Vehicle Preprocessor | Robust 14-feature encoder handling dummy baselines and unseen categories | M1 | ORIGINAL_REQUEST R4 |
| 5 | Model Service | Encapsulated model artifact loading (`car_price_model.joblib`) and caching | M1 | ORIGINAL_REQUEST R4 |
| 6 | Dependencies & UTF-8 Encoding | Re-encode `requirements.txt` to UTF-8 and add `pydantic`, `pytest`, `flask-limiter`, etc. | M1 | Survey Finding |
| 7 | Application Factory | `create_app(config_name)` factory initializing extensions, blueprints, and services | M2 | ORIGINAL_REQUEST R1 |
| 8 | Pydantic Request Validation | Strict Pydantic v2 validation rejecting invalid types and out-of-range values | M2 | ORIGINAL_REQUEST R2 |
| 9 | Structured 400 Error Handling | Return HTTP 400 with structured JSON error details instead of HTTP 200 | M2 | ORIGINAL_REQUEST R2, AC |
| 10 | Versioned Prediction API | `POST /api/v1/predict` returning structured prediction response | M2 | ORIGINAL_REQUEST R2 |
| 11 | Legacy API Alias | `POST /predict` backward-compatible endpoint returning `{"price": "..."}` | M2 | ORIGINAL_REQUEST R2, AC |
| 12 | UI Blueprint & Hierarchy | `GET /` rendering `index.html` with brand/model hierarchy dropdowns | M2 | ORIGINAL_REQUEST R1, AC |
| 13 | OpenAPI / Swagger Documentation | Interactive Swagger UI at `/apidocs` and JSON spec at `/openapi.json` | M2 | ORIGINAL_REQUEST R2, AC |
| 14 | Health Check Endpoints | `GET /health/live` (process) and `GET /health/ready` (model readiness) returning 200 OK | M3 | ORIGINAL_REQUEST R5, AC |
| 15 | Structured Request Logging | Contextual JSON/structured logging with request ID and prediction duration | M3 | ORIGINAL_REQUEST R5 |
| 16 | Security Headers | Middleware injecting security headers (`X-Content-Type-Options`, `X-Frame-Options`) | M3 | ORIGINAL_REQUEST R5 |
| 17 | Rate Limiting | Rate limiting (Flask-Limiter) protecting prediction endpoints | M3 | ORIGINAL_REQUEST R5 |
| 18 | Production WSGI Entrypoint | `wsgi.py` production server entrypoint compatible with Waitress/Gunicorn | M3 | ORIGINAL_REQUEST R5, AC |
| 19 | Automated CI Workflow | GitHub Actions CI workflow running pytest on push/PR (excluding Docker) | M3 | ORIGINAL_REQUEST R5 |
| 20 | Frontend Inline Error Alerts | Accessible inline error alert box replacing browser `alert()` dialogs | M4 | ORIGINAL_REQUEST R6 |
| 21 | Submission Spinner & Debouncing | Disable submit button and show spinner during pending prediction requests | M4 | ORIGINAL_REQUEST R6 |
| 22 | Mobile Responsive Design | Responsive styling for form inputs and prediction cards across screen sizes | M4 | ORIGINAL_REQUEST R6 |
| 23 | E2E Test Suite (Tiers 1-4) | Opaque-box test suite with 100% pass rate covering features, boundaries, combinations | E2E Track & M5 | ORIGINAL_REQUEST R3, AC |
| 24 | Adversarial Hardening (Tier 5) | White-box stress testing and edge-case verification | M5 | Project Pattern Phase 2 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Services, Preprocessing & Config | Config classes, constants, Preprocessor, ModelService, PricingService, requirements.txt | None | DONE |
| M2 | App Factory, Pydantic Validation & API | `create_app`, Pydantic schemas, `api_v1`, UI blueprint, OpenAPI docs, legacy `/predict` | M1 | PLANNED |
| M3 | Observability, Health, Security & WSGI | `/health/live`, `/health/ready`, logging, security headers, rate limiting, `wsgi.py`, CI | M2 | PLANNED |
| M4 | Frontend Reliability & UX Enhancements | Inline alerts, spinner, button debouncing, responsive layout in `templates/` and `static/` | M2 | PLANNED |
| M5 | Final Verification & Adversarial Hardening | 100% E2E test suite pass (Tiers 1-4), Tier 5 adversarial hardening, forensic audit | M3, M4, TEST_READY | PLANNED |

*Note: E2E Testing Track executes in parallel to design the test infrastructure and author Tiers 1-4 test cases, publishing `TEST_READY.md` before M5 begins.*

---

## Interface Contracts

### 1. `car_price_predictor.services.preprocessor.VehiclePreprocessor`
```python
class VehiclePreprocessor:
    def __init__(self, feature_columns: list[str], brand_map: dict[str, float], model_map: dict[str, float]):
        ...
    def preprocess(self, payload: dict) -> pd.DataFrame:
        """
        Transforms validated payload dictionary into a single-row DataFrame with exact 14 columns:
        ['km_driven', 'mileage', 'engine', 'max_power', 'seats', 'brand_encoded', 'model_encoded',
         'seller_type_Individual', 'seller_type_Trustmark Dealer',
         'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol',
         'transmission_Manual']
        Handles unseen brands/models by assigning 0.0 frequency encoding.
        Baseline dummy levels (CNG, Dealer, Automatic) result in 0.0 for their respective flags.
        """
```

### 2. `car_price_predictor.services.pricing.PricingService`
```python
class PricingService:
    @staticmethod
    def calculate_price(base_prediction: float, year: int, mileage: float, owner: str) -> float:
        """
        Applies owner, year, and mileage heuristic multipliers to base_prediction:
        - owner: First Owner (1.0), Second Owner (0.95), Third Owner (0.90), Fourth & Above Owner (0.85)
        - year: clamp(1 + (year - 2018) * 0.05, 0.1, 2.0)
        - mileage: clamp(1 + (mileage - 18.0) * 0.01, 0.8, 1.2)
        Returns: unrounded final price float.
        """

    @staticmethod
    def format_price(final_price: float) -> str:
        """
        Returns formatted price string e.g. '508,560.15'.
        """
```

### 3. `car_price_predictor.services.model_service.ModelService`
```python
class ModelService:
    def __init__(self, model_path: str = "car_price_model.joblib"):
        ...
    def is_ready(self) -> bool:
        """Returns True if model artifact is successfully loaded and ready for inference."""
    def predict(self, input_df: pd.DataFrame) -> float:
        """Runs scikit-learn model.predict(input_df)[0]."""
```

### 4. `car_price_predictor.schemas.prediction.PredictionRequest`
```python
class PredictionRequest(BaseModel):
    brand: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    year: int = Field(..., ge=1980, le=2030)
    km_driven: float = Field(..., ge=0)
    mileage: float = Field(..., ge=0, le=100)
    engine: float = Field(..., ge=100, le=10000)
    max_power: float = Field(..., ge=10, le=2000)
    seats: int = Field(..., ge=1, le=20)
    fuel: str = Field(...)  # Petrol, Diesel, CNG, LPG, Electric
    transmission: str = Field(...)  # Manual, Automatic
    seller_type: str = Field(...)  # Individual, Dealer, Trustmark Dealer
    owner: str = Field(...)  # First Owner, Second Owner, Third Owner, Fourth & Above Owner
```

### 5. API Response Contracts
- `POST /api/v1/predict`:
  - Request: JSON matching `PredictionRequest`
  - Success Response: `HTTP 200 OK`
    ```json
    {
      "price": "508,560.15",
      "valuation": 508560.15,
      "currency": "INR",
      "status": "success"
    }
    ```
  - Validation Failure: `HTTP 400 Bad Request`
    ```json
    {
      "error": "Validation Error",
      "details": [
        {"field": "year", "message": "Input should be greater than or equal to 1980", "type": "greater_than_equal"}
      ]
    }
    ```
- `POST /predict` (Backward-Compatible Alias):
  - Request: JSON matching `PredictionRequest`
  - Success Response: `HTTP 200 OK`
    ```json
    {
      "price": "508,560.15"
    }
    ```
  - Validation Failure: `HTTP 400 Bad Request`
    ```json
    {
      "error": "Validation Error",
      "details": [...]
    }
    ```
- `GET /health/live`: `HTTP 200 OK` `{"status": "alive"}`
- `GET /health/ready`: `HTTP 200 OK` `{"status": "ready", "model_loaded": true}` (or `HTTP 503` if model not ready)

---

## Code Layout
```
d:\projects\CarPricePredictor\
├── car_price_predictor\
│   ├── __init__.py                # create_app application factory
│   ├── config.py                  # BaseConfig, DevelopmentConfig, ProductionConfig, TestingConfig
│   ├── blueprints\
│   │   ├── __init__.py
│   │   ├── api_v1\
│   │   │   ├── __init__.py
│   │   │   └── routes.py          # /api/v1/predict & /predict routes
│   │   ├── ui\
│   │   │   ├── __init__.py
│   │   │   └── routes.py          # / route rendering index.html
│   │   ├── health\
│   │   │   ├── __init__.py
│   │   │   └── routes.py          # /health/live, /health/ready routes
│   │   └── docs\
│   │       ├── __init__.py
│   │       └── routes.py          # /apidocs, /openapi.json Swagger UI
│   ├── schemas\
│   │   ├── __init__.py
│   │   └── prediction.py          # Pydantic v2 schemas
│   ├── services\
│   │   ├── __init__.py
│   │   ├── constants.py           # Named constants for pricing heuristics
│   │   ├── pricing.py             # PricingService
│   │   ├── preprocessor.py        # VehiclePreprocessor
│   │   └── model_service.py       # ModelService
│   └── utils\
│       ├── __init__.py
│       ├── logging.py             # Structured logger setup
│       └── security.py            # Security headers middleware
├── templates\
│   └── index.html                 # Frontend template with inline alert & spinner
├── static\
│   ├── script.js                  # Frontend JS with debounce & inline feedback
│   └── style.css                  # Responsive UI styling & spinner
├── tests\
│   ├── conftest.py                # Pytest fixtures (client, test app, valid payloads)
│   ├── unit\
│   │   ├── test_pricing_rules.py  # Pricing rule unit tests
│   │   └── test_preprocessor.py   # Preprocessing & encoding unit tests
│   ├── integration\
│   │   ├── test_api_predict.py    # /api/v1/predict & /predict endpoints
│   │   ├── test_validation.py     # 400 Bad Request error cases
│   │   └── test_health.py         # /health/live and /health/ready
│   ├── regression\
│   │   └── test_backward_compat.py# 8 golden benchmark assertions
│   └── e2e\                       # Opaque-box E2E test suite (Tiers 1-4)
├── app.py                         # Top-level legacy entrypoint running create_app()
├── wsgi.py                        # Production WSGI entrypoint
├── requirements.txt               # UTF-8 encoded dependency manifest
├── .env.example                   # Environment configuration example
├── .github\
│   └── workflows\
│       └── ci.yml                 # Automated CI workflow
└── car_price_model.joblib         # Serialized ML model artifact
```
