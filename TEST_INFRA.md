# E2E Test Infra: CarPricePredictor

## Test Philosophy
- **Opaque-Box & Requirement-Driven**: Tests derive directly from `ORIGINAL_REQUEST.md` and `PROJECT.md` specifications, exercising the application purely through external entry points (`POST /api/v1/predict`, `POST /predict`, `GET /`, `GET /health/live`, `GET /health/ready`, `GET /apidocs`).
- **Zero Internal Dependency**: Test verification does not mock internal components or depend on private implementation details.
- **Methodology**: Systematic combination of Category-Partition Testing, Boundary Value Analysis (BVA), Pairwise Combinatorial Testing, and Real-World Workload Scenarios.
- **Progressive Testability**: Verification mechanisms do not require features more complex than what is being tested.

---

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Scenario) |
|---|---------|---------------------|:----------------:|:-----------------:|:-----------------:|:-----------------:|
| 1 | Configuration Management | ORIGINAL_REQUEST R1, R5 | 5 | 5 | ✓ | ✓ |
| 2 | Named Pricing Constants | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ | ✓ |
| 3 | Pricing Rules Heuristics | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ | ✓ |
| 4 | Vehicle Preprocessing | ORIGINAL_REQUEST R4 | 5 | 5 | ✓ | ✓ |
| 5 | Model Service & Loading | ORIGINAL_REQUEST R4 | 5 | 5 | ✓ | ✓ |
| 6 | Application Factory (`create_app`) | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ | ✓ |
| 7 | Pydantic Request Validation | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 8 | Structured 400 Bad Request Errors | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 9 | Versioned API (`/api/v1/predict`) | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 10 | Legacy Alias (`/predict`) | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 11 | UI Blueprint & Brand/Model Dropdowns | ORIGINAL_REQUEST R1, R6 | 5 | 5 | ✓ | ✓ |
| 12 | OpenAPI / Swagger Documentation | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 13 | Health Check (`/health/live`) | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 14 | Health Check (`/health/ready`) | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 15 | Structured Request Logging | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 16 | Security Headers | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 17 | Rate Limiting | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 18 | Production WSGI Entrypoint | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 19 | Automated CI Workflow File | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 20 | Frontend Inline Error Alerts | ORIGINAL_REQUEST R6 | 5 | 5 | ✓ | ✓ |
| 21 | Submission Spinner & Debouncing | ORIGINAL_REQUEST R6 | 5 | 5 | ✓ | ✓ |
| 22 | Mobile Responsive Design | ORIGINAL_REQUEST R6 | 5 | 5 | ✓ | ✓ |
| 23 | Ground Truth Regression Baselines | ORIGINAL_REQUEST R3 | 5 | 5 | ✓ | ✓ |
| 24 | Unseen Category Fallbacks | ORIGINAL_REQUEST R4 | 5 | 5 | ✓ | ✓ |

---

## Test Architecture
- **Framework**: `pytest`
- **Environment**: Executed via `.venv\Scripts\python.exe -m pytest`
- **Directory Layout**:
  ```
  tests/
  ├── conftest.py                    # Global fixtures (test client, valid payloads, golden vectors)
  ├── unit/                          # Isolated unit tests
  │   ├── test_pricing_rules.py
  │   └── test_preprocessor.py
  ├── integration/                   # Integration and contract tests
  │   ├── test_api_predict.py
  │   ├── test_validation.py
  │   ├── test_health.py
  │   └── test_openapi.py
  ├── regression/                    # Regression tests
  │   └── test_backward_compat.py    # 8 golden benchmark assertions
  └── e2e/                           # Requirement-driven opaque-box test tiers
      ├── test_tier1_features.py     # Tier 1: Pure feature coverage
      ├── test_tier2_boundaries.py   # Tier 2: Boundary value and edge cases
      ├── test_tier3_pairwise.py     # Tier 3: Cross-feature pairwise interactions
      └── test_tier4_workloads.py    # Tier 4: Real-world user workloads and end-to-end flows
  ```

---

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Description |
|---|----------|--------------------|-------------|
| 1 | Standard Car Buyer Flow | F10, F11, F23 | User loads UI `/`, fetches dropdowns, enters typical Maruti Swift 2019 details, verifies valuation matches ₹508,560.15. |
| 2 | Fleet Luxury Appraisal | F9, F7, F23 | Enterprise API client sends batch-like requests to `/api/v1/predict` for BMW 5 Series and Audi models, verifying valuation and metadata. |
| 3 | Faulty API Client Robustness | F7, F8, F9, F10 | Malformed JSON, non-numeric values, negative kilometers, missing required keys; all return structured 400 Bad Request with field errors without server crashes. |
| 4 | Platform Health & Monitoring | F13, F14, F15, F16 | Synthetic monitoring hits `/health/live` and `/health/ready`, checks response times, security headers, and structured logging. |
| 5 | Edge Fleet Evaluation (Outliers) | F4, F24, F7, F2 | Extremely high mileage (clamped at 1.2 factor), old vehicle year (clamped at 0.1 factor), fourth & above owner, unseen brand/model. |
| 6 | High-Rate Submission & Security | F16, F17, F10 | Rapid consecutive requests to `/predict` to test rate limiting behavior and security header compliance. |

---

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: ≥5 test cases per feature category (total ≥ 120 tests).
- **Tier 2 (Boundary & Corner)**: ≥5 test cases per boundary condition (numeric limits, boundary years, extreme mileage, malformed strings, null bytes, special characters).
- **Tier 3 (Cross-Feature Combinations)**: Pairwise coverage across fuel types, transmissions, owner tiers, and seller types.
- **Tier 4 (Real-World Workloads)**: ≥6 comprehensive application workflows simulating end-to-end user and enterprise integration.
- **Pass Criterion**: 100% test pass rate with zero fatal errors.
