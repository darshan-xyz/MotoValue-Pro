# Test Infrastructure Ready: CarPricePredictor E2E & Regression Suite

**Status**: READY FOR TESTING & MILESTONE VERIFICATION  
**Author**: Test Writer E2E Track (`test_writer_e2e_1`)  
**Target Environment**: `d:\projects\CarPricePredictor\.venv\Scripts\python.exe`  
**Test Framework**: Pytest 9.1.1  

---

## 1. Test Runner Command

To execute the entire automated test suite:
```powershell
d:\projects\CarPricePredictor\.venv\Scripts\python.exe -m pytest tests/
```

To run individual tiers or regression suites:
```powershell
# Golden Benchmarks & Regression
d:\projects\CarPricePredictor\.venv\Scripts\python.exe -m pytest tests/regression/test_backward_compat.py

# Tier 1: Feature Coverage (120 tests)
d:\projects\CarPricePredictor\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier1_features.py

# Tier 2: Boundary Value Analysis (54 tests)
d:\projects\CarPricePredictor\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier2_boundaries.py

# Tier 3: Pairwise Combinatorial Interactions (36 tests)
d:\projects\CarPricePredictor\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier3_pairwise.py

# Tier 4: Real-World Workload Scenarios (6 scenarios)
d:\projects\CarPricePredictor\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier4_workloads.py
```

---

## 2. Test Suite Summary & Counts

| Tier / Suite | Target Scope | Test Count | Pass Rate (Current) | Milestone Coverage |
|--------------|--------------|:----------:|:-------------------:|:------------------:|
| **Regression** | Backward Compatibility (8 Golden Benchmarks) | 12 | 11 Passed / 1 Skipped* | Baseline & M2 |
| **Tier 1** | Feature Coverage (24 Features ≥ 5 tests each) | 120 | 33 Passed / 87 Skipped* | M1, M2, M3, M4 |
| **Tier 2** | Boundary Value Analysis & Extreme Corners | 54 | 54 Passed (100%) | Full Scope |
| **Tier 3** | Pairwise Combinatorial (Fuel, Trans, Owner, Seller) | 36 | 36 Passed (100%) | Full Scope |
| **Tier 4** | Real-World Application Workloads & Recovery | 6 | 5 Passed / 1 Skipped* | Full Workflows |
| **TOTAL** | **Comprehensive Opaque-Box E2E Suite** | **228** | **139 Passed / 89 Skipped / 0 Failed** | **M1 - M5** |

*\*Note: Skipped tests represent forward-looking opaque-box contract assertions designed for pending milestones (M1 Core Services, M2 Pydantic Validation & App Factory, M3 Observability & WSGI, M4 Frontend Reliability). They are cleanly guarded with informative skip reasons and will automatically transition to passing as workers complete their respective milestone implementations.*

---

## 3. Feature Coverage Checklist (Tiers 1-4)

| # | Feature | Requirement | Tests Defined | Test Class / Location |
|---|---------|-------------|:-------------:|-----------------------|
| 1 | Configuration Management | ORIGINAL_REQUEST R1, R5 | 5 | `TestFeature01ConfigurationManagement` |
| 2 | Named Pricing Constants | ORIGINAL_REQUEST R1 | 5 | `TestFeature02NamedPricingConstants` |
| 3 | Pricing Rules Heuristics | ORIGINAL_REQUEST R1 | 5 | `TestFeature03PricingRulesHeuristics` |
| 4 | Vehicle Preprocessing | ORIGINAL_REQUEST R4 | 5 | `TestFeature04VehiclePreprocessing` |
| 5 | Model Service & Loading | ORIGINAL_REQUEST R4 | 5 | `TestFeature05ModelService` |
| 6 | Application Factory (`create_app`) | ORIGINAL_REQUEST R1 | 5 | `TestFeature06ApplicationFactory` |
| 7 | Pydantic Request Validation | ORIGINAL_REQUEST R2 | 5 | `TestFeature07PydanticRequestValidation` |
| 8 | Structured 400 Bad Request Errors | ORIGINAL_REQUEST R2 | 5 | `TestFeature08Structured400Errors` |
| 9 | Versioned Prediction API (`/api/v1/predict`) | ORIGINAL_REQUEST R2 | 5 | `TestFeature09VersionedPredictionAPI` |
| 10 | Legacy Prediction Alias (`/predict`) | ORIGINAL_REQUEST R2 | 5 | `TestFeature10LegacyPredictionAlias` |
| 11 | UI Blueprint & Hierarchy | ORIGINAL_REQUEST R1, R6 | 5 | `TestFeature11UIBlueprintAndHierarchy` |
| 12 | OpenAPI / Swagger Documentation | ORIGINAL_REQUEST R2 | 5 | `TestFeature12OpenAPIDocs` |
| 13 | Health Check (`/health/live`) | ORIGINAL_REQUEST R5 | 5 | `TestFeature13HealthLive` |
| 14 | Health Check (`/health/ready`) | ORIGINAL_REQUEST R5 | 5 | `TestFeature14HealthReady` |
| 15 | Structured Request Logging | ORIGINAL_REQUEST R5 | 5 | `TestFeature15StructuredRequestLogging` |
| 16 | Security Headers Middleware | ORIGINAL_REQUEST R5 | 5 | `TestFeature16SecurityHeaders` |
| 17 | Rate Limiting Protection | ORIGINAL_REQUEST R5 | 5 | `TestFeature17RateLimiting` |
| 18 | Production WSGI Entrypoint (`wsgi.py`) | ORIGINAL_REQUEST R5 | 5 | `TestFeature18ProductionWSGIEntrypoint` |
| 19 | Automated CI Workflow (`ci.yml`) | ORIGINAL_REQUEST R5 | 5 | `TestFeature19AutomatedCIWorkflow` |
| 20 | Frontend Inline Error Alerts | ORIGINAL_REQUEST R6 | 5 | `TestFeature20FrontendInlineErrorAlerts` |
| 21 | Submission Spinner & Debouncing | ORIGINAL_REQUEST R6 | 5 | `TestFeature21SubmissionSpinnerAndDebounce` |
| 22 | Mobile Responsive Design | ORIGINAL_REQUEST R6 | 5 | `TestFeature22MobileResponsiveDesign` |
| 23 | Ground Truth Regression Baselines | ORIGINAL_REQUEST R3 | 5 | `TestFeature23GroundTruthBaselines` |
| 24 | Unseen Category Fallbacks | ORIGINAL_REQUEST R4 | 5 | `TestFeature24UnseenCategoryFallbacks` |

---

## 4. Boundary Value Analysis (Tier 2 Coverage)

- **Year Limits**: Tested at 1980 (min), 2030 (max), 1981, 2029, 1979 (invalid), 2031 (invalid), 1800, 3000.
- **Mileage Limits**: Tested at 0.0 (min), 100.0 (max), 18.55 (fractional), -0.1 (invalid), 100.1 (invalid), 500.0 (outlier).
- **Engine Displacement**: Tested at 100 CC (min), 10000 CC (max), 796, 4999, 99.9 (invalid), 10000.1 (invalid), 0, negative.
- **Max Power**: Tested at 10 bhp (min), 2000 bhp (max), 119.35 (decimal), 9.9 (invalid), 2000.1 (invalid), 0, negative.
- **Seats**: Tested at 1 (min), 20 (max), 2, 4, 5, 7, 8, 0 (invalid), 21 (invalid), negative.
- **Kilometers Driven**: Tested at 0 (min), 15k, 45k, 120k, 1,000,000 (extreme high), -1 (invalid), -50,000 (invalid), non-numeric.
- **String Boundaries**: Empty strings, whitespace-only, SQL/script injection patterns, 1000-char strings, Unicode & emoji.
- **Pricing Clamps**: Year floor clamp (<=1998 -> 0.1), Year ceiling clamp (>=2038 -> 2.0), Mileage floor clamp (<=0 -> 0.8), Mileage ceiling clamp (>=38 -> 1.2), Combined extremes.
- **Payload Stress**: Empty `{}` JSON, missing required fields, unexpected fields, array bodies, non-JSON headers.

---

## 5. Pairwise Combinatorial Matrix (Tier 3 Coverage)

The test suite contains an orthogonal 20-vector covering array that mathematically covers 100% of all 71 two-way pairs across:
- **Fuel**: Petrol, Diesel, CNG, LPG, Electric (5 options)
- **Transmission**: Manual, Automatic (2 options)
- **Owner**: First Owner, Second Owner, Third Owner, Fourth & Above Owner (4 options)
- **Seller Type**: Individual, Dealer, Trustmark Dealer (3 options)

Includes meta-test `test_pairwise_coverage_completeness` verifying 0 missing pairs out of 71 total possible pairs.

---

## 6. Real-World Application Scenarios (Tier 4 Coverage)

1. **Scenario 1 (Standard Retail Buyer Flow)**: Loads `/`, inspects brand & car hierarchy, inputs Maruti Swift 2019, submits to `/predict`, validates ₹508,560.15 estimate.
2. **Scenario 2 (Fleet Luxury Appraisal)**: Batch-appraises BMW 5 Series, Audi A4, Mercedes E-Class, Volvo XC60; checks ground truth and luxury price consistency.
3. **Scenario 3 (Faulty API Client Recovery)**: Sends 6 sequential malformed requests (missing fields, wrong types, negative numbers); asserts zero 500 crashes and verifies immediate system recovery on valid request.
4. **Scenario 4 (Health & APM Heartbeat)**: Probes `/health/live` and `/health/ready`, checks response times (<100ms) and status payload.
5. **Scenario 5 (Outlier Edge Fleet)**: Appraises high-mileage (45 kmpl), vintage car (1980), electric automatic, and unseen hypercar without crashes.
6. **Scenario 6 (Burst Rate & Stability)**: Sends burst of 15 rapid consecutive prediction requests; verifies latency and server stability.

---

## 7. Golden Benchmarks (Regression Suite)

All 8 ground-truth benchmarks from `baseline_benchmarks.json` are verified for exact string match and numerical precision:
- `TC-01`: Maruti Swift 2019 -> `508,560.15` (Passed)
- `TC-02`: BMW 5 Series 2021 -> `3,312,029.37` (Passed)
- `TC-03`: Hyundai i10 2015 -> `209,300.94` (Passed)
- `TC-04`: Maruti Wagon 2017 -> `263,809.22` (Passed)
- `TC-05`: Toyota Innova 2018 -> `523,794.72` (Passed)
- `TC-06`: Honda City 2020 -> `1,165,296.00` (Passed)
- `TC-07`: Maruti Alto 2005 -> `31,279.17` (Passed)
- `TC-08`: Renault KWID 2022 -> `540,771.45` (Passed)
