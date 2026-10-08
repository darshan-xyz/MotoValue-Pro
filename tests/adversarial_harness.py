"""
Adversarial Stress Test Harness for CarPricePredictor.
Executed by challenger_ml_2 to empirically stress-test:
1. Valuation Confidence Intervals across extreme vehicles and random boundary payloads.
2. Explainability Service response time (<100ms) and mathematical consistency of contributions and pricing adjustments.
3. Legacy POST /predict endpoint isolation (absence of 'valuation' and 'confidence_interval').
"""
import time
import random
import statistics
from car_price_predictor import create_app
from car_price_predictor.services.explainability import ExplainabilityService

app = create_app("testing")
client = app.test_client()

print("=" * 60)
print("CHALLENGER STRESS HARNESS EXECUTION")
print("=" * 60)

# -------------------------------------------------------------
# 1. LATENCY AUDIT OF POST /api/v1/explain
# -------------------------------------------------------------
print("\n--- 1. POST /api/v1/explain LATENCY AUDIT ---")
benchmark_payload = {
    "brand": "Maruti",
    "model": "Swift",
    "year": 2019,
    "km_driven": 45000,
    "mileage": 21.21,
    "engine": 1197,
    "max_power": 81.86,
    "seats": 5,
    "fuel": "Petrol",
    "transmission": "Manual",
    "seller_type": "Individual",
    "owner": "First Owner",
}

latencies = []
for i in range(10):
    t0 = time.perf_counter()
    res = client.post("/api/v1/explain", json=benchmark_payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latencies.append(elapsed_ms)
    assert res.status_code == 200

print(f"10 Invocations Latencies (ms): {[round(x, 1) for x in latencies]}")
print(f"Min: {min(latencies):.2f} ms | Median: {statistics.median(latencies):.2f} ms | Mean: {statistics.mean(latencies):.2f} ms | Max: {max(latencies):.2f} ms")
print(f"SLA Target (<100ms): {'PASS' if max(latencies) < 100 else 'FAIL (Exceeds SLA)'}")


# -------------------------------------------------------------
# 2. MATHEMATICAL CONSISTENCY OF CONTRIBUTIONS & PRICING ADJUSTMENTS
# -------------------------------------------------------------
print("\n--- 2. MATHEMATICAL CONSISTENCY OF CONTRIBUTIONS & PRICING ADJUSTMENTS ---")
test_vehicles = [
    ("Maruti Swift (Standard)", benchmark_payload),
    ("BMW X5 (Luxury)", {
        "brand": "BMW", "model": "X5", "year": 2022, "km_driven": 15000, "mileage": 12.0,
        "engine": 2998, "max_power": 335.0, "seats": 5, "fuel": "Petrol",
        "transmission": "Automatic", "seller_type": "Dealer", "owner": "First Owner"
    }),
    ("Old Innova (Depreciated)", {
        "brand": "Toyota", "model": "Innova", "year": 2010, "km_driven": 250000, "mileage": 12.8,
        "engine": 2494, "max_power": 102.0, "seats": 7, "fuel": "Diesel",
        "transmission": "Manual", "seller_type": "Individual", "owner": "Fourth & Above Owner"
    }),
]

for vname, vpayload in test_vehicles:
    print(f"\nAnalyzing: {vname}")
    res = client.post("/api/v1/explain", json=vpayload)
    assert res.status_code == 200
    data = res.get_json()
    val = data["valuation"]
    base_ml = data["base_ml_valuation"]
    pa = data["pricing_adjustments"]

    # Check 1: Multiplier consistency
    expected_mult = pa["owner_multiplier"] * pa["year_factor"] * pa["mileage_factor"]
    mult_consistent = abs(pa["total_heuristic_multiplier"] - expected_mult) < 1e-4
    print(f"  Multiplier consistency: total={pa['total_heuristic_multiplier']:.6f}, product={expected_mult:.6f} -> {'PASS' if mult_consistent else 'FAIL'}")

    # Check 2: Final valuation vs base * total_multiplier
    expected_val = round(base_ml * pa["total_heuristic_multiplier"], 2)
    val_consistent = abs(val - expected_val) <= 0.05
    print(f"  Valuation consistency: val={val}, base*mult={expected_val} -> {'PASS' if val_consistent else 'FAIL'}")

    # Check 3: Net adjustment amount vs valuation - base_ml
    expected_net = round(val - base_ml, 2)
    net_consistent = abs(pa["net_adjustment_amount"] - expected_net) <= 0.05
    print(f"  Net adjustment amount: reported={pa['net_adjustment_amount']}, val - base={expected_net} -> {'PASS' if net_consistent else 'FAIL'}")

    # Check 4: Feature contributions decomposition
    pricing_rule_sum = 0.0
    ml_sum = 0.0
    for fc in data["feature_contributions"]:
        if fc["source"] == "pricing_rule":
            pricing_rule_sum += fc["impact_amount"]
        else:
            ml_sum += fc["impact_amount"]

    print(f"  Pricing rule contributions sum: {pricing_rule_sum:.2f} (reported net_adjustment: {pa['net_adjustment_amount']})")
    print(f"  ML feature contributions sum: {ml_sum:.2f}")
    print(f"  Total contributions sum (pricing + ML): {pricing_rule_sum + ml_sum:.2f}")

    # Check missing pricing rule for mileage
    pricing_rule_features = [fc["feature"] for fc in data["feature_contributions"] if fc["source"] == "pricing_rule"]
    print(f"  Pricing rule features present: {pricing_rule_features} (Note: mileage has pricing rule multiplier {pa['mileage_factor']} but is NOT present as a pricing rule contribution)")


# -------------------------------------------------------------
# 3. CONFIDENCE INTERVAL BOUNDS INVARIANT ACROSS EXTREMES
# -------------------------------------------------------------
print("\n--- 3. CONFIDENCE INTERVAL BOUNDS INVARIANT (0 <= lower <= val <= upper) ---")
extreme_cases = [
    ("Vintage 1980 car", {"brand": "Maruti", "model": "800", "year": 1980, "km_driven": 500000, "mileage": 15.0, "engine": 800, "max_power": 35.0, "seats": 4, "fuel": "Petrol", "transmission": "Manual", "seller_type": "Individual", "owner": "Fourth & Above Owner"}),
    ("Luxury 2025 car", {"brand": "BMW", "model": "X5", "year": 2025, "km_driven": 500, "mileage": 12.0, "engine": 2998, "max_power": 375.0, "seats": 5, "fuel": "Petrol", "transmission": "Automatic", "seller_type": "Dealer", "owner": "First Owner"}),
    ("Zero mileage efficiency", {"brand": "Hyundai", "model": "i20", "year": 2020, "km_driven": 10000, "mileage": 0.0, "engine": 1197, "max_power": 82.0, "seats": 5, "fuel": "Petrol", "transmission": "Manual", "seller_type": "Individual", "owner": "First Owner"}),
    ("100 kmpl mileage efficiency", {"brand": "Hyundai", "model": "i20", "year": 2020, "km_driven": 10000, "mileage": 100.0, "engine": 1197, "max_power": 82.0, "seats": 5, "fuel": "Petrol", "transmission": "Manual", "seller_type": "Individual", "owner": "First Owner"}),
    ("Min engine power (10 bhp)", {"brand": "Tata", "model": "Nano", "year": 2015, "km_driven": 30000, "mileage": 25.0, "engine": 624, "max_power": 10.0, "seats": 4, "fuel": "Petrol", "transmission": "Manual", "seller_type": "Individual", "owner": "First Owner"}),
    ("Hypercar (2000 bhp)", {"brand": "Bugatti", "model": "Chiron", "year": 2024, "km_driven": 1000, "mileage": 5.0, "engine": 7993, "max_power": 2000.0, "seats": 2, "fuel": "Petrol", "transmission": "Automatic", "seller_type": "Trustmark Dealer", "owner": "First Owner"}),
    ("Zero km driven", {"brand": "Honda", "model": "City", "year": 2023, "km_driven": 0, "mileage": 18.0, "engine": 1498, "max_power": 119.0, "seats": 5, "fuel": "Petrol", "transmission": "Manual", "seller_type": "Dealer", "owner": "First Owner"}),
    ("5 Million km driven", {"brand": "Toyota", "model": "Innova", "year": 2010, "km_driven": 5000000, "mileage": 12.0, "engine": 2494, "max_power": 102.0, "seats": 7, "fuel": "Diesel", "transmission": "Manual", "seller_type": "Individual", "owner": "Fourth & Above Owner"}),
]

random.seed(42)
for i in range(100):
    extreme_cases.append((
        f"Random Case {i+1}",
        {
            "brand": random.choice(["Maruti", "Hyundai", "Honda", "Tata", "Toyota", "BMW", "Ford", "Audi", "RandomBrand"]),
            "model": random.choice(["Swift", "City", "Innova", "i20", "Corolla", "Fortuner", "RandomModel"]),
            "year": random.randint(1980, 2030),
            "km_driven": float(random.choice([0, 10, 50000, 300000, 2000000])),
            "mileage": round(random.uniform(0.0, 100.0), 2),
            "engine": float(random.choice([100, 600, 1197, 1998, 4951, 10000])),
            "max_power": float(random.choice([10, 35, 82, 180, 450, 2000])),
            "seats": random.choice([1, 2, 4, 5, 7, 8, 10, 20]),
            "fuel": random.choice(["Petrol", "Diesel", "CNG", "LPG", "Electric"]),
            "transmission": random.choice(["Manual", "Automatic"]),
            "seller_type": random.choice(["Individual", "Dealer", "Trustmark Dealer"]),
            "owner": random.choice(["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
        }
    ))

ci_violations = 0
for name, p in extreme_cases:
    res = client.post("/api/v1/predict", json=p)
    assert res.status_code == 200
    data = res.get_json()
    val = data["valuation"]
    ci = data["confidence_interval"]
    lb = ci["lower_bound"]
    ub = ci["upper_bound"]
    if not (0.0 <= lb <= val <= ub):
        print(f"  VIOLATION in {name}: lb={lb}, val={val}, ub={ub}")
        ci_violations += 1

print(f"Tested {len(extreme_cases)} extreme & randomized cases.")
print(f"Invariant (0 <= lower_bound <= valuation <= upper_bound) violations: {ci_violations}")


# -------------------------------------------------------------
# 4. LEGACY POST /predict ISOLATION AUDIT
# -------------------------------------------------------------
print("\n--- 4. LEGACY POST /predict ISOLATION AUDIT ---")
legacy_violations = 0
for name, p in extreme_cases:
    res = client.post("/predict", json=p)
    assert res.status_code == 200
    data = res.get_json()
    keys = set(data.keys())
    if keys != {"price"}:
        print(f"  VIOLATION in {name}: keys = {keys}")
        legacy_violations += 1
    if "valuation" in data or "confidence_interval" in data:
        print(f"  LEAK in {name}: {data}")
        legacy_violations += 1

print(f"Tested {len(extreme_cases)} payloads against legacy POST /predict.")
print(f"Violations (unexpected keys or leaks): {legacy_violations}")

print("\n" + "=" * 60)
print("CHALLENGER STRESS HARNESS COMPLETED")
print("=" * 60)
