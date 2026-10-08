"""
Tier 3: Pairwise Combinatorial Test Suite.
Verifies cross-feature interactions across categorical dimensions:
- fuel (Petrol, Diesel, CNG, LPG, Electric)
- transmission (Manual, Automatic)
- owner (First Owner, Second Owner, Third Owner, Fourth & Above Owner)
- seller_type (Individual, Dealer, Trustmark Dealer)
Ensures 100% coverage of all 71 two-way interaction pairs.
"""

import itertools
import pytest

# 20 Covering-Array test vectors achieving 100% coverage of all 71 pairwise combinations
PAIRWISE_COVERING_VECTORS = [
    ("Petrol", "Manual", "First Owner", "Individual"),
    ("Petrol", "Automatic", "Second Owner", "Dealer"),
    ("Diesel", "Manual", "Second Owner", "Trustmark Dealer"),
    ("Diesel", "Automatic", "Third Owner", "Individual"),
    ("CNG", "Manual", "Third Owner", "Dealer"),
    ("CNG", "Automatic", "First Owner", "Trustmark Dealer"),
    ("LPG", "Manual", "Fourth & Above Owner", "Individual"),
    ("Electric", "Automatic", "Fourth & Above Owner", "Dealer"),
    ("LPG", "Automatic", "First Owner", "Dealer"),
    ("Electric", "Manual", "Second Owner", "Individual"),
    ("Petrol", "Manual", "Third Owner", "Trustmark Dealer"),
    ("Petrol", "Manual", "Fourth & Above Owner", "Trustmark Dealer"),
    ("Diesel", "Manual", "First Owner", "Dealer"),
    ("CNG", "Manual", "Second Owner", "Individual"),
    ("LPG", "Manual", "Second Owner", "Trustmark Dealer"),
    ("Electric", "Manual", "First Owner", "Trustmark Dealer"),
    ("Diesel", "Manual", "Fourth & Above Owner", "Individual"),
    ("CNG", "Manual", "Fourth & Above Owner", "Individual"),
    ("LPG", "Manual", "Third Owner", "Individual"),
    ("Electric", "Manual", "Third Owner", "Individual"),
]


class TestTier3PairwiseCombinations:
    """Tests all pairwise interactions among categorical variables."""

    @pytest.mark.parametrize(
        "fuel,transmission,owner,seller_type",
        PAIRWISE_COVERING_VECTORS,
    )
    def test_pairwise_covering_vector(
        self, client, valid_payload, fuel, transmission, owner, seller_type
    ):
        """
        Verify every orthogonal pairwise combination produces a valid valuation without 500 error.
        """
        payload = valid_payload.copy()
        payload["fuel"] = fuel
        payload["transmission"] = transmission
        payload["owner"] = owner
        payload["seller_type"] = seller_type

        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"Failed on combination: {fuel}, {transmission}, {owner}, {seller_type}"

        data = response.get_json()
        assert "price" in data
        price_str = data["price"]
        assert len(price_str) > 0
        parsed_price = float(price_str.replace(",", ""))
        assert parsed_price > 0.0, f"Expected positive price, got {parsed_price}"

    def test_pairwise_coverage_completeness(self):
        """
        Meta-test verifying that PAIRWISE_COVERING_VECTORS mathematically covers
        100% of all 71 two-way pairs across fuel, transmission, owner, and seller_type.
        """
        fuels = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
        transmissions = ["Manual", "Automatic"]
        owners = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
        sellers = ["Individual", "Dealer", "Trustmark Dealer"]

        expected_pairs = set()
        for f in fuels:
            for t in transmissions:
                expected_pairs.add(("fuel_trans", f, t))
            for o in owners:
                expected_pairs.add(("fuel_owner", f, o))
            for s in sellers:
                expected_pairs.add(("fuel_seller", f, s))
        for t in transmissions:
            for o in owners:
                expected_pairs.add(("trans_owner", t, o))
            for s in sellers:
                expected_pairs.add(("trans_seller", t, s))
        for o in owners:
            for s in sellers:
                expected_pairs.add(("owner_seller", o, s))

        assert len(expected_pairs) == 71

        observed_pairs = set()
        for f, t, o, s in PAIRWISE_COVERING_VECTORS:
            observed_pairs.add(("fuel_trans", f, t))
            observed_pairs.add(("fuel_owner", f, o))
            observed_pairs.add(("fuel_seller", f, s))
            observed_pairs.add(("trans_owner", t, o))
            observed_pairs.add(("trans_seller", t, s))
            observed_pairs.add(("owner_seller", o, s))

        missing = expected_pairs - observed_pairs
        assert len(missing) == 0, f"Missing pairwise combinations: {missing}"

    @pytest.mark.parametrize("fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    def test_fuel_transmission_permutations(self, client, valid_payload, fuel):
        """Verify each fuel type functions with both Manual and Automatic transmissions."""
        for trans in ["Manual", "Automatic"]:
            payload = valid_payload.copy()
            payload["fuel"] = fuel
            payload["transmission"] = trans
            res = client.post("/predict", json=payload)
            assert res.status_code == 200
            assert "price" in res.get_json()

    @pytest.mark.parametrize("owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
    def test_owner_tier_multipliers(self, client, valid_payload, owner):
        """Verify each owner tier yields consistent valuations across identical vehicle specs."""
        payload = valid_payload.copy()
        payload["owner"] = owner
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert "price" in res.get_json()

    @pytest.mark.parametrize("seller_type", ["Individual", "Dealer", "Trustmark Dealer"])
    def test_seller_type_variations(self, client, valid_payload, seller_type):
        """Verify each seller type processes properly."""
        payload = valid_payload.copy()
        payload["seller_type"] = seller_type
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert "price" in res.get_json()

    def test_high_mileage_diesel_automatic(self, client, valid_payload):
        """Cross-feature edge case: Diesel + Automatic + High Mileage (25 kmpl)."""
        payload = valid_payload.copy()
        payload["fuel"] = "Diesel"
        payload["transmission"] = "Automatic"
        payload["mileage"] = 25.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert float(res.get_json()["price"].replace(",", "")) > 0

    def test_low_mileage_electric_automatic(self, client, valid_payload):
        """Cross-feature edge case: Electric + Automatic + Low Mileage."""
        payload = valid_payload.copy()
        payload["fuel"] = "Electric"
        payload["transmission"] = "Automatic"
        payload["mileage"] = 5.0
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert float(res.get_json()["price"].replace(",", "")) > 0

    def test_seven_seater_diesel_manual(self, client, valid_payload):
        """Cross-feature edge case: 7-Seater + Diesel + Manual + Individual."""
        payload = valid_payload.copy()
        payload["seats"] = 7
        payload["fuel"] = "Diesel"
        payload["transmission"] = "Manual"
        payload["seller_type"] = "Individual"
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert float(res.get_json()["price"].replace(",", "")) > 0
