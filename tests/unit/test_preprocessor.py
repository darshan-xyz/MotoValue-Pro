"""
Unit tests for VehiclePreprocessor transformation, encoding, and dummy baselines.
"""

import pytest
import pandas as pd
from car_price_predictor.services.preprocessor import VehiclePreprocessor


class TestPreprocessorOutputStructure:
    """Verify DataFrame structure, shape, columns, and exclusions."""

    def test_output_is_dataframe(self, preprocessor, sample_payload):
        df = preprocessor.preprocess(sample_payload)
        assert isinstance(df, pd.DataFrame)

    def test_output_shape_is_single_row_14_cols(self, preprocessor, sample_payload):
        df = preprocessor.preprocess(sample_payload)
        assert df.shape == (1, 14)

    def test_output_columns_exact_order(self, preprocessor, sample_payload, feature_columns):
        df = preprocessor.preprocess(sample_payload)
        assert list(df.columns) == feature_columns

    def test_year_and_owner_strictly_excluded(self, preprocessor, sample_payload):
        df = preprocessor.preprocess(sample_payload)
        assert "year" not in df.columns
        assert "owner" not in df.columns

    def test_no_null_or_nan_values(self, preprocessor, sample_payload):
        df = preprocessor.preprocess(sample_payload)
        assert not df.isna().any().any()
        assert not df.isnull().any().any()

    def test_all_columns_numeric_dtypes(self, preprocessor, sample_payload):
        df = preprocessor.preprocess(sample_payload)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric dtype"


class TestDummyVariableZeroing:
    """Verify reference levels drop to 0.0 across all dummy flags."""

    def test_fuel_cng_all_fuel_dummies_zero(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["fuel"] = "CNG"
        df = preprocessor.preprocess(payload)

        assert df["fuel_Diesel"].iloc[0] == 0.0
        assert df["fuel_Electric"].iloc[0] == 0.0
        assert df["fuel_LPG"].iloc[0] == 0.0
        assert df["fuel_Petrol"].iloc[0] == 0.0

    def test_seller_type_dealer_all_seller_dummies_zero(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["seller_type"] = "Dealer"
        df = preprocessor.preprocess(payload)

        assert df["seller_type_Individual"].iloc[0] == 0.0
        assert df["seller_type_Trustmark Dealer"].iloc[0] == 0.0

    def test_transmission_automatic_dummy_zero(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["transmission"] = "Automatic"
        df = preprocessor.preprocess(payload)

        assert df["transmission_Manual"].iloc[0] == 0.0

    def test_simultaneous_reference_baselines(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["fuel"] = "CNG"
        payload["seller_type"] = "Dealer"
        payload["transmission"] = "Automatic"
        df = preprocessor.preprocess(payload)

        dummy_cols = [
            "seller_type_Individual",
            "seller_type_Trustmark Dealer",
            "fuel_Diesel",
            "fuel_Electric",
            "fuel_LPG",
            "fuel_Petrol",
            "transmission_Manual",
        ]
        for col in dummy_cols:
            assert df[col].iloc[0] == 0.0, f"Expected 0.0 for baseline dummy {col}"


class TestOneHotActiveIndicators:
    """Verify non-reference categories activate exactly one indicator column."""

    @pytest.mark.parametrize(
        "fuel_val,active_col",
        [
            ("Diesel", "fuel_Diesel"),
            ("Electric", "fuel_Electric"),
            ("LPG", "fuel_LPG"),
            ("Petrol", "fuel_Petrol"),
        ],
    )
    def test_fuel_active_flags(self, preprocessor, sample_payload, fuel_val, active_col):
        payload = sample_payload.copy()
        payload["fuel"] = fuel_val
        df = preprocessor.preprocess(payload)

        all_fuel_cols = ["fuel_Diesel", "fuel_Electric", "fuel_LPG", "fuel_Petrol"]
        for col in all_fuel_cols:
            if col == active_col:
                assert df[col].iloc[0] == 1.0, f"Expected 1.0 for active flag {col}"
            else:
                assert df[col].iloc[0] == 0.0, f"Expected 0.0 for inactive flag {col}"

    @pytest.mark.parametrize(
        "seller_val,active_col",
        [
            ("Individual", "seller_type_Individual"),
            ("Trustmark Dealer", "seller_type_Trustmark Dealer"),
        ],
    )
    def test_seller_type_active_flags(self, preprocessor, sample_payload, seller_val, active_col):
        payload = sample_payload.copy()
        payload["seller_type"] = seller_val
        df = preprocessor.preprocess(payload)

        all_seller_cols = ["seller_type_Individual", "seller_type_Trustmark Dealer"]
        for col in all_seller_cols:
            if col == active_col:
                assert df[col].iloc[0] == 1.0
            else:
                assert df[col].iloc[0] == 0.0

    def test_transmission_manual_active_flag(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["transmission"] = "Manual"
        df = preprocessor.preprocess(payload)
        assert df["transmission_Manual"].iloc[0] == 1.0


class TestFrequencyEncodingAndFallbacks:
    """Verify frequency encoding for known categories and 0.0 fallback for unseen categories."""

    def test_known_brand_and_model_encoding(self, preprocessor, sample_payload, brand_map, model_map):
        df = preprocessor.preprocess(sample_payload)
        expected_brand_val = brand_map["Maruti"]
        expected_model_val = model_map["Swift"]

        assert df["brand_encoded"].iloc[0] == pytest.approx(expected_brand_val)
        assert df["model_encoded"].iloc[0] == pytest.approx(expected_model_val)

    def test_unseen_brand_fallback_to_zero(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["brand"] = "NonExistentBrand"
        df = preprocessor.preprocess(payload)

        assert df["brand_encoded"].iloc[0] == 0.0

    def test_unseen_model_fallback_to_zero(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["model"] = "NonExistentModel"
        df = preprocessor.preprocess(payload)

        assert df["model_encoded"].iloc[0] == 0.0

    def test_both_unseen_fallback_to_zero(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["brand"] = "AlienMotors"
        payload["model"] = "UFO"
        df = preprocessor.preprocess(payload)

        assert df["brand_encoded"].iloc[0] == 0.0
        assert df["model_encoded"].iloc[0] == 0.0

    @pytest.mark.parametrize(
        "brand_input",
        ["maruti", "MARUTI", " Maruti ", "  Maruti\t"],
    )
    def test_case_and_whitespace_insensitive_brand(self, preprocessor, sample_payload, brand_input, brand_map):
        payload = sample_payload.copy()
        payload["brand"] = brand_input
        df = preprocessor.preprocess(payload)

        assert df["brand_encoded"].iloc[0] == pytest.approx(brand_map["Maruti"])

    @pytest.mark.parametrize(
        "model_input",
        ["swift", "SWIFT", " Swift ", "  Swift\t"],
    )
    def test_case_and_whitespace_insensitive_model(self, preprocessor, sample_payload, model_input, model_map):
        payload = sample_payload.copy()
        payload["model"] = model_input
        df = preprocessor.preprocess(payload)

        assert df["model_encoded"].iloc[0] == pytest.approx(model_map["Swift"])


class TestNumericFeaturesPreservation:
    """Verify numeric feature values are preserved and properly cast."""

    def test_numeric_values_preserved(self, preprocessor, sample_payload):
        df = preprocessor.preprocess(sample_payload)

        assert df["km_driven"].iloc[0] == pytest.approx(45000.0)
        assert df["mileage"].iloc[0] == pytest.approx(18.5)
        assert df["engine"].iloc[0] == pytest.approx(1197.0)
        assert df["max_power"].iloc[0] == pytest.approx(82.0)
        assert df["seats"].iloc[0] == pytest.approx(5.0)

    def test_numeric_string_inputs_coerced(self, preprocessor, sample_payload):
        payload = sample_payload.copy()
        payload["km_driven"] = "50000"
        payload["mileage"] = "19.2"
        payload["engine"] = "1498"
        payload["max_power"] = "90.5"
        payload["seats"] = "7"
        df = preprocessor.preprocess(payload)

        assert df["km_driven"].iloc[0] == pytest.approx(50000.0)
        assert df["mileage"].iloc[0] == pytest.approx(19.2)
        assert df["engine"].iloc[0] == pytest.approx(1498.0)
        assert df["max_power"].iloc[0] == pytest.approx(90.5)
        assert df["seats"].iloc[0] == pytest.approx(7.0)


class TestPricingContextExtraction:
    """Verify extraction of post-model heuristic pricing context."""

    def test_extract_pricing_context(self, preprocessor):
        payload = {"year": "2019", "mileage": "18.5", "owner": "Second Owner"}
        ctx = preprocessor.extract_pricing_context(payload)

        assert ctx["year"] == 2019
        assert ctx["mileage"] == 18.5
        assert ctx["owner"] == "Second Owner"

    def test_extract_pricing_inputs_tuple(self, preprocessor):
        payload = {"year": 2021, "mileage": 17.4, "owner": "First Owner"}
        year, mileage, owner = preprocessor.extract_pricing_inputs(payload)

        assert year == 2021
        assert mileage == 17.4
        assert owner == "First Owner"

    def test_empty_payload_safety(self, preprocessor):
        df = preprocessor.preprocess({})
        assert df.shape == (1, 14)
        assert not df.isna().any().any()

    def test_batch_preprocessing(self, preprocessor, sample_payload):
        p1 = sample_payload.copy()
        p2 = sample_payload.copy()
        p2["fuel"] = "Diesel"
        p3 = sample_payload.copy()
        p3["fuel"] = "CNG"

        df_batch = preprocessor.preprocess_batch([p1, p2, p3])
        assert df_batch.shape == (3, 14)
        assert df_batch["fuel_Petrol"].iloc[0] == 1.0
        assert df_batch["fuel_Diesel"].iloc[1] == 1.0
        assert df_batch["fuel_Diesel"].iloc[2] == 0.0
