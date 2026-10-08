"""
Unit tests for ML Training Pipeline and Ingestion (R1).
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from car_price_predictor.services.constants import FEATURE_COLUMNS
from car_price_predictor.training.data_loader import (
    clean_raw_dataframe,
    generate_synthetic_data,
    generate_synthetic_sample,
    load_dataset,
    parse_numeric_unit,
)
from car_price_predictor.training.pipeline import TrainingConfig, TrainingPipeline
from car_price_predictor.training.preprocessor_adapter import TrainingPreprocessorAdapter
from car_price_predictor.training.versioning import resolve_next_version, save_versioned_artifact


class TestDataLoaderAndIngestion:
    """Tests data cleaning, unit stripping, and synthetic data generation."""

    def test_parse_numeric_unit_valid_strings(self):
        """Extracts float values cleanly from unit-bearing strings."""
        assert parse_numeric_unit("18.5 kmpl") == 18.5
        assert parse_numeric_unit("21.14 km/kg") == 21.14
        assert parse_numeric_unit("1197 CC") == 1197.0
        assert parse_numeric_unit("82 bhp") == 82.0
        assert parse_numeric_unit("45,000") == 45000.0
        assert parse_numeric_unit(1200) == 1200.0
        assert parse_numeric_unit(82.5) == 82.5

    def test_parse_numeric_unit_null_and_malformed(self):
        """Returns None for null, empty, or unparseable input."""
        assert parse_numeric_unit(None) is None
        assert parse_numeric_unit("") is None
        assert parse_numeric_unit("   ") is None
        assert parse_numeric_unit(float("nan")) is None
        assert parse_numeric_unit("N/A") is None

    def test_clean_raw_dataframe_column_aliases_and_dropping(self):
        """Normalizes aliases and drops non-feature/index columns."""
        raw_df = pd.DataFrame(
            {
                "Unnamed: 0": [0],
                "car_name": ["Maruti Swift VXI"],
                "fuel_type": ["Petrol"],
                "transmission_type": ["Manual"],
                "vehicle_age": [5],
                "owner_type": ["First Owner"],
                "selling_price": ["500,000"],
                "km_driven": ["45,000"],
                "mileage": ["18.5 kmpl"],
                "engine": ["1197 CC"],
                "max_power": ["82 bhp"],
                "seats": [5],
                "min_cost_price": [400000],
            }
        )
        cleaned = clean_raw_dataframe(raw_df)

        assert "unnamed:_0" not in cleaned.columns
        assert "min_cost_price" not in cleaned.columns
        assert "brand" in cleaned.columns and cleaned["brand"].iloc[0] == "Maruti"
        assert "model" in cleaned.columns and cleaned["model"].iloc[0] == "Swift"
        assert cleaned["fuel"].iloc[0] == "Petrol"
        assert cleaned["transmission"].iloc[0] == "Manual"
        assert cleaned["owner"].iloc[0] == "First Owner"
        assert cleaned["selling_price"].iloc[0] == 500000.0
        assert cleaned["mileage"].iloc[0] == 18.5
        assert cleaned["engine"].iloc[0] == 1197.0
        assert cleaned["max_power"].iloc[0] == 82.0

    def test_generate_synthetic_data_schema_and_reproducibility(self):
        """Generates deterministic DataFrame with expected columns and distributions."""
        df1 = generate_synthetic_data(n_samples=50, random_state=42)
        df2 = generate_synthetic_sample(n_samples=50, random_state=42)

        assert len(df1) == 50
        assert len(df2) == 50
        pd.testing.assert_frame_equal(df1, df2)

        required_cols = [
            "car_name", "brand", "model", "year", "selling_price",
            "km_driven", "fuel", "seller_type", "transmission", "owner",
            "mileage", "engine", "max_power", "seats",
        ]
        for col in required_cols:
            assert col in df1.columns

        assert (df1["selling_price"] > 0).all()
        assert (df1["km_driven"] >= 0).all()
        assert (df1["max_power"] > 0).all()

    def test_load_dataset_sample_mode(self):
        """load_dataset with use_sample=True returns generated DataFrame."""
        df = load_dataset(use_sample=True, n_samples=30, random_state=99)
        assert len(df) == 30
        assert "selling_price" in df.columns

    def test_load_dataset_missing_file_raises_error(self):
        """load_dataset with non-existent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(data_path="non_existent_dataset_file.csv")


class TestPreprocessorAdapter:
    """Tests TrainingPreprocessorAdapter consistency and column alignment."""

    def test_adapter_fit_generates_frequency_maps(self):
        """Fitting computes normalized frequencies and brand-to-model hierarchy."""
        sample_df = pd.DataFrame(
            {
                "brand": ["Maruti", "Maruti", "Hyundai"],
                "model": ["Swift", "WagonR", "i10"],
            }
        )
        adapter = TrainingPreprocessorAdapter()
        adapter.fit(sample_df)

        assert adapter.brand_map["Maruti"] == pytest.approx(2 / 3)
        assert adapter.brand_map["Hyundai"] == pytest.approx(1 / 3)
        assert adapter.model_map["Swift"] == pytest.approx(1 / 3)
        assert "Swift" in adapter.brand_to_models["Maruti"]
        assert "WagonR" in adapter.brand_to_models["Maruti"]
        assert "i10" in adapter.brand_to_models["Hyundai"]

    def test_adapter_transform_exact_14_columns(self):
        """Transform produces DataFrame matching exact 14 FEATURE_COLUMNS in order."""
        sample_df = generate_synthetic_data(n_samples=20, random_state=42)
        adapter = TrainingPreprocessorAdapter()
        transformed = adapter.fit_transform(sample_df)

        assert list(transformed.columns) == list(FEATURE_COLUMNS)
        assert len(transformed) == 20
        assert not transformed.isna().any().any()

    def test_adapter_transform_unseen_categories_zeroed(self):
        """Unseen brands/models in validation get encoded as 0.0 frequency."""
        train_df = pd.DataFrame(
            {
                "brand": ["Maruti"],
                "model": ["Swift"],
                "km_driven": [45000],
                "mileage": [18.5],
                "engine": [1197],
                "max_power": [82.0],
                "seats": [5],
                "fuel": ["Petrol"],
                "transmission": ["Manual"],
                "seller_type": ["Individual"],
            }
        )
        adapter = TrainingPreprocessorAdapter()
        adapter.fit(train_df)

        val_df = pd.DataFrame(
            {
                "brand": ["Tesla"],
                "model": ["Model3"],
                "km_driven": [10000],
                "mileage": [0.0],
                "engine": [0.0],
                "max_power": [280.0],
                "seats": [5],
                "fuel": ["Electric"],
                "transmission": ["Automatic"],
                "seller_type": ["Dealer"],
            }
        )
        transformed = adapter.transform(val_df)

        assert transformed["brand_encoded"].iloc[0] == 0.0
        assert transformed["model_encoded"].iloc[0] == 0.0
        # Reference dropped: Automatic transmission -> transmission_Manual is 0.0
        assert transformed["transmission_Manual"].iloc[0] == 0.0
        # Reference dropped: Dealer seller -> all seller dummies 0.0
        assert transformed["seller_type_Individual"].iloc[0] == 0.0
        assert transformed["seller_type_Trustmark Dealer"].iloc[0] == 0.0


class TestArtifactVersioning:
    """Tests model artifact serialization, versioning, and metadata JSON generation."""

    def test_resolve_next_version_autoincrement(self, tmp_path):
        """Computes incremental version numbers from existing files."""
        assert resolve_next_version(tmp_path) == "v1"

        (tmp_path / "model_v1.joblib").touch()
        assert resolve_next_version(tmp_path) == "v2"

        (tmp_path / "model_v2.joblib").touch()
        assert resolve_next_version(tmp_path) == "v3"

    def test_save_versioned_artifact_five_keys_contract(self, tmp_path):
        """Serialized joblib preserves 5 core keys and companion metadata JSON."""
        from sklearn.dummy import DummyRegressor

        model = DummyRegressor(strategy="mean")
        columns = list(FEATURE_COLUMNS)
        brand_map = {"Maruti": 0.5, "Hyundai": 0.5}
        model_map = {"Swift": 0.5, "i10": 0.5}
        brand_to_models = {"Maruti": ["Swift"], "Hyundai": ["i10"]}

        joblib_path, meta_path = save_versioned_artifact(
            model=model,
            columns=columns,
            brand_map=brand_map,
            model_map=model_map,
            brand_to_models=brand_to_models,
            output_dir=tmp_path,
            version_tag="v1-test",
        )

        assert joblib_path.exists()
        assert meta_path.exists()

        # Verify joblib payload
        payload = joblib.load(joblib_path)
        assert isinstance(payload, dict)
        for key in ["model", "columns", "brand_map", "model_map", "brand_to_models"]:
            assert key in payload

        assert payload["columns"] == columns
        assert payload["brand_map"] == brand_map
        assert payload["version"] == "v1-test"

        # Verify metadata JSON
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["version"] == "v1-test"
        assert meta["model_class"] == "DummyRegressor"
        assert "feature_columns" in meta
        assert "timestamp" in meta


class TestTrainingPipelineExecution:
    """Tests end-to-end TrainingPipeline execution with synthetic samples."""

    def test_pipeline_run_end_to_end_fast(self, tmp_path):
        """Runs fast training pipeline with small synthetic sample and saves artifact."""
        config = TrainingConfig(
            use_sample=True,
            n_samples=100,
            n_estimators=5,
            random_state=42,
            output_dir=tmp_path,
            version="v99-test",
            enforce_gates=False,
        )
        pipeline = TrainingPipeline(config)
        result = pipeline.run()

        assert result["status"] == "success"
        assert "metrics" in result
        assert "train" in result["metrics"]
        assert "validation" in result["metrics"]
        assert result["metrics"]["train"]["r2"] > 0.80

        assert result["joblib_path"] is not None
        assert Path(result["joblib_path"]).exists()
        assert Path(result["metadata_path"]).exists()

    def test_pipeline_dry_run_does_not_write_files(self, tmp_path):
        """dry_run=True computes metrics without serializing artifacts to disk."""
        config = TrainingConfig(
            use_sample=True,
            n_samples=60,
            n_estimators=5,
            random_state=42,
            output_dir=tmp_path,
            dry_run=True,
            enforce_gates=False,
        )
        pipeline = TrainingPipeline(config)
        result = pipeline.run()

        assert result["status"] == "success"
        assert result["joblib_path"] is None
        assert list(tmp_path.glob("*.joblib")) == []

    def test_pipeline_missing_target_raises_error(self):
        """Pipeline raises ValueError if selling_price target column is absent."""
        df_no_target = pd.DataFrame({"brand": ["Maruti"], "km_driven": [10000]})
        config = TrainingConfig(use_sample=False)
        pipeline = TrainingPipeline(config)

        # Mock load_dataset returning df_no_target
        import unittest.mock as mock
        with mock.patch("car_price_predictor.training.pipeline.load_dataset", return_value=df_no_target):
            with pytest.raises(ValueError, match="selling_price"):
                pipeline.run()

    def test_preprocessor_adapter_empty_dataframe_raises(self):
        """Fitting adapter on empty DataFrame raises ValueError."""
        adapter = TrainingPreprocessorAdapter()
        with pytest.raises(ValueError, match="empty"):
            adapter.fit(pd.DataFrame())

    def test_preprocessor_adapter_not_fitted_transform_raises(self):
        """Calling transform before fit raises RuntimeError."""
        adapter = TrainingPreprocessorAdapter()
        with pytest.raises(RuntimeError, match="must be fitted"):
            adapter.transform(pd.DataFrame({"brand": ["Maruti"]}))

    def test_preprocessor_adapter_missing_brand_or_model_handled(self):
        """Adapter handles DataFrames with missing brand or model columns gracefully."""
        adapter = TrainingPreprocessorAdapter()
        df = pd.DataFrame({"km_driven": [10000.0], "selling_price": [200000.0]})
        adapter.fit(df)
        assert adapter.brand_map == {}
        assert adapter.model_map == {}
        transformed = adapter.transform(df)
        assert len(transformed) == 1
        assert transformed["brand_encoded"].iloc[0] == 0.0

    def test_save_versioned_artifact_promote_flag(self, tmp_path):
        """promote=True copies model to designated promote_path."""
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor(strategy="mean")
        promote_target = tmp_path / "promoted_model.joblib"

        joblib_path, _ = save_versioned_artifact(
            model=model,
            columns=list(FEATURE_COLUMNS),
            brand_map={},
            model_map={},
            brand_to_models={},
            output_dir=tmp_path / "artifacts",
            version_tag="v_promote",
            promote=True,
            promote_path=promote_target,
        )

        assert joblib_path.exists()
        assert promote_target.exists()

    def test_clean_raw_dataframe_vehicle_age_sub100_conversion(self):
        """Vehicle age <= 100 converts to calendar year properly."""
        raw_df = pd.DataFrame({
            "car_name": ["Ford EcoSport"],
            "vehicle_age": [4],
            "selling_price": [450000],
            "km_driven": [30000],
            "max_power": [90],
        })
        cleaned = clean_raw_dataframe(raw_df)
        assert "year" in cleaned.columns
        assert cleaned["year"].iloc[0] == 2020  # 2024 - 4
