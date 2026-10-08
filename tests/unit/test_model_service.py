"""
Unit tests for ModelService lifecycle, artifact loading, and inference.
"""
from concurrent.futures import ThreadPoolExecutor
import pytest
import pandas as pd
from car_price_predictor.services.model_service import ModelService


class TestModelServiceLifecycle:
    """Verifies artifact loading and readiness state."""

    def test_model_service_loads_successfully(self, model_service):
        assert model_service.is_ready() is True
        assert model_service.model is not None

    def test_model_service_feature_columns_and_maps(self, model_service):
        assert len(model_service.feature_columns) == 14
        assert len(model_service.brand_map) == 32
        assert "Maruti" in model_service.brand_map
        assert "BMW" in model_service.brand_map
        assert len(model_service.model_map) > 100
        assert "Swift" in model_service.model_map

    def test_model_service_missing_file_fails_gracefully(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.joblib"
        service = ModelService(model_path=missing_path, load_on_init=True)
        assert service.is_ready() is False

    def test_model_service_predict_not_ready_raises_error(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.joblib"
        service = ModelService(model_path=missing_path, load_on_init=False)
        dummy_df = pd.DataFrame(0.0, index=[0], columns=service.feature_columns)
        with pytest.raises(RuntimeError, match="not loaded or ready"):
            service.predict(dummy_df)

    def test_model_service_resolve_path_with_subdirectories_from_other_cwd(self, tmp_path, monkeypatch):
        """Relative model path containing subdirectories resolves relative to project root regardless of CWD."""
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        test_sub = project_root / "test_subdir_artifact"
        test_sub.mkdir(exist_ok=True)
        test_file = test_sub / "dummy.joblib"
        test_file.write_text("test", encoding="utf-8")
        try:
            monkeypatch.chdir(tmp_path)
            service = ModelService(model_path="test_subdir_artifact/dummy.joblib", load_on_init=False)
            resolved = service._resolve_path()
            assert resolved == test_file.resolve()
            assert resolved.exists()
        finally:
            if test_file.exists():
                test_file.unlink()
            if test_sub.exists():
                test_sub.rmdir()


class TestModelServiceHierarchy:
    """Verifies brand and model accessors."""

    def test_get_brands_alphabetical(self, model_service):
        brands = model_service.get_brands()
        assert isinstance(brands, list)
        assert len(brands) == 32
        assert brands == sorted(brands)
        assert "Maruti" in brands

    def test_get_hierarchy_mapping(self, model_service):
        hierarchy = model_service.get_hierarchy()
        assert isinstance(hierarchy, dict)
        assert "Maruti" in hierarchy
        assert "Swift" in hierarchy["Maruti"]

    def test_get_models_for_brand(self, model_service):
        maruti_models = model_service.get_models_for_brand("Maruti")
        assert isinstance(maruti_models, list)
        assert "Swift" in maruti_models

        unknown_models = model_service.get_models_for_brand("NonExistentBrand")
        assert unknown_models == []


class TestModelServiceInference:
    """Verifies model prediction and thread safety."""

    def test_predict_returns_valid_float(self, model_service, valid_payload):
        df = model_service.preprocessor.preprocess(valid_payload)
        prediction = model_service.predict(df)

        assert isinstance(prediction, float)
        assert prediction > 0.0

    def test_predict_thread_safety(self, model_service, valid_payload):
        df = model_service.preprocessor.preprocess(valid_payload)

        def worker():
            return model_service.predict(df)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            results = [f.result() for f in futures]

        assert len(results) == 20
        # All thread predictions for the same input should be identical
        first = results[0]
        assert all(r == pytest.approx(first) for r in results)
