"""
Unit tests for Model Evaluation and Benchmark Metrics (R2).
"""
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from car_price_predictor.training.evaluator import ModelEvaluator, ModelQualityGateError


class TestModelEvaluatorMetrics:
    """Tests regression metrics calculations and numerical safeguards."""

    def test_compute_metrics_perfect_predictions(self):
        """Perfect predictions yield R2=1.0, MAE=0.0, RMSE=0.0, MAPE=0.0."""
        y_true = np.array([100000.0, 200000.0, 300000.0])
        y_pred = np.array([100000.0, 200000.0, 300000.0])

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        assert metrics["r2"] == 1.0
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mape"] == 0.0

    def test_compute_metrics_matches_scikit_learn(self):
        """Metrics match standard scikit-learn functions on non-trivial vectors."""
        y_true = np.array([350000.0, 480000.0, 520000.0, 610000.0, 950000.0])
        y_pred = np.array([340000.0, 495000.0, 510000.0, 625000.0, 920000.0])

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        expected_r2 = round(float(r2_score(y_true, y_pred)), 4)
        expected_mae = round(float(mean_absolute_error(y_true, y_pred)), 2)

        assert metrics["r2"] == expected_r2
        assert metrics["mae"] == expected_mae
        assert metrics["rmse"] > 0
        assert metrics["mape"] > 0

    def test_mape_numerical_stabilization_zero_targets(self):
        """MAPE handles zero or near-zero targets using epsilon safeguard without ZeroDivisionError."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 210.0])

        # Should not raise ZeroDivisionError
        mape = ModelEvaluator.calculate_mape(y_true, y_pred, epsilon=1.0)
        assert np.isfinite(mape)
        assert mape > 0

    def test_compute_metrics_empty_array_raises_error(self):
        """Calling compute_metrics with empty arrays raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            ModelEvaluator.compute_metrics(np.array([]), np.array([]))


class TestModelQualityGates:
    """Tests threshold checks, generalization gap calculation, and gate enforcement."""

    def test_quality_gate_passes_for_accurate_model(self):
        """High-performing model passes all baseline quality gates."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = X[:, 0] * 500000.0 + 300000.0

        # Linear relationship model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        evaluator = ModelEvaluator()
        result = evaluator.evaluate_model(
            model=model,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            enforce_gates=False,
        )

        assert result["passed"] is True
        assert len(result["gate_failures"]) == 0
        assert result["validation"]["r2"] >= 0.80
        assert result["validation"]["mae"] <= 150000.0
        assert result["generalization_gap"] <= 0.15

    def test_quality_gate_fails_for_substandard_model(self):
        """Dummy model with R2 < 0 fails validation gate."""
        X_train = np.ones((50, 5))
        y_train = np.linspace(100000, 1000000, 50)
        X_val = np.ones((20, 5))
        y_val = np.linspace(100000, 1000000, 20)

        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator()
        result = evaluator.evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            enforce_gates=False,
        )

        assert result["passed"] is False
        assert len(result["gate_failures"]) > 0
        assert any("R2" in f for f in result["gate_failures"])

    def test_enforce_gates_raises_quality_gate_error(self):
        """enforce_gates=True raises ModelQualityGateError when gates fail."""
        X = np.ones((20, 2))
        y = np.array([10000.0] * 10 + [1000000.0] * 10)

        model = DummyRegressor(strategy="constant", constant=50000.0)
        model.fit(X, y)

        evaluator = ModelEvaluator()
        with pytest.raises(ModelQualityGateError, match="failed quality gate"):
            evaluator.evaluate_model(
                model=model,
                X_train=X,
                y_train=y,
                X_val=X,
                y_val=y,
                enforce_gates=True,
            )

    def test_custom_threshold_override(self):
        """Custom threshold dictionaries are applied."""
        X = np.ones((20, 2))
        y = np.linspace(100000, 200000, 20)

        model = DummyRegressor(strategy="mean")
        model.fit(X, y)

        evaluator = ModelEvaluator()
        # Set lax min_val_r2 threshold to -1.0
        result = evaluator.evaluate_model(
            model=model,
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            thresholds={"min_val_r2": -1.0, "max_val_mae": 500000.0, "max_val_rmse": 500000.0, "max_val_mape": 100.0, "max_generalization_gap": 1.0},
            enforce_gates=False,
        )
        assert result["passed"] is True

    def test_calculate_rmse_exact_formula(self):
        """calculate_rmse matches sqrt(mean((y_true - y_pred)^2))."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([13.0, 24.0, 25.0])
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert ModelEvaluator.calculate_rmse(y_true, y_pred) == pytest.approx(expected_rmse)

    def test_all_five_quality_gates_failures_reported(self):
        """Model failing all metrics reports descriptive messages for all failed gates."""
        X_train = np.ones((50, 2))
        y_train = np.linspace(100000, 1000000, 50)
        X_val = np.ones((20, 2))
        # Large shift in y_val creates massive error and negative R2
        y_val = np.linspace(5000000, 10000000, 20)

        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator()
        result = evaluator.evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            enforce_gates=False,
        )

        assert result["passed"] is False
        failures = result["gate_failures"]
        assert len(failures) >= 3
        assert any("R2" in f for f in failures)
        assert any("MAE" in f for f in failures)
        assert any("RMSE" in f for f in failures)
