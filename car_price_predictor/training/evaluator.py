"""
Model Evaluation and Benchmark Metrics Module (R2).

Computes regression metrics (R2, MAE, RMSE, MAPE) across train and validation splits,
and enforces quality gate thresholds before artifact publication.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ModelQualityGateError(ValueError):
    """Raised when a trained candidate model fails quality gate thresholds."""
    pass


class ModelEvaluator:
    """
    Evaluates regression model performance across train and validation splits,
    calculates benchmark metrics, and enforces deployment quality gates.
    """

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "min_val_r2": 0.80,
        "max_val_mae": 150000.0,
        "max_val_rmse": 275000.0,
        "max_val_mape": 25.0,
        "max_generalization_gap": 0.15,
    }

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates Root Mean Squared Error robustly across scikit-learn versions."""
        y_true_arr = np.asarray(y_true, dtype=np.float64)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        return float(np.sqrt(mse))

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
        """
        Calculates Mean Absolute Percentage Error (percentage 0 to 100).
        Numerically stabilized with epsilon to prevent division by zero on zero or low target values.
        """
        y_true_arr = np.asarray(y_true, dtype=np.float64)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)
        denominator = np.maximum(np.abs(y_true_arr), float(epsilon))
        mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100.0)
        return mape

    @classmethod
    def compute_metrics(
        cls,
        y_true: np.ndarray | pd.Series | list[float],
        y_pred: np.ndarray | pd.Series | list[float],
        epsilon: float = 1.0,
    ) -> dict[str, float]:
        """
        Calculates R2, MAE, RMSE, and MAPE metrics for predictions vs ground truth.
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        if len(y_t) == 0:
            raise ValueError("Cannot compute metrics on empty arrays")

        r2 = float(r2_score(y_t, y_p))
        mae = float(mean_absolute_error(y_t, y_p))
        rmse = cls.calculate_rmse(y_t, y_p)
        mape = cls.calculate_mape(y_t, y_p, epsilon=epsilon)

        return {
            "r2": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
        }

    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        thresholds: dict[str, float] | None = None,
        enforce_gates: bool = False,
    ) -> dict[str, Any]:
        """
        Generates predictions for train and validation splits, computes metrics,
        and checks against quality gate thresholds.
        """
        active_thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            active_thresholds.update(thresholds)

        # Inferences
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Metric calculations
        train_metrics = self.compute_metrics(y_train, y_train_pred)
        val_metrics = self.compute_metrics(y_val, y_val_pred)

        # Generalization gap
        generalization_gap = round(float(train_metrics["r2"] - val_metrics["r2"]), 4)

        # Check quality gates
        gate_failures: list[str] = []

        min_r2 = active_thresholds.get("min_val_r2", 0.80)
        if val_metrics["r2"] < min_r2:
            gate_failures.append(
                f"Validation R2 ({val_metrics['r2']:.4f}) below threshold ({min_r2:.4f})"
            )

        max_mae = active_thresholds.get("max_val_mae", 150000.0)
        if val_metrics["mae"] > max_mae:
            gate_failures.append(
                f"Validation MAE ({val_metrics['mae']:,.2f}) exceeds threshold ({max_mae:,.2f})"
            )

        max_rmse = active_thresholds.get("max_val_rmse", 275000.0)
        if val_metrics["rmse"] > max_rmse:
            gate_failures.append(
                f"Validation RMSE ({val_metrics['rmse']:,.2f}) exceeds threshold ({max_rmse:,.2f})"
            )

        max_mape = active_thresholds.get("max_val_mape", 25.0)
        if val_metrics["mape"] > max_mape:
            gate_failures.append(
                f"Validation MAPE ({val_metrics['mape']:.2f}%) exceeds threshold ({max_mape:.2f}%)"
            )

        max_gap = active_thresholds.get("max_generalization_gap", 0.15)
        if generalization_gap > max_gap:
            gate_failures.append(
                f"Generalization gap ({generalization_gap:.4f}) exceeds threshold ({max_gap:.4f})"
            )

        passed = len(gate_failures) == 0

        summary: dict[str, Any] = {
            "train": train_metrics,
            "validation": val_metrics,
            "generalization_gap": generalization_gap,
            "passed": passed,
            "thresholds": active_thresholds,
            "gate_failures": gate_failures,
        }

        if enforce_gates and not passed:
            err_msg = "Model failed quality gate checks:\n - " + "\n - ".join(gate_failures)
            logger.error(err_msg)
            raise ModelQualityGateError(err_msg)

        return summary
