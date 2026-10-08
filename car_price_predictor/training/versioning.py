"""
Model Artifact Versioning and Serialization Module (R1).

Serializes trained model artifacts preserving the 5 core dictionary keys required by ModelService,
and logs structured metadata JSON companion files.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any
import joblib
import sklearn

logger = logging.getLogger(__name__)


def resolve_next_version(output_dir: str | Path = "artifacts") -> str:
    """
    Scans output_dir for existing model_v{X}.joblib artifacts and computes the next
    incremental version string (e.g. 'v1', 'v2').
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        return "v1"

    existing_versions: list[int] = []
    for f in out_path.glob("model_v*.joblib"):
        match = re.search(r"model_v(\d+)\.joblib", f.name)
        if match:
            try:
                existing_versions.append(int(match.group(1)))
            except ValueError:
                pass

    if not existing_versions:
        return "v1"

    return f"v{max(existing_versions) + 1}"


def save_versioned_artifact(
    model: Any,
    columns: list[str],
    brand_map: dict[str, float],
    model_map: dict[str, float],
    brand_to_models: dict[str, list[str]],
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts",
    version_tag: str | None = None,
    promote: bool = False,
    promote_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Serializes versioned .joblib artifact and companion metadata JSON file.
    Preserves all 5 required dictionary keys for 100% ModelService compatibility.

    Returns:
        tuple[Path, Path]: (joblib_path, metadata_json_path)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine version tag
    if not version_tag:
        version_tag = resolve_next_version(out_dir)
    elif not version_tag.startswith("v"):
        version_tag = f"v{version_tag}"

    # Extract feature importances if available
    feature_importances: dict[str, float] = {}
    if hasattr(model, "feature_importances_") and hasattr(model, "n_features_in_"):
        fi_values = model.feature_importances_
        if len(fi_values) == len(columns):
            feature_importances = {
                str(col): round(float(val), 6) for col, val in zip(columns, fi_values)
            }

    # Extract hyperparameters
    hyperparameters = model.get_params() if hasattr(model, "get_params") else {}

    # Build structured metadata dictionary
    meta: dict[str, Any] = {
        "version": version_tag,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_class": model.__class__.__name__,
        "sklearn_version": sklearn.__version__,
        "hyperparameters": hyperparameters,
        "feature_columns": list(columns),
        "feature_importances": feature_importances,
        "num_brands": len(brand_map),
        "num_models": len(model_map),
    }

    if metadata:
        meta.update(metadata)

    # Build joblib serialization dictionary (strictly maintaining 5 core keys)
    artifact_payload: dict[str, Any] = {
        "model": model,
        "columns": list(columns),
        "brand_map": dict(brand_map),
        "model_map": dict(model_map),
        "brand_to_models": dict(brand_to_models),
        # Non-breaking additions:
        "version": version_tag,
        "metadata": meta,
    }

    # Target file paths
    joblib_path = out_dir / f"model_{version_tag}.joblib"
    metadata_path = out_dir / f"model_{version_tag}_metadata.json"

    # Write joblib
    logger.info("Saving model artifact to %s", joblib_path)
    joblib.dump(artifact_payload, joblib_path, compress=3)

    # Write companion metadata JSON
    logger.info("Saving metadata record to %s", metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Promote to active primary model if requested
    if promote:
        if promote_path is None:
            workspace_root = Path(__file__).resolve().parent.parent.parent
            promote_path = workspace_root / "car_price_model.joblib"
        else:
            promote_path = Path(promote_path)

        logger.info("Promoting artifact to %s", promote_path)
        shutil.copy2(joblib_path, promote_path)

    return joblib_path, metadata_path
