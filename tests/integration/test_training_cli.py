"""
Integration tests for Training CLI entrypoint (R1).
"""
import json
from pathlib import Path
import joblib
import pytest

from car_price_predictor.training.cli import build_parser, main


class TestTrainingCliIntegration:
    """Verifies CLI argument parsing, end-to-end execution, and exit codes."""

    def test_cli_parser_defaults(self):
        """Argument parser defines expected default options."""
        parser = build_parser()
        args = parser.parse_args([])

        assert args.data is None
        assert args.sample is False
        assert args.output_dir == "artifacts"
        assert args.min_r2 == 0.80
        assert args.n_estimators == 100
        assert args.dry_run is False

    def test_cli_main_sample_execution(self, tmp_path):
        """CLI main with --sample flag completes with exit code 0 and writes artifacts."""
        exit_code = main(
            [
                "--sample",
                "--output-dir", str(tmp_path),
                "--version", "v1-cli-test",
                "--n-estimators", "5",
                "--min-r2", "0.50",
            ]
        )
        assert exit_code == 0

        joblib_path = tmp_path / "model_v1-cli-test.joblib"
        meta_path = tmp_path / "model_v1-cli-test_metadata.json"

        assert joblib_path.exists()
        assert meta_path.exists()

        # Validate saved payload structure
        data = joblib.load(joblib_path)
        assert "model" in data
        assert "columns" in data

    def test_cli_main_dry_run_flag(self, tmp_path):
        """CLI main with --dry-run completes with exit code 0 without writing files."""
        exit_code = main(
            [
                "--sample",
                "--dry-run",
                "--output-dir", str(tmp_path),
                "--n-estimators", "5",
                "--min-r2", "0.50",
            ]
        )
        assert exit_code == 0
        assert list(tmp_path.glob("*.joblib")) == []

    def test_cli_main_invalid_data_file_returns_nonzero(self):
        """CLI main with non-existent dataset file returns exit code 1."""
        exit_code = main(["--data", "non_existent_file_xyz_123.csv"])
        assert exit_code == 1

    def test_cli_metadata_hyperparameters_recorded(self, tmp_path):
        """Hyperparameters configured via CLI flags are serialized in metadata JSON."""
        exit_code = main(
            [
                "--sample",
                "--output-dir", str(tmp_path),
                "--version", "v2-params-test",
                "--n-estimators", "7",
                "--random-state", "123",
                "--min-r2", "0.50",
            ]
        )
        assert exit_code == 0

        meta_path = tmp_path / "model_v2-params-test_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["hyperparameters"]["n_estimators"] == 7
        assert metadata["hyperparameters"]["random_state"] == 123
