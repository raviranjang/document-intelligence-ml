"""Tests for deterministic temperature fitting and artifact lineage."""

from pathlib import Path

from document_intelligence.calibration import (
    CalibrationProvenance,
    CalibrationSample,
    fit_temperature_scaling,
    load_temperature_scaling_config,
)

CONFIG_PATH = Path("configs/calibration/temperature_scaling.toml")


def _provenance() -> CalibrationProvenance:
    return CalibrationProvenance(
        base_model_name="invoice-layout",
        base_model_version="1.0.0",
        calibration_dataset_name="approved-calibration-split",
        calibration_dataset_version="1.0.0",
        source_commit="e" * 40,
        calibration_config_version="1.0.0",
    )


def test_overconfident_predictions_fit_temperature_above_identity() -> None:
    samples = (
        CalibrationSample("correct-1", (5.0, 0.0), 0),
        CalibrationSample("correct-2", (5.0, 0.0), 0),
        CalibrationSample("wrong-1", (5.0, 0.0), 1),
        CalibrationSample("wrong-2", (5.0, 0.0), 1),
    )

    artifact = fit_temperature_scaling(
        samples=samples,
        provenance=_provenance(),
        config=load_temperature_scaling_config(CONFIG_PATH),
    )

    assert artifact.temperature > 1.0
    assert (
        artifact.metrics_after.negative_log_likelihood
        <= artifact.metrics_before.negative_log_likelihood
    )
    assert (
        artifact.metrics_after.expected_calibration_error
        < artifact.metrics_before.expected_calibration_error
    )
    assert sum(artifact.calibrate((5.0, 0.0))) == 1.0


def test_underconfident_correct_predictions_fit_temperature_below_identity() -> None:
    samples = (
        CalibrationSample("class-0", (0.2, 0.0), 0),
        CalibrationSample("class-1", (0.0, 0.2), 1),
    )

    artifact = fit_temperature_scaling(
        samples=samples,
        provenance=_provenance(),
        config=load_temperature_scaling_config(CONFIG_PATH),
    )

    assert artifact.temperature < 1.0
    assert (
        artifact.metrics_after.negative_log_likelihood
        < artifact.metrics_before.negative_log_likelihood
    )


def test_artifact_serialization_excludes_raw_samples() -> None:
    artifact = fit_temperature_scaling(
        samples=(CalibrationSample("private-sample-id", (1.0, 0.0), 0),),
        provenance=_provenance(),
        config=load_temperature_scaling_config(CONFIG_PATH),
    )

    payload = artifact.to_dict()

    assert payload["method"] == "temperature_scaling"
    assert payload["calibration_dataset"] == {
        "name": "approved-calibration-split",
        "version": "1.0.0",
    }
    assert payload["parameters"]["temperature"] == artifact.temperature
    assert "private-sample-id" not in str(payload)
    assert "logits" not in str(payload)


def test_repository_calibration_config_is_versioned_and_loadable() -> None:
    config = load_temperature_scaling_config(CONFIG_PATH)

    assert config.schema_version == "1.0.0"
    assert config.method == "temperature_scaling"
    assert config.minimum_temperature < 1 < config.maximum_temperature
    assert config.reliability_bin_count == 15
