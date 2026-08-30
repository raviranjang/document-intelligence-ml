"""Tests for held-out calibration evaluation reports."""

from pathlib import Path

from document_intelligence.calibration import (
    CalibrationEvaluationProvenance,
    CalibrationProvenance,
    CalibrationSample,
    build_calibration_metrics_report,
    fit_temperature_scaling,
    load_temperature_scaling_config,
)


def test_report_evaluates_held_out_aggregate_and_sorted_cohorts() -> None:
    artifact = fit_temperature_scaling(
        samples=(
            CalibrationSample("fit-1", (2.0, 0.0), 0),
            CalibrationSample("fit-2", (2.0, 0.0), 1),
        ),
        provenance=CalibrationProvenance(
            base_model_name="invoice-layout",
            base_model_version="1.0.0",
            calibration_dataset_name="calibration-split",
            calibration_dataset_version="1.0.0",
            source_commit="e" * 40,
            calibration_config_version="1.0.0",
        ),
        config=load_temperature_scaling_config(
            Path("configs/calibration/temperature_scaling.toml")
        ),
    )
    report = build_calibration_metrics_report(
        provenance=CalibrationEvaluationProvenance(
            evaluation_dataset_name="held-out-evaluation",
            evaluation_dataset_version="1.0.0",
            source_commit="f" * 40,
            evaluation_config_version="1.0.0",
        ),
        artifact=artifact,
        samples=(
            CalibrationSample("eval-private-1", (1.0, 0.0), 0, ("clean",)),
            CalibrationSample("eval-private-2", (1.0, 0.0), 1, ("noisy",)),
        ),
    ).to_dict()

    assert report["evaluation_dataset"] == {
        "name": "held-out-evaluation",
        "version": "1.0.0",
    }
    assert report["calibration"]["dataset"]["name"] == "calibration-split"
    assert report["aggregate"]["sample_count"] == 2
    assert list(report["slices"]) == ["clean", "noisy"]
    assert "eval-private" not in str(report)
    assert "logits" not in str(report)
