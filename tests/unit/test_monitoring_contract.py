"""Tests for backend-neutral monitoring records."""

import pytest

from document_intelligence.common.telemetry import (
    InferenceStage,
    MonitoringRecord,
    MonitoringStatus,
)


def test_monitoring_record_accepts_safe_success_dimensions() -> None:
    record = MonitoringRecord(
        stage=InferenceStage.OCR,
        status=MonitoringStatus.SUCCESS,
        duration_seconds=0.125,
        document_id="synthetic-1",
        model_name="ocr-baseline",
        model_version="1.0.0",
        model_source="official_pretrained",
        output_count=2,
        output_categories=("printed",),
        mean_confidence=0.9,
    )

    assert record.duration_seconds == 0.125
    assert record.mean_confidence == 0.9


def test_monitoring_record_requires_complete_model_identity() -> None:
    with pytest.raises(ValueError, match="provided together"):
        MonitoringRecord(
            stage=InferenceStage.OCR,
            status=MonitoringStatus.SUCCESS,
            duration_seconds=0.1,
            model_name="ocr-baseline",
        )


def test_error_record_rejects_prediction_observations() -> None:
    with pytest.raises(ValueError, match="must not include output"):
        MonitoringRecord(
            stage=InferenceStage.EXTRACTION,
            status=MonitoringStatus.ERROR,
            duration_seconds=0.1,
            error_type="RuntimeError",
            output_count=1,
        )


@pytest.mark.parametrize("duration", [-0.1, float("inf"), float("nan")])
def test_monitoring_record_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration_seconds"):
        MonitoringRecord(
            stage=InferenceStage.PIPELINE,
            status=MonitoringStatus.SUCCESS,
            duration_seconds=duration,
        )
