"""Versioned configuration for the untouched PaddleOCR baseline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PaddleOCRConfig:
    """Configuration that uniquely identifies the pretrained OCR baseline."""

    schema_version: str
    paddleocr_version: str
    paddlepaddle_version: str
    ocr_version: str
    detection_model_name: str
    recognition_model_name: str
    model_source: str
    language: str
    device: str
    engine: str
    use_doc_orientation_classify: bool
    use_doc_unwarping: bool
    use_textline_orientation: bool
    text_rec_score_threshold: float

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported OCR configuration schema_version")
        if self.model_source != "official_pretrained":
            raise ValueError("the baseline requires untouched official pretrained weights")
        for field_name in (
            "paddleocr_version",
            "paddlepaddle_version",
            "ocr_version",
            "detection_model_name",
            "recognition_model_name",
            "language",
            "device",
            "engine",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.engine not in {"paddle", "paddle_static", "paddle_dynamic"}:
            raise ValueError("baseline engine must be a Paddle inference engine")
        if isinstance(self.text_rec_score_threshold, bool) or not isinstance(
            self.text_rec_score_threshold, (int, float)
        ):
            raise TypeError("text_rec_score_threshold must be a real number")
        threshold = float(self.text_rec_score_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("text_rec_score_threshold must be between 0 and 1")
        object.__setattr__(self, "text_rec_score_threshold", threshold)

    @property
    def model_name(self) -> str:
        """Return the combined detector and recognizer identity."""
        return f"{self.detection_model_name}+{self.recognition_model_name}"

    @property
    def model_version(self) -> str:
        """Return the framework and OCR-family version identity."""
        return (
            f"paddleocr-{self.paddleocr_version}/paddlepaddle-{self.paddlepaddle_version}/"
            f"{self.ocr_version}"
        )

    def to_paddle_options(self) -> dict[str, object]:
        """Translate stable configuration into PaddleOCR constructor options."""
        return {
            "text_detection_model_name": self.detection_model_name,
            "text_recognition_model_name": self.recognition_model_name,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "text_rec_score_thresh": self.text_rec_score_threshold,
            "lang": self.language,
            "ocr_version": self.ocr_version,
            "device": self.device,
            "engine": self.engine,
        }


def load_paddle_ocr_config(path: Path) -> PaddleOCRConfig:
    """Load a strict PaddleOCR baseline configuration from TOML."""
    with path.open("rb") as config_file:
        document: dict[str, Any] = tomllib.load(config_file)

    expected_root_fields = {"schema_version", "pipeline"}
    unexpected_root_fields = set(document) - expected_root_fields
    if unexpected_root_fields:
        raise ValueError(f"unsupported OCR configuration fields: {sorted(unexpected_root_fields)}")
    if set(document) != expected_root_fields:
        missing_fields = expected_root_fields - set(document)
        raise ValueError(f"missing OCR configuration fields: {sorted(missing_fields)}")

    pipeline = document["pipeline"]
    if not isinstance(pipeline, dict):
        raise TypeError("pipeline must be a TOML table")

    expected_pipeline_fields = {
        "paddleocr_version",
        "paddlepaddle_version",
        "ocr_version",
        "detection_model_name",
        "recognition_model_name",
        "model_source",
        "language",
        "device",
        "engine",
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_textline_orientation",
        "text_rec_score_threshold",
    }
    unexpected_pipeline_fields = set(pipeline) - expected_pipeline_fields
    if unexpected_pipeline_fields:
        raise ValueError(f"unsupported OCR pipeline fields: {sorted(unexpected_pipeline_fields)}")
    missing_pipeline_fields = expected_pipeline_fields - set(pipeline)
    if missing_pipeline_fields:
        raise ValueError(f"missing OCR pipeline fields: {sorted(missing_pipeline_fields)}")

    return PaddleOCRConfig(
        schema_version=_require_type(document["schema_version"], str, "schema_version"),
        paddleocr_version=_require_type(pipeline["paddleocr_version"], str, "paddleocr_version"),
        paddlepaddle_version=_require_type(
            pipeline["paddlepaddle_version"], str, "paddlepaddle_version"
        ),
        ocr_version=_require_type(pipeline["ocr_version"], str, "ocr_version"),
        detection_model_name=_require_type(
            pipeline["detection_model_name"], str, "detection_model_name"
        ),
        recognition_model_name=_require_type(
            pipeline["recognition_model_name"], str, "recognition_model_name"
        ),
        model_source=_require_type(pipeline["model_source"], str, "model_source"),
        language=_require_type(pipeline["language"], str, "language"),
        device=_require_type(pipeline["device"], str, "device"),
        engine=_require_type(pipeline["engine"], str, "engine"),
        use_doc_orientation_classify=_require_bool(
            pipeline["use_doc_orientation_classify"], "use_doc_orientation_classify"
        ),
        use_doc_unwarping=_require_bool(pipeline["use_doc_unwarping"], "use_doc_unwarping"),
        use_textline_orientation=_require_bool(
            pipeline["use_textline_orientation"], "use_textline_orientation"
        ),
        text_rec_score_threshold=_require_number(
            pipeline["text_rec_score_threshold"], "text_rec_score_threshold"
        ),
    )


def _require_type(value: Any, expected_type: type[str], field_name: str) -> str:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be a {expected_type.__name__}")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    return float(value)
