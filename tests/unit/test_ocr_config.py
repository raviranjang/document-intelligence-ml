"""Tests for the versioned PaddleOCR baseline configuration."""

from pathlib import Path

import pytest

from document_intelligence.ocr.config import PaddleOCRConfig, load_paddle_ocr_config

REPOSITORY_ROOT = Path(__file__).parents[2]
BASELINE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "ocr" / "baseline.toml"


def test_load_baseline_config_identifies_untouched_models() -> None:
    config = load_paddle_ocr_config(BASELINE_CONFIG_PATH)

    assert config.paddleocr_version == "3.7.0"
    assert config.paddlepaddle_version == "3.3.1"
    assert config.model_name == "PP-OCRv6_medium_det+PP-OCRv6_medium_rec"
    assert config.model_version == "paddleocr-3.7.0/paddlepaddle-3.3.1/PP-OCRv6"
    assert config.model_source == "official_pretrained"
    assert config.device == "cpu"


def test_baseline_config_translates_to_documented_paddle_options() -> None:
    config = load_paddle_ocr_config(BASELINE_CONFIG_PATH)

    assert config.to_paddle_options() == {
        "text_detection_model_name": "PP-OCRv6_medium_det",
        "text_recognition_model_name": "PP-OCRv6_medium_rec",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_rec_score_thresh": 0.0,
        "lang": "en",
        "ocr_version": "PP-OCRv6",
        "device": "cpu",
        "engine": "paddle_static",
    }


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_config_rejects_invalid_recognition_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        PaddleOCRConfig(
            schema_version="1.0.0",
            paddleocr_version="3.7.0",
            paddlepaddle_version="3.3.1",
            ocr_version="PP-OCRv6",
            detection_model_name="detector",
            recognition_model_name="recognizer",
            model_source="official_pretrained",
            language="en",
            device="cpu",
            engine="paddle_static",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_threshold=threshold,
        )


def test_config_rejects_non_pretrained_model_source(tmp_path: Path) -> None:
    config_text = BASELINE_CONFIG_PATH.read_text(encoding="utf-8")
    config_text = config_text.replace(
        'model_source = "official_pretrained"', 'model_source = "local"'
    )
    temporary_config = tmp_path / "invalid-baseline.toml"
    temporary_config.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="official pretrained"):
        load_paddle_ocr_config(temporary_config)
