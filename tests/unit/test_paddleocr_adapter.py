"""Unit tests for conversion at the PaddleOCR framework boundary."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from document_intelligence.ocr.config import PaddleOCRConfig
from document_intelligence.ocr.paddleocr import OCRAdapterError, PaddleOCRBaseline


class FakeResult:
    """Minimal stand-in for PaddleOCR's result object."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.json = {"res": payload}


class FakePipeline:
    """Deterministic in-memory implementation of the Paddle predict interface."""

    def __init__(self, results: Iterable[object]) -> None:
        self._results = tuple(results)
        self.calls: list[str] = []

    def predict(self, input_document: str) -> Iterable[object]:
        self.calls.append(input_document)
        return self._results


class FailingPipeline:
    """Pipeline used to verify third-party failures retain context."""

    def predict(self, input_document: str) -> Iterable[object]:
        raise RuntimeError(f"engine failed for {input_document}")


@pytest.fixture
def config() -> PaddleOCRConfig:
    return PaddleOCRConfig(
        schema_version="1.0.0",
        paddleocr_version="3.7.0",
        paddlepaddle_version="3.3.1",
        ocr_version="PP-OCRv6",
        detection_model_name="PP-OCRv6_medium_det",
        recognition_model_name="PP-OCRv6_medium_rec",
        model_source="official_pretrained",
        language="en",
        device="cpu",
        engine="paddle_static",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_threshold=0.0,
    )


@pytest.fixture
def input_image(tmp_path: Path) -> Path:
    image = tmp_path / "invoice.png"
    image.write_bytes(b"synthetic image fixture")
    return image


def test_adapter_converts_boxes_scores_and_reading_order(
    config: PaddleOCRConfig, input_image: Path
) -> None:
    pipeline = FakePipeline(
        [
            FakeResult(
                {
                    "page_index": None,
                    "rec_texts": ["Invoice", " ", "42"],
                    "rec_scores": [0.98, 0.25, 0.91],
                    "rec_boxes": [[10, 20, 70, 40], [1, 1, 2, 2], [80, 20, 100, 40]],
                }
            )
        ]
    )
    adapter = PaddleOCRBaseline(pipeline, config)

    document = adapter.predict(input_image, document_id="invoice-42")

    assert pipeline.calls == [str(input_image)]
    assert document.document_id == "invoice-42"
    assert document.model.name == "PP-OCRv6_medium_det+PP-OCRv6_medium_rec"
    assert [token.text for token in document.pages[0].tokens] == ["Invoice", "42"]
    assert [token.token_index for token in document.pages[0].tokens] == [0, 1]
    assert document.pages[0].tokens[0].bounding_box.x_min == 10.0
    assert document.pages[0].tokens[1].confidence == 0.91


def test_adapter_derives_axis_aligned_box_from_polygon(
    config: PaddleOCRConfig, input_image: Path
) -> None:
    pipeline = FakePipeline(
        [
            {
                "page_index": 0,
                "rec_texts": ["Total"],
                "rec_scores": [0.9],
                "rec_polys": [[[12, 20], [50, 18], [52, 30], [10, 32]]],
            }
        ]
    )

    document = PaddleOCRBaseline(pipeline, config).predict(input_image)

    bounding_box = document.pages[0].tokens[0].bounding_box
    assert (bounding_box.x_min, bounding_box.y_min) == (10.0, 18.0)
    assert (bounding_box.x_max, bounding_box.y_max) == (52.0, 32.0)


def test_adapter_preserves_pdf_page_indexes(config: PaddleOCRConfig, tmp_path: Path) -> None:
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"synthetic pdf fixture")
    pipeline = FakePipeline(
        [
            FakeResult(
                {
                    "page_index": page_index,
                    "rec_texts": [],
                    "rec_scores": [],
                    "rec_boxes": [],
                }
            )
            for page_index in range(2)
        ]
    )

    document = PaddleOCRBaseline(pipeline, config).predict(pdf)

    assert [page.page_index for page in document.pages] == [0, 1]


def test_adapter_rejects_misaligned_paddle_result_arrays(
    config: PaddleOCRConfig, input_image: Path
) -> None:
    pipeline = FakePipeline(
        [
            FakeResult(
                {
                    "page_index": 0,
                    "rec_texts": ["Invoice"],
                    "rec_scores": [],
                    "rec_boxes": [[0, 0, 10, 10]],
                }
            )
        ]
    )

    with pytest.raises(OCRAdapterError, match="lengths must match"):
        PaddleOCRBaseline(pipeline, config).predict(input_image)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_adapter_rejects_invalid_recognition_confidence(
    config: PaddleOCRConfig, input_image: Path, confidence: float
) -> None:
    pipeline = FakePipeline(
        [
            FakeResult(
                {
                    "page_index": 0,
                    "rec_texts": ["Invoice"],
                    "rec_scores": [confidence],
                    "rec_boxes": [[0, 0, 10, 10]],
                }
            )
        ]
    )

    with pytest.raises(OCRAdapterError, match="invalid PaddleOCR recognition confidence"):
        PaddleOCRBaseline(pipeline, config).predict(input_image)


def test_adapter_rejects_empty_result_sequence(config: PaddleOCRConfig, input_image: Path) -> None:
    with pytest.raises(OCRAdapterError, match="no page results"):
        PaddleOCRBaseline(FakePipeline([]), config).predict(input_image)


def test_adapter_wraps_framework_prediction_failure(
    config: PaddleOCRConfig, input_image: Path
) -> None:
    with pytest.raises(OCRAdapterError, match=r"prediction failed for invoice\.png"):
        PaddleOCRBaseline(FailingPipeline(), config).predict(input_image)


def test_adapter_rejects_missing_document(config: PaddleOCRConfig, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PaddleOCRBaseline(FakePipeline([]), config).predict(tmp_path / "missing.png")


def test_adapter_rejects_unsupported_document_type(
    config: PaddleOCRConfig, tmp_path: Path
) -> None:
    text_file = tmp_path / "invoice.txt"
    text_file.write_text("not an OCR input", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported OCR input type"):
        PaddleOCRBaseline(FakePipeline([]), config).predict(text_file)
