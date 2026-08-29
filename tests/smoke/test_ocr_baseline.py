"""Lightweight smoke coverage for the baseline OCR output contract."""

from collections.abc import Iterable
from pathlib import Path

from document_intelligence.ocr.config import load_paddle_ocr_config
from document_intelligence.ocr.paddleocr import PaddleOCRBaseline

REPOSITORY_ROOT = Path(__file__).parents[2]


class SmokePipeline:
    """Framework-free pipeline fixture matching PaddleOCR's documented JSON result."""

    def predict(self, input_document: str) -> Iterable[object]:
        return (
            {
                "res": {
                    "input_path": input_document,
                    "page_index": None,
                    "rec_texts": ["INVOICE", "100.00"],
                    "rec_scores": [0.99, 0.93],
                    "rec_boxes": [[10, 10, 90, 30], [120, 60, 180, 80]],
                }
            },
        )


def test_baseline_emits_canonical_ocr_document(tmp_path: Path) -> None:
    input_image = tmp_path / "synthetic-invoice.png"
    input_image.write_bytes(b"synthetic image fixture")
    config = load_paddle_ocr_config(REPOSITORY_ROOT / "configs" / "ocr" / "baseline.toml")

    output = PaddleOCRBaseline(SmokePipeline(), config).predict(
        input_image, document_id="synthetic-invoice"
    )

    serialized = output.to_dict()
    assert serialized["schema_version"] == "1.0.0"
    assert serialized["document_id"] == "synthetic-invoice"
    assert serialized["model"]["source"] == "official_pretrained"
    assert [token["text"] for token in serialized["pages"][0]["tokens"]] == [
        "INVOICE",
        "100.00",
    ]
