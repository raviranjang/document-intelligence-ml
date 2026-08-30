"""Framework-free smoke test for the complete inference adapter chain."""

from collections.abc import Iterable
from pathlib import Path

from document_intelligence.classification import (
    DocumentLabel,
    KeywordInvoiceClassifier,
    load_keyword_classifier_config,
)
from document_intelligence.extraction import (
    DeterministicExtractor,
    EntityType,
    load_deterministic_extraction_config,
)
from document_intelligence.inference import DocumentInferencePipeline
from document_intelligence.ocr.config import load_paddle_ocr_config
from document_intelligence.ocr.paddleocr import PaddleOCRBaseline

REPOSITORY_ROOT = Path(__file__).parents[2]


class _SyntheticPaddlePipeline:
    def predict(self, input_document: str) -> Iterable[object]:
        return (
            {
                "res": {
                    "input_path": input_document,
                    "page_index": None,
                    "rec_texts": ["INVOICE", "Invoice No: INV-42", "Subtotal"],
                    "rec_scores": [0.99, 0.97, 0.96],
                    "rec_boxes": [
                        [10, 10, 90, 30],
                        [10, 40, 180, 60],
                        [10, 70, 90, 90],
                    ],
                }
            },
        )


def test_inference_pipeline_returns_versioned_model_evidence(tmp_path: Path) -> None:
    input_image = tmp_path / "synthetic-invoice.png"
    input_image.write_bytes(b"synthetic image fixture")
    pipeline = DocumentInferencePipeline(
        ocr=PaddleOCRBaseline(
            _SyntheticPaddlePipeline(),
            load_paddle_ocr_config(REPOSITORY_ROOT / "configs" / "ocr" / "baseline.toml"),
        ),
        classifier=KeywordInvoiceClassifier(
            load_keyword_classifier_config(
                REPOSITORY_ROOT / "configs" / "classification" / "keyword_invoice_baseline.toml"
            )
        ),
        extractor=DeterministicExtractor(
            load_deterministic_extraction_config(
                REPOSITORY_ROOT / "configs" / "extraction" / "deterministic_baseline.toml"
            )
        ),
    )

    result = pipeline.predict(input_image, document_id="synthetic-invoice")

    assert result.schema_version == "1.0.0"
    assert result.classification.label is DocumentLabel.INVOICE
    assert result.extraction.entities[0].entity_type is EntityType.INVOICE_NUMBER
    assert result.extraction.entities[0].raw_value == "INV-42"
    assert result.ocr.model.version
    assert result.classification.model.version == "1.0.0"
    assert result.extraction.model.version == "1.0.0"
