"""Tests for the thin pretrained OCR command-line workflow."""

from pathlib import Path

import pytest

from document_intelligence.common.types import OCRDocument, OCRModelMetadata, OCRPage
from document_intelligence.ocr import cli
from document_intelligence.ocr.config import PaddleOCRConfig

REPOSITORY_ROOT = Path(__file__).parents[2]
BASELINE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "ocr" / "baseline.toml"


class FakeAdapter:
    """Adapter fixture that avoids loading model weights in CLI tests."""

    def predict(self, document: Path, *, document_id: str | None = None) -> OCRDocument:
        return OCRDocument(
            document_id=document_id or document.name,
            pages=(OCRPage(page_index=0, tokens=()),),
            model=OCRModelMetadata(
                name="detector+recognizer",
                version="paddleocr-3.7.0/paddlepaddle-3.3.1/PP-OCRv6",
                source="official_pretrained",
            ),
        )


def _replace_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    def create_adapter(config: PaddleOCRConfig) -> FakeAdapter:
        assert config.model_source == "official_pretrained"
        return FakeAdapter()

    monkeypatch.setattr("document_intelligence.ocr.cli.PaddleOCRBaseline.create", create_adapter)


def test_cli_prints_canonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_image = tmp_path / "invoice.png"
    input_image.write_bytes(b"synthetic")
    _replace_adapter(monkeypatch)

    exit_code = cli.run(
        [
            str(input_image),
            "--config",
            str(BASELINE_CONFIG_PATH),
            "--document-id",
            "invoice-1",
        ]
    )

    assert exit_code == 0
    assert '"document_id": "invoice-1"' in capsys.readouterr().out


def test_cli_writes_new_output_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_image = tmp_path / "invoice.png"
    input_image.write_bytes(b"synthetic")
    output_path = tmp_path / "results" / "invoice.json"
    _replace_adapter(monkeypatch)

    assert (
        cli.run(
            [
                str(input_image),
                "--config",
                str(BASELINE_CONFIG_PATH),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    with pytest.raises(FileExistsError):
        cli.run(
            [
                str(input_image),
                "--config",
                str(BASELINE_CONFIG_PATH),
                "--output",
                str(output_path),
            ]
        )
