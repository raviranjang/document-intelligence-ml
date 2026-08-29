"""Thin command-line entry point for the pretrained OCR baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from document_intelligence.ocr.config import load_paddle_ocr_config
from document_intelligence.ocr.paddleocr import PaddleOCRBaseline


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline OCR argument parser."""
    parser = argparse.ArgumentParser(
        prog="document-intelligence-ocr",
        description="Run the versioned untouched PaddleOCR baseline.",
    )
    parser.add_argument("input", type=Path, help="Local image or PDF to process.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr/baseline.toml"),
        help="Versioned OCR configuration file.",
    )
    parser.add_argument("--document-id", help="Stable dataset record ID; defaults to filename.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write canonical OCR JSON to this new file; defaults to standard output.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Run baseline OCR and emit its canonical JSON representation."""
    arguments = build_parser().parse_args(argv)
    config = load_paddle_ocr_config(arguments.config)
    adapter = PaddleOCRBaseline.create(config)
    document = adapter.predict(arguments.input, document_id=arguments.document_id)
    serialized_document = json.dumps(document.to_dict(), ensure_ascii=False, indent=2)

    if arguments.output is None:
        print(serialized_document)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(serialized_document)
            output_file.write("\n")
    return 0


def main() -> int:
    """Execute the console entry point."""
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
