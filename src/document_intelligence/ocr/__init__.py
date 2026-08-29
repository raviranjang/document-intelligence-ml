"""OCR detection, recognition, and pipeline adapters."""

from document_intelligence.ocr.config import PaddleOCRConfig, load_paddle_ocr_config
from document_intelligence.ocr.paddleocr import OCRAdapterError, PaddleOCRBaseline

__all__ = [
    "OCRAdapterError",
    "PaddleOCRBaseline",
    "PaddleOCRConfig",
    "load_paddle_ocr_config",
]
