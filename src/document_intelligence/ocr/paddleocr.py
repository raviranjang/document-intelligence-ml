"""PaddleOCR adapter for the untouched pretrained OCR baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from typing import Any, Protocol, cast

from document_intelligence.common.types import (
    BoundingBox,
    OCRDocument,
    OCRModelMetadata,
    OCRPage,
    OCRToken,
)
from document_intelligence.ocr.config import PaddleOCRConfig

SUPPORTED_DOCUMENT_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff"})


class OCRAdapterError(RuntimeError):
    """Raised when PaddleOCR cannot produce the canonical output contract."""


class PaddlePipeline(Protocol):
    """Narrow interface used from the third-party PaddleOCR pipeline."""

    def predict(self, input_document: str) -> Iterable[object]:
        """Return PaddleOCR result objects for an input path."""


class PaddleOCRBaseline:
    """Long-lived adapter around untouched official PaddleOCR weights."""

    def __init__(self, pipeline: PaddlePipeline, config: PaddleOCRConfig) -> None:
        self._pipeline = pipeline
        self._config = config

    @classmethod
    def create(cls, config: PaddleOCRConfig) -> PaddleOCRBaseline:
        """Load PaddleOCR once using the configured official pretrained models."""
        try:
            installed_version = version("paddleocr")
            installed_paddle_version = version("paddlepaddle")
            paddleocr_module = import_module("paddleocr")
        except (PackageNotFoundError, ModuleNotFoundError) as error:
            raise OCRAdapterError(
                "PaddleOCR is not installed; install the project with the 'ocr' extra"
            ) from error
        if installed_version != config.paddleocr_version:
            raise OCRAdapterError(
                f"configured PaddleOCR {config.paddleocr_version} but found {installed_version}"
            )
        if installed_paddle_version != config.paddlepaddle_version:
            raise OCRAdapterError(
                f"configured PaddlePaddle {config.paddlepaddle_version} but found "
                f"{installed_paddle_version}"
            )

        pipeline_factory: Any = getattr(paddleocr_module, "PaddleOCR", None)
        if pipeline_factory is None:
            raise OCRAdapterError("installed paddleocr package does not expose PaddleOCR")
        try:
            pipeline: PaddlePipeline = pipeline_factory(**config.to_paddle_options())
        except Exception as error:
            raise OCRAdapterError(f"failed to initialize PaddleOCR: {error}") from error
        return cls(pipeline, config)

    def predict(self, document: Path, *, document_id: str | None = None) -> OCRDocument:
        """Run OCR and convert Paddle results into the canonical document contract."""
        self._validate_document_path(document)
        try:
            raw_results = self._pipeline.predict(str(document))
            pages = tuple(
                self._convert_page(result, fallback_page_index=index)
                for index, result in enumerate(raw_results)
            )
        except OCRAdapterError:
            raise
        except Exception as error:
            raise OCRAdapterError(
                f"PaddleOCR prediction failed for {document.name}: {error}"
            ) from error

        if not pages:
            raise OCRAdapterError("PaddleOCR returned no page results")
        return OCRDocument(
            document_id=document_id or document.name,
            pages=pages,
            model=OCRModelMetadata(
                name=self._config.model_name,
                version=self._config.model_version,
                source=self._config.model_source,
            ),
        )

    @staticmethod
    def _validate_document_path(document: Path) -> None:
        if not document.exists():
            raise FileNotFoundError(document)
        if not document.is_file():
            raise ValueError("OCR input must be a file")
        if document.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
            raise ValueError(
                f"unsupported OCR input type {document.suffix!r}; expected one of {supported}"
            )

    @staticmethod
    def _convert_page(result: object, *, fallback_page_index: int) -> OCRPage:
        payload = _extract_result_payload(result)
        raw_page_index = payload.get("page_index")
        page_index = fallback_page_index if raw_page_index is None else raw_page_index
        if isinstance(page_index, bool) or not isinstance(page_index, int):
            raise OCRAdapterError("PaddleOCR page_index must be an integer or null")

        texts = _as_list(payload.get("rec_texts"), "rec_texts")
        scores = _as_list(payload.get("rec_scores"), "rec_scores")
        raw_boxes = payload.get("rec_boxes")
        boxes = (
            _as_list(raw_boxes, "rec_boxes")
            if raw_boxes is not None
            else _as_list(payload.get("rec_polys"), "rec_polys")
        )
        if not (len(texts) == len(scores) == len(boxes)):
            raise OCRAdapterError(
                "PaddleOCR rec_texts, rec_scores, and geometry lengths must match"
            )

        tokens: list[OCRToken] = []
        for text, score, geometry in zip(texts, scores, boxes, strict=True):
            if not isinstance(text, str):
                raise OCRAdapterError("PaddleOCR rec_texts must contain strings")
            if not text.strip():
                continue
            tokens.append(
                OCRToken(
                    text=text,
                    bounding_box=_to_bounding_box(geometry),
                    confidence=_to_confidence(score),
                    page_index=page_index,
                    token_index=len(tokens),
                )
            )
        return OCRPage(page_index=page_index, tokens=tuple(tokens))


def _extract_result_payload(result: object) -> Mapping[str, Any]:
    result_json: object = result if isinstance(result, Mapping) else getattr(result, "json", None)
    if not isinstance(result_json, Mapping):
        raise OCRAdapterError("PaddleOCR result must expose a mapping through its json attribute")

    payload = result_json.get("res", result_json)
    if not isinstance(payload, Mapping):
        raise OCRAdapterError("PaddleOCR result 'res' value must be a mapping")
    return cast(Mapping[str, Any], payload)


def _as_list(value: object, field_name: str) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise OCRAdapterError(f"PaddleOCR {field_name} must be an array")
    try:
        return list(cast(Iterable[Any], value))
    except TypeError as error:
        raise OCRAdapterError(f"PaddleOCR {field_name} must be an array") from error


def _to_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(cast(Any, value))
        except (TypeError, ValueError) as error:
            raise OCRAdapterError("PaddleOCR rec_scores must contain real numbers") from error
    confidence = float(value)
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise OCRAdapterError(f"invalid PaddleOCR recognition confidence: {confidence}")
    return confidence


def _to_bounding_box(geometry: object) -> BoundingBox:
    values = _as_list(geometry, "recognition geometry")
    try:
        if len(values) == 4 and all(not _is_coordinate_pair(value) for value in values):
            x_min, y_min, x_max, y_max = (float(cast(Any, value)) for value in values)
        else:
            points = [_as_list(point, "polygon point") for point in values]
            if len(points) < 3 or any(len(point) != 2 for point in points):
                raise OCRAdapterError("PaddleOCR polygon must contain at least three x/y points")
            x_coordinates = [float(cast(Any, point[0])) for point in points]
            y_coordinates = [float(cast(Any, point[1])) for point in points]
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)
        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    except OCRAdapterError:
        raise
    except (TypeError, ValueError) as error:
        raise OCRAdapterError("invalid PaddleOCR recognition geometry") from error


def _is_coordinate_pair(value: object) -> bool:
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        return len(cast(Any, value)) == 2
    except TypeError:
        return False
