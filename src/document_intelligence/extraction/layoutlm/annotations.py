"""Strict loading and operational validation for entity annotations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from document_intelligence.common.types import BoundingBox
from document_intelligence.extraction.labels import SemanticLabelSchema, validate_bio_sequence

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnnotationIssue:
    """One actionable annotation validation failure."""

    code: str
    location: str
    message: str


class AnnotationValidationError(ValueError):
    """Raised with every detectable issue in one annotation artifact."""

    def __init__(self, issues: Iterable[AnnotationIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("AnnotationValidationError requires at least one issue")
        details = "\n".join(
            f"[{issue.code}] {issue.location}: {issue.message}" for issue in self.issues
        )
        super().__init__(
            f"annotation validation failed with {len(self.issues)} issue(s):\n{details}"
        )


@dataclass(frozen=True, slots=True)
class AnnotatedToken:
    """One canonical OCR token and its semantic BIO label."""

    token_index: int
    text: str
    bounding_box: BoundingBox
    label: str


@dataclass(frozen=True, slots=True)
class AnnotatedPage:
    """One image page with aligned OCR token annotations."""

    page_index: int
    image_width: int
    image_height: int
    tokens: tuple[AnnotatedToken, ...]


@dataclass(frozen=True, slots=True)
class AnnotatedDocument:
    """Validated semantic annotation ready for feature preparation."""

    document_id: str
    label_schema_version: str
    pages: tuple[AnnotatedPage, ...]
    schema_version: str = "1.0.0"
    source_ocr_schema_version: str = "1.0.0"


class EntityAnnotationLoader:
    """Validate structural, geometry, index, and BIO annotation invariants."""

    def __init__(
        self, annotation_schema: Mapping[str, Any], label_schema: SemanticLabelSchema
    ) -> None:
        try:
            Draft202012Validator.check_schema(annotation_schema)
        except SchemaError as error:
            raise ValueError(f"invalid entity annotation schema: {error.message}") from error
        self._schema_validator = Draft202012Validator(annotation_schema)
        self._label_schema = label_schema

    @classmethod
    def from_schema_file(
        cls, annotation_schema_path: Path, label_schema: SemanticLabelSchema
    ) -> EntityAnnotationLoader:
        """Create a loader from a UTF-8 JSON Schema file."""
        return cls(_load_json_object(annotation_schema_path), label_schema)

    def load(self, annotation_path: Path) -> AnnotatedDocument:
        """Load and validate one UTF-8 annotation artifact."""
        try:
            document = _load_json_object(annotation_path)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            raise AnnotationValidationError(
                (AnnotationIssue("invalid_annotation_file", "$", str(error)),)
            ) from error
        return self.validate(document)

    def validate(self, document: Mapping[str, Any]) -> AnnotatedDocument:
        """Validate an in-memory annotation and return typed immutable values."""
        schema_issues = tuple(
            AnnotationIssue(
                code="schema_violation",
                location=_format_json_path(error.absolute_path),
                message=error.message,
            )
            for error in sorted(
                self._schema_validator.iter_errors(document),
                key=lambda item: tuple(str(component) for component in item.absolute_path),
            )
        )
        if schema_issues:
            raise AnnotationValidationError(schema_issues)

        typed_document = cast(JsonObject, document)
        issues: list[AnnotationIssue] = []
        if typed_document["label_schema_version"] != self._label_schema.schema_version:
            issues.append(
                AnnotationIssue(
                    "label_schema_mismatch",
                    "$.label_schema_version",
                    "annotation and configured label schema versions must match",
                )
            )

        pages: list[AnnotatedPage] = []
        for page_position, raw_page in enumerate(cast(list[JsonObject], typed_document["pages"])):
            page_location = f"$.pages[{page_position}]"
            page_index = cast(int, raw_page["page_index"])
            image_width = cast(int, raw_page["image_width"])
            image_height = cast(int, raw_page["image_height"])
            if page_index != page_position:
                issues.append(
                    AnnotationIssue(
                        "non_contiguous_page_index",
                        f"{page_location}.page_index",
                        f"expected page_index {page_position}",
                    )
                )

            tokens: list[AnnotatedToken] = []
            labels: list[str] = []
            for token_position, raw_token in enumerate(cast(list[JsonObject], raw_page["tokens"])):
                token_location = f"{page_location}.tokens[{token_position}]"
                token_index = cast(int, raw_token["token_index"])
                text = cast(str, raw_token["text"])
                label = cast(str, raw_token["label"])
                if token_index != token_position:
                    issues.append(
                        AnnotationIssue(
                            "non_contiguous_token_index",
                            f"{token_location}.token_index",
                            f"expected token_index {token_position}",
                        )
                    )
                if not text.strip():
                    issues.append(
                        AnnotationIssue(
                            "blank_token_text",
                            f"{token_location}.text",
                            "token text must contain a non-whitespace character",
                        )
                    )
                raw_box = cast(JsonObject, raw_token["bounding_box"])
                try:
                    box = BoundingBox(
                        x_min=cast(float, raw_box["x_min"]),
                        y_min=cast(float, raw_box["y_min"]),
                        x_max=cast(float, raw_box["x_max"]),
                        y_max=cast(float, raw_box["y_max"]),
                    )
                except (TypeError, ValueError) as error:
                    issues.append(
                        AnnotationIssue(
                            "invalid_bounding_box",
                            f"{token_location}.bounding_box",
                            str(error),
                        )
                    )
                    labels.append(label)
                    continue
                if box.x_max > image_width or box.y_max > image_height:
                    issues.append(
                        AnnotationIssue(
                            "box_outside_page",
                            f"{token_location}.bounding_box",
                            "bounding box must remain within image dimensions",
                        )
                    )
                tokens.append(AnnotatedToken(token_index, text, box, label))
                labels.append(label)
            try:
                validate_bio_sequence(tuple(labels), schema=self._label_schema)
            except ValueError as error:
                issues.append(AnnotationIssue("invalid_bio_sequence", page_location, str(error)))
            pages.append(AnnotatedPage(page_index, image_width, image_height, tuple(tokens)))

        if issues:
            raise AnnotationValidationError(issues)
        return AnnotatedDocument(
            document_id=cast(str, typed_document["document_id"]),
            label_schema_version=cast(str, typed_document["label_schema_version"]),
            pages=tuple(pages),
            schema_version=cast(str, typed_document["schema_version"]),
            source_ocr_schema_version=cast(str, typed_document["source_ocr_schema_version"]),
        )


def _load_json_object(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as json_file:
        value: Any = json.load(json_file)
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return cast(JsonObject, value)


def _format_json_path(path: Iterable[str | int]) -> str:
    formatted_path = "$"
    for component in path:
        formatted_path += f"[{component}]" if isinstance(component, int) else f".{component}"
    return formatted_path
