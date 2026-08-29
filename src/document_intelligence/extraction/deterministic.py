"""Deterministic field-candidate extraction from canonical OCR."""

from __future__ import annotations

import re
from dataclasses import dataclass

from document_intelligence.common.types import OCRDocument, OCRPage
from document_intelligence.extraction.config import DeterministicExtractionConfig
from document_intelligence.extraction.types import (
    ExtractedEntity,
    ExtractionDocument,
    ExtractionModelMetadata,
    TokenReference,
)


@dataclass(frozen=True, slots=True)
class _TokenSpan:
    token_index: int
    start: int
    end: int


class DeterministicExtractor:
    """Apply versioned lexical rules and retain source-token provenance."""

    def __init__(self, config: DeterministicExtractionConfig) -> None:
        self._config = config
        self._compiled_rules = tuple(
            (rule, re.compile(rule.pattern, flags=re.IGNORECASE)) for rule in config.rules
        )

    def predict(self, document: OCRDocument) -> ExtractionDocument:
        """Return all unique rule matches in stable page/rule/match order."""
        entities: list[ExtractedEntity] = []
        seen: set[tuple[object, ...]] = set()
        for page in document.pages:
            page_text, spans = _page_text_and_spans(page)
            for rule, pattern in self._compiled_rules:
                for match in pattern.finditer(page_text):
                    raw_value = match.group("value").strip()
                    if not raw_value:
                        continue
                    value_start, value_end = match.span("value")
                    token_references = tuple(
                        TokenReference(page_index=page.page_index, token_index=span.token_index)
                        for span in spans
                        if span.start < value_end and span.end > value_start
                    )
                    if not token_references:
                        continue
                    identity = (
                        rule.entity_type,
                        raw_value,
                        page.page_index,
                        token_references,
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    entities.append(
                        ExtractedEntity(
                            entity_type=rule.entity_type,
                            raw_value=raw_value,
                            token_references=token_references,
                            rule_id=rule.rule_id,
                        )
                    )
        return ExtractionDocument(
            document_id=document.document_id,
            entities=tuple(entities),
            model=ExtractionModelMetadata(
                name=self._config.model_name,
                version=self._config.model_version,
                source=self._config.model_source,
            ),
        )


def _page_text_and_spans(page: OCRPage) -> tuple[str, tuple[_TokenSpan, ...]]:
    text_parts: list[str] = []
    spans: list[_TokenSpan] = []
    cursor = 0
    for token in page.tokens:
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token.text)
        cursor += len(token.text)
        spans.append(_TokenSpan(token_index=token.token_index, start=start, end=cursor))
    return "".join(text_parts), tuple(spans)
