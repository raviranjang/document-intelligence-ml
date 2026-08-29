"""Framework-independent LayoutLMv3 page-feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from document_intelligence.extraction.labels import SemanticLabelSchema
from document_intelligence.extraction.layoutlm.alignment import (
    AlignedPageFeatures,
    TokenizerEncoding,
    align_page_encoding,
)
from document_intelligence.extraction.layoutlm.annotations import (
    AnnotatedDocument,
    AnnotatedPage,
)
from document_intelligence.extraction.layoutlm.config import LayoutLMDatasetConfig
from document_intelligence.extraction.layoutlm.geometry import normalize_bounding_box


class WordTokenizer(Protocol):
    """Narrow adapter contract for a tokenizer with source-word alignment."""

    def encode_words(self, words: tuple[str, ...], *, max_length: int) -> TokenizerEncoding:
        """Encode pre-split words and return aligned word IDs."""


@dataclass(frozen=True, slots=True)
class LayoutLMPageExample:
    """One validated document page transformed for LayoutLMv3 token classification."""

    document_id: str
    page_index: int
    image_width: int
    image_height: int
    features: AlignedPageFeatures


class LayoutLMDatasetBuilder:
    """Build training/inference-compatible page features from annotations."""

    def __init__(
        self,
        *,
        tokenizer: WordTokenizer,
        label_schema: SemanticLabelSchema,
        config: LayoutLMDatasetConfig,
    ) -> None:
        self._tokenizer = tokenizer
        self._label_schema = label_schema
        self._config = config

    def build_document(self, document: AnnotatedDocument) -> tuple[LayoutLMPageExample, ...]:
        """Transform every validated page without silently dropping examples."""
        if document.label_schema_version != self._label_schema.schema_version:
            raise ValueError("annotation and configured label schema versions must match")
        return tuple(self._build_page(document.document_id, page) for page in document.pages)

    def _build_page(self, document_id: str, page: AnnotatedPage) -> LayoutLMPageExample:
        words = tuple(token.text for token in page.tokens)
        labels = tuple(token.label for token in page.tokens)
        boxes = tuple(
            normalize_bounding_box(
                token.bounding_box,
                image_width=page.image_width,
                image_height=page.image_height,
                scale=self._config.bounding_box_scale,
            )
            for token in page.tokens
        )
        encoding = self._tokenizer.encode_words(words, max_length=self._config.max_length)
        if len(encoding.input_ids) > self._config.max_length:
            raise ValueError("tokenizer output exceeds configured max_length")
        features = align_page_encoding(
            encoding=encoding,
            word_boxes=boxes,
            word_labels=labels,
            label_schema=self._label_schema,
            ignore_label_id=self._config.ignore_label_id,
        )
        return LayoutLMPageExample(
            document_id=document_id,
            page_index=page.page_index,
            image_width=page.image_width,
            image_height=page.image_height,
            features=features,
        )
