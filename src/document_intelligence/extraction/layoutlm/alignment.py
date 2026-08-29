"""Tokenizer-independent subword label and geometry alignment."""

from __future__ import annotations

from dataclasses import dataclass

from document_intelligence.extraction.labels import (
    LabelPrefix,
    SemanticLabelSchema,
)
from document_intelligence.extraction.layoutlm.geometry import NormalizedBoundingBox

IGNORE_LABEL_ID = -100
SPECIAL_TOKEN_BOX = (0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class TokenizerEncoding:
    """Minimal tokenizer output required for deterministic word alignment."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    word_ids: tuple[int | None, ...]
    truncated: bool

    def __post_init__(self) -> None:
        lengths = {len(self.input_ids), len(self.attention_mask), len(self.word_ids)}
        if len(lengths) != 1:
            raise ValueError("tokenizer output lengths must match")
        if not self.input_ids:
            raise ValueError("tokenizer output must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.input_ids):
            raise TypeError("input_ids must contain integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
            for value in self.attention_mask
        ):
            raise ValueError("attention_mask must contain only zero or one")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a bool")


@dataclass(frozen=True, slots=True)
class AlignedPageFeatures:
    """Model-ready textual, spatial, and supervised page features."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    bounding_boxes: tuple[tuple[int, int, int, int], ...]
    label_ids: tuple[int, ...]
    word_ids: tuple[int | None, ...]
    truncated: bool


def align_page_encoding(
    *,
    encoding: TokenizerEncoding,
    word_boxes: tuple[NormalizedBoundingBox, ...],
    word_labels: tuple[str, ...],
    label_schema: SemanticLabelSchema,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> AlignedPageFeatures:
    """Align source-word boxes and BIO labels to tokenizer subwords."""
    if len(word_boxes) != len(word_labels):
        raise ValueError("word_boxes and word_labels lengths must match")
    if not word_boxes:
        raise ValueError("page must contain at least one source word")
    if isinstance(ignore_label_id, bool) or not isinstance(ignore_label_id, int):
        raise TypeError("ignore_label_id must be an integer")

    observed_word_ids = tuple(word_id for word_id in encoding.word_ids if word_id is not None)
    if not observed_word_ids:
        raise ValueError("tokenizer output contains no source words")
    if any(
        isinstance(word_id, bool) or not isinstance(word_id, int) for word_id in observed_word_ids
    ):
        raise TypeError("word_ids must contain integers or None")
    if any(word_id < 0 or word_id >= len(word_boxes) for word_id in observed_word_ids):
        raise ValueError("word_ids reference a source word outside the page")
    if tuple(sorted(observed_word_ids)) != observed_word_ids:
        raise ValueError("word_ids must be non-decreasing")
    observed_unique_ids = tuple(dict.fromkeys(observed_word_ids))
    expected_ids = tuple(range(observed_unique_ids[-1] + 1))
    if observed_unique_ids != expected_ids:
        raise ValueError("tokenizer output must not skip source words")
    if not encoding.truncated and len(observed_unique_ids) != len(word_boxes):
        raise ValueError("untruncated tokenizer output must include every source word")
    if encoding.truncated and len(observed_unique_ids) == len(word_boxes):
        raise ValueError("truncated output must omit at least one trailing source word")

    aligned_boxes: list[tuple[int, int, int, int]] = []
    aligned_labels: list[int] = []
    previous_word_id: int | None = None
    for word_id in encoding.word_ids:
        if word_id is None:
            aligned_boxes.append(SPECIAL_TOKEN_BOX)
            aligned_labels.append(ignore_label_id)
            continue
        aligned_boxes.append(word_boxes[word_id].as_tuple())
        label = label_schema.require_label(word_labels[word_id])
        if word_id == previous_word_id and label.prefix is LabelPrefix.BEGIN:
            if label.entity_type is None:
                raise ValueError("begin label must have an entity type")
            label = label_schema.require_label(f"I-{label.entity_type.value}")
        aligned_labels.append(label.label_id)
        previous_word_id = word_id

    return AlignedPageFeatures(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        bounding_boxes=tuple(aligned_boxes),
        label_ids=tuple(aligned_labels),
        word_ids=encoding.word_ids,
        truncated=encoding.truncated,
    )
