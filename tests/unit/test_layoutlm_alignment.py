"""Tests for explicit source-word to subword alignment."""

from pathlib import Path

import pytest

from document_intelligence.extraction import load_semantic_label_schema
from document_intelligence.extraction.layoutlm import (
    IGNORE_LABEL_ID,
    NormalizedBoundingBox,
    TokenizerEncoding,
    align_page_encoding,
)

LABEL_SCHEMA_PATH = Path("configs/extraction/semantic_labels_v1.json")


def _box(index: int) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(index, index, index + 10, index + 10)


def test_normal_words_and_special_tokens_align_to_source_contract() -> None:
    schema = load_semantic_label_schema(LABEL_SCHEMA_PATH)

    features = align_page_encoding(
        encoding=TokenizerEncoding(
            input_ids=(101, 10, 11, 102),
            attention_mask=(1, 1, 1, 1),
            word_ids=(None, 0, 1, None),
            truncated=False,
        ),
        word_boxes=(_box(0), _box(20)),
        word_labels=("O", "B-INVOICE_NUMBER"),
        label_schema=schema,
    )

    assert features.bounding_boxes == (
        (0, 0, 0, 0),
        _box(0).as_tuple(),
        _box(20).as_tuple(),
        (0, 0, 0, 0),
    )
    assert features.label_ids == (
        IGNORE_LABEL_ID,
        schema.label_to_id["O"],
        schema.label_to_id["B-INVOICE_NUMBER"],
        IGNORE_LABEL_ID,
    )


def test_split_word_converts_begin_label_to_inside_for_continuation() -> None:
    schema = load_semantic_label_schema(LABEL_SCHEMA_PATH)

    features = align_page_encoding(
        encoding=TokenizerEncoding(
            input_ids=(101, 20, 21, 102),
            attention_mask=(1, 1, 1, 1),
            word_ids=(None, 0, 0, None),
            truncated=False,
        ),
        word_boxes=(_box(0),),
        word_labels=("B-SELLER_NAME",),
        label_schema=schema,
    )

    assert features.label_ids == (
        IGNORE_LABEL_ID,
        schema.label_to_id["B-SELLER_NAME"],
        schema.label_to_id["I-SELLER_NAME"],
        IGNORE_LABEL_ID,
    )


def test_padding_uses_special_box_and_ignored_label() -> None:
    schema = load_semantic_label_schema(LABEL_SCHEMA_PATH)

    features = align_page_encoding(
        encoding=TokenizerEncoding(
            input_ids=(101, 20, 102, 0),
            attention_mask=(1, 1, 1, 0),
            word_ids=(None, 0, None, None),
            truncated=False,
        ),
        word_boxes=(_box(0),),
        word_labels=("O",),
        label_schema=schema,
    )

    assert features.bounding_boxes[-1] == (0, 0, 0, 0)
    assert features.label_ids[-1] == IGNORE_LABEL_ID


def test_right_truncation_may_remove_only_trailing_words() -> None:
    schema = load_semantic_label_schema(LABEL_SCHEMA_PATH)

    features = align_page_encoding(
        encoding=TokenizerEncoding(
            input_ids=(101, 10, 11, 102),
            attention_mask=(1, 1, 1, 1),
            word_ids=(None, 0, 1, None),
            truncated=True,
        ),
        word_boxes=(_box(0), _box(20), _box(40)),
        word_labels=("O", "O", "B-TOTAL_AMOUNT"),
        label_schema=schema,
    )

    assert features.truncated is True
    assert 2 not in features.word_ids


@pytest.mark.parametrize(
    "encoding",
    [
        TokenizerEncoding((101, 10, 102), (1, 1, 1), (None, 0, None), False),
        TokenizerEncoding((101, 10, 12, 102), (1, 1, 1, 1), (None, 0, 2, None), True),
        TokenizerEncoding((101, 10, 11, 102), (1, 1, 1, 1), (None, 0, 1, None), True),
    ],
)
def test_alignment_rejects_missing_middle_or_inconsistent_truncation(
    encoding: TokenizerEncoding,
) -> None:
    schema = load_semantic_label_schema(LABEL_SCHEMA_PATH)

    with pytest.raises(
        ValueError, match=r"every source word|skip source words|omit|outside the page"
    ):
        align_page_encoding(
            encoding=encoding,
            word_boxes=(_box(0), _box(20)),
            word_labels=("O", "O"),
            label_schema=schema,
        )


def test_alignment_rejects_empty_source_words() -> None:
    schema = load_semantic_label_schema(LABEL_SCHEMA_PATH)

    with pytest.raises(ValueError, match="at least one source word"):
        align_page_encoding(
            encoding=TokenizerEncoding((101, 102), (1, 1), (None, None), False),
            word_boxes=(),
            word_labels=(),
            label_schema=schema,
        )
