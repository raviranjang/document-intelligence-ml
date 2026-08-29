"""Shared text normalization for classification training and inference parity."""

from __future__ import annotations

import re

NON_WORD_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_classifier_text(value: str) -> str:
    """Case-fold text and collapse punctuation and whitespace boundaries."""
    return " ".join(NON_WORD_PATTERN.sub(" ", value.casefold()).split())
