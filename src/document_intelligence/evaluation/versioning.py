"""Shared validation patterns for lineage-bearing evaluation artifacts."""

from __future__ import annotations

import re

SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?$"
)
COMMIT_PATTERN = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
