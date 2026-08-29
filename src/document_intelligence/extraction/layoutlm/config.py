"""Versioned configuration for LayoutLMv3 feature preparation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class LayoutLMDatasetConfig:
    """Training/inference-shared dataset transform configuration."""

    schema_version: str
    max_length: int
    bounding_box_scale: int
    ignore_label_id: int
    subword_label_policy: str
    truncation_side: str

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported LayoutLM dataset configuration schema_version")
        for field_name in ("max_length", "bounding_box_scale", "ignore_label_id"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if self.max_length < 3:
            raise ValueError("max_length must leave room for content and special tokens")
        if self.bounding_box_scale != 1000:
            raise ValueError("bounding_box_scale must be 1000 for LayoutLMv3")
        if self.ignore_label_id != -100:
            raise ValueError("ignore_label_id must be -100")
        if self.subword_label_policy != "propagate_bio":
            raise ValueError("subword_label_policy must be propagate_bio")
        if self.truncation_side != "right":
            raise ValueError("truncation_side must be right")


def load_layoutlm_dataset_config(path: Path) -> LayoutLMDatasetConfig:
    """Load the strict LayoutLMv3 dataset transform configuration."""
    with path.open("rb") as config_file:
        document: dict[str, Any] = tomllib.load(config_file)
    expected_fields = {
        "schema_version",
        "max_length",
        "bounding_box_scale",
        "ignore_label_id",
        "subword_label_policy",
        "truncation_side",
    }
    unexpected_fields = set(document) - expected_fields
    if unexpected_fields:
        raise ValueError(f"unsupported LayoutLM dataset fields: {sorted(unexpected_fields)}")
    missing_fields = expected_fields - set(document)
    if missing_fields:
        raise ValueError(f"missing LayoutLM dataset fields: {sorted(missing_fields)}")
    return LayoutLMDatasetConfig(
        schema_version=_require_string(document["schema_version"], "schema_version"),
        max_length=_require_integer(document["max_length"], "max_length"),
        bounding_box_scale=_require_integer(document["bounding_box_scale"], "bounding_box_scale"),
        ignore_label_id=_require_integer(document["ignore_label_id"], "ignore_label_id"),
        subword_label_policy=_require_string(
            document["subword_label_policy"], "subword_label_policy"
        ),
        truncation_side=_require_string(document["truncation_side"], "truncation_side"),
    )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return cast(int, value)
