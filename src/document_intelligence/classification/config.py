"""Strict configuration for the deterministic document-classifier baseline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from document_intelligence.classification.normalization import normalize_classifier_text


@dataclass(frozen=True, slots=True)
class KeywordClassifierConfig:
    """Versioned keyword baseline configuration."""

    schema_version: str
    model_name: str
    model_version: str
    model_source: str
    decision_threshold: float
    invoice_signals: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported classification configuration schema_version")
        for field_name in ("model_name", "model_version", "model_source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.model_source != "deterministic_rules":
            raise ValueError("keyword baseline model_source must be deterministic_rules")
        if isinstance(self.decision_threshold, bool) or not isinstance(
            self.decision_threshold, (int, float)
        ):
            raise TypeError("decision_threshold must be a real number")
        threshold = float(self.decision_threshold)
        if not isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("decision_threshold must be finite and in the interval (0, 1]")
        if not isinstance(self.invoice_signals, tuple):
            raise TypeError("invoice_signals must be a tuple")
        if not self.invoice_signals:
            raise ValueError("invoice_signals must not be empty")
        if any(
            not isinstance(signal, str) or not signal.strip() for signal in self.invoice_signals
        ):
            raise ValueError("invoice_signals must contain non-blank strings")
        canonical_signals = tuple(
            normalize_classifier_text(signal) for signal in self.invoice_signals
        )
        if any(not signal for signal in canonical_signals):
            raise ValueError("invoice_signals must contain word characters")
        if len(set(canonical_signals)) != len(canonical_signals):
            raise ValueError("invoice_signals must be unique after normalization")
        object.__setattr__(self, "decision_threshold", threshold)
        object.__setattr__(self, "invoice_signals", canonical_signals)


def load_keyword_classifier_config(path: Path) -> KeywordClassifierConfig:
    """Load a keyword classifier configuration from a strict TOML document."""
    with path.open("rb") as config_file:
        document: dict[str, Any] = tomllib.load(config_file)

    expected_root_fields = {"schema_version", "classifier"}
    _require_exact_fields(document, expected_root_fields, "classification configuration")
    classifier = document["classifier"]
    if not isinstance(classifier, dict):
        raise TypeError("classifier must be a TOML table")
    expected_classifier_fields = {
        "model_name",
        "model_version",
        "model_source",
        "decision_threshold",
        "invoice_signals",
    }
    _require_exact_fields(classifier, expected_classifier_fields, "classifier")

    raw_signals = classifier["invoice_signals"]
    if not isinstance(raw_signals, list) or not all(
        isinstance(signal, str) for signal in raw_signals
    ):
        raise TypeError("invoice_signals must be an array of strings")
    return KeywordClassifierConfig(
        schema_version=_require_string(document["schema_version"], "schema_version"),
        model_name=_require_string(classifier["model_name"], "model_name"),
        model_version=_require_string(classifier["model_version"], "model_version"),
        model_source=_require_string(classifier["model_source"], "model_source"),
        decision_threshold=_require_number(classifier["decision_threshold"], "decision_threshold"),
        invoice_signals=tuple(raw_signals),
    )


def _require_exact_fields(
    document: dict[str, Any], expected_fields: set[str], context: str
) -> None:
    unexpected_fields = set(document) - expected_fields
    if unexpected_fields:
        raise ValueError(f"unsupported {context} fields: {sorted(unexpected_fields)}")
    missing_fields = expected_fields - set(document)
    if missing_fields:
        raise ValueError(f"missing {context} fields: {sorted(missing_fields)}")


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    return float(value)
