"""Strict configuration for deterministic semantic extraction rules."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_intelligence.extraction.types import EntityType


@dataclass(frozen=True, slots=True)
class ExtractionRule:
    """One versioned regex rule with a required named value group."""

    rule_id: str
    entity_type: EntityType
    pattern: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must be a non-empty string")
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("entity_type must be an EntityType")
        if not isinstance(self.pattern, str) or not self.pattern.strip():
            raise ValueError("pattern must be a non-empty string")
        try:
            compiled_pattern = re.compile(self.pattern, flags=re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"invalid pattern for rule {self.rule_id!r}: {error}") from error
        if "value" not in compiled_pattern.groupindex:
            raise ValueError(f"rule {self.rule_id!r} must define a named 'value' group")


@dataclass(frozen=True, slots=True)
class DeterministicExtractionConfig:
    """Versioned configuration for the extraction baseline."""

    schema_version: str
    model_name: str
    model_version: str
    model_source: str
    rules: tuple[ExtractionRule, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported extraction configuration schema_version")
        for field_name in ("model_name", "model_version", "model_source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.model_source != "deterministic_rules":
            raise ValueError("extraction baseline model_source must be deterministic_rules")
        if not isinstance(self.rules, tuple):
            raise TypeError("rules must be a tuple")
        if not self.rules:
            raise ValueError("rules must not be empty")
        if not all(isinstance(rule, ExtractionRule) for rule in self.rules):
            raise TypeError("rules must contain ExtractionRule values")
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("rule_id values must be unique")


def load_deterministic_extraction_config(path: Path) -> DeterministicExtractionConfig:
    """Load a deterministic extraction configuration from strict TOML."""
    with path.open("rb") as config_file:
        document: dict[str, Any] = tomllib.load(config_file)

    expected_root_fields = {"schema_version", "extractor", "rules"}
    _require_exact_fields(document, expected_root_fields, "extraction configuration")
    extractor = document["extractor"]
    if not isinstance(extractor, dict):
        raise TypeError("extractor must be a TOML table")
    _require_exact_fields(extractor, {"model_name", "model_version", "model_source"}, "extractor")
    raw_rules = document["rules"]
    if not isinstance(raw_rules, list):
        raise TypeError("rules must be an array of TOML tables")
    rules = tuple(_load_rule(raw_rule, index=index) for index, raw_rule in enumerate(raw_rules))
    return DeterministicExtractionConfig(
        schema_version=_require_string(document["schema_version"], "schema_version"),
        model_name=_require_string(extractor["model_name"], "model_name"),
        model_version=_require_string(extractor["model_version"], "model_version"),
        model_source=_require_string(extractor["model_source"], "model_source"),
        rules=rules,
    )


def _load_rule(raw_rule: Any, *, index: int) -> ExtractionRule:
    if not isinstance(raw_rule, dict):
        raise TypeError(f"rules[{index}] must be a TOML table")
    _require_exact_fields(raw_rule, {"rule_id", "entity_type", "pattern"}, f"rules[{index}]")
    raw_entity_type = _require_string(raw_rule["entity_type"], "entity_type")
    try:
        entity_type = EntityType(raw_entity_type)
    except ValueError as error:
        raise ValueError(f"unsupported entity_type: {raw_entity_type!r}") from error
    return ExtractionRule(
        rule_id=_require_string(raw_rule["rule_id"], "rule_id"),
        entity_type=entity_type,
        pattern=_require_string(raw_rule["pattern"], "pattern"),
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
