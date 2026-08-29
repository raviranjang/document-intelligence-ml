"""Contract tests for the versioned dataset manifest schema."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

REPOSITORY_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "datasets" / "schemas" / "dataset-manifest.schema.json"
FIXTURE_PATH = REPOSITORY_ROOT / "datasets" / "fixtures" / "synthetic-ocr-manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as json_file:
        value: dict[str, Any] = json.load(json_file)
    return value


@pytest.fixture
def schema() -> dict[str, Any]:
    return _load_json(SCHEMA_PATH)


@pytest.fixture
def manifest() -> dict[str, Any]:
    return _load_json(FIXTURE_PATH)


@pytest.fixture
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_manifest_schema_is_valid_json_schema(schema: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)


def test_synthetic_manifest_satisfies_schema(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    validator.validate(manifest)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema_version", "2.0.0"),
        ("name", "Contains Spaces"),
        ("version", "latest"),
        ("created_at", "2026-08-29"),
        ("tasks", ["unsupported_task"]),
    ],
)
def test_manifest_rejects_invalid_identity_and_lineage_fields(
    validator: Draft202012Validator,
    manifest: dict[str, Any],
    field_name: str,
    invalid_value: Any,
) -> None:
    invalid_manifest = deepcopy(manifest)
    invalid_manifest[field_name] = invalid_value

    with pytest.raises(ValidationError):
        validator.validate(invalid_manifest)


def test_manifest_rejects_undeclared_fields(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    invalid_manifest = deepcopy(manifest)
    invalid_manifest["owner_email"] = "not-for-public-manifests@example.invalid"

    with pytest.raises(ValidationError, match="Additional properties"):
        validator.validate(invalid_manifest)


def test_grouped_split_requires_grouping_keys(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    invalid_manifest = deepcopy(manifest)
    del invalid_manifest["split_strategy"]["grouping_keys"]

    with pytest.raises(ValidationError, match="grouping_keys"):
        validator.validate(invalid_manifest)


@pytest.mark.parametrize("checksum", ["ABCDEF" * 10 + "ABCD", "0" * 63, "0" * 65])
def test_artifact_requires_lowercase_sha256(
    validator: Draft202012Validator, manifest: dict[str, Any], checksum: str
) -> None:
    invalid_manifest = deepcopy(manifest)
    invalid_manifest["records"][0]["document"]["sha256"] = checksum

    with pytest.raises(ValidationError, match="does not match"):
        validator.validate(invalid_manifest)


def test_record_rejects_unknown_split(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    invalid_manifest = deepcopy(manifest)
    invalid_manifest["records"][0]["split"] = "development"

    with pytest.raises(ValidationError):
        validator.validate(invalid_manifest)
