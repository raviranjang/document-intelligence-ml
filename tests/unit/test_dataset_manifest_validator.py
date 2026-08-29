"""Unit tests for operational dataset manifest validation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from document_intelligence.data.validation import (
    DatasetManifestValidator,
    DatasetValidationError,
    LocalArtifactReader,
    UnsupportedArtifactURIError,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "datasets" / "schemas" / "dataset-manifest.schema.json"


class MemoryArtifactReader:
    """Small deterministic artifact reader for validator unit tests."""

    def __init__(self, artifacts: dict[str, bytes]) -> None:
        self._artifacts = artifacts

    def read_bytes(self, uri: str) -> bytes:
        try:
            return self._artifacts[uri]
        except KeyError as error:
            raise FileNotFoundError(uri) from error


class UnreadableArtifactReader:
    """Artifact reader that simulates a storage permission failure."""

    def read_bytes(self, uri: str) -> bytes:
        raise PermissionError(f"permission denied: {uri}")


def _artifact(uri: str, contents: bytes, media_type: str) -> dict[str, Any]:
    return {
        "uri": uri,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "media_type": media_type,
        "size_bytes": len(contents),
    }


@pytest.fixture
def document_contents() -> bytes:
    return b"synthetic image bytes"


@pytest.fixture
def annotation_contents() -> bytes:
    return b'{"tokens": [{"text": "Invoice"}]}'


@pytest.fixture
def manifest(document_contents: bytes, annotation_contents: bytes) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "name": "validator-fixture",
        "version": "1.0.0",
        "created_at": "2026-08-29T00:00:00Z",
        "tasks": ["ocr_recognition"],
        "sources": [{"name": "synthetic", "version": "1.0.0"}],
        "split_strategy": {
            "method": "grouped",
            "grouping_keys": ["template_family"],
        },
        "records": [
            {
                "id": "record-1",
                "split": "training",
                "groups": {"template_family": "template-a"},
                "document": _artifact("documents/record-1.png", document_contents, "image/png"),
                "annotation": _artifact(
                    "annotations/record-1.json", annotation_contents, "application/json"
                ),
            }
        ],
    }


@pytest.fixture
def artifacts(document_contents: bytes, annotation_contents: bytes) -> dict[str, bytes]:
    return {
        "documents/record-1.png": document_contents,
        "annotations/record-1.json": annotation_contents,
    }


@pytest.fixture
def validator() -> DatasetManifestValidator:
    return DatasetManifestValidator.from_schema_file(SCHEMA_PATH)


def _issue_codes(error: DatasetValidationError) -> set[str]:
    return {issue.code for issue in error.issues}


def test_validator_accepts_valid_manifest(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))


def test_validate_file_loads_manifest_and_local_artifacts(
    tmp_path: Path,
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    for uri, contents in artifacts.items():
        artifact_path = tmp_path / uri
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(contents)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest = validator.validate_file(manifest_path)

    assert loaded_manifest == manifest


def test_validator_reports_schema_violations_before_artifact_checks(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
) -> None:
    del manifest["version"]

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader({}))

    assert _issue_codes(raised_error.value) == {"schema_violation"}
    assert raised_error.value.issues[0].location == "$"


def test_validator_rejects_duplicate_records_and_artifacts(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    manifest["records"].append(deepcopy(manifest["records"][0]))

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert {"duplicate_record_id", "duplicate_artifact"} <= _issue_codes(raised_error.value)


def test_validator_rejects_group_values_shared_across_splits(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    second_record = deepcopy(manifest["records"][0])
    second_record["id"] = "record-2"
    second_record["split"] = "test"
    second_record["document"] = _artifact("documents/record-2.png", b"second", "image/png")
    second_record.pop("annotation")
    manifest["records"].append(second_record)
    artifacts["documents/record-2.png"] = b"second"

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert "split_leakage" in _issue_codes(raised_error.value)


def test_validator_requires_each_grouping_value(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    manifest["records"][0]["groups"] = {"seller": "seller-a"}

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert "missing_group_value" in _issue_codes(raised_error.value)


def test_validator_reports_missing_artifact(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    del artifacts["documents/record-1.png"]

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert "missing_artifact" in _issue_codes(raised_error.value)


def test_validator_reports_unreadable_artifacts(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
) -> None:
    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=UnreadableArtifactReader())

    assert _issue_codes(raised_error.value) == {"unreadable_artifact"}


def test_validator_rejects_empty_artifact(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    artifacts["documents/record-1.png"] = b""
    manifest["records"][0]["document"] = _artifact("documents/record-1.png", b"", "image/png")

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert _issue_codes(raised_error.value) == {"empty_artifact"}


@pytest.mark.parametrize(
    ("mutated_field", "expected_code"),
    [("sha256", "checksum_mismatch"), ("size_bytes", "artifact_size_mismatch")],
)
def test_validator_checks_artifact_integrity(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
    mutated_field: str,
    expected_code: str,
) -> None:
    manifest["records"][0]["document"][mutated_field] = (
        "0" * 64 if mutated_field == "sha256" else 999
    )

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert expected_code in _issue_codes(raised_error.value)


@pytest.mark.parametrize("annotation_contents", [b"{", b"not-json", b"\xff"])
def test_validator_rejects_corrupt_json_annotations(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
    annotation_contents: bytes,
) -> None:
    artifacts["annotations/record-1.json"] = annotation_contents
    manifest["records"][0]["annotation"] = _artifact(
        "annotations/record-1.json", annotation_contents, "application/json"
    )

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert "corrupt_annotation" in _issue_codes(raised_error.value)


@pytest.mark.parametrize("annotation_contents", [b"null", b'""', b"[]", b"{}"])
def test_validator_rejects_empty_json_annotations(
    validator: DatasetManifestValidator,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
    annotation_contents: bytes,
) -> None:
    artifacts["annotations/record-1.json"] = annotation_contents
    manifest["records"][0]["annotation"] = _artifact(
        "annotations/record-1.json", annotation_contents, "application/json"
    )

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate(manifest, artifact_reader=MemoryArtifactReader(artifacts))

    assert "empty_annotation" in _issue_codes(raised_error.value)


@pytest.mark.parametrize(
    "uri",
    [
        "../outside.json",
        "%2e%2e/outside.json",
        "/absolute/path.json",
        "https://example.invalid/artifact.json",
        "windows\\path.json",
    ],
)
def test_local_reader_rejects_unsafe_or_unsupported_uris(tmp_path: Path, uri: str) -> None:
    reader = LocalArtifactReader(tmp_path)

    with pytest.raises(UnsupportedArtifactURIError):
        reader.read_bytes(uri)


def test_validate_file_reports_invalid_json(
    tmp_path: Path, validator: DatasetManifestValidator
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{", encoding="utf-8")

    with pytest.raises(DatasetValidationError) as raised_error:
        validator.validate_file(manifest_path)

    assert _issue_codes(raised_error.value) == {"invalid_manifest_file"}
