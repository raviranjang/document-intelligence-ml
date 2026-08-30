"""Tests for atomic, checksum-verified serving bundle export."""

import json
from pathlib import Path

import pytest

from document_intelligence.export import (
    ArtifactRole,
    ArtifactSource,
    ModelTask,
    ReleaseMetadata,
    export_serving_bundle,
    validate_serving_bundle,
)


def _metadata() -> ReleaseMetadata:
    return ReleaseMetadata(
        model_name="keyword-invoice-baseline",
        model_version="1.0.0",
        model_family="deterministic-keyword-classifier",
        task=ModelTask.DOCUMENT_CLASSIFICATION,
        source_commit="a" * 40,
        config_version="1.0.0",
        output_schema_name="DocumentClassification",
        output_schema_version="1.0.0",
        evaluation_report_uri="reports/classification-1.0.0.json",
    )


def test_export_copies_sources_and_records_stable_checksums(tmp_path: Path) -> None:
    model = tmp_path / "rules.json"
    model.write_text('{"threshold": 0.3}\n', encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text('schema_version = "1.0.0"\n', encoding="utf-8")
    destination = tmp_path / "release"

    manifest = export_serving_bundle(
        destination=destination,
        metadata=_metadata(),
        sources=(
            ArtifactSource(ArtifactRole.MODEL, model, "application/json"),
            ArtifactSource(ArtifactRole.CONFIG, config, "application/toml"),
        ),
    )

    assert destination.is_dir()
    assert (destination / "manifest.json").is_file()
    assert [item.role for item in manifest.artifacts] == [ArtifactRole.CONFIG, ArtifactRole.MODEL]
    assert all(len(item.sha256) == 64 for item in manifest.artifacts)
    loaded = validate_serving_bundle(destination)
    assert loaded["model"]["name"] == "keyword-invoice-baseline"
    assert loaded["lineage"]["dataset"] is None


def test_export_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "release"
    destination.mkdir()
    source = tmp_path / "model.bin"
    source.write_bytes(b"synthetic")

    with pytest.raises(FileExistsError, match="already exists"):
        export_serving_bundle(
            destination=destination,
            metadata=_metadata(),
            sources=(ArtifactSource(ArtifactRole.MODEL, source, "application/octet-stream"),),
        )


def test_validation_detects_artifact_tampering(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"original")
    destination = tmp_path / "release"
    export_serving_bundle(
        destination=destination,
        metadata=_metadata(),
        sources=(ArtifactSource(ArtifactRole.MODEL, source, "application/octet-stream"),),
    )
    (destination / "files" / "model.bin").write_bytes(b"tampered")

    with pytest.raises(ValueError, match=r"size mismatch|checksum mismatch"):
        validate_serving_bundle(destination)


def test_validation_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"original")
    destination = tmp_path / "release"
    export_serving_bundle(
        destination=destination,
        metadata=_metadata(),
        sources=(ArtifactSource(ArtifactRole.MODEL, source, "application/octet-stream"),),
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../model.bin"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        validate_serving_bundle(destination)


def test_export_rejects_duplicate_roles_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    destination = tmp_path / "release"

    with pytest.raises(ValueError, match="roles must be unique"):
        export_serving_bundle(
            destination=destination,
            metadata=_metadata(),
            sources=(
                ArtifactSource(ArtifactRole.MODEL, first, "application/octet-stream"),
                ArtifactSource(ArtifactRole.MODEL, second, "application/octet-stream"),
            ),
        )
    assert not destination.exists()
