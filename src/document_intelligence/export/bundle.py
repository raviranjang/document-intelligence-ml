"""Atomic, checksum-verified serving artifact bundle export."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from document_intelligence.evaluation.versioning import COMMIT_PATTERN, SEMANTIC_VERSION_PATTERN

MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


class ModelTask(StrEnum):
    """Independently releasable model tasks."""

    OCR = "ocr"
    DOCUMENT_CLASSIFICATION = "document_classification"
    SEMANTIC_EXTRACTION = "semantic_extraction"


class ArtifactRole(StrEnum):
    """Supported serving bundle file roles."""

    MODEL = "model"
    TOKENIZER = "tokenizer"
    PROCESSOR = "processor"
    LABELS = "labels"
    CALIBRATION = "calibration"
    CONFIG = "config"


@dataclass(frozen=True, slots=True)
class ReleaseMetadata:
    """Lineage and serving contract shared by every bundle artifact."""

    model_name: str
    model_version: str
    model_family: str
    task: ModelTask
    source_commit: str
    config_version: str
    output_schema_name: str
    output_schema_version: str
    evaluation_report_uri: str
    training_run_id: str | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None

    def __post_init__(self) -> None:
        if MODEL_NAME_PATTERN.fullmatch(self.model_name) is None:
            raise ValueError("model_name must be a lowercase slug")
        for field_name in ("model_version", "config_version", "output_schema_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or SEMANTIC_VERSION_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a semantic version")
        for field_name in (
            "model_family",
            "output_schema_name",
            "evaluation_report_uri",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")
        if not isinstance(self.task, ModelTask):
            raise TypeError("task must be a ModelTask")
        if (
            not isinstance(self.source_commit, str)
            or COMMIT_PATTERN.fullmatch(self.source_commit) is None
        ):
            raise ValueError("source_commit must be a lowercase Git commit hash")
        if (self.dataset_name is None) != (self.dataset_version is None):
            raise ValueError("dataset_name and dataset_version must be provided together")
        if self.dataset_version is not None and (
            SEMANTIC_VERSION_PATTERN.fullmatch(self.dataset_version) is None
        ):
            raise ValueError("dataset_version must be a semantic version")
        for field_name in ("training_run_id", "dataset_name"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-blank string or None")


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    """One validated source file selected for export."""

    role: ArtifactRole
    path: Path
    media_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, ArtifactRole):
            raise TypeError("role must be an ArtifactRole")
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        if not isinstance(self.media_type, str) or "/" not in self.media_type:
            raise ValueError("media_type must be a non-blank type/subtype")


@dataclass(frozen=True, slots=True)
class BundleArtifact:
    """Immutable checksum entry in a serving manifest."""

    role: ArtifactRole
    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ServingBundleManifest:
    """Versioned serving bundle identity, lineage, and file inventory."""

    metadata: ReleaseMetadata
    artifacts: tuple[BundleArtifact, ...]
    created_at: str
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical public manifest contract."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "model": {
                "name": self.metadata.model_name,
                "version": self.metadata.model_version,
                "family": self.metadata.model_family,
                "task": self.metadata.task.value,
            },
            "lineage": {
                "source_commit": self.metadata.source_commit,
                "config_version": self.metadata.config_version,
                "training_run_id": self.metadata.training_run_id,
                "dataset": (
                    {
                        "name": self.metadata.dataset_name,
                        "version": self.metadata.dataset_version,
                    }
                    if self.metadata.dataset_name is not None
                    else None
                ),
                "evaluation_report_uri": self.metadata.evaluation_report_uri,
            },
            "output_contract": {
                "name": self.metadata.output_schema_name,
                "version": self.metadata.output_schema_version,
            },
            "artifacts": [
                {
                    "role": item.role.value,
                    "path": item.relative_path,
                    "media_type": item.media_type,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in self.artifacts
            ],
        }


def export_serving_bundle(
    *,
    destination: Path,
    metadata: ReleaseMetadata,
    sources: tuple[ArtifactSource, ...],
) -> ServingBundleManifest:
    """Copy existing artifacts into a new atomic, non-overwriting bundle."""
    if destination.exists():
        raise FileExistsError(f"serving bundle destination already exists: {destination}")
    if not sources:
        raise ValueError("serving bundle requires at least one artifact source")
    if len({source.role for source in sources}) != len(sources):
        raise ValueError("artifact roles must be unique")
    if len({source.path.name for source in sources}) != len(sources):
        raise ValueError("artifact filenames must be unique")
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(source.path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        files_directory = staging / "files"
        files_directory.mkdir()
        artifacts: list[BundleArtifact] = []
        for source in sources:
            target = files_directory / source.path.name
            shutil.copyfile(source.path, target)
            contents = target.read_bytes()
            artifacts.append(
                BundleArtifact(
                    role=source.role,
                    relative_path=target.relative_to(staging).as_posix(),
                    media_type=source.media_type,
                    size_bytes=len(contents),
                    sha256=hashlib.sha256(contents).hexdigest(),
                )
            )
        manifest = ServingBundleManifest(
            metadata=metadata,
            artifacts=tuple(sorted(artifacts, key=lambda item: item.role.value)),
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    validate_serving_bundle(destination)
    return manifest


def validate_serving_bundle(bundle: Path) -> dict[str, Any]:
    """Validate manifest structure and every artifact checksum before loading."""
    manifest_path = bundle / "manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        raw_manifest: Any = json.load(manifest_file)
    if not isinstance(raw_manifest, dict) or raw_manifest.get("schema_version") != "1.0.0":
        raise ValueError("unsupported or invalid serving bundle manifest")
    raw_artifacts = raw_manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("serving bundle manifest requires artifacts")
    resolved_bundle = bundle.resolve()
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            raise TypeError(f"artifacts[{index}] must be an object")
        relative_path = raw_artifact.get("path")
        if not isinstance(relative_path, str) or "\\" in relative_path:
            raise ValueError(f"artifacts[{index}].path must be a relative POSIX path")
        artifact_path = (resolved_bundle / relative_path).resolve()
        if not artifact_path.is_relative_to(resolved_bundle):
            raise ValueError(f"artifacts[{index}].path escapes the serving bundle")
        contents = artifact_path.read_bytes()
        if raw_artifact.get("size_bytes") != len(contents):
            raise ValueError(f"artifacts[{index}] size mismatch")
        if raw_artifact.get("sha256") != hashlib.sha256(contents).hexdigest():
            raise ValueError(f"artifacts[{index}] checksum mismatch")
    return cast(dict[str, Any], raw_manifest)
