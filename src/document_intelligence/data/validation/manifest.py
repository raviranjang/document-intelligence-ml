"""Operational validation for versioned dataset manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import unquote, urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single actionable dataset validation failure."""

    code: str
    message: str
    location: str
    record_id: str | None = None

    def __str__(self) -> str:
        record_context = f" record={self.record_id!r}" if self.record_id is not None else ""
        return f"[{self.code}] {self.location}{record_context}: {self.message}"


class DatasetValidationError(ValueError):
    """Raised when a manifest or one of its referenced artifacts is invalid."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("DatasetValidationError requires at least one issue")
        details = "\n".join(str(issue) for issue in self.issues)
        super().__init__(f"dataset validation failed with {len(self.issues)} issue(s):\n{details}")


class UnsupportedArtifactURIError(ValueError):
    """Raised when a local reader cannot safely resolve an artifact URI."""


class ArtifactReader(Protocol):
    """Read artifact bytes without exposing storage details to the validator."""

    def read_bytes(self, uri: str) -> bytes:
        """Return bytes for an artifact URI."""


@dataclass(frozen=True, slots=True)
class LocalArtifactReader:
    """Read relative artifact URIs from one controlled local root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    def read_bytes(self, uri: str) -> bytes:
        """Read a relative URI while preventing traversal outside the root."""
        parsed_uri = urlsplit(uri)
        if parsed_uri.scheme or parsed_uri.netloc or parsed_uri.query or parsed_uri.fragment:
            raise UnsupportedArtifactURIError("local validation requires a relative artifact URI")

        decoded_path = unquote(parsed_uri.path)
        if "\\" in decoded_path:
            raise UnsupportedArtifactURIError("artifact URIs must use forward slashes")
        relative_path = Path(decoded_path)
        if relative_path.is_absolute():
            raise UnsupportedArtifactURIError("absolute artifact paths are not allowed")

        artifact_path = (self.root / relative_path).resolve()
        if not artifact_path.is_relative_to(self.root):
            raise UnsupportedArtifactURIError("artifact URI escapes the configured dataset root")
        return artifact_path.read_bytes()


class DatasetManifestValidator:
    """Validate manifest structure, dataset invariants, and artifact integrity."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            message = f"invalid dataset manifest schema: {error.message}"
            raise ValueError(message) from error
        self._schema_validator = Draft202012Validator(schema, format_checker=FormatChecker())

    @classmethod
    def from_schema_file(cls, schema_path: Path) -> DatasetManifestValidator:
        """Create a validator from a UTF-8 JSON Schema file."""
        schema = _load_json_object(schema_path, description="schema")
        return cls(schema)

    def validate_file(
        self,
        manifest_path: Path,
        *,
        artifact_root: Path | None = None,
    ) -> JsonObject:
        """Load and validate a manifest and its locally referenced artifacts."""
        try:
            manifest = _load_json_object(manifest_path, description="manifest")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            issue = ValidationIssue(
                code="invalid_manifest_file",
                message=str(error),
                location=str(manifest_path),
            )
            raise DatasetValidationError((issue,)) from error

        reader = LocalArtifactReader(artifact_root or manifest_path.parent)
        self.validate(manifest, artifact_reader=reader)
        return manifest

    def validate(self, manifest: Mapping[str, Any], *, artifact_reader: ArtifactReader) -> None:
        """Validate an already loaded manifest and every referenced artifact."""
        schema_issues = self._validate_schema(manifest)
        if schema_issues:
            raise DatasetValidationError(schema_issues)

        records = cast(list[JsonObject], manifest["records"])
        issues = [
            *self._validate_unique_records(records),
            *self._validate_grouped_splits(manifest, records),
            *self._validate_artifacts(records, artifact_reader),
        ]
        if issues:
            raise DatasetValidationError(issues)

    def _validate_schema(self, manifest: Mapping[str, Any]) -> list[ValidationIssue]:
        errors = sorted(
            self._schema_validator.iter_errors(manifest),
            key=lambda error: tuple(str(component) for component in error.absolute_path),
        )
        return [
            ValidationIssue(
                code="schema_violation",
                message=error.message,
                location=_format_json_path(error.absolute_path),
            )
            for error in errors
        ]

    @staticmethod
    def _validate_unique_records(records: list[JsonObject]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        seen_record_ids: set[str] = set()
        seen_artifacts: dict[tuple[str, str], str] = {}

        for index, record in enumerate(records):
            record_id = cast(str, record["id"])
            if record_id in seen_record_ids:
                issues.append(
                    ValidationIssue(
                        code="duplicate_record_id",
                        message="record IDs must be unique",
                        location=f"$.records[{index}].id",
                        record_id=record_id,
                    )
                )
            seen_record_ids.add(record_id)

            for role in ("document", "annotation"):
                artifact = cast(JsonObject | None, record.get(role))
                if artifact is None:
                    continue
                uri = cast(str, artifact["uri"])
                artifact_key = (role, uri)
                first_record_id = seen_artifacts.get(artifact_key)
                if first_record_id is not None:
                    issues.append(
                        ValidationIssue(
                            code="duplicate_artifact",
                            message=f"{role} URI is already used by record {first_record_id!r}",
                            location=f"$.records[{index}].{role}.uri",
                            record_id=record_id,
                        )
                    )
                else:
                    seen_artifacts[artifact_key] = record_id
        return issues

    @staticmethod
    def _validate_grouped_splits(
        manifest: Mapping[str, Any], records: list[JsonObject]
    ) -> list[ValidationIssue]:
        split_strategy = cast(JsonObject, manifest["split_strategy"])
        if split_strategy["method"] != "grouped":
            return []

        grouping_keys = cast(list[str], split_strategy["grouping_keys"])
        group_splits: dict[tuple[str, str], tuple[str, str]] = {}
        issues: list[ValidationIssue] = []
        for index, record in enumerate(records):
            record_id = cast(str, record["id"])
            split = cast(str, record["split"])
            groups = cast(dict[str, str], record.get("groups", {}))
            for grouping_key in grouping_keys:
                group_value = groups.get(grouping_key)
                if group_value is None:
                    issues.append(
                        ValidationIssue(
                            code="missing_group_value",
                            message=f"grouped split requires a {grouping_key!r} value",
                            location=f"$.records[{index}].groups",
                            record_id=record_id,
                        )
                    )
                    continue

                group_identity = (grouping_key, group_value)
                existing = group_splits.get(group_identity)
                if existing is not None and existing[0] != split:
                    first_split, first_record_id = existing
                    issues.append(
                        ValidationIssue(
                            code="split_leakage",
                            message=(
                                f"group {grouping_key}={group_value!r} occurs in {first_split!r} "
                                f"for record {first_record_id!r} and in {split!r}"
                            ),
                            location=f"$.records[{index}].groups.{grouping_key}",
                            record_id=record_id,
                        )
                    )
                else:
                    group_splits[group_identity] = (split, record_id)
        return issues

    @staticmethod
    def _validate_artifacts(
        records: list[JsonObject], artifact_reader: ArtifactReader
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for index, record in enumerate(records):
            record_id = cast(str, record["id"])
            for role in ("document", "annotation"):
                artifact = cast(JsonObject | None, record.get(role))
                if artifact is None:
                    continue
                location = f"$.records[{index}].{role}"
                issues.extend(
                    _validate_artifact(
                        artifact,
                        artifact_reader=artifact_reader,
                        role=role,
                        location=location,
                        record_id=record_id,
                    )
                )
        return issues


def _validate_artifact(
    artifact: JsonObject,
    *,
    artifact_reader: ArtifactReader,
    role: str,
    location: str,
    record_id: str,
) -> list[ValidationIssue]:
    uri = cast(str, artifact["uri"])
    try:
        contents = artifact_reader.read_bytes(uri)
    except UnsupportedArtifactURIError as error:
        return [ValidationIssue("unsupported_artifact_uri", str(error), location, record_id)]
    except FileNotFoundError:
        return [
            ValidationIssue(
                "missing_artifact", f"artifact does not exist: {uri}", location, record_id
            )
        ]
    except OSError as error:
        return [ValidationIssue("unreadable_artifact", str(error), location, record_id)]

    issues: list[ValidationIssue] = []
    if not contents:
        issues.append(ValidationIssue("empty_artifact", "artifact is empty", location, record_id))

    expected_size = artifact.get("size_bytes")
    if expected_size is not None and len(contents) != expected_size:
        issues.append(
            ValidationIssue(
                "artifact_size_mismatch",
                f"expected {expected_size} bytes but read {len(contents)}",
                location,
                record_id,
            )
        )

    actual_checksum = hashlib.sha256(contents).hexdigest()
    if actual_checksum != artifact["sha256"]:
        issues.append(
            ValidationIssue(
                "checksum_mismatch",
                f"SHA-256 mismatch for {uri}",
                location,
                record_id,
            )
        )

    if role == "annotation" and artifact["media_type"] == "application/json" and contents:
        issues.extend(_validate_json_annotation(contents, location=location, record_id=record_id))
    return issues


def _validate_json_annotation(
    contents: bytes, *, location: str, record_id: str
) -> list[ValidationIssue]:
    try:
        annotation = json.loads(contents)
    except (UnicodeError, json.JSONDecodeError) as error:
        return [ValidationIssue("corrupt_annotation", str(error), location, record_id)]

    if annotation is None or annotation == "" or annotation == [] or annotation == {}:
        return [
            ValidationIssue(
                "empty_annotation", "annotation JSON has no content", location, record_id
            )
        ]
    return []


def _load_json_object(path: Path, *, description: str) -> JsonObject:
    with path.open(encoding="utf-8") as json_file:
        value: Any = json.load(json_file)
    if not isinstance(value, dict):
        raise TypeError(f"{description} root must be a JSON object")
    return cast(JsonObject, value)


def _format_json_path(path: Iterable[str | int]) -> str:
    formatted_path = "$"
    for component in path:
        formatted_path += f"[{component}]" if isinstance(component, int) else f".{component}"
    return formatted_path
