"""Explicit transitions from production feedback to regression metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from document_intelligence.evaluation.versioning import COMMIT_PATTERN, SEMANTIC_VERSION_PATTERN

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
DATASET_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class FeedbackSignal(StrEnum):
    """Signals that may justify human feedback review."""

    LOW_CONFIDENCE = "low_confidence"
    HUMAN_CORRECTION = "human_correction"
    MODEL_DOMAIN_DISAGREEMENT = "model_domain_disagreement"
    FALSE_ACCEPT = "false_accept"
    FALSE_REVIEW = "false_review"
    NEW_LAYOUT = "new_layout"
    RARE_LABEL = "rare_label"
    OCR_CONFUSION = "ocr_confusion"


class FailureComponent(StrEnum):
    """Mutually explicit owners for a verified failure."""

    SOURCE_QUALITY = "source_quality"
    OCR_DETECTION = "ocr_detection"
    OCR_RECOGNITION = "ocr_recognition"
    CLASSIFICATION = "classification"
    SEMANTIC_EXTRACTION = "semantic_extraction"
    CALIBRATION = "calibration"
    DOWNSTREAM_BUSINESS_LOGIC = "downstream_business_logic"
    INFRASTRUCTURE = "infrastructure"


class VerificationStatus(StrEnum):
    """Outcome of independent feedback verification."""

    VERIFIED = "verified"
    REJECTED = "rejected"


class RegressionPriority(StrEnum):
    """Review priority without implying model-release approval."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


ACTIONABLE_ML_COMPONENTS = frozenset(
    {
        FailureComponent.OCR_DETECTION,
        FailureComponent.OCR_RECOGNITION,
        FailureComponent.CLASSIFICATION,
        FailureComponent.SEMANTIC_EXTRACTION,
        FailureComponent.CALIBRATION,
    }
)


class FeedbackWorkflowError(ValueError):
    """Raised when feedback cannot make the requested state transition."""


class FeedbackEligibilityError(FeedbackWorkflowError):
    """Raised when verified feedback does not belong in an ML dataset."""


@dataclass(frozen=True, slots=True)
class ModelReference:
    """Identity of the model whose evidence triggered feedback."""

    name: str
    version: str
    source: str
    serving_bundle_uri: str

    def __post_init__(self) -> None:
        _require_non_blank(self.name, "name")
        if (
            not isinstance(self.version, str)
            or SEMANTIC_VERSION_PATTERN.fullmatch(self.version) is None
        ):
            raise ValueError("version must be a semantic version")
        _require_non_blank(self.source, "source")
        _require_reference(self.serving_bundle_uri, "serving_bundle_uri")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "serving_bundle_uri": self.serving_bundle_uri,
        }


@dataclass(frozen=True, slots=True)
class ProductionFeedback:
    """Privacy-minimized intake metadata pointing to protected evidence."""

    feedback_id: str
    received_at: datetime
    signal: FeedbackSignal
    model: ModelReference
    document_artifact_uri: str
    document_sha256: str
    prediction_reference_uri: str

    def __post_init__(self) -> None:
        _require_identifier(self.feedback_id, "feedback_id")
        _require_aware_datetime(self.received_at, "received_at")
        if not isinstance(self.signal, FeedbackSignal):
            raise TypeError("signal must be a FeedbackSignal")
        if not isinstance(self.model, ModelReference):
            raise TypeError("model must be a ModelReference")
        _require_reference(self.document_artifact_uri, "document_artifact_uri")
        if (
            not isinstance(self.document_sha256, str)
            or SHA256_PATTERN.fullmatch(self.document_sha256) is None
        ):
            raise ValueError("document_sha256 must be a lowercase SHA-256 digest")
        _require_reference(self.prediction_reference_uri, "prediction_reference_uri")


@dataclass(frozen=True, slots=True)
class FeedbackVerification:
    """Independent verification and component attribution for one intake record."""

    feedback_id: str
    status: VerificationStatus
    reviewed_at: datetime
    reviewer_reference: str
    evidence_uri: str
    failure_component: FailureComponent | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.feedback_id, "feedback_id")
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("status must be a VerificationStatus")
        _require_aware_datetime(self.reviewed_at, "reviewed_at")
        _require_non_blank(self.reviewer_reference, "reviewer_reference")
        _require_reference(self.evidence_uri, "evidence_uri")
        if self.status is VerificationStatus.VERIFIED and not isinstance(
            self.failure_component, FailureComponent
        ):
            raise ValueError("verified feedback requires failure_component")
        if self.status is VerificationStatus.REJECTED and self.failure_component is not None:
            raise ValueError("rejected feedback must not assign failure_component")


@dataclass(frozen=True, slots=True)
class DatasetCandidate:
    """Verified metadata awaiting inclusion in an immutable dataset manifest."""

    candidate_id: str
    feedback_id: str
    signal: FeedbackSignal
    target_component: FailureComponent
    source_model: ModelReference
    document_artifact_uri: str
    document_sha256: str
    verification_evidence_uri: str
    cohorts: tuple[str, ...]
    priority: RegressionPriority
    created_at: datetime
    workflow_state: str = "dataset_candidate"

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        _require_identifier(self.feedback_id, "feedback_id")
        if not isinstance(self.signal, FeedbackSignal):
            raise TypeError("signal must be a FeedbackSignal")
        if self.target_component not in ACTIONABLE_ML_COMPONENTS:
            raise FeedbackEligibilityError("target_component is not actionable by an ML component")
        if not isinstance(self.source_model, ModelReference):
            raise TypeError("source_model must be a ModelReference")
        _require_reference(self.document_artifact_uri, "document_artifact_uri")
        if (
            not isinstance(self.document_sha256, str)
            or SHA256_PATTERN.fullmatch(self.document_sha256) is None
        ):
            raise ValueError("document_sha256 must be a lowercase SHA-256 digest")
        _require_reference(self.verification_evidence_uri, "verification_evidence_uri")
        _require_cohorts(self.cohorts)
        if not isinstance(self.priority, RegressionPriority):
            raise TypeError("priority must be a RegressionPriority")
        _require_aware_datetime(self.created_at, "created_at")
        if self.workflow_state != "dataset_candidate":
            raise ValueError("unsupported workflow_state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "feedback_id": self.feedback_id,
            "signal": self.signal.value,
            "target_component": self.target_component.value,
            "source_model": self.source_model.to_dict(),
            "document_artifact_uri": self.document_artifact_uri,
            "document_sha256": self.document_sha256,
            "verification_evidence_uri": self.verification_evidence_uri,
            "cohorts": list(self.cohorts),
            "priority": self.priority.value,
            "created_at": _format_datetime(self.created_at),
            "workflow_state": self.workflow_state,
        }


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """A fixed, verified failure retained as a future release guard."""

    case_id: str
    candidate: DatasetCandidate
    fixed_model: ModelReference
    expected_output_uri: str
    fix_evaluation_uri: str
    registered_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.case_id, "case_id")
        if not isinstance(self.candidate, DatasetCandidate):
            raise TypeError("candidate must be a DatasetCandidate")
        if not isinstance(self.fixed_model, ModelReference):
            raise TypeError("fixed_model must be a ModelReference")
        if self.fixed_model.name != self.candidate.source_model.name:
            raise FeedbackWorkflowError(
                "fixed_model must identify the same independently versioned model"
            )
        if self.fixed_model.version == self.candidate.source_model.version:
            raise FeedbackWorkflowError(
                "fixed_model version must differ from the failing model version"
            )
        _require_reference(self.expected_output_uri, "expected_output_uri")
        _require_reference(self.fix_evaluation_uri, "fix_evaluation_uri")
        _require_aware_datetime(self.registered_at, "registered_at")
        if self.registered_at < self.candidate.created_at:
            raise FeedbackWorkflowError("registered_at must not precede candidate creation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "feedback_id": self.candidate.feedback_id,
            "signal": self.candidate.signal.value,
            "target_component": self.candidate.target_component.value,
            "priority": self.candidate.priority.value,
            "cohorts": list(self.candidate.cohorts),
            "document_artifact_uri": self.candidate.document_artifact_uri,
            "document_sha256": self.candidate.document_sha256,
            "verification_evidence_uri": self.candidate.verification_evidence_uri,
            "expected_output_uri": self.expected_output_uri,
            "fix_evaluation_uri": self.fix_evaluation_uri,
            "failing_model": self.candidate.source_model.to_dict(),
            "fixed_model": self.fixed_model.to_dict(),
            "registered_at": _format_datetime(self.registered_at),
        }


@dataclass(frozen=True, slots=True)
class RegressionManifest:
    """Immutable, lineage-bearing registry of fixed known failures."""

    name: str
    version: str
    source_commit: str
    created_at: datetime
    cases: tuple[RegressionCase, ...]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or DATASET_NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError("name must be a lowercase slug")
        if (
            not isinstance(self.version, str)
            or SEMANTIC_VERSION_PATTERN.fullmatch(self.version) is None
        ):
            raise ValueError("version must be a semantic version")
        if (
            not isinstance(self.source_commit, str)
            or COMMIT_PATTERN.fullmatch(self.source_commit) is None
        ):
            raise ValueError("source_commit must be a lowercase Git commit hash")
        _require_aware_datetime(self.created_at, "created_at")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("cases must be a non-empty tuple")
        if not all(isinstance(case, RegressionCase) for case in self.cases):
            raise TypeError("cases must contain RegressionCase values")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id values must be unique")
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported regression manifest schema_version")
        if any(case.registered_at > self.created_at for case in self.cases):
            raise ValueError("created_at must not precede regression case registration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "source_commit": self.source_commit,
            "created_at": _format_datetime(self.created_at),
            "cases": [
                case.to_dict() for case in sorted(self.cases, key=lambda item: item.case_id)
            ],
        }


def create_dataset_candidate(
    feedback: ProductionFeedback,
    verification: FeedbackVerification,
    *,
    candidate_id: str,
    cohorts: tuple[str, ...],
    priority: RegressionPriority,
    created_at: datetime,
) -> DatasetCandidate:
    """Promote only independently verified, ML-actionable feedback metadata."""
    if feedback.feedback_id != verification.feedback_id:
        raise FeedbackWorkflowError("feedback and verification identifiers must match")
    if verification.status is not VerificationStatus.VERIFIED:
        raise FeedbackWorkflowError("only verified feedback can become a dataset candidate")
    if verification.reviewed_at < feedback.received_at:
        raise FeedbackWorkflowError("verification must not precede feedback receipt")
    _require_aware_datetime(created_at, "created_at")
    if created_at < verification.reviewed_at:
        raise FeedbackWorkflowError("candidate creation must not precede verification")
    component = verification.failure_component
    if component not in ACTIONABLE_ML_COMPONENTS:
        component_name = component.value if component is not None else "unassigned"
        raise FeedbackEligibilityError(
            f"failure component {component_name!r} is not actionable by an ML component"
        )
    return DatasetCandidate(
        candidate_id=candidate_id,
        feedback_id=feedback.feedback_id,
        signal=feedback.signal,
        target_component=component,
        source_model=feedback.model,
        document_artifact_uri=feedback.document_artifact_uri,
        document_sha256=feedback.document_sha256,
        verification_evidence_uri=verification.evidence_uri,
        cohorts=cohorts,
        priority=priority,
        created_at=created_at,
    )


def register_regression_case(
    candidate: DatasetCandidate,
    *,
    case_id: str,
    fixed_model: ModelReference,
    expected_output_uri: str,
    fix_evaluation_uri: str,
    registered_at: datetime,
) -> RegressionCase:
    """Register a guard only after a different model version has a documented fix."""
    return RegressionCase(
        case_id=case_id,
        candidate=candidate,
        fixed_model=fixed_model,
        expected_output_uri=expected_output_uri,
        fix_evaluation_uri=fix_evaluation_uri,
        registered_at=registered_at,
    )


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a stable identifier")


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _require_reference(value: str, field_name: str) -> None:
    _require_non_blank(value, field_name)
    if any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must not contain whitespace")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


def _require_cohorts(cohorts: tuple[str, ...]) -> None:
    if (
        not isinstance(cohorts, tuple)
        or not cohorts
        or any(
            not isinstance(cohort, str) or DATASET_NAME_PATTERN.fullmatch(cohort) is None
            for cohort in cohorts
        )
    ):
        raise ValueError("cohorts must be a tuple of lowercase slugs")
    if len(set(cohorts)) != len(cohorts):
        raise ValueError("cohorts must not contain duplicates")


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
