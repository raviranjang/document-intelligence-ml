"""Tests for controlled feedback and regression state transitions."""

from datetime import UTC, datetime

import pytest

from document_intelligence.data.feedback import (
    DatasetCandidate,
    FailureComponent,
    FeedbackEligibilityError,
    FeedbackSignal,
    FeedbackVerification,
    FeedbackWorkflowError,
    ModelReference,
    ProductionFeedback,
    RegressionManifest,
    RegressionPriority,
    VerificationStatus,
    create_dataset_candidate,
    register_regression_case,
)

NOW = datetime(2026, 8, 30, tzinfo=UTC)
SHA256 = "a" * 64


def _model(version: str = "1.0.0") -> ModelReference:
    return ModelReference(
        name="invoice-classifier",
        version=version,
        source="deterministic_rules",
        serving_bundle_uri=f"registry://invoice-classifier/{version}",
    )


def _feedback() -> ProductionFeedback:
    return ProductionFeedback(
        feedback_id="feedback-001",
        received_at=NOW,
        signal=FeedbackSignal.HUMAN_CORRECTION,
        model=_model(),
        document_artifact_uri="protected://documents/document-001",
        document_sha256=SHA256,
        prediction_reference_uri="protected://predictions/prediction-001",
    )


def _verification(
    component: FailureComponent = FailureComponent.CLASSIFICATION,
    *,
    status: VerificationStatus = VerificationStatus.VERIFIED,
) -> FeedbackVerification:
    return FeedbackVerification(
        feedback_id="feedback-001",
        status=status,
        reviewed_at=NOW,
        reviewer_reference="review-system://reviews/review-001",
        evidence_uri="protected://verification/evidence-001",
        failure_component=(component if status is VerificationStatus.VERIFIED else None),
    )


def _candidate() -> DatasetCandidate:
    return create_dataset_candidate(
        _feedback(),
        _verification(),
        candidate_id="candidate-001",
        cohorts=("production_feedback", "human_correction"),
        priority=RegressionPriority.HIGH,
        created_at=NOW,
    )


def test_verified_ml_failure_becomes_traceable_dataset_candidate() -> None:
    candidate = _candidate()

    payload = candidate.to_dict()
    assert payload["workflow_state"] == "dataset_candidate"
    assert payload["target_component"] == "classification"
    assert payload["document_sha256"] == SHA256
    assert "training" not in payload


def test_rejected_feedback_cannot_become_dataset_candidate() -> None:
    with pytest.raises(FeedbackWorkflowError, match="only verified"):
        create_dataset_candidate(
            _feedback(),
            _verification(status=VerificationStatus.REJECTED),
            candidate_id="candidate-001",
            cohorts=("production_feedback",),
            priority=RegressionPriority.LOW,
            created_at=NOW,
        )


@pytest.mark.parametrize(
    "component",
    [
        FailureComponent.SOURCE_QUALITY,
        FailureComponent.DOWNSTREAM_BUSINESS_LOGIC,
        FailureComponent.INFRASTRUCTURE,
    ],
)
def test_non_ml_failures_are_rejected_as_dataset_candidates(
    component: FailureComponent,
) -> None:
    with pytest.raises(FeedbackEligibilityError, match="not actionable"):
        create_dataset_candidate(
            _feedback(),
            _verification(component),
            candidate_id="candidate-001",
            cohorts=("production_feedback",),
            priority=RegressionPriority.MEDIUM,
            created_at=NOW,
        )


def test_feedback_and_verification_identifiers_must_match() -> None:
    verification = FeedbackVerification(
        feedback_id="feedback-002",
        status=VerificationStatus.VERIFIED,
        reviewed_at=NOW,
        reviewer_reference="review-system://reviews/review-002",
        evidence_uri="protected://verification/evidence-002",
        failure_component=FailureComponent.CLASSIFICATION,
    )

    with pytest.raises(FeedbackWorkflowError, match="identifiers must match"):
        create_dataset_candidate(
            _feedback(),
            verification,
            candidate_id="candidate-001",
            cohorts=("production_feedback",),
            priority=RegressionPriority.HIGH,
            created_at=NOW,
        )


def test_candidate_creation_must_follow_verification() -> None:
    with pytest.raises(FeedbackWorkflowError, match="must not precede verification"):
        create_dataset_candidate(
            _feedback(),
            _verification(),
            candidate_id="candidate-001",
            cohorts=("production_feedback",),
            priority=RegressionPriority.HIGH,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
        )


def test_feedback_requires_immutable_document_checksum() -> None:
    with pytest.raises(ValueError, match="document_sha256"):
        ProductionFeedback(
            feedback_id="feedback-001",
            received_at=NOW,
            signal=FeedbackSignal.HUMAN_CORRECTION,
            model=_model(),
            document_artifact_uri="protected://documents/document-001",
            document_sha256="not-a-checksum",
            prediction_reference_uri="protected://predictions/prediction-001",
        )


def test_regression_case_requires_a_different_fixed_model_version() -> None:
    with pytest.raises(FeedbackWorkflowError, match="version must differ"):
        register_regression_case(
            _candidate(),
            case_id="regression-001",
            fixed_model=_model(),
            expected_output_uri="approved://annotations/expected-001",
            fix_evaluation_uri="reports://evaluation/fix-001",
            registered_at=NOW,
        )


def test_regression_manifest_sorts_fixed_cases_and_retains_lineage() -> None:
    candidate = _candidate()
    case_b = register_regression_case(
        candidate,
        case_id="regression-002",
        fixed_model=_model("1.1.0"),
        expected_output_uri="approved://annotations/expected-002",
        fix_evaluation_uri="reports://evaluation/fix-002",
        registered_at=NOW,
    )
    case_a = register_regression_case(
        candidate,
        case_id="regression-001",
        fixed_model=_model("1.1.0"),
        expected_output_uri="approved://annotations/expected-001",
        fix_evaluation_uri="reports://evaluation/fix-001",
        registered_at=NOW,
    )
    manifest = RegressionManifest(
        name="invoice-classifier-regression",
        version="1.0.0",
        source_commit="b" * 40,
        created_at=NOW,
        cases=(case_b, case_a),
    )

    payload = manifest.to_dict()
    assert [case["case_id"] for case in payload["cases"]] == [
        "regression-001",
        "regression-002",
    ]
    assert payload["source_commit"] == "b" * 40
    assert payload["cases"][0]["failing_model"]["version"] == "1.0.0"
    assert payload["cases"][0]["fixed_model"]["version"] == "1.1.0"


def test_regression_manifest_rejects_duplicate_case_ids() -> None:
    case = register_regression_case(
        _candidate(),
        case_id="regression-001",
        fixed_model=_model("1.1.0"),
        expected_output_uri="approved://annotations/expected-001",
        fix_evaluation_uri="reports://evaluation/fix-001",
        registered_at=NOW,
    )

    with pytest.raises(ValueError, match="case_id values must be unique"):
        RegressionManifest(
            name="invoice-classifier-regression",
            version="1.0.0",
            source_commit="b" * 40,
            created_at=NOW,
            cases=(case, case),
        )
