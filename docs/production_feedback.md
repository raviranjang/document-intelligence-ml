# Production feedback and regression workflow

Production feedback is evidence for review, never a command to train. This repository implements
explicit typed transitions and contains no path from feedback intake to an optimizer, checkpoint, or
model registry update.

```text
production signal
  -> protected feedback reference
  -> independent verification
  -> failure-component classification
  -> dataset candidate
  -> approved immutable dataset manifest
  -> justified experiment and evaluation
  -> candidate model review
  -> fixed-case regression guard
```

## Intake and verification

`ProductionFeedback` records a stable feedback identifier, timestamp, signal type, exact source-model
identity and serving-bundle reference, protected document URI and SHA-256, and protected prediction
reference. Raw documents, OCR text, extracted field values, human corrections, and customer data do
not belong in Git or in these metadata objects.

`FeedbackVerification` is a separate record with a reviewer-system reference, evidence reference,
review timestamp, outcome, and one failure component. Rejected feedback cannot receive a component
or become a dataset candidate. Verification and candidate timestamps must preserve causal order.

Verified failures are routed to source quality, OCR detection, OCR recognition, classification,
semantic extraction, calibration, downstream business logic, or infrastructure. Source-quality,
downstream-business, and infrastructure failures are explicitly rejected from the ML dataset path;
they must be handled by their owning systems.

## Dataset candidates

Only verified OCR, classification, extraction, or calibration failures can become a
`DatasetCandidate`. A candidate retains the signal, component, source model, protected artifact
reference and checksum, verification evidence, review priority, and bounded cohort tags.

The `dataset_candidate` state is not training approval and is not itself a released dataset. Before
use, authorized data owners must verify the annotation, license/consent, retention policy, duplicate
status, cohort assignment, and split-leakage risk. Accepted examples are copied into an immutable
dataset version described by the existing dataset manifest schema and must pass
`DatasetManifestValidator`. Rejected or quarantined examples never enter training.

Retraining still requires a documented observed problem, hypothesis, dataset change, primary metric,
regression guards, versioned configuration, tracked run, and evaluation against the established
baseline. Calendar age or one feedback item is not sufficient justification.

## Regression registration

A `RegressionCase` can be registered only from an eligible dataset candidate after a different
version of the same independently released model has a documented expected-output artifact and fix
evaluation. A versioned `RegressionManifest` records the failing and fixed model identities, source
commit, evidence references, cohorts, component, and priority.

Regression artifacts remain in approved storage; only sanitized synthetic fixtures or explicitly
approved small cases may enter Git. This repository currently contains no verified production failure
that can honestly be registered as fixed, so no production-derived regression fixture or result is
published. New model candidates must execute the applicable versioned regression manifest before
promotion and report regressions rather than silently dropping cases.
