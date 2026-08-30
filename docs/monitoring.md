# Inference monitoring plan

Inference monitoring is exposed through a backend-neutral `MonitoringSink`. Deployments can adapt
the typed records to OpenTelemetry spans and metrics or to structured logs without adding an
observability SDK to the model package. The default no-op sink stores nothing.

The lifecycle-scoped inference pipeline emits one record for each OCR, classification, and extraction
stage plus one record for the complete pipeline. Records contain only:

- stage, status, and elapsed seconds;
- model name, version, and source for completed model stages;
- output counts and category names;
- mean OCR token confidence when confidence is available; and
- exception type for failures.

OCR text, extracted values, document bytes, filesystem paths, and exception messages are excluded.
Document identifiers are also excluded by default. A deployment may set
`include_document_id_in_monitoring=True` only when its privacy and cardinality policy permits that
identifier to enter telemetry.

Classification `decision_score` is deliberately not reported as confidence because the deterministic
baseline does not produce a calibrated probability. Calibration metrics remain separate, versioned
evaluation artifacts.

## Backend mapping

An OpenTelemetry adapter should map duration to a histogram and status/model/category fields to
bounded attributes. Error counts and request counts should be counters. Document identifiers must
not become metric attributes; when explicitly enabled, they belong only on appropriately protected
logs or traces. A sink is expected to be lifecycle-scoped and must not load model artifacts.

Recommended dashboards include P50/P95 latency and error rate by stage and model version, OCR token
count and mean-confidence distributions, classifier label distribution, extraction count and entity
type distribution, and traffic volume. Cohort dimensions may be joined downstream only from approved,
bounded metadata; raw document-derived values are not monitoring dimensions.

## Drift and alerts

Alert thresholds require a measured production baseline and are therefore not invented in this
repository. Deployments should compare rolling distributions with a documented reference window and
alert on sustained changes rather than individual requests. Important signals include latency/error
regression, OCR confidence or token-count shifts, classifier label shifts, extraction coverage shifts,
new source/template cohorts, and delayed ground-truth quality degradation.

Every deployment must assign alert ownership, escalation routing, retention, and dashboard links in
its runbook before production approval. The owning team should first classify a signal as source
quality, OCR, classification, extraction, calibration, downstream logic, or infrastructure. Monitoring
must never trigger training or update weights directly; verified feedback enters the versioned
feedback and regression workflow.
