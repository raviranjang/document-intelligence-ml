# Confidence calibration

This module provides deterministic multiclass temperature scaling for model logits. Calibration is a
versioned artifact tied to one base model and one immutable held-out calibration split. A calibrated
probability is model evidence, not a downstream business acceptance decision.

## Method

Temperature scaling divides every class logit by one positive scalar before applying softmax. A
temperature above one softens confidence; a temperature below one sharpens it. Because the same scalar
is applied to every class, predicted class ordering and accuracy are unchanged.

`fit_temperature_scaling` minimizes mean multiclass negative log likelihood in log-temperature space
using deterministic bounded golden-section search. Bounds and iteration count come from
[`configs/calibration/temperature_scaling.toml`](../configs/calibration/temperature_scaling.toml).
Identity temperature `1.0` is always an explicit candidate, so fitting cannot select a temperature with
worse calibration-split NLL merely due to numerical optimization error.

No random seed is needed because the fitting algorithm is deterministic. This does not make upstream
model training perfectly reproducible.

## Metrics

Calibration metrics include:

- mean multiclass negative log likelihood;
- expected calibration error (ECE);
- multiclass Brier score;
- mean top-class confidence;
- top-class accuracy;
- fixed-width reliability bins with count, mean confidence, and accuracy.

ECE uses equal-width bins over `[0, 1]` and weights each absolute confidence/accuracy gap by its sample
fraction. Confidence exactly `1.0` belongs to the final bin. Empty bins remain present with `null`
confidence and accuracy so reliability charts have a stable shape. The Brier score is the mean sum of
squared error across all classes; it is not divided by class count.

The configured bin count is part of evaluation behavior. Calibration results produced with different
binning are not directly interchangeable.

## Artifact lineage

`TemperatureCalibrationArtifact` schema `1.0.0` records:

- base model name and version;
- calibration dataset name and immutable version;
- source commit and calibration configuration version;
- method and fitted temperature;
- aggregate fitting diagnostics before and after scaling.

Serialized artifacts do not include logits, raw documents, or sample identifiers. Fitting diagnostics
describe the calibration split and must not be presented as held-out generalization performance.

## Held-out evaluation and cohorts

`CalibrationMetricsReport` evaluates a fitted artifact on a separate immutable evaluation split. It
records evaluation lineage, calibration-dataset lineage, aggregate metrics, and alphabetically ordered
cohort slices. Important slices include entity type, seller/template family, image quality, language,
source, unseen templates, recent production, and known regression cases when approved labels exist.

The calibration split, model-selection validation split, and final evaluation split must not be
silently reused. Any base-model weight, label mapping, logit definition, calibration dataset, fitting
configuration, or temperature change creates a different calibration artifact.

## Current status

The implementation is validated with synthetic contract tests only. No approved immutable model logits,
calibration split, or independent evaluation split are available, so this repository publishes no
temperature, ECE improvement, reliability claim, or production-readiness claim.
