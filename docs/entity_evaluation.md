# Semantic entity reconstruction and evaluation

This module reconstructs entity spans from page-level BIO labels and evaluates semantic extraction at
token, entity, and document-field levels. It can compare the deterministic baseline or a future
LayoutLMv3 model on the same immutable dataset without conflating different failure types.

## BIO reconstruction

`reconstruct_entity_spans` requires canonical OCR tokens with contiguous zero-based token indexes and
a same-length label sequence validated against the versioned semantic label schema. Entities cannot
cross page boundaries. `B-ENTITY` starts a new span, `I-ENTITY` continues the same typed span, and `O`
closes any active span.

Raw entity text is reconstructed by joining the exact source OCR token strings with one ASCII space.
No case folding, identifier repair, date interpretation, currency parsing, or business reconciliation
occurs. Changing token joining is a versioned postprocessing behavior and requires regression review.

## Metric semantics

All precision, recall, and F1 values are micro-averaged counts over the evaluated corpus. Reports also
contain the same metrics for every entity type.

### Token level

A true positive requires the same document, entity type, page index, and token index in reference and
prediction. A correct token assigned to the wrong entity type is one false positive and one false
negative. `O` tokens do not inflate the metric.

### Entity level

A true positive requires the same document, entity type, and complete ordered token span. Raw value is
not part of boundary matching, allowing boundary/model attribution to remain separate from OCR text or
postprocessing differences.

### Field exact match

For each document/entity-type pair present in either reference or prediction, the sorted multisets of
raw candidate values must be identical. Comparison is case-sensitive with no normalization. This
penalizes missing, extra, and text-mismatched candidates. A field absent from both sides is not counted
and cannot inflate the rate.

Undefined precision, recall, F1, or field rates serialize as `null` when their denominator is absent.
When predictions and references exist but have no true positives, F1 is `0.0`, not unavailable.

## Report lineage and slices

`EntityMetricsReport` schema `1.0.0` records dataset name/version, label-schema version, source commit,
evaluation-config version, and extractor name/version/source. It contains aggregate metrics and
alphabetically ordered cohort slices. Important cohorts include seller, template family, image quality,
language, source, unseen templates, recent production, and regression cases when approved annotations
provide them.

Reports contain aggregate counts and rates, not raw documents or token text. This does not grant
permission to publish source datasets or sample-level sensitive values.

## Training and baseline status

No approved immutable semantic training/evaluation dataset is currently available. Consequently, the
repository has not trained LayoutLMv3 and makes no comparison or quality claim. When data is approved,
the training run must preserve dataset/configuration/code/model lineage and compare against the
deterministic extraction baseline using this exact evaluation contract.
