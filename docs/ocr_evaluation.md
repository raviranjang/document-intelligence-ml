# OCR evaluation

OCR detection and recognition are evaluated separately so failures are assigned to the model stage
that produced them. This module reports measured evidence only; it does not apply business rules or
claim that a pretrained baseline improved.

## Recognition metrics

Recognition samples contain a non-blank reference transcription, a prediction, optional cohorts,
and an optional identifier type.

- Character error rate (CER) is the corpus sum of Unicode-code-point Levenshtein edits divided by
  the corpus reference-character count.
- Word error rate (WER) is the corpus sum of Levenshtein edits over whitespace-split words divided
  by the corpus reference-word count.
- Exact match is case-sensitive equality of the complete, unnormalized strings.
- Identifier exact match applies the same strict equality only to samples with an identifier type.

The initial metrics intentionally perform no case folding, punctuation removal, Unicode
normalization, or domain reconciliation. Such transformations can hide meaningful OCR failures and
must become explicit versioned evaluation configuration if later justified.

## Detection metrics

Detection compares axis-aligned `BoundingBox` values with a configurable intersection-over-union
(IoU) threshold. Maximum-cardinality one-to-one matching prevents either a prediction or reference
region from being counted more than once. The report includes true positives, false positives,
false negatives, precision, recall, and F1.

When a metric denominator is empty, its value is JSON `null`; the evaluator does not report a
fabricated zero or perfect score. The aggregate counts remain available to explain why the value is
undefined.

## Slices and lineage

Every named cohort is evaluated independently in addition to the micro-averaged aggregate. Cohorts
may represent template family, image quality, identifier type, language, source, long-tail cases,
or regression cases. A report records:

- immutable dataset name and semantic version;
- full 40- or 64-character source commit;
- evaluation-configuration semantic version;
- model name, version, and source;
- aggregate and alphabetically ordered cohort metrics.

`OCRMetricsReport` schema version `1.0.0` omits unavailable detection or recognition sections rather
than inventing results. The next workflow stage will load approved ground truth, execute this
evaluator, and classify measured errors. No sample report is committed because illustrative values
could be mistaken for real baseline performance.
