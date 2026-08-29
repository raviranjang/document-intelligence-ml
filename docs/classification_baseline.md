# Document-classification baseline

The initial classifier is an auditable deterministic baseline over the canonical `OCRDocument`.
It distinguishes only the product-approved labels `INVOICE` and `NOT_INVOICE`; adding document
types requires a versioned label-contract change.

## Decision pipeline

For each OCR page, the classifier case-folds text, replaces punctuation boundaries with spaces, and
collapses repeated whitespace. Versioned invoice signals are matched as complete normalized token
phrases within a page. A phrase cannot match across page boundaries, and repeated occurrences of a
signal do not increase its weight.

The decision score is the fraction of configured signals present in the document. The document is
classified as `INVOICE` when the score meets the configured threshold and `NOT_INVOICE` otherwise.
This score is deterministic evidence, not a calibrated probability. Downstream services must not
interpret it as confidence or use it as a business-acceptance decision.

The baseline configuration is
[`configs/classification/keyword_invoice_baseline.toml`](../configs/classification/keyword_invoice_baseline.toml).
It identifies the model name, version, source, threshold, and lexical signals. Any behavioral config
change creates a new baseline version and requires evaluation.

## Output contract

`DocumentClassification` schema `1.0.0` contains:

- document identifier;
- typed document label;
- decision score and threshold;
- unique matched signals for explainability;
- classifier model name, version, and source.

It consumes canonical OCR rather than framework-specific Paddle results. The output deliberately
contains no extracted invoice fields or business reconciliation.

## Evaluation

Evaluation reports require an immutable dataset version, source commit, evaluation-config version,
and classifier identity. Reports contain a complete confusion matrix plus precision, recall, and F1
for each class, aggregate accuracy and macro F1, and alphabetically ordered cohort slices. Undefined
class denominators remain `null`; they are not replaced with invented values.

Important cohorts should include source, language, scan quality, template family, recent production,
and unseen templates where those annotations are approved. Accuracy must never be reported alone.

No quality claim is made in this repository because an approved classifier evaluation dataset is not
yet available. The configured threshold establishes an initial behavior to measure, not an optimized
threshold.

## Limitations and next decision

Keyword evidence is sensitive to OCR recognition errors, unfamiliar terminology, sparse documents,
and non-invoice documents that reuse invoice vocabulary. Correlated signals are not independent and
the score is not calibrated. A learned classifier is justified only after a versioned baseline report
identifies measurable failures and defines success metrics and regression guards.

OCR recognition and detection fine-tuning also remain deferred until the OCR baseline is evaluated on
approved data. This preserves the required separation between OCR, classification, semantic
extraction, and downstream business failures.
