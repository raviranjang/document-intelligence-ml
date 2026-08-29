# Model card: deterministic extraction baseline

## Identity

- Name: `deterministic-invoice-extractor`
- Version: `1.0.0`
- Source: deterministic rules
- Input: canonical `OCRDocument` schema `1.0.0`
- Output: `ExtractionDocument` schema `1.0.0`

## Intended use

This baseline supplies traceable raw field candidates for controlled evaluation and integration. It
provides a minimum benchmark that a more complex layout-aware extractor must outperform.

## Supported fields

Invoice number, order reference, total amount, and numeric invoice date when accompanied by a
configured lexical label. Seller name is deliberately unsupported in this version.

## Evaluation status

Not evaluated. No approved immutable entity-labelled dataset or lineage-bearing evaluation report is
available. Synthetic tests are not evidence of production quality.

## Limitations

- Depends on OCR recognition and reading order.
- Supports only configured English lexical anchors and limited value formats.
- Does not normalize amounts, currencies, identifiers, or ambiguous dates.
- Does not use geometry, images, or wider document semantics when selecting candidates.
- Returns alternatives without deciding which candidate is business truth.

## Release gate

Define and approve the versioned entity-label schema, annotate immutable evaluation data, measure
token/entity precision, recall, F1, and field exact match, review cohort failures, and retain code,
configuration, OCR, dataset, and report lineage. Downstream business validation remains out of scope.
