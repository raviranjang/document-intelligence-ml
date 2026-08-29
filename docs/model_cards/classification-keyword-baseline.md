# Model card: keyword invoice baseline

## Identity

- Name: `keyword-invoice-baseline`
- Version: `1.0.0`
- Source: deterministic rules
- Labels: `INVOICE`, `NOT_INVOICE`
- Input: canonical `OCRDocument` schema `1.0.0`
- Output: `DocumentClassification` schema `1.0.0`

## Intended use

This model is a transparent starting point for measuring binary document-classification failures. It
may provide classification evidence to controlled evaluation and integration workflows. It is not a
business acceptance rule and its score is not a calibrated probability.

## Evaluation status

Not evaluated. No approved immutable classification dataset or lineage-bearing metrics report is
available. Do not infer quality, fairness, robustness, or production readiness from synthetic tests.

## Known limitations

- Depends on OCR text quality and configured English invoice vocabulary.
- Does not use layout, images, issuer identity, or negative lexical evidence.
- May miss unfamiliar invoices and match non-invoices containing invoice terms.
- The initial threshold is unoptimized and must remain versioned.

## Release gate

Before promotion, run aggregate and cohort evaluation with class precision, recall, F1, and the full
confusion matrix. Review false positives and false negatives, select a measured threshold, define
regression guards, and preserve dataset, configuration, code, and model lineage.
