# Model card: untouched PP-OCRv6 baseline

## Identity

- Status: baseline pending evaluation
- Framework: PaddleOCR 3.7.0 with PaddlePaddle 3.3.1
- Detector: `PP-OCRv6_medium_det`
- Recognizer: `PP-OCRv6_medium_rec`
- Source: official pretrained weights
- Configuration: [`configs/ocr/baseline.toml`](../../configs/ocr/baseline.toml)
- Output contract: `OCRDocument` schema `1.0.0`

## Intended use

Establish reproducible OCR detection and recognition evidence for document-intelligence research,
evaluation, and later error analysis. Output is probabilistic evidence and must not make business
acceptance or reconciliation decisions.

## Evaluation status

Not yet evaluated in this repository. Accuracy, character error rate, word error rate, exact match,
latency, and slice results must remain unreported until the versioned evaluation pipeline runs on
an immutable manifest.

## Known limitations

- Baseline preprocessing intentionally disables orientation classification and image unwarping.
- Results depend on document language, scan quality, typography, layout, and hardware/runtime.
- Polygon geometry is represented by its enclosing axis-aligned box in the current contract.
- Recognition confidence is not a calibrated business probability.
- Official weights are downloaded externally and are not stored in Git.

## Ethical and data considerations

Evaluation fixtures and datasets must be synthetic, sanitized, or explicitly approved. Do not
commit production documents, personal data, customer identifiers, credentials, or signed storage
URLs. Slice evaluation must inspect representative document sources and known failure cohorts.
