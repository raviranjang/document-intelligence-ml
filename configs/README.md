# Configurations

Store version-controlled training, inference, export, and evaluation configuration here. Each
configuration must identify its model family and version; illustrative values must be labeled as
such. Do not place machine-specific filesystem paths or secrets in configuration files.

`ocr/baseline.toml` identifies the untouched official PaddleOCR baseline, including dependency,
model-family, detector, recognizer, engine, device, preprocessing switches, and recognition
threshold. Changing any of these values creates a different baseline and requires evaluation.

`classification/keyword_invoice_baseline.toml` defines the initial deterministic binary document
classifier. Its signal list and decision threshold are versioned behavior, not trained or calibrated
parameters. Changing either requires a model-version change and evaluation.

`extraction/deterministic_baseline.toml` defines traceable lexical field-candidate rules. Patterns and
rule ordering are versioned extractor behavior. The baseline preserves raw OCR evidence and does not
perform locale-sensitive normalization or downstream business validation.

`extraction/semantic_labels_v1.json` is the canonical BIO label-to-ID mapping for semantic token
classification. Released IDs are stable model interfaces; semantic or ordering changes require a new
label-schema version and corresponding annotation documentation.
