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
