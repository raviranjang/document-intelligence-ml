# Inference adapters

The inference package defines narrow structural contracts for OCR, document classification, and
semantic extraction. Concrete implementations keep PaddlePaddle, Transformers, ONNX, or other
framework values behind those contracts and return the existing versioned domain objects.

`DocumentInferencePipeline` receives already initialized adapters and retains them for its entire
lifecycle. Construct the pipeline once when a worker or model server starts, then reuse it for every
prediction. Model loading, tokenizer construction, and rule compilation must not happen inside a
request handler.

An optional lifecycle-scoped monitoring sink receives privacy-conscious stage records. Document
identifiers are omitted unless the deployment explicitly opts in; see
[`monitoring.md`](monitoring.md) for the telemetry contract and operational plan.

For each document, the pipeline:

1. invokes OCR once;
2. passes the same canonical `OCRDocument` instance to the classifier and extractor;
3. verifies that every output retains the same document identifier; and
4. returns `DocumentInference`, which preserves the schema and model identity of every component.

The aggregate output is evidence, not a business decision. It contains OCR, classification, and raw
extraction results without order validation, seller reconciliation, approval decisions, or other
domain rules.

Before constructing an adapter from an exported release, call `validate_serving_bundle`. Loading is
model-family-specific: deterministic baselines use their strict TOML loaders, PaddleOCR uses its
version-pinned factory, and future LayoutLMv3 or ONNX adapters must validate export parity before
release. A generic loader must not guess how to deserialize an unknown artifact family.

The smoke suite uses synthetic bytes and a framework-free Paddle-compatible fixture. It exercises
the complete adapter chain without downloading weights, requiring a GPU, or publishing model-quality
claims.
