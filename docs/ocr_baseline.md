# Pretrained OCR baseline

The first OCR baseline runs the official `PP-OCRv6_medium_det` detector and
`PP-OCRv6_medium_rec` recognizer distributed for PaddleOCR 3.7.0. The repository does not modify,
fine-tune, or commit these weights. [`configs/ocr/baseline.toml`](../configs/ocr/baseline.toml)
records the complete model and inference identity.

No accuracy or performance claim is attached to this baseline yet. It exists to establish a
reproducible reference for the versioned evaluation dataset and error taxonomy that follow.

## Install

Install the locked CPU baseline dependencies separately from the lightweight development stack:

```bash
uv sync --locked --extra ocr
```

PaddleOCR downloads its official pretrained artifacts on first initialization and stores them in
its external model cache. Model files, caches, customer documents, and generated OCR outputs must
not be committed to this repository.

## Run

Process one local image or PDF and write a new canonical OCR result:

```bash
uv run --extra ocr document-intelligence-ocr input/invoice.png \
  --config configs/ocr/baseline.toml \
  --document-id synthetic-invoice-0001 \
  --output output/synthetic-invoice-0001.json
```

Omit `--output` to write JSON to standard output. The command refuses to overwrite an existing
file so that baseline evidence is not replaced accidentally. Use a stable dataset record ID rather
than a customer identifier for `--document-id`.

The adapter loads the pipeline once per process, processes all returned pages, and converts the
document to `OCRDocument` schema version `1.0.0`. Paddle arrays and result objects do not cross the
adapter boundary.

## Deliberate baseline behavior

- Document orientation classification, document unwarping, and text-line orientation are disabled.
- The inference engine is `paddle_static` on CPU.
- Recognition threshold is zero, preserving scored non-blank recognition results.
- Blank recognition strings are omitted because they cannot satisfy the canonical token contract.
- Rectangular `rec_boxes` are preferred; polygon results are reduced to enclosing axis-aligned
  boxes without model-coordinate normalization.
- Token and page order are preserved and validated as contiguous, zero-based indexes.
- Detection confidence is not inferred from recognition results; it will be represented only when
  Paddle provides an unambiguous region-level mapping.

These choices are baseline configuration, not business acceptance rules. Any change requires a
new configuration/model identity and evaluation against the same immutable dataset.

## Verification

Normal CI uses framework-free fixtures shaped like PaddleOCR's documented result payload. This
checks configuration parsing, adapter conversion, multi-page ordering, invalid output handling,
CLI serialization, and the canonical contract without downloading large model binaries.

Before recording a baseline report, run the optional PaddleOCR environment on the controlled
evaluation dataset, retain the exact configuration and dependency lock, and record model artifact
lineage. Do not interpret example output as a measured model result.
