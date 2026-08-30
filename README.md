# Document Intelligence ML

Production-oriented machine-learning code for OCR, document classification, layout-aware entity
extraction, calibration, evaluation, export, and inference adapters.

This repository produces probabilistic document evidence. Upload APIs, customer-facing services,
order reconciliation, and deterministic business acceptance rules belong to other repositories.

## Current scope

Development follows measured milestones. The first milestone establishes an untouched pretrained
PaddleOCR baseline:

```text
document input
  -> pretrained text detection
  -> pretrained text recognition
  -> canonical OCRDocument
  -> evaluation
  -> baseline report and error taxonomy
```

Fine-tuning, LayoutLMv3 extraction, and business validation are intentionally outside that first
milestone. No model-quality claims are made until a versioned dataset and evaluation report exist.

The pretrained OCR adapter, canonical output contract, locked optional dependencies, execution
command, and current limitations are documented in
[`docs/ocr_baseline.md`](docs/ocr_baseline.md).

The metric definitions and lineage-aware report contract are documented in
[`docs/ocr_evaluation.md`](docs/ocr_evaluation.md). No metric values are reported until an immutable,
approved evaluation dataset is available.

The deterministic error taxonomy, privacy-preserving aggregate report, and fine-tuning decision
gate are documented in [`docs/ocr_error_analysis.md`](docs/ocr_error_analysis.md). The tooling is
implemented, but an actual baseline report still requires an approved evaluation dataset.

The binary `INVOICE` / `NOT_INVOICE` classifier now has a versioned deterministic baseline,
canonical output contract, and per-class evaluation report. See
[`docs/classification_baseline.md`](docs/classification_baseline.md). No classifier quality claim is
made before evaluation on an approved immutable dataset.

The deterministic semantic extraction baseline produces raw, token-traceable invoice field
candidates without confidence, normalization, or business acceptance decisions. Its supported
fields and limitations are documented in
[`docs/deterministic_extraction_baseline.md`](docs/deterministic_extraction_baseline.md).

Semantic token annotations now use a versioned BIO vocabulary and strict artifact schema. Entity
definitions, boundary rules, ambiguity handling, OCR-error policy, exclusions, and schema evolution
are specified in [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).

The framework-independent LayoutLMv3 dataset pipeline validates semantic annotations, normalizes
geometry, and aligns BIO labels across tokenizer subwords with explicit truncation behavior. See
[`docs/layoutlmv3_dataset_pipeline.md`](docs/layoutlmv3_dataset_pipeline.md).

BIO entity reconstruction and lineage-aware token, entity, field-exact, and cohort evaluation are
defined in [`docs/entity_evaluation.md`](docs/entity_evaluation.md). No values are reported until an
approved immutable entity dataset is available.

Generic multiclass temperature scaling, calibration artifacts, reliability analysis, and held-out
cohort reports are documented in
[`docs/confidence_calibration.md`](docs/confidence_calibration.md). No fitted temperature or quality
claim is published without approved model logits and separate calibration/evaluation splits.

## Development setup

The project requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependency versions are
recorded in `uv.lock`.

```bash
uv sync --locked --dev
uv run --locked pytest
```

Run the complete local quality gate with:

```bash
make validate
```

On systems without `make`, run the commands under the `validate` target in `Makefile`. CI runs the
same lock, formatting, lint, type, test, and package-build checks.

## Repository structure

```text
src/document_intelligence/  Python package and stable internal contracts
configs/                    Version-controlled model and evaluation configuration
datasets/                   Dataset schemas, manifests, and documentation only
docs/                       ADRs, annotation guidance, and model cards
notebooks/                  Purposeful exploration; reusable logic moves to src/
scripts/                    Thin command-line entry points
tests/                      Unit, integration, regression, and smoke coverage
docker/                     Container definitions introduced when serving requires them
infra/                      ML infrastructure definitions introduced with a concrete need
```

Large datasets, model weights, checkpoints, local MLflow runs, and production documents must not
be committed. Model artifacts must retain lineage to source commit, configuration, dataset
version, training run, and evaluation report.

## Git workflow

Normal development uses a short-lived, purpose-named branch and a focused pull request. Keep
`main` usable, use Conventional Commits, update from `main` before merge, and rerun the full
quality gate after resolving conflicts. See `AGENTS.md` for the complete engineering policy.
