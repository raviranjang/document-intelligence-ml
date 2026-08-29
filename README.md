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
