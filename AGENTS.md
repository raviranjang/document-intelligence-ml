# AGENTS.md — Document Intelligence ML

## Purpose

This repository contains the machine-learning codebase for the Document Intelligence platform.

It owns:
- OCR text detection
- OCR text recognition
- document classification
- layout-aware semantic entity extraction
- confidence calibration
- dataset preparation and validation
- training
- evaluation
- experiment tracking
- model export
- inference adapters
- regression testing
- model release metadata

It does not own:
- document upload APIs
- customer-facing HTTP APIs
- order or seller services
- business reconciliation APIs
- manual-review applications
- production business decision rules

Those responsibilities belong to separate services.

## Engineering principles

Treat this repository as a production ML codebase, not as a collection of experiments or notebooks.

Optimize for:
- reproducibility
- explicit model and dataset lineage
- readable code
- stable internal contracts
- testability
- measurable experiments
- production inference compatibility
- safe model evolution
- observability
- separation of ML inference from deterministic business logic

Prefer simple, explicit implementations over clever abstractions.

## Repository strategy

Keep all closely related document-intelligence models in this repository.

Do not create a separate repository merely because a new model is introduced.

Each model must still have:
- its own module
- its own training configuration
- its own dataset requirements
- its own evaluation
- its own version
- its own model artifacts
- its own release lifecycle

Same repository does not mean same model lifecycle.

Split a model into a separate repository only when ownership, release cadence, infrastructure, or product scope becomes materially independent.

## Initial models

### OCR Text Detector

Responsibility: identify where text exists in the document image.

Initial implementation:
- PaddleOCR / PP-OCR pretrained detector
- start from pretrained weights
- establish an untouched baseline before fine-tuning

Expected output:
- text-region polygons or bounding boxes
- detection confidence where available

### OCR Text Recognizer

Responsibility: convert detected text regions into characters and strings.

Initial implementation:
- PaddleOCR / PP-OCR pretrained recognizer
- start from pretrained weights
- fine-tune independently from detection when justified

Expected output:
- recognized text
- confidence
- associated geometry

### Document Classifier

Responsibility: determine the semantic document type.

Initial labels:
- INVOICE
- NOT_INVOICE

Expand only when product requirements justify additional classes.

### Layout-Aware Entity Extractor

Responsibility: extract semantic fields from OCR tokens using text, geometry, and document context.

Initial model family:
- LayoutLMv3-style token classification

Example entities:
- ORDER_REFERENCE
- INVOICE_NUMBER
- SELLER_NAME
- TOTAL_AMOUNT
- INVOICE_DATE

The label schema must be versioned.

## Calibration

Confidence calibration is a versioned ML artifact, not necessarily another neural model.

Possible implementation:
- temperature scaling
- held-out calibration dataset

Track:
- base model version
- calibration dataset version
- calibration method
- calibration parameters
- calibration metrics

Do not confuse raw softmax scores with calibrated probabilities.

## ML / business boundary

ML components produce probabilistic evidence.

Business systems enforce deterministic truth.

The ML repository must not contain order-domain decision rules such as:
- seller must equal order seller
- amount must equal order amount
- order must exist
- invoice should be auto-approved

The ML layer may expose:
- extracted value
- model confidence
- calibrated confidence
- alternative predictions
- model version

It must not decide business acceptance.

## Technology stack

Use:
- Python 3.11
- PyTorch where required
- Hugging Face Transformers
- PaddleOCR / PaddlePaddle for OCR
- PyMuPDF where required
- Pillow
- OpenCV only when justified
- NumPy
- Pandas for dataset/evaluation utilities
- MLflow
- pytest
- Ruff
- mypy
- ONNX where appropriate
- OpenTelemetry-compatible instrumentation where needed

Prefer `pyproject.toml` and a lockfile.

For a new project, prefer `uv` unless repository constraints require another tool.

## Expected repository layout

```text
document-intelligence-ml/
├── src/
│   └── document_intelligence/
│       ├── common/
│       │   ├── config/
│       │   ├── logging/
│       │   ├── telemetry/
│       │   └── types/
│       ├── data/
│       │   ├── ingestion/
│       │   ├── preprocessing/
│       │   ├── annotation/
│       │   ├── validation/
│       │   ├── manifests/
│       │   └── splitting/
│       ├── ocr/
│       │   ├── detection/
│       │   ├── recognition/
│       │   └── pipeline/
│       ├── classification/
│       ├── extraction/
│       │   ├── layoutlm/
│       │   ├── labels/
│       │   └── postprocessing/
│       ├── calibration/
│       ├── evaluation/
│       │   ├── metrics/
│       │   ├── slicing/
│       │   └── reports/
│       ├── training/
│       ├── inference/
│       └── export/
├── configs/
│   ├── ocr_detection/
│   ├── ocr_recognition/
│   ├── classification/
│   ├── extraction/
│   └── evaluation/
├── datasets/
│   └── README.md
├── notebooks/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── smoke/
├── docs/
│   ├── adr/
│   ├── annotation_guidelines.md
│   └── model_cards/
├── docker/
├── infra/
├── pyproject.toml
├── Makefile
├── README.md
└── AGENTS.md
```

Use the `src/` layout.

Do not place substantial Python modules at repository root.

## Python naming standards

Files and modules:
```text
snake_case.py
```

Functions:
```python
normalize_bbox()
build_training_dataset()
calculate_entity_f1()
```

Variables:
```python
ocr_tokens
normalized_boxes
entity_logits
training_examples
model_version
dataset_version
```

Classes:
```python
OCRToken
OCRRecognizer
InvoiceDataset
EntityExtractor
TrainingConfig
```

Constants:
```python
MAX_SEQUENCE_LENGTH = 512
IGNORE_LABEL_ID = -100
```

Avoid ambiguous names such as:
```text
data
data2
tmp
final
result1
new_model
model_final
```

## Type safety

Use type annotations for public and non-trivial internal functions.

Prefer typed domain objects over anonymous dictionaries.

Shared types should live under:
```text
src/document_intelligence/common/types/
```

Do not create duplicate definitions of core contracts in multiple model modules.

## Internal model interfaces

Hide framework-specific implementations behind stable internal interfaces where practical.

Concrete adapters may include:
- PaddleOCRRecognizer
- LayoutLMv3EntityExtractor

Application-level code should not require knowledge of PaddlePaddle, Transformers, or ONNX implementation details unless necessary.

## Configuration

Training and inference configuration must not be scattered through source code.

Use version-controlled configuration files.

Example:
```yaml
model:
  base_model: microsoft/layoutlmv3-base

training:
  learning_rate: 2.0e-5
  batch_size: 16
  epochs: 10
  seed: 42
```

Do not hide experiment-critical parameters inside notebooks.

Do not hard-code machine-specific filesystem paths.

## Model versioning

Never use names such as:
```text
model_final
final_model_v2
latest_model
best_final
```

Use explicit model identities:
```text
invoice-ocr-det-v1
invoice-ocr-rec-v3
invoice-classifier-v2
invoice-layout-v5
```

A model identity must map to:
- model family
- training run
- dataset version
- configuration
- source commit
- evaluation report
- exported artifact

Model versions are independent across models.

## Dataset versioning

Every trained model must reference an immutable dataset version.

Never mutate a released dataset in place.

Git should contain:
- manifests
- schemas
- small fixtures
- dataset-building code
- checksums
- documentation

Large source artifacts must not be committed directly to Git.

## Dataset splitting

Do not default to naive random train/test splits.

Prefer grouping by relevant structure such as:
- seller
- template family
- source
- business cohort

Maintain meaningful evaluation cohorts:
- training
- validation
- test
- golden
- long_tail
- unseen_template
- recent_production
- regression

Do not tune repeatedly against the final test set.

## Annotation

Maintain:
```text
docs/annotation_guidelines.md
```

Document:
- supported labels
- entity boundaries
- ambiguous cases
- multi-token values
- OCR errors
- missing fields
- duplicate candidate fields
- annotation exclusions
- examples

Changes to label semantics require a label-schema version change.

## Data validation

Every training pipeline must validate data before training.

Validate at minimum:
- missing files
- unreadable files
- empty examples
- duplicate records
- invalid labels
- invalid bounding boxes
- coordinates outside accepted ranges
- token/label length mismatches
- corrupt annotations
- unsupported values

Fail loudly when input data violates required invariants.

Do not silently skip large numbers of invalid samples.

## OCR baseline first

The first OCR milestone must use untouched pretrained weights.

Initial sequence:
```text
Input Document
      ↓
Pretrained OCR
      ↓
Canonical OCR Output
      ↓
Evaluation
      ↓
Error Taxonomy
```

Only fine-tune after establishing measurable baseline failures.

Distinguish:
- detection failure
- recognition failure
- semantic extraction failure

Do not fine-tune OCR to solve a semantic extraction problem.

## OCR training

Detection and recognition are separate learning problems.

Detection training data:
```text
image + text polygons
```

Recognition training data:
```text
cropped text region + transcription
```

Track separate metrics.

Detection:
- precision
- recall

Recognition:
- character error rate
- word error rate
- exact match
- identifier-specific exact match

## Semantic extraction

The layout-aware extractor consumes a canonical representation of OCR output.

Conceptually:
```text
OCR text
+
geometry
+
document image
      ↓
layout-aware model
      ↓
token labels
      ↓
entity reconstruction
```

Keep preprocessing, inference, entity reconstruction, and normalization distinguishable.

Do not mix domain reconciliation into semantic extraction.

## Label alignment

Transformer tokenization may split source tokens into multiple subwords.

Subword label alignment must be explicit and tested.

Add unit tests for:
- normal words
- split words
- special tokens
- empty tokens
- truncation

## Bounding-box normalization

Use one canonical implementation for converting source coordinates to model coordinates.

Do not duplicate bounding-box normalization in training and inference.

Test:
- normal coordinates
- image boundaries
- zero dimensions
- invalid boxes
- rounding behavior

## Training / inference parity

Training and production inference must use compatible:
- preprocessing
- tokenization
- normalization
- label mappings
- postprocessing
- thresholds

Shared transforms should be reused where practical.

## Experiment discipline

Every meaningful experiment should have a hypothesis.

Before modifying a model, identify:
- observed problem
- hypothesis
- proposed change
- primary success metric
- regression guards

Avoid random hyperparameter experimentation with no stated objective.

## Experiment tracking

Use MLflow or the approved experiment-tracking system.

Every training run should record where feasible:
- run ID
- experiment name
- source commit
- Python version
- dependency environment
- dataset version
- base model
- configuration
- random seed
- hyperparameters
- hardware information
- training metrics
- validation metrics
- evaluation artifacts
- checkpoints

Do not rely on terminal output as experiment history.

## Experiment naming

Good:
```text
ocr_rec_low_resolution_aug_v1
ocr_det_small_text_sampling_v2
layout_order_ref_hard_negatives_v1
```

Bad:
```text
test1
experiment2
new
try_again
final
```

## Reproducibility

Set random seeds where supported.

Do not claim perfect reproducibility merely because a seed is set.

Record environment and hardware details.

## Evaluation philosophy

Never report only a generic `accuracy`.

Use metrics appropriate to the model.

OCR detection:
- precision
- recall

OCR recognition:
- CER
- WER
- exact match

Classification:
- class precision
- recall
- F1
- confusion matrix

Entity extraction:
- token precision/recall/F1
- entity-level precision/recall/F1
- field exact match

Calibration:
- calibration error
- reliability analysis
- confidence distribution

Do not optimize aggregate metrics while ignoring critical fields.

## Slice evaluation

Evaluate important cohorts separately.

Examples:
- seller
- template family
- image quality
- field
- identifier type
- language
- source
- recent production
- unseen templates

## Baselines

Always maintain a baseline.

Examples:
- untouched pretrained OCR
- deterministic extraction baseline
- previous production model
- simple classifier baseline

A more complex model must demonstrate measurable value against a relevant baseline.

## Regression dataset

Known important failures that are fixed should be considered for the regression set.

Do not let new releases silently reintroduce resolved critical failures.

## Hard-example mining

Useful candidates include:
- low-confidence predictions
- human corrections
- model/domain disagreement
- false accepts
- false reviews
- new layouts
- rare labels
- repeated OCR confusions

Do not automatically train on every production failure.

First determine whether the failure belongs to:
- source quality
- OCR detection
- OCR recognition
- classification
- semantic extraction
- calibration
- downstream business logic
- infrastructure

Only train the component capable of learning from that failure.

## Production feedback

Production feedback must not directly update model weights.

Required path:
```text
Production feedback
      ↓
Verification
      ↓
Failure classification
      ↓
Dataset candidate
      ↓
Versioned dataset
      ↓
Training
      ↓
Evaluation
      ↓
Candidate model
```

## Retraining

Do not retrain merely because a calendar interval elapsed.

Periodic evaluation is acceptable.

Retraining should be justified by evidence such as:
- meaningful degradation
- sustained drift
- new important cohort
- sufficient verified hard examples
- newly supported labels
- material model weakness

Every retraining run should document why it was initiated.

## Training code vs inference code

Keep training-only and inference code distinguishable.

Training may contain:
- losses
- optimizers
- callbacks
- augmentation
- checkpointing
- training loops

Inference should contain only what is required to:
- load the artifact
- preprocess
- predict
- postprocess
- emit a stable result

Production inference must not import an unnecessary training stack.

## Model loading

Model artifacts should load once per process or model-server lifecycle.

Do not load a model inside every prediction call.

## Inference contracts

Inference outputs must use stable typed contracts.

Do not expose raw framework tensors outside model adapters unless explicitly needed.

Include model-version metadata where relevant.

## Postprocessing

Postprocessing is part of model behavior.

Track significant changes such as:
- BIO entity reconstruction
- whitespace normalization
- amount normalization
- token joining
- confidence aggregation

## Export

Production-serving artifacts may differ from training checkpoints.

Lifecycle:
```text
training checkpoint
      ↓
validated checkpoint
      ↓
export
      ↓
ONNX or serving artifact
      ↓
serving validation
```

Successful serialization is not sufficient.

Validate exported predictions against the source framework within accepted tolerance.

## Model cards

Every model intended for production should have a model card under:
```text
docs/model_cards/
```

Include:
- purpose
- model family
- inputs
- outputs
- training dataset
- evaluation datasets
- primary metrics
- slice results
- known limitations
- unsupported use
- model owner
- release/version information

## Architecture Decision Records

Use:
```text
docs/adr/
```

for significant choices.

Each ADR should contain:
- context
- decision
- alternatives
- consequences

## Notebooks

Notebooks are acceptable for:
- exploratory data analysis
- visualization
- error analysis
- rapid hypothesis investigation

Reusable logic must move to `src/`.

Notebooks must not become the only implementation of production behavior.

## Scripts

Scripts should be thin command-line entry points.

Substantial logic belongs in package modules.

## Testing

### Unit tests
Examples:
- bounding-box normalization
- label alignment
- entity reconstruction
- data validators
- metrics
- configuration parsing
- calibration transformations

### Integration tests
Examples:
- dataset processor + tokenizer
- checkpoint loading
- model prediction
- export
- MLflow adapter

### Regression tests
Use known failure cases.

### Smoke tests
At minimum:
```text
load model
process a small fixture
return expected output schema
```

Unit tests should normally run without a GPU.

## Test fixtures

Keep small, synthetic, or explicitly approved fixtures under version control.

Do not commit sensitive production documents merely to simplify testing.

## CI expectations

Every pull request should run at minimum:
- Ruff formatting check
- Ruff lint
- mypy
- unit tests
- lightweight integration tests
- configuration validation

Do not run expensive full model training on every pull request.

## Error handling

Do not hide failures.

Avoid:
```python
try:
    ...
except Exception:
    pass
```

Use domain-specific exceptions where useful.

Report which dataset/sample and stage failed.

## Logging

Use structured logging.

Useful context:
- run_id
- model_version
- dataset_version
- experiment_name
- document_id where permitted
- stage
- duration

Do not log full raw documents or unnecessary sensitive extracted contents.

## Secrets

Never commit:
- AWS credentials
- MLflow passwords
- API tokens
- database passwords
- private keys

Use environment variables, workload identity, or secret management.

## Dependency discipline

Before adding a package, consider:
- maintenance
- security
- transitive dependencies
- runtime image size
- licensing
- GPU compatibility
- Python compatibility

Explain meaningful new dependencies.

## Hardware awareness

Training and inference code should explicitly handle device selection.

Do not assume CUDA exists during unit tests or developer setup.

GPU-required workflows should fail with a clear message when no supported accelerator is available.

## Performance

Benchmark before optimizing.

Relevant inference measures:
- P50 latency
- P95 latency
- throughput
- GPU utilization
- GPU memory
- batch efficiency

Model selection should consider:
```text
quality
latency
throughput
memory
cost
maintainability
```

## Monitoring readiness

Before a model is production-ready, define:
- model outputs to monitor
- important confidence distributions
- important cohort slices
- known failure indicators
- delayed ground-truth sources
- drift signals
- alert ownership

## Definition of done — experiment

An experiment is complete when:
- hypothesis is documented
- configuration is versioned
- dataset version is known
- run is tracked
- evaluation is complete
- important slices are inspected
- result is recorded
- conclusion is stated

## Definition of done — candidate model

A candidate model requires:
- reproducible training run
- immutable dataset version
- model artifact
- model metadata
- aggregate evaluation
- slice evaluation
- regression evaluation
- comparison to current baseline
- model card update
- export validation when applicable
- inference smoke test
- known limitations documented

## Definition of done — production model

A production model additionally requires:
- approved candidate
- serving artifact
- versioned configuration
- monitoring plan
- deployment strategy
- rollback path
- traceable lineage from production artifact to training run

## Initial implementation sequence

Unless requirements change:

```text
1. Bootstrap repository
2. Configure Python / uv / pyproject
3. Configure Ruff, mypy and pytest
4. Create shared domain types
5. Create dataset manifest schema
6. Add dataset validators
7. Implement pretrained OCR baseline
8. Define canonical OCR output
9. Implement OCR evaluation
10. Produce baseline error analysis
11. Add OCR recognition fine-tuning
12. Add OCR detection fine-tuning if justified
13. Implement document classifier
14. Establish deterministic extraction baseline
15. Define semantic extraction label schema
16. Build LayoutLMv3 dataset pipeline
17. Train layout-aware extractor
18. Implement entity evaluation
19. Add calibration
20. Export serving artifacts
21. Add inference adapters
22. Add monitoring hooks
23. Add production feedback / regression workflow
```

Do not build everything simultaneously.

## First milestone

The first model milestone is:

```text
document input
      ↓
pretrained OCR detector
      ↓
pretrained OCR recognizer
      ↓
canonical OCRDocument
      ↓
evaluation
      ↓
baseline report
```

No OCR fine-tuning yet.
No LayoutLMv3 yet.
No business validation.

The objective is to establish a reproducible baseline before changing weights.

## Coding agent instructions

When working as a coding agent:

1. Read this file before proposing or modifying code.
2. Inspect existing repository conventions before creating new structures.
3. Do not rewrite working architecture without a concrete reason.
4. For substantial changes, briefly state the intended design before editing.
5. Identify the files that will be created or changed.
6. Implement the smallest coherent change.
7. Add or update tests in the same change.
8. Run relevant formatting, linting, type checks, and tests.
9. Report failing checks explicitly.
10. Do not hide errors or weaken tests merely to obtain a green build.
11. Do not introduce new frameworks without explaining why.
12. Avoid unrelated refactoring.
13. Preserve stable internal contracts unless the task explicitly changes them.
14. Keep framework-specific details behind adapters where practical.
15. Never invent dataset statistics, model metrics, or experimental results.
16. Mark illustrative configuration values as illustrative.
17. Do not claim a model improved unless evaluation demonstrates it.
18. Do not introduce production business rules into ML code.
19. Do not train automatically from raw production feedback.
20. Do not create a separate repository or deployable service without an explicit architectural requirement.

## When asked to build a new model

Before coding, determine:
```text
problem
input
output
baseline
dataset
labels
metric
evaluation slices
expected serving contract
```

Start with the smallest meaningful baseline.

## When asked to fine-tune

Before modifying training code:
1. identify the baseline failure,
2. identify whether the failure belongs to this model,
3. state the training hypothesis,
4. identify the dataset change,
5. identify the success metric,
6. identify regression guards.

Then implement.

## When asked to optimize

Do not optimize from intuition alone.

First establish a measurement.

Examples:
```text
latency
throughput
memory
CER
F1
exact match
```

Compare against the baseline.

## When asked to refactor

Preserve:
- model behavior
- public internal interfaces
- reproducibility
- experiment metadata
- training/inference parity

Add or strengthen tests before risky refactors.

## Documentation expectations

Update relevant documentation when changing:
- model interfaces
- label schema
- dataset format
- training commands
- configuration
- evaluation metrics
- model release process
- export process

Code and documentation should describe the same architecture.

## Core principles

Always prefer:

> reproducibility over cleverness

> measured failure analysis over random model changes

> stable contracts over framework leakage

> representative datasets over impressive demos

> slice-level evaluation over one aggregate accuracy number

> versioned datasets over mutable training folders

> controlled retraining over automatic retraining

> independent model lifecycles even when models share one repository

> deterministic business validation outside the ML layer

> simple architecture until scale or ownership proves a need for complexity
# 69. Git Workflow and Repository Hygiene

Treat Git history as part of the engineering artifact.

The repository should read like a real, maintained production ML codebase:
- incremental changes,
- clear intent,
- reviewable commits,
- understandable history,
- no giant unexplained code drops,
- no fake authorship,
- no misleading commit metadata.

Do not attempt to disguise generated code by falsifying contributors, timestamps, or review history.

The goal is professional engineering hygiene, not artificial provenance.

---

# 70. Branching Strategy

Use short-lived branches.

Preferred branch categories:

```text
feature/
fix/
refactor/
test/
docs/
chore/
experiment/
```

Examples:

```text
feature/ocr-baseline
feature/layoutlm-dataset-pipeline
feature/document-classifier
fix/bbox-normalization
fix/ocr-token-ordering
refactor/inference-contracts
test/entity-regression-suite
docs/model-card-layout-v1
chore/configure-ruff
experiment/ocr-low-resolution-augmentation
```

Avoid:

```text
my-branch
temp
test
new
final
changes
codex-work
ai-generated
```

Branch names should communicate the engineering intent.

---

# 71. Main Branch Policy

The main branch should remain in a usable state.

Do not commit directly to `main` for normal development.

Prefer:

```text
branch
  ↓
local validation
  ↓
pull request
  ↓
review
  ↓
merge
```

Emergency fixes may use an expedited path, but should still preserve reviewability where possible.

---

# 72. One Branch, One Intent

A branch should solve one coherent problem.

Good:

```text
feature/ocr-baseline
```

contains:
- OCR adapter
- canonical OCR contract
- config
- tests
- baseline CLI

Bad:

```text
feature/ml-platform
```

contains:
- OCR
- classifier
- LayoutLM
- Docker
- MLflow
- Terraform
- documentation
- unrelated refactors

Do not bundle unrelated work simply because it was generated in one coding session.

---

# 73. Commit Philosophy

Commits should tell the story of the implementation.

Prefer multiple meaningful commits over one enormous commit.

A healthy sequence might look like:

```text
chore: bootstrap Python project with uv

chore: configure ruff, mypy and pytest

feat: add canonical OCR domain types

feat: add PaddleOCR baseline adapter

test: add OCR result contract tests

feat: add OCR baseline evaluation command

docs: document OCR baseline workflow
```

This is preferable to:

```text
feat: implement entire ML platform
```

---

# 74. Commit Size

Each commit should be small enough that another engineer can reasonably review it.

A commit should usually represent one of:

- one capability,
- one refactor,
- one bug fix,
- one test addition,
- one documentation change,
- one configuration change.

Avoid commits that touch dozens of unrelated files.

Large generated changes must be decomposed into logical commits before merge.

---

# 75. Commit Messages

Use Conventional Commit-style messages.

Preferred types:

```text
feat:
fix:
refactor:
test:
docs:
chore:
perf:
ci:
build:
revert:
```

Examples:

```text
feat: add canonical OCR token contract

feat: add PaddleOCR recognition adapter

fix: preserve token order after OCR normalization

fix: reject invalid normalized bounding boxes

test: add regression case for split order reference

refactor: extract shared bbox normalization utility

docs: add annotation guidelines for order references

chore: configure mypy strictness for domain modules

perf: batch OCR recognition crops

ci: run unit tests on Python 3.11
```

---

# 76. Commit Message Quality

Commit messages should explain the engineering change, not the editing process.

Good:

```text
feat: add grouped dataset split by seller template
```

Bad:

```text
update files
```

Bad:

```text
changes from prompt
```

Bad:

```text
AI generated code
```

Bad:

```text
fix stuff
```

Bad:

```text
final changes
```

Do not mention the coding agent in every commit message unless that information is actually relevant to repository policy.

---

# 77. Commit Body

Use a commit body when the change needs context.

Example:

```text
feat: add grouped extraction dataset split

Group examples by seller-template identifier before splitting to
reduce template leakage between training and evaluation cohorts.

The split remains deterministic for a configured random seed.
```

Use the body to explain:
- why the change exists,
- constraints,
- non-obvious behavior,
- migration impact,
- relevant trade-offs.

Do not repeat the diff line-by-line.

---

# 78. Commit Ordering

When implementing a larger feature, prefer dependency order.

Example:

```text
1. chore: add required dependency
2. feat: add domain types
3. feat: add implementation
4. test: add coverage
5. docs: document usage
```

Do not create commits whose intermediate state is unnecessarily broken if it can reasonably be avoided.

Where practical, each commit should build and pass relevant tests independently.

---

# 79. Avoid Generated-Code Dump Commits

A coding agent must not produce a large tree of files and commit everything as one change merely because generation happened in one operation.

Before committing:
1. inspect all generated files,
2. remove unnecessary abstractions,
3. remove unused code,
4. group changes by intent,
5. run checks,
6. commit incrementally.

The final history should represent engineering decisions, not tool invocation boundaries.

---

# 80. No Fake History

Never fabricate:
- commit authors,
- contributor names,
- timestamps,
- code-review approvals,
- issue numbers,
- ticket references,
- release tags,
- changelog entries claiming events that did not happen.

Do not create fake merge commits or fake reviewer attribution to make the repository appear older or more collaborative.

Professional commit hygiene is encouraged.

False provenance is not.

---

# 81. Pull Request Scope

A pull request should have one primary purpose.

Recommended PR size:

```text
small to medium
```

Prefer several focused PRs over one extremely large PR.

Examples:

```text
PR 1 — Repository bootstrap
PR 2 — Canonical OCR contracts
PR 3 — Pretrained OCR baseline
PR 4 — OCR evaluation pipeline
PR 5 — OCR fine-tuning support
```

This is preferable to:

```text
PR — Build complete invoice intelligence platform
```

---

# 82. Pull Request Description

Every meaningful PR should include:

```text
What changed
Why it changed
How it was tested
Risks / limitations
Follow-up work
```

Recommended template:

```markdown
## Summary

Brief description of the change.

## Why

Engineering or product reason for the change.

## Changes

- change 1
- change 2
- change 3

## Validation

- [ ] unit tests
- [ ] integration tests
- [ ] lint
- [ ] type checks
- [ ] model/evaluation checks where applicable

## Risks

Known risks or limitations.

## Follow-up

Work intentionally left for later.
```

---

# 83. Reviewability

Write code so a reviewer can understand the change without reconstructing the coding session.

Before opening a PR:
- remove dead code,
- remove commented-out experiments,
- remove debug prints,
- remove temporary files,
- remove unused imports,
- clean notebook outputs where appropriate,
- ensure names are intentional,
- ensure tests explain behavior.

Do not leave artifacts such as:

```text
tmp.py
scratch.py
try2.py
final_fix.py
debug_output.json
```

unless they are intentionally part of the repository.

---

# 84. Commit Before Refactor

When a feature is working and a refactor is needed, prefer separating them.

Example:

```text
feat: add OCR result normalization

refactor: simplify OCR result normalization pipeline
```

Do not mix:
- behavior change,
- large renaming,
- folder moves,
- formatting changes,
- and test rewrites

in one commit unless strongly justified.

---

# 85. Formatting-Only Changes

Do not mix repository-wide formatting with functional changes.

If a broad formatter change is required, commit it separately.

Example:

```text
chore: apply ruff formatting
```

Then implement functional changes afterward.

This keeps `git blame` and code review useful.

---

# 86. Dependency Changes

Dependency additions or upgrades should be explicit.

Preferred commit:

```text
build: add mlflow dependency for experiment tracking
```

or:

```text
build: upgrade transformers to 4.x
```

Document meaningful compatibility implications.

Avoid silently changing many package versions while implementing unrelated features.

---

# 87. Schema and Contract Changes

Changes to stable schemas should be isolated and obvious.

Examples:

```text
feat: add model_version to OCRDocument contract

feat: version entity label schema
```

Update:
- implementation,
- tests,
- documentation,
- migration/compatibility notes where applicable

in the same PR.

---

# 88. Experimental Work

Experiments may use branches such as:

```text
experiment/layout-hard-negatives
experiment/ocr-augmentation-v2
```

Experimental branches do not automatically justify merging experimental code into main.

Before merge:
- extract reusable code,
- remove throwaway instrumentation,
- convert parameters into configuration,
- add tests,
- document the result.

The merged implementation should look production-ready even if the discovery work was exploratory.

---

# 89. Notebook Hygiene

Before committing notebooks:
- clear unnecessary execution outputs,
- remove credentials,
- remove large embedded artifacts,
- remove irrelevant exploratory cells,
- ensure the notebook has a clear purpose.

Do not use notebook commit history as a substitute for production code history.

If useful code emerges from a notebook, move it into `src/` and commit that code separately.

---

# 90. Generated Files

Do not commit generated files unless there is a clear repository need.

Usually exclude:
- Python caches
- local virtual environments
- model checkpoints
- large datasets
- MLflow local artifacts
- IDE metadata
- local logs
- temporary exports

Maintain an appropriate `.gitignore`.

Possible entries:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
mlruns/
artifacts/
checkpoints/
*.onnx
*.pt
*.pth
*.ckpt
.env
.env.*
```

Do not exclude required example configuration files.

---

# 91. Model Artifact Commits

Do not commit large trained model binaries directly into Git unless the repository has an explicit artifact-management policy requiring it.

Prefer:
- object storage,
- model registry,
- artifact repository,
- MLflow artifact storage.

Git should record enough metadata to locate and reproduce the artifact.

---

# 92. Dataset Commits

Do not commit full production datasets.

Allow only:
- small test fixtures,
- synthetic examples,
- approved sanitized samples,
- manifests,
- annotation schemas.

Large datasets belong outside Git.

---

# 93. Rebase and Merge Hygiene

Before merging:
- update the branch from main,
- resolve conflicts intentionally,
- rerun relevant checks,
- inspect the final diff.

Prefer a repository-standard merge strategy.

For a small team or portfolio project, squash merge is acceptable for very small PRs.

For features where incremental history is useful, preserve meaningful commits.

Do not preserve noisy commits such as:

```text
fix
fix again
oops
wip
final fix
```

Clean them before merge through interactive rebase or squash.

---

# 94. WIP Commits

Temporary local commits are acceptable during development.

Examples:

```text
wip: experiment with OCR normalization
```

But they should normally be cleaned before merging.

The shared history should preserve engineering intent rather than development noise.

---

# 95. Code Review Checklist

Before requesting review, verify:

```text
[ ] branch has one coherent purpose
[ ] commit history is understandable
[ ] no unrelated files changed
[ ] formatting passes
[ ] lint passes
[ ] type checks pass
[ ] unit tests pass
[ ] relevant integration tests pass
[ ] documentation is updated
[ ] no secrets are committed
[ ] no large artifacts are committed
[ ] no dead/debug code remains
[ ] model metrics are not invented
[ ] generated code has been inspected
```

---

# 96. Coding Agent Git Behavior

When a coding agent is used:

1. Do not commit automatically unless explicitly instructed.
2. First show or summarize the intended file changes.
3. Implement the smallest coherent unit.
4. Run relevant checks.
5. Suggest an appropriate branch name.
6. Suggest a commit message.
7. Do not create fake ticket numbers or issue references.
8. Do not create fake contributor/reviewer metadata.
9. Do not bundle unrelated generated changes.
10. If the change is large, propose a commit breakdown before committing.
11. Preserve human-readable code and repository history.
12. Avoid comments such as "generated by AI" unless repository policy explicitly requires them.
13. Do not remove legitimate attribution or licensing information.
14. Do not rewrite Git history without explicit instruction.
15. Do not force-push unless explicitly instructed and safe.

---

# 97. Suggested Branch and Commit Flow for This Repository

For initial implementation, prefer:

```text
main
 │
 ├── chore/bootstrap-project
 │     ├── chore: initialize Python 3.11 project with uv
 │     ├── chore: configure ruff, mypy and pytest
 │     └── chore: add base repository structure
 │
 ├── feature/shared-domain-types
 │     ├── feat: add bounding box domain type
 │     ├── feat: add canonical OCR token contract
 │     └── test: add domain contract tests
 │
 ├── feature/ocr-baseline
 │     ├── feat: add PaddleOCR baseline adapter
 │     ├── feat: add OCR pipeline configuration
 │     ├── test: add OCR smoke coverage
 │     └── docs: document baseline OCR execution
 │
 └── feature/ocr-evaluation
       ├── feat: add OCR evaluation metrics
       ├── feat: add evaluation report generation
       └── test: add metric validation cases
```

This allows the repository to evolve in a realistic, reviewable sequence.

---

# 98. Repository History Principle

A healthy repository history should let an engineer answer:

```text
Why was this introduced?
What problem did it solve?
What changed afterward?
Which model/dataset version did it affect?
Was the change tested?
```

If Git history cannot help answer those questions, improve the commit structure.

The objective is not to make the project merely look realistic.

The objective is to make it behave like a professionally maintained engineering codebase.
