# LayoutLMv3 dataset pipeline

This pipeline converts versioned semantic annotations into deterministic page-level textual, spatial,
and supervision features. It deliberately has no PyTorch or Transformers dependency. The next training
milestone supplies pinned model/tokenizer and image-processor adapters behind the narrow interfaces
defined here.

## Inputs and validation

`EntityAnnotationLoader` validates annotation JSON against the versioned artifact schema, then enforces
operational invariants that JSON Schema cannot express:

- annotation and label-schema versions match;
- page and token indexes are contiguous and zero-based;
- token text is not blank;
- bounding boxes have positive area and remain inside the page image;
- every page label sequence has valid BIO transitions.

Structural problems are reported before conversion. Operational problems are aggregated into stable
issue codes; examples are never silently skipped. Manifest artifact existence, checksums, duplicate
records, and grouped split leakage remain the responsibility of `DatasetManifestValidator` before this
task-specific loader runs.

## Geometry normalization

`normalize_bounding_box` is the canonical training/inference transform from source-image coordinates
to LayoutLM's integer `0..1000` grid. It uses floor for minimum edges and ceiling for maximum edges,
which preserves positive source boxes after rounding. Coordinates outside the page are rejected rather
than clipped. Zero or invalid image dimensions fail immediately.

This transform must be reused by the inference adapter; duplicating the formula would risk silent
training/serving skew.

## Subword alignment

The tokenizer adapter receives already split OCR words and returns input IDs, attention mask, source
word IDs, and an explicit truncation flag. Alignment behavior is versioned as `propagate_bio`:

- the first subword receives the source token label;
- continuation pieces repeat `O` or `I-ENTITY`;
- a continuation of `B-ENTITY` is converted to `I-ENTITY`;
- special and padding positions receive box `[0, 0, 0, 0]` and label ID `-100`;
- every subword for a source token receives the same normalized box.

Right truncation may omit only a contiguous suffix of source words. Missing middle words, out-of-range
word IDs, non-monotonic alignment, an empty encoded page, or a truncation flag inconsistent with the
observed words is an error. This prevents empty-token behavior and tokenizer truncation from silently
shifting labels.

## Feature contract

`LayoutLMDatasetBuilder` emits one `LayoutLMPageExample` per annotated page. It retains document/page
identity and source image dimensions alongside:

- input IDs;
- attention mask;
- normalized bounding boxes;
- aligned label IDs;
- source word IDs;
- truncation status.

The versioned transform configuration is
[`configs/extraction/layoutlmv3_dataset.toml`](../configs/extraction/layoutlmv3_dataset.toml).
Changing maximum length, geometry scale, ignored label ID, subword policy, or truncation side changes
dataset behavior and requires a configuration/schema version review.

## Training handoff

This PR does not download `microsoft/layoutlmv3-base`, select an unpinned model revision, decode PDFs,
or create image tensors. Training must add a pinned tokenizer/image processor, join approved manifest
document artifacts to validated annotation pages, record dataset and configuration lineage, and test
the concrete adapter against these framework-independent contracts. Large images and datasets remain
outside Git.

No training or quality result is claimed by this pipeline.

The downstream reconstruction and evaluation contract is documented in
[`entity_evaluation.md`](entity_evaluation.md). Actual LayoutLMv3 training remains deferred until an
approved immutable semantic dataset is available.
