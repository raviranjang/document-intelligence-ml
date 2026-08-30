# Serving artifact export

Serving bundles are immutable, framework-neutral release directories. Export packages existing
validated files; it does not train a model, select a checkpoint, or treat successful serialization as
prediction parity.

Each `manifest.json` records model name, independent semantic version, family, task, source commit,
configuration version, optional training run and dataset lineage, evaluation report reference, output
contract, creation time, and a role-based file inventory. Every file records a relative POSIX path,
media type, byte size, and SHA-256 checksum.

Export writes into a staging directory and atomically renames it only after all source files have been
copied and hashed. Existing destinations are never overwritten. Artifact roles and filenames must be
unique, sources must already exist, and paths may not escape the bundle.

`validate_serving_bundle` must run before an inference adapter loads any file. It rejects unsupported
manifests, missing files, path traversal, size changes, and checksum changes. Training-only code is not
part of the bundle.

The `MODEL`, `TOKENIZER`, `PROCESSOR`, `LABELS`, `CALIBRATION`, and `CONFIG` roles support independent
model lifecycles. A release includes only roles required by that model. Deterministic baselines may
have no training run or dataset, while trained releases must retain those lineage fields under the
candidate-model release policy.

Framework export such as checkpoint-to-ONNX conversion remains model-adapter work. It must compare
exported predictions with the validated source framework within an explicitly approved tolerance.
This repository has no trained LayoutLMv3 checkpoint, so it publishes no neural serving bundle or
parity result.
