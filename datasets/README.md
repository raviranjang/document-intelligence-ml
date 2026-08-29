# Datasets

Git may contain dataset schemas, immutable manifests, checksums, documentation, and small
synthetic or explicitly approved fixtures. Full source datasets and production documents belong
in controlled external storage.

Every training or evaluation run must reference an immutable dataset version. Released datasets
must never be mutated in place.

## Dataset manifest

[`schemas/dataset-manifest.schema.json`](schemas/dataset-manifest.schema.json) defines the
version `1.0.0` manifest contract using JSON Schema Draft 2020-12. A manifest records:

- an immutable semantic dataset version and creation timestamp;
- supported ML tasks and source lineage;
- the split strategy, including grouping keys used to prevent leakage;
- stable record identifiers, training/validation/test membership, and evaluation cohorts;
- document and optional annotation URIs with SHA-256 checksums and media types.

Artifact URIs may be relative to a controlled dataset root or point to remote object storage. Do
not place credentials, signed URLs, personal data, customer identifiers, or machine-specific
absolute paths in a manifest. Group values must be pseudonymized when their source values are
sensitive.

The file at [`fixtures/synthetic-ocr-manifest.json`](fixtures/synthetic-ocr-manifest.json) is a
sanitized schema example. Its artifact references and checksums are illustrative; the referenced
files are intentionally not production data.

Schema validation establishes structural compatibility. Operational dataset validation—such as
checking duplicate records, artifact availability, checksum integrity, annotation correctness,
and split leakage—is handled separately and must run before training or evaluation.
