# OCR baseline error analysis

This module turns versioned OCR evaluation samples into a reproducible, aggregate error taxonomy.
It does not contain a benchmark result: values may be published only after running against an
approved immutable dataset with the provenance required by [`ocr_evaluation.md`](ocr_evaluation.md).

## Failure ownership

The report assigns only failures that can be measured from OCR references:

| Stage | Category | Meaning |
| --- | --- | --- |
| Detection | `missed_region` | A reference region has no one-to-one match at the configured IoU threshold. |
| Detection | `spurious_region` | A predicted region has no one-to-one match at the configured IoU threshold. |
| Recognition | `character_substitution` | The minimum alignment contains substitutions only. |
| Recognition | `character_insertion` | The minimum alignment contains insertions only. |
| Recognition | `character_deletion` | The minimum alignment contains deletions only. |
| Recognition | `mixed_character_edits` | The minimum alignment contains more than one edit-operation type. |
| Recognition | `identifier_mismatch` | A labelled identifier is not an exact, case-sensitive match. |

`semantic_extraction` is a distinct failure stage, but this OCR analysis never emits it. A correct
OCR token mapped to the wrong business field is an extraction error and must be measured against
the later entity-label schema. OCR must not be fine-tuned to repair that failure.

## Deterministic classification

Detection diagnostics reuse the same maximum-cardinality, one-to-one region assignment as the
published detection metrics. Candidate pairs must meet the report's IoU threshold. Assignment
details expose matched indices and unmatched reference/prediction indices so metrics and analysis
cannot silently disagree.

Recognition diagnostics use a minimum Levenshtein alignment over Unicode code points. Ties are
resolved consistently in substitution, deletion, insertion order. Exact predictions produce no
error observation. Identifier mismatches are counted in addition to their character-edit category
because they identify a high-impact cohort, not a mutually exclusive edit operation.

## Report contract

`build_ocr_error_analysis_report` requires measured recognition or detection samples and emits
schema version `1.0.0`. The aggregate categories and cohort slices are ranked by descending error
occurrences, followed by stable stage/category ordering. Each row records both:

- `affected_samples`: distinct evaluation samples containing the category;
- `occurrences`: regions or edit operations measured for the category.

The report retains evaluated cohorts with no observed failures as empty slices. It includes dataset,
model, source-commit, evaluation-config, and IoU-threshold lineage. It intentionally excludes raw
reference text, predictions, document images, and sample identifiers, making the aggregate artifact
safer to review without treating it as permission to publish source data.

## Decision gate

A baseline report should rank aggregate categories and important cohorts such as scan quality,
language, template family, identifier type, and document source. A fine-tuning proposal must cite a
measured failure from that report, show that the failure belongs to the proposed model component,
define the immutable train/validation/test split, and state the success and regression criteria.

Until an approved evaluation dataset is available, this repository makes no claims about baseline
quality and does not justify detection or recognition fine-tuning.
