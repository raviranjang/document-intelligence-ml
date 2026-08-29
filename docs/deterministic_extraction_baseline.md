# Deterministic extraction baseline

The initial semantic extraction baseline applies versioned lexical rules to canonical `OCRDocument`
tokens. It establishes a transparent benchmark for future layout-aware models; it is not a business
validation or reconciliation engine.

## Supported evidence

Version `1.0.0` emits raw candidates for fields with explicit lexical anchors:

- `INVOICE_NUMBER`
- `ORDER_REFERENCE`
- `TOTAL_AMOUNT`
- `INVOICE_DATE`

`SELLER_NAME` is part of the initial entity vocabulary but is not guessed by this baseline. Reliable
seller identification generally needs layout, document context, or approved seller data. Omitting the
field is safer and more measurable than treating the first prominent text as a seller.

Rules are stored in
[`configs/extraction/deterministic_baseline.toml`](../configs/extraction/deterministic_baseline.toml).
Each rule has a stable identifier, entity type, and regular expression with a required named `value`
group. Configuration is strict: unknown fields, duplicate rule identifiers, unsupported entity types,
invalid expressions, and expressions without traceable values are rejected.

## Processing and output

Rules run independently within each OCR page, in configured order. They may match values spanning
adjacent OCR tokens but cannot create evidence across page boundaries. Every result contains:

- the entity type;
- the exact captured OCR value;
- page/token references to the source evidence;
- the rule identifier;
- extractor name, version, and source at document level.

All unique matches are returned in stable page/rule/match order. The baseline does not select one
candidate as authoritative when multiple values match.

`ExtractionDocument` schema `1.0.0` intentionally has no normalized value or confidence. Currency
interpretation, locale-sensitive amount parsing, ambiguous numeric dates, candidate ranking, and
confidence aggregation are separate versioned behaviors. A deterministic rule match is not a
calibrated probability.

## ML and business boundary

The output is document evidence. This repository does not compare order references with an order
system, decide seller ownership, apply payment tolerances, accept an invoice, or route a business
workflow. Those policies belong to downstream services.

## Evaluation status

No approved immutable entity dataset is available, so this repository publishes no precision,
recall, F1, exact-match, or coverage result for the baseline. Synthetic tests verify contracts,
configuration parsing, matching behavior, provenance, page isolation, and candidate preservation;
they do not demonstrate production quality.

The next milestone defines the versioned BIO label schema and annotation semantics. Later entity
evaluation must compare this deterministic baseline and the layout-aware model on identical splits
and important cohorts, using token/entity precision, recall, F1, and field exact match.
