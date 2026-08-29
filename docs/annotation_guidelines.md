# Semantic extraction annotation guidelines

## Contract and version

These guidelines define semantic label schema `1.0.0` for invoice entity extraction. The canonical
label-to-ID mapping is [`semantic_labels_v1.json`](../configs/extraction/semantic_labels_v1.json),
and annotation artifacts must satisfy
[`entity-annotation.schema.json`](../datasets/schemas/entity-annotation.schema.json).

Annotations align to canonical `OCRDocument` schema `1.0.0`, not to independently re-tokenized or
manually corrected text. Page and token indexes must be contiguous, zero-based, and identical to the
source OCR artifact. Bounding boxes and token text must be copied without silent correction.

The tagging scheme is BIO:

- `O` means the token is outside every supported entity.
- `B-<ENTITY>` starts an entity, including every single-token entity.
- `I-<ENTITY>` continues the immediately preceding entity of the same type.

An `I-` label cannot start a page, follow `O`, or follow a different entity type. Entities never span
pages. Label ID zero is permanently assigned to `O`; existing IDs must never be reordered or reused.

## Supported entities

### `ORDER_REFERENCE`

The buyer or customer order identifier that the invoice is billing against, commonly labelled
purchase order, PO number, order reference, or customer order number.

Include only the identifier value. Exclude its lexical label, surrounding punctuation, sales-order
identifiers owned only by the seller, invoice numbers, shipment numbers, and line-item identifiers.
When the document does not establish that an unlabelled identifier is an order reference, label it
`O` and route the case for review.

### `INVOICE_NUMBER`

The identifier assigned to the invoice by its issuer. Include only the complete identifier value,
including meaningful prefixes, suffixes, separators, and leading zeros. Exclude the words invoice,
invoice number, or invoice no; account, tax, order, quotation, credit-note, and customer identifiers;
and barcode text unless the document explicitly identifies it as the invoice number.

### `SELLER_NAME`

The legal or trading name of the organization issuing the invoice. Include the complete name and
legal suffix when present, such as `Example Supplies Ltd`. Exclude logos without OCR text, postal
addresses, tax-registration numbers, contact details, bank names, buyers, ship-to parties, and
marketplaces that are not the invoice issuer.

Prominence or position alone is insufficient when multiple organizations are shown. Ambiguous issuer
identity must be reviewed rather than guessed.

### `TOTAL_AMOUNT`

The final amount payable for the invoice. Include a currency symbol or currency code when it is part
of the same adjacent value expression. Preserve the exact OCR representation, including separators
and decimal digits. Exclude the lexical label and exclude subtotal, tax, discount, balance carried
forward, amount paid, unit price, and line-item totals.

When several totals exist, use the value explicitly identified as grand total, invoice total, total
amount, or amount due. If conflicting candidates are equally authoritative, do not guess; send the
document to annotation review.

### `INVOICE_DATE`

The date the invoice was issued. Include the complete written or numeric value and preserve its OCR
format. Exclude the lexical label, due date, delivery date, order date, tax-point date, service period,
payment date, and dates found only in unrelated references. Ambiguous numeric ordering is not
normalized during annotation.

## Entity boundaries and multi-token values

Annotate the smallest complete token span representing the value. Do not include a preceding label,
colon, or decorative punctuation when it is a separate OCR token. A single-token value receives a
`B-` label. For multiple tokens, the first receives `B-` and every remaining token receives `I-` of
the same entity type.

Examples:

| OCR tokens | Labels |
| --- | --- |
| `Invoice`, `No:`, `INV-001` | `O`, `O`, `B-INVOICE_NUMBER` |
| `Seller:`, `Example`, `Supplies`, `Ltd` | `O`, `B-SELLER_NAME`, `I-SELLER_NAME`, `I-SELLER_NAME` |
| `Amount`, `Due:`, `USD`, `1,250.00` | `O`, `O`, `B-TOTAL_AMOUNT`, `I-TOTAL_AMOUNT` |
| `Invoice`, `Date:`, `29`, `August`, `2026` | `O`, `O`, `B-INVOICE_DATE`, `I-INVOICE_DATE`, `I-INVOICE_DATE` |

Whitespace and punctuation are not normalized in the annotation artifact. Normalization and entity
reconstruction are separate, versioned postprocessing stages.

## OCR errors

Annotate the semantic role visible in context even when OCR misspells a value, as long as the value
boundary is representable by whole OCR tokens. Never edit token text or geometry inside an annotation
to repair OCR. Record systematic OCR failures for the OCR error dataset separately.

If one OCR token merges a field label and value, splits multiple semantic fields into one token, or
otherwise makes the correct boundary impossible at token level, exclude that sample from training
until an explicit partial-token policy and schema version exist. Do not assign a misleading whole-token
label merely to retain the example.

## Missing, repeated, and ambiguous fields

Missing supported fields require no placeholder entity; all unrelated tokens remain `O`. Do not
invent a value from document metadata or another system.

Annotate every occurrence that unambiguously expresses the same supported field, including a repeated
invoice number in a page header. Candidate selection is postprocessing, not annotation. When distinct
values compete for one semantic field and the document does not clearly establish authority, route the
document for review and exclude the disputed field rather than choosing by position.

Overlapping or nested entities are unsupported by BIO schema `1.0.0`. If a token span would require
two entity types, exclude the ambiguous span and raise a schema-review case.

## Annotation exclusions

Exclude an example from training when any required invariant cannot be represented faithfully,
including:

- missing or unreadable source image/OCR artifact;
- corrupt, empty, duplicated, or non-contiguous OCR token data;
- invalid or out-of-page geometry;
- merged tokens that prevent correct entity boundaries;
- unresolved issuer or field ambiguity;
- unsupported language or document type for the approved dataset scope;
- sensitive data that lacks approval for model development.

Exclusion is explicit dataset curation, never a silent validator skip. The reason must be recorded in
the approved annotation workflow without placing confidential content in this public repository.

## Quality review and schema evolution

Every annotation batch must pass structural validation, BIO transition validation, token/label length
checks, geometry checks, and a documented human review sample before dataset versioning. Reviewers
should pay particular attention to seller/buyer confusion, subtotal versus final total, order versus
invoice identifiers, and invoice date versus due date.

Any change to entity meaning, boundary policy, ambiguity policy, or tagging scheme requires a new
label-schema version and updated guidelines. New labels append stable IDs; existing released IDs are
never reassigned. Dataset manifests and model artifacts must record the exact label-schema version.
