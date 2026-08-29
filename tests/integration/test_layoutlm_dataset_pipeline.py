"""Integration coverage for annotation-to-LayoutLM feature preparation."""

from pathlib import Path

from document_intelligence.extraction import load_semantic_label_schema
from document_intelligence.extraction.layoutlm import (
    EntityAnnotationLoader,
    LayoutLMDatasetBuilder,
    TokenizerEncoding,
    load_layoutlm_dataset_config,
)

ANNOTATION_SCHEMA_PATH = Path("datasets/schemas/entity-annotation.schema.json")
ANNOTATION_FIXTURE_PATH = Path("datasets/fixtures/synthetic-entity-annotation.json")
LABEL_SCHEMA_PATH = Path("configs/extraction/semantic_labels_v1.json")
DATASET_CONFIG_PATH = Path("configs/extraction/layoutlmv3_dataset.toml")


class FixtureTokenizer:
    """Small tokenizer double that keeps one subword per source token."""

    def encode_words(self, words: tuple[str, ...], *, max_length: int) -> TokenizerEncoding:
        input_ids = (101, *(range(1000, 1000 + len(words))), 102)
        if len(input_ids) > max_length:
            raise ValueError("fixture exceeds max_length")
        return TokenizerEncoding(
            input_ids=tuple(input_ids),
            attention_mask=(1,) * len(input_ids),
            word_ids=(None, *range(len(words)), None),
            truncated=False,
        )


def test_loader_and_builder_create_model_ready_page_features() -> None:
    schema = load_semantic_label_schema(LABEL_SCHEMA_PATH)
    loader = EntityAnnotationLoader.from_schema_file(ANNOTATION_SCHEMA_PATH, schema)
    document = loader.load(ANNOTATION_FIXTURE_PATH)
    builder = LayoutLMDatasetBuilder(
        tokenizer=FixtureTokenizer(),
        label_schema=schema,
        config=load_layoutlm_dataset_config(DATASET_CONFIG_PATH),
    )

    examples = builder.build_document(document)

    assert len(examples) == 1
    assert examples[0].document_id == "synthetic-invoice-001"
    assert len(examples[0].features.input_ids) == 6
    assert examples[0].features.bounding_boxes[0] == (0, 0, 0, 0)
    assert examples[0].features.bounding_boxes[1] == (20, 14, 100, 29)
    assert examples[0].features.label_ids[1] == schema.label_to_id["B-SELLER_NAME"]
