"""LayoutLMv3 dataset preparation and shared transforms."""

from document_intelligence.extraction.layoutlm.alignment import (
    IGNORE_LABEL_ID,
    SPECIAL_TOKEN_BOX,
    AlignedPageFeatures,
    TokenizerEncoding,
    align_page_encoding,
)
from document_intelligence.extraction.layoutlm.annotations import (
    AnnotatedDocument,
    AnnotatedPage,
    AnnotatedToken,
    AnnotationIssue,
    AnnotationValidationError,
    EntityAnnotationLoader,
)
from document_intelligence.extraction.layoutlm.config import (
    LayoutLMDatasetConfig,
    load_layoutlm_dataset_config,
)
from document_intelligence.extraction.layoutlm.dataset import (
    LayoutLMDatasetBuilder,
    LayoutLMPageExample,
    WordTokenizer,
)
from document_intelligence.extraction.layoutlm.geometry import (
    NormalizedBoundingBox,
    normalize_bounding_box,
)

__all__ = [
    "IGNORE_LABEL_ID",
    "SPECIAL_TOKEN_BOX",
    "AlignedPageFeatures",
    "AnnotatedDocument",
    "AnnotatedPage",
    "AnnotatedToken",
    "AnnotationIssue",
    "AnnotationValidationError",
    "EntityAnnotationLoader",
    "LayoutLMDatasetBuilder",
    "LayoutLMDatasetConfig",
    "LayoutLMPageExample",
    "NormalizedBoundingBox",
    "TokenizerEncoding",
    "WordTokenizer",
    "align_page_encoding",
    "load_layoutlm_dataset_config",
    "normalize_bounding_box",
]
