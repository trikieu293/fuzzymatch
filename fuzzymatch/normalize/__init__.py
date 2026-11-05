from .legal import DEFAULT_LEGAL_SUFFIXES, strip_legal_suffixes
from .pipeline import Normalizer
from .steps import (
    accent_fold,
    casefold,
    replace_words,
    strip_punctuation,
    strip_whitespace,
    unicode_normalize,
)

__all__ = [
    "DEFAULT_LEGAL_SUFFIXES",
    "Normalizer",
    "accent_fold",
    "casefold",
    "replace_words",
    "strip_legal_suffixes",
    "strip_punctuation",
    "strip_whitespace",
    "unicode_normalize",
]
