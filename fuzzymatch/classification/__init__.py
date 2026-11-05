"""Classification layer: Fellegi-Sunter and sklearn classifiers."""

from .fs_classifier import FSClassifier
from .sklearn_classifier import SklearnClassifier

__all__ = ["FSClassifier", "SklearnClassifier"]
