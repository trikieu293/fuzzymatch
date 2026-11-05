"""Evaluation metrics for fuzzy matching pipelines."""

from .blocking import block_quality_metrics
from .cluster import cluster_metrics
from .pairwise import pairwise_metrics

__all__ = ["block_quality_metrics", "cluster_metrics", "pairwise_metrics"]
