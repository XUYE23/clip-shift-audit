"""Evaluate cached vision-language embeddings without loading a model."""
from .core import audit, cosine_logits, confidence, fit_threshold, auroc
__all__ = ["audit", "cosine_logits", "confidence", "fit_threshold", "auroc"]
