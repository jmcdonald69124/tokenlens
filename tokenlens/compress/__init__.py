"""Cache-aware, content-safe compression for Anthropic requests.

Rungs:
  0-2  safe floor   (whitespace/decorative, deterministic, model-free)
  5    llmlingua2   (small local token-pruning model, optional)
"""

from .pipeline import CompressionResult, compress_request

__all__ = ["CompressionResult", "compress_request"]
