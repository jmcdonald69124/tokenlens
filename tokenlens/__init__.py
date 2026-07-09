"""TokenLens — a local BYOK proxy that measures (and, later, compresses) the
tokens sent to a frontier model.

Milestone 1: forward every request unchanged to the Anthropic API and log the
token counts + cost. This establishes the baseline the compression ladder is
measured against.
"""

__version__ = "0.1.0"
