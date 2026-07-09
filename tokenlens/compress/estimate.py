"""Local, dependency-free helpers: rough token estimation and a conservative
"is this natural-language prose?" heuristic.

Token estimates are approximate (used only for the savings log line). The
ground-truth token count always comes from the `usage` block in the real API
response.
"""

from __future__ import annotations

import re

# ~4 characters per token is a reasonable rule of thumb for English prose.
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


# Signals that a text block is code / structured data rather than prose. We err
# strongly toward "not prose" — compressing code or structured data is unsafe.
_CODE_FENCE = re.compile(r"```")
_LOOKS_JSON = re.compile(r"^\s*[{\[]")
_LOOKS_XML = re.compile(r"^\s*<[a-zA-Z/!?]")
_LOOKS_DIFF = re.compile(r"(?m)^(diff --git|@@ |[+-]{3} )")
# An indented line (tab or 4+ leading spaces before non-space) is a strong
# code/structured-content signal — prose almost never indents this way.
_INDENTED_LINE = re.compile(r"(?m)^(\t| {4,})\S")
# characters/keywords that are common in code but rare in prose
_CODEY = re.compile(r"[{}<>|`\\]|=>|::|;\s*$")
# Only unambiguous code syntax — narrow enough not to fire on prose that
# happens to contain words like "from", "return", or "class".
_CODE_KEYWORD = re.compile(
    r"\b(def|function|func|fn)\s+\w+\s*\("      # function definitions
    r"|\b(const|let|var)\s+\w+\s*="             # variable declarations
    r"|\bimport\s+[\w.]+"                        # imports
    r"|\bfrom\s+[\w.]+\s+import\b"               # python from-import
)


def is_prose(text: str) -> bool:
    """Conservative test: return True only when we're confident this is prose."""
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if _CODE_FENCE.search(text):
        return False
    if _LOOKS_JSON.match(stripped) or _LOOKS_XML.match(stripped):
        return False
    if _LOOKS_DIFF.search(text):
        return False
    if _INDENTED_LINE.search(text):
        return False
    if _CODE_KEYWORD.search(text):
        return False
    # A few "codey" punctuation markers is telling even in short text.
    if len(_CODEY.findall(text)) >= 2:
        return False
    # Several very long unbroken tokens (paths, hashes, base64) suggest non-prose.
    # A single long token (a URL, a divider) shouldn't disqualify a whole doc.
    long_tokens = sum(1 for tok in text.split() if len(tok) > 40)
    if long_tokens >= 3 or any(len(tok) > 200 for tok in text.split()):
        return False
    return True
