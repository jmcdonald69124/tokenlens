"""LLM-as-judge scoring: did compression cost us anything?

The metric we want is *retention*, not raw quality:

    retention = grade(answer from compressed input) / grade(answer from cleartext input)

Grading the compressed answer on its own tells you nothing useful — a 71/100 is
only alarming if cleartext scored 95, and perfectly fine if cleartext also
scored 71 because the task is hard. Compression is only ever accused of the
*difference*. So the judge always sees both answers and grades them together.

Two things keep the judge honest:

  * The judge is shown the ORIGINAL, uncompressed request. It is never told
    which answer came from the compressed side. Otherwise you are measuring the
    judge's prior about compression, not the compression.
  * The A/B position is randomised per case, deterministically from the case
    key, so a rerun of the same eval shuffles identically (reproducible) while
    position bias cannot systematically favour the compressed arm.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .api import ApiError, Client, text_of

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

# Anthropic structured outputs: no numeric constraints (minimum/maximum) and no
# string-length constraints are supported, so the 0-100 range lives in the
# description and is clamped on our side. `reasoning` is deliberately the first
# property: it gives the judge somewhere to think before it commits to a number.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Two or three sentences comparing the two answers against the "
                           "request and reference: what each gets right, what each omits, "
                           "invents, or gets wrong.",
        },
        "grade_a": {
            "type": "integer",
            "description": "Quality of ANSWER A on a 0-100 scale. 100 = fully correct, "
                           "complete, and follows every instruction in the request. "
                           "Deduct for missing required facts, wrong facts, invented "
                           "facts, and violated formatting or content constraints.",
        },
        "grade_b": {
            "type": "integer",
            "description": "Quality of ANSWER B on the same 0-100 scale, graded by exactly "
                           "the same standard as ANSWER A.",
        },
        "note": {
            "type": "string",
            "description": "One short sentence naming the single most important difference "
                           "between the two answers, or 'no material difference'.",
        },
    },
    "required": ["reasoning", "grade_a", "grade_b", "note"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a strict, impartial grader. You are given a request, optionally a reference "
    "answer, and two candidate answers to that request. Grade each candidate independently "
    "on a 0-100 scale for how well it fulfils the request.\n\n"
    "Rules:\n"
    "- The reference answer, when present, is the ground truth. An answer that contradicts "
    "it is wrong, however fluent it is.\n"
    "- Judge substance and instruction-following, not style, length, or tone. A terse "
    "correct answer beats a verbose one.\n"
    "- Missing a required fact, inventing a fact, or violating an explicit formatting or "
    "content constraint in the request are all serious deductions.\n"
    "- The two answers are in random order. Do not assume anything about where either "
    "came from, and do not let order influence you.\n"
    "- Grade the two answers by the same standard. If they are equally good, give them the "
    "same grade — do not manufacture a difference."
)

# Judge input is bounded: a 200k-token context would otherwise make the judge
# cost more than the thing it is judging.
_MAX_REQUEST_CHARS = 8000
_MAX_ANSWER_CHARS = 6000


@dataclass
class Judgement:
    cleartext_grade: int
    compressed_grade: int
    note: str
    reasoning: str
    swapped: bool  # True when the compressed answer was shown as ANSWER A

    @property
    def retention(self) -> float:
        """Compressed quality as a fraction of cleartext quality.

        Capped at 1.0: compression getting *luckier* than cleartext on some case
        is noise, and letting it above 1.0 would let one lucky case paper over a
        real regression elsewhere in the mean.
        """
        if self.cleartext_grade <= 0:
            # Cleartext scored zero — the task is broken, not the compression.
            return 1.0
        return min(1.0, self.compressed_grade / self.cleartext_grade)

    @property
    def delta(self) -> int:
        return self.compressed_grade - self.cleartext_grade


def _trim(text: str, limit: int) -> str:
    """Keep the head and tail — the instruction usually lives at one end."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n\n[... {len(text) - limit:,} characters elided by the judge ...]\n\n{text[-tail:]}"


def _swap_for(key: str) -> bool:
    """Deterministic per-case coin flip for A/B position."""
    return hashlib.sha256(key.encode("utf-8")).digest()[0] % 2 == 1


def _clamp(value) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def build_prompt(request_text: str, reference: str | None,
                 answer_a: str, answer_b: str) -> str:
    parts = [
        "<request>",
        _trim(request_text, _MAX_REQUEST_CHARS),
        "</request>",
    ]
    if reference:
        parts += ["", "<reference_answer>", reference.strip(), "</reference_answer>"]
    parts += [
        "",
        "<answer_a>",
        _trim(answer_a, _MAX_ANSWER_CHARS) or "(the model returned nothing)",
        "</answer_a>",
        "",
        "<answer_b>",
        _trim(answer_b, _MAX_ANSWER_CHARS) or "(the model returned nothing)",
        "</answer_b>",
        "",
        "Grade ANSWER A and ANSWER B.",
    ]
    return "\n".join(parts)


def grade(
    client: Client,
    *,
    case_key: str,
    request_text: str,
    cleartext_answer: str,
    compressed_answer: str,
    reference: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 1024,
) -> Judgement:
    """Grade a cleartext/compressed answer pair. Raises ApiError on failure."""
    swapped = _swap_for(case_key)
    answer_a, answer_b = (
        (compressed_answer, cleartext_answer) if swapped
        else (cleartext_answer, compressed_answer)
    )

    payload = {
        "model": judge_model,
        "max_tokens": max_tokens,
        "system": _SYSTEM,
        "messages": [{
            "role": "user",
            "content": build_prompt(request_text, reference, answer_a, answer_b),
        }],
        "output_config": {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    }

    parsed = parse_response(client.messages(payload))
    grade_a, grade_b = parsed["grade_a"], parsed["grade_b"]
    compressed, cleartext = (
        (grade_a, grade_b) if swapped else (grade_b, grade_a)
    )
    return Judgement(
        cleartext_grade=cleartext,
        compressed_grade=compressed,
        note=parsed["note"],
        reasoning=parsed["reasoning"],
        swapped=swapped,
    )


def parse_response(message: dict) -> dict:
    """Pull the judge's JSON out of a Messages API response and sanitise it."""
    import json

    raw = text_of(message)
    if not raw:
        raise ApiError(None, "judge returned an empty response")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ApiError(None, f"judge did not return JSON: {raw[:200]}") from None
    if not isinstance(data, dict):
        raise ApiError(None, "judge returned JSON that is not an object")
    return {
        "grade_a": _clamp(data.get("grade_a")),
        "grade_b": _clamp(data.get("grade_b")),
        "note": str(data.get("note", "")).strip(),
        "reasoning": str(data.get("reasoning", "")).strip(),
    }
