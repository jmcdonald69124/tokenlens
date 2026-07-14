"""The TokenLens eval harness.

The proxy can tell you how many tokens compression removed. It cannot tell you
whether the answer was still any good. That question is what this package
answers, and until it does, every compression rung above the safe floor is a
guess.

The loop (DESIGN.md §2, "calibration"):

    golden task  ->  cleartext request  ->  model  ->  answer A
                 ->  compressed request ->  model  ->  answer B
                                                          |
                    answer A + answer B + reference -> judge -> retention

Run every task through every rung/rate, and you get a compression-ratio vs
quality curve — and from the curve, a policy: the most aggressive setting per
task class whose quality still lands within tolerance of cleartext.
"""

from .harness import Arm, CaseResult, EvalReport, run_eval
from .tasks import GoldenTask, load_tasks

__all__ = [
    "Arm",
    "CaseResult",
    "EvalReport",
    "GoldenTask",
    "load_tasks",
    "run_eval",
]
