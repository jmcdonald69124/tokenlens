"""Golden task loading.

A golden task is the smallest unit that can falsify a compression setting:

    question   what we ask the model to do
    context    the bulky prose the compressor is allowed to chew on
    reference  the answer a competent human would accept
    task_class the bucket the policy is chosen per (long-doc QA behaves very
               differently from extraction under pruning, so they must not be
               averaged together)

The bundled set (goldens/default.jsonl) is small, synthetic and self-contained
on purpose: it ships in the package, needs no download, and every reference
answer is checkable against the context. It is a smoke-grade calibration set,
not a benchmark — point `--tasks` at your own traffic-shaped set before you
trust a policy in production.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

DEFAULT_TASKS = os.path.join(os.path.dirname(__file__), "goldens", "default.jsonl")

_REQUIRED = ("id", "task_class", "question", "context", "reference")


@dataclass(frozen=True)
class GoldenTask:
    id: str
    task_class: str
    question: str
    context: str
    reference: str

    @property
    def summary(self) -> dict:
        """A lightweight, publishable description — what the task asks and how big
        its context is, without dragging the full prose onto the dashboard."""
        return {
            "id": self.id,
            "task_class": self.task_class,
            "question": self.question,
            "reference": self.reference,
            "context_chars": len(self.context),
        }

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "GoldenTask":
        missing = [k for k in _REQUIRED if not d.get(k)]
        if missing:
            raise ValueError(f"{where}: task is missing {', '.join(missing)}")
        return cls(
            id=str(d["id"]),
            task_class=str(d["task_class"]),
            question=str(d["question"]),
            context=str(d["context"]),
            reference=str(d["reference"]),
        )


def load_tasks(path: str | None = None, classes: list[str] | None = None) -> list[GoldenTask]:
    """Load a JSONL golden set. `classes` filters by task_class."""
    path = path or DEFAULT_TASKS
    tasks: list[GoldenTask] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({e})") from None
            tasks.append(GoldenTask.from_dict(raw, f"{path}:{lineno}"))

    if not tasks:
        raise ValueError(f"{path}: no tasks found")

    if classes:
        wanted = set(classes)
        tasks = [t for t in tasks if t.task_class in wanted]
        if not tasks:
            raise ValueError(f"{path}: no tasks match classes {sorted(wanted)}")

    seen = set()
    for t in tasks:
        if t.id in seen:
            raise ValueError(f"{path}: duplicate task id {t.id!r}")
        seen.add(t.id)
    return tasks
