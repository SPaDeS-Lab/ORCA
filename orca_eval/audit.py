"""Audit oracle with a global cost budget."""
from __future__ import annotations

from .tasks import Task


class AuditOracle:
    """Exact verifier for verifiable tasks. Each check of one candidate costs
    ``task.verify_cost`` and is charged against a shared budget."""

    def __init__(self, budget: float):
        self.budget = float(budget)
        self.spent = 0.0
        self.n_checks = 0

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    def can_afford(self, task: Task) -> bool:
        return task.verifiable and task.verify_cost <= self.remaining + 1e-9

    def check(self, task: Task, candidate: str) -> bool:
        if not self.can_afford(task):
            raise RuntimeError("audit not affordable / task unverifiable")
        self.spent += task.verify_cost
        self.n_checks += 1
        return candidate == task.answer
