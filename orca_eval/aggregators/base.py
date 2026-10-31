from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..agents import AgentPool, Proposal
from ..audit import AuditOracle
from ..tasks import Task


@dataclass
class Env:
    rng: np.random.Generator
    pool: AgentPool
    oracle: AuditOracle
    t: int = 0
    T: int = 0


@dataclass
class Result:
    answer: Optional[str]
    n_audits: int = 0


class Aggregator:
    name = "base"

    def reset(self, env: Env) -> None:
        pass

    def aggregate(self, task: Task, proposals: List[Proposal], env: Env) -> Result:
        raise NotImplementedError


def group_by_answer(proposals: List[Proposal]) -> Dict[str, List[int]]:
    g: Dict[str, List[int]] = defaultdict(list)
    for p in proposals:
        g[p.answer].append(p.agent_id)
    return g


def plurality(scores: Dict[str, float], rng: np.random.Generator) -> str:
    """argmax with uniformly random tie-breaking."""
    best = max(scores.values())
    tied = sorted(k for k, v in scores.items() if abs(v - best) < 1e-12)
    return tied[rng.integers(len(tied))] if len(tied) > 1 else tied[0]
