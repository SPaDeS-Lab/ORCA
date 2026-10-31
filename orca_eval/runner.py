"""Evaluation runner for real Claude Haiku 5.5 experiments on ORCA and baselines.

Protocol per paper:
- n <= 10 agents (default n=10), all instantiated with Claude Haiku 5.5.
- Tasks arrive online in random order.
- Each agent submits one proposal per task.
- Compromised fraction f in [0.1, 0.5] (default f=0.3) or f=0.
- All methods receive identical agent proposals within a seed (paired comparison).
- Exact audit verifier with shared global budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .agents import AgentPool, ClaudeHaikuClient, Proposal
from .aggregators import Env, ORCA, all_methods
from .audit import AuditOracle
from .tasks import Task, full_audit_cost

TIERS = ("cheap", "medium", "expensive", "unverifiable")


@dataclass(frozen=True)
class Scenario:
    n_agents: int = 10
    adv_frac: float = 0.0
    attack: str = "none"
    budget_frac: float = 0.15     # audit budget as a fraction of the full-audit cost
    orca_loss: float = 40.0       # L=40 in paper
    model: str = "claude-3-5-haiku-20241022"
    mock: bool = False            # True for offline dry-run testing without API charges
    error_corr: float = 0.0       # Deprecated simulation param, kept for backward compat


def run_scenario(tasks: List[Task], sc: Scenario, seed: int,
                 client: Optional[ClaudeHaikuClient] = None,
                 progress_cb: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
    """Execute one (scenario, seed) run over tasks."""
    rng = np.random.default_rng([seed, 99])
    order = rng.permutation(len(tasks))
    seq = [tasks[i] for i in order]
    T = len(seq)

    # Initialize agent pool with Claude Haiku 5.5 agents
    if client is None:
        client = ClaudeHaikuClient(model=sc.model, mock=sc.mock)
    pool = AgentPool(n_agents=sc.n_agents, adv_frac=sc.adv_frac, attack=sc.attack,
                     seed=seed, client=client)

    # Online proposal collection: Claude Haiku agents propose answers
    proposals: List[List[Proposal]] = []
    prop_rng = np.random.default_rng([seed, 5])
    for t, task in enumerate(seq):
        props = pool.propose_all(task, t, T, prop_rng)
        proposals.append(props)
        if progress_cb is not None and (t + 1) % 50 == 0:
            progress_cb(t + 1, T)

    budget = sc.budget_frac * full_audit_cost(tasks)

    rows = []
    for m_idx, method in enumerate(all_methods(loss=sc.orca_loss)):
        env = Env(rng=np.random.default_rng([seed, 1000 + m_idx]), pool=pool,
                  oracle=AuditOracle(budget), T=T)
        method.reset(env)
        correct = np.zeros(T, dtype=bool)
        for t, task in enumerate(seq):
            env.t = t
            res = method.aggregate(task, proposals[t], env)
            correct[t] = (res.answer == task.answer)

        row = dict(asdict(sc), seed=seed, method=method.name,
                   acc=float(correct.mean()), audit_cost=float(env.oracle.spent),
                   n_audits=int(env.oracle.n_checks))

        for tier in TIERS:
            idx = [i for i, t_ in enumerate(seq) if t_.tier == tier]
            row[f"acc_{tier}"] = float(correct[idx].mean()) if idx else 0.0

        for kind in sorted({t_.kind for t_ in seq}):
            idx = [i for i, t_ in enumerate(seq) if t_.kind == kind]
            row[f"kind_{kind}"] = float(correct[idx].mean()) if idx else 0.0

        if isinstance(method, ORCA) and pool.adversary_ids:
            rep = method.reputation
            adv = np.array([i in pool.adversary_ids for i in range(len(pool))])
            hon_r, adv_r = rep[~adv], rep[adv]
            if len(hon_r) and len(adv_r):
                gt = (hon_r[:, None] > adv_r[None, :]).mean()
                eq = (hon_r[:, None] == adv_r[None, :]).mean()
                row["det_auc"] = float(gt + 0.5 * eq)
            row["rep_adv"] = float(adv_r.mean()) if len(adv_r) else float("nan")
            row["rep_hon"] = float(hon_r.mean()) if len(hon_r) else float("nan")

        rows.append(row)
    return rows
