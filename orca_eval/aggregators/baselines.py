"""The five baselines."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from ..agents import Proposal
from ..tasks import Task
from .base import Aggregator, Env, Result, group_by_answer, plurality


class MajorityVote(Aggregator):
    name = "Majority vote"

    def aggregate(self, task, proposals, env):
        cnt = Counter(p.answer for p in proposals)
        return Result(plurality({k: float(v) for k, v in cnt.items()}, env.rng))


class MultiAgentDebate(Aggregator):
    """Agents see each other's answers for ``rounds`` rounds, may revise
    (conformity + persuasion by correct answers), then majority vote."""
    name = "Multi-agent debate"

    def __init__(self, rounds: int = 2):
        self.rounds = rounds

    def aggregate(self, task, proposals, env):
        cur = {p.agent_id: p.answer for p in proposals}
        conf = {p.agent_id: p.confidence for p in proposals}
        for _ in range(self.rounds):
            new = {}
            for ag in env.pool.agents:
                peers = [a for i, a in cur.items() if i != ag.id]
                new[ag.id] = ag.revise(task, cur[ag.id], peers, conf[ag.id], env.rng) if peers else cur[ag.id]
            cur = new
        cnt = Counter(cur.values())
        return Result(plurality({k: float(v) for k, v in cnt.items()}, env.rng))


class LastWriterWins(Aggregator):
    """Agents overwrite a shared answer slot in sequence; the last write stands.
    ``order='random'``: random write order.  ``order='adv_last'``: worst case in
    which adversaries are scheduled last."""
    name = "Last-writer-wins"

    def __init__(self, order: str = "random"):
        self.order = order

    def aggregate(self, task, proposals, env):
        ids = list(range(len(proposals)))
        env.rng.shuffle(ids)
        if self.order == "adv_last":
            ids.sort(key=lambda i: i in env.pool.adversary_ids)
        return Result(proposals[ids[-1]].answer)


class AgreementWeightedVote(Aggregator):
    """Unsupervised reliability: an agent's weight is its running rate of
    agreeing with the (leave-one-out) plurality of the other agents."""
    name = "Agreement-weighted"

    def reset(self, env):
        n = len(env.pool)
        self.agree = np.zeros(n)
        self.cnt = np.zeros(n)

    def aggregate(self, task, proposals, env):
        w = (self.agree + 1.0) / (self.cnt + 2.0)          # prior 0.5
        scores: Dict[str, float] = {}
        for p in proposals:
            scores[p.answer] = scores.get(p.answer, 0.0) + w[p.agent_id]
        answer = plurality(scores, env.rng)
        for p in proposals:                                   # update after deciding
            others = Counter(q.answer for q in proposals if q.agent_id != p.agent_id)
            if others:
                top = max(others.values())
                loo = sorted(k for k, v in others.items() if v == top)
                self.agree[p.agent_id] += float(p.answer in loo) / len(loo)
                self.cnt[p.agent_id] += 1
        return Result(answer)


class AuditPlurality(Aggregator):
    """Plurality vote; whenever the agents are not unanimous and the task is
    verifiable and affordable, audit the leading answer and fall back to the
    next-most-voted answer if the audit fails (up to ``max_checks``).  No
    reputation learning and no cost-awareness beyond the global budget."""
    name = "Audit-Plurality"

    def __init__(self, max_checks: int = 3):
        self.max_checks = max_checks

    def aggregate(self, task, proposals, env):
        groups = group_by_answer(proposals)
        order = sorted(groups, key=lambda a: (-len(groups[a]), env.rng.random()))
        n_aud = 0
        if len(order) > 1 and task.verifiable:
            for cand in order[: self.max_checks]:
                if not env.oracle.can_afford(task):
                    break
                n_aud += 1
                if env.oracle.check(task, cand):
                    return Result(cand, n_aud)
            # every audited candidate failed, or budget ran out
            remaining = [c for c in order[n_aud:]]
            return Result(remaining[0] if remaining else order[0], n_aud)
        return Result(order[0], n_aud)
