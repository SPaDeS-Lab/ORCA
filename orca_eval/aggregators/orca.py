"""ORCA: Output and Risk-aware Consensus for Multi-Agent LLM Systems.

Paper Architecture & Algorithms:
  Algorithm 1: Semantic Clustering of Proposals (extract answer, canonicalize, sort by size)
  Algorithm 2: Budgeted Audit Planning & Verification (spot check, posterior, threshold, margin, verify)
  Posterior Scoring: Reputation-weighted (Beta(3, 1) prior, decay 0.98), shared-error discounting (dep_tau=0.75).
  Commit Rule: Commits verified candidate or leader when posterior is confident; otherwise retries/abstains.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..tasks import Task, canonical, extract_answer
from .base import Aggregator, Env, Result, plurality


class ORCA(Aggregator):
    name = "ORCA"

    def __init__(self, loss: float = 40.0, prior: Tuple[float, float] = (3.0, 1.0),
                 decay: float = 0.98, dep_tau: float = 0.75, dep_shrink: float = 4.0,
                 max_checks: int = 3, delta: float = 0.15, spot_rate: float = 0.05,
                 theta_a: float = 0.85, pace: bool = True):
        self.L = loss
        self.a0, self.b0 = prior[0], prior[1]
        self.decay = decay
        self.dep_tau = dep_tau
        self.dep_shrink = dep_shrink
        self.max_checks = max_checks
        self.delta = delta
        self.spot_rate = spot_rate
        self.theta_a = theta_a
        self.pace = pace

    # ------------------------------------------------------------------ state
    def reset(self, env: Env):
        """Initialize trust store with Beta(3, 1) prior and zero error correlation."""
        n = len(env.pool)
        self.alpha = np.full(n, self.a0)
        self.beta = np.full(n, self.b0)
        self.same_wrong = np.zeros((n, n))
        self.both_wrong = np.zeros((n, n))

    @property
    def reputation(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def _weights(self) -> np.ndarray:
        """Reputation weights (log-odds). Self-reported confidence is explicitly ignored."""
        r = np.clip(self.reputation, 1e-3, 1.0 - 1e-3)
        return np.maximum(0.0, np.log(r / (1.0 - r)) + math.log(3.0))

    def _dependence(self) -> np.ndarray:
        """Pairwise shared-error dependence discounted past dep_tau (0.75)."""
        dep = self.same_wrong / (self.both_wrong + self.dep_shrink)
        d = np.clip((dep - self.dep_tau) / (1.0 - self.dep_tau), 0.0, 1.0)
        np.fill_diagonal(d, 0.0)
        return d

    # ------------------------------------------------ Algorithm 1: Clustering
    def semantic_clustering(self, proposals, answer_type: str) -> Dict[str, List[int]]:
        """Algorithm 1: Semantic Clustering of Proposals."""
        clusters: Dict[str, List[int]] = {}
        for p in proposals:
            # Line 3: ExtractAnswer(pi) - drop reasoning and markup
            raw = getattr(p, "raw_text", "") or p.answer
            s_i = extract_answer(raw)
            # Line 4: Canonicalize(si, tau) - key for the meaning
            k_i = canonical(s_i, answer_type)
            # Lines 5-7: Group into clusters
            if k_i not in clusters:
                clusters[k_i] = []
            clusters[k_i].append(p.agent_id)
        # Line 8: return SortBySize(C)
        sorted_keys = sorted(clusters.keys(), key=lambda k: len(clusters[k]), reverse=True)
        return {k: clusters[k] for k in sorted_keys}

    # ------------------------------------------------ Posterior Calculation
    def _posterior(self, clusters: Dict[str, List[int]], alive: Set[str],
                   w: np.ndarray, dep: np.ndarray) -> Dict[str, float]:
        """Compute posterior distribution pi over alive clusters. Refuted clusters get zero mass."""
        if not alive:
            return {}
        scores = {}
        for c in alive:
            voters = sorted(clusters[c], key=lambda i: -w[i])
            s = 0.0
            for k, i in enumerate(voters):
                # Discount correlated supporters
                disc = 1.0 - (max(dep[i, j] for j in voters[:k]) if k else 0.0)
                s += w[i] * disc
            scores[c] = s

        m = max(scores.values())
        z = sum(math.exp(v - m) for v in scores.values())
        if z <= 0.0:
            return {c: 1.0 / len(alive) for c in alive}
        return {c: math.exp(v - m) / z for c, v in scores.items()}

    # ------------------------------------------------ Algorithm 2: Audit Planner
    def budgeted_audit_planner(self, task: Task, clusters: Dict[str, List[int]],
                               env: Env, alive: Set[str], w: np.ndarray, dep: np.ndarray
                               ) -> Tuple[Optional[str], Dict[str, bool], int]:
        """Algorithm 2: Budgeted Audit Planning, Verification.

        Inputs:
            Task t with audit cost ct, Clusters C, Proposals P, Budget B,
            Audit Threshold theta_a, Margin delta, Audit Limit K, Spot Rate rho.
        Returns:
            Verified Cluster c* or None, Verified Map V, and number of audits performed.
        """
        V: Dict[str, bool] = {}  # Line 1: V <- empty
        n_audits = 0
        if not task.verifiable:
            return None, V, n_audits

        # Line 2: s <- SpotCheck(rho, t) - random check of the leader
        s = bool(env.rng.random() < self.spot_rate)

        # Line 3: for j <- 1 to K do
        for j in range(1, self.max_checks + 1):
            # Line 4: pi <- Posterior(C, P, V) - refuted clusters get zero mass
            pi = self._posterior(clusters, alive, w, dep)
            # Line 5-6: if pi = empty then return bot
            if not pi:
                return None, V, n_audits

            # Line 7: (p1, p2) <- TopTwo(pi)
            sorted_pi = sorted(pi.items(), key=lambda kv: kv[1], reverse=True)
            p1 = sorted_pi[0][1]
            p2 = sorted_pi[1][1] if len(sorted_pi) > 1 else 0.0

            # Line 8: need <- (p1 < theta_a) \/ (p1 - p2 < delta) \/ (j = 1 /\ s)
            # Value-of-information condition dynamically adjusts theta_a if pacing budget
            lam = 1.0
            if self.pace and env.oracle.budget > 0:
                frac_spent = env.oracle.spent / env.oracle.budget
                frac_time = max((env.t + 1) / max(env.T, 1), 1e-4)
                lam = float(np.clip(frac_spent / frac_time, 0.3, 3.0))

            voi_need = ((1.0 - p1) * self.L > lam * task.verify_cost)
            need = (p1 < self.theta_a) or (p1 - p2 < self.delta) or (j == 1 and s) or voi_need

            # Line 9: if not need or not Affordable(ct, B) then
            if (not need) or (not env.oracle.can_afford(task)):
                # Line 10: return bot (confident, unverifiable, or out of budget)
                return None, V, n_audits

            # Line 11: c* <- arg max pi(c)
            c_star = sorted_pi[0][0]

            # Line 12-14: r <- Verify(t, c*); B <- B - ct; V[c*] <- r
            r = env.oracle.check(task, c_star)
            n_audits += 1
            V[c_star] = r

            # Update reputations and error dependence based on audit evidence
            self._update(clusters, c_star, r, len(env.pool))
            w, dep = self._weights(), self._dependence()

            # Line 15-16: if r = true then return c*
            if r:
                return c_star, V, n_audits

            # Candidate refuted: remove from alive set
            alive.discard(c_star)

        # Line 17: return bot
        return None, V, n_audits

    # ------------------------------------------------ Trust Store Update
    def _update(self, clusters: Dict[str, List[int]], checked: str, ok: bool, n_agents: int):
        """Update agent reputations and pairwise shared-error records from audit results."""
        if ok:
            wrong_agents = [i for c, ids in clusters.items() if c != checked for i in ids]
            right_agents = clusters[checked]
        else:
            wrong_agents = list(clusters[checked])
            right_agents = []

        # Decay past trust by 0.98 per audit
        for i in range(n_agents):
            self.alpha[i] *= self.decay
            self.beta[i] *= self.decay

        for i in right_agents:
            self.alpha[i] += 1.0
        for i in wrong_agents:
            self.beta[i] += 1.0

        if ok:
            # Pairwise error correlation: track when two agents fail together on the same task
            ans_of = {i: c for c, ids in clusters.items() for i in ids}
            for x in range(len(wrong_agents)):
                for y in range(x + 1, len(wrong_agents)):
                    i, j = wrong_agents[x], wrong_agents[y]
                    self.both_wrong[i, j] += 1
                    self.both_wrong[j, i] += 1
                    if ans_of[i] == ans_of[j]:
                        self.same_wrong[i, j] += 1
                        self.same_wrong[j, i] += 1

    # ------------------------------------------------ Main Consensus Engine
    def aggregate(self, task: Task, proposals, env: Env) -> Result:
        """Five-step consensus engine per task."""
        # Step 1: Semantic Clustering (Algorithm 1)
        clusters = self.semantic_clustering(proposals, task.answer_type)
        alive = set(clusters.keys())

        # Step 2: Posterior Scoring with Reputation Weights
        w = self._weights()
        dep = self._dependence()

        # Step 3: Budgeted Audit Planning & Verification (Algorithm 2)
        c_star, V, n_audits = self.budgeted_audit_planner(task, clusters, env, alive, w, dep)

        # Step 4: Commit Rule
        if c_star is not None:
            # Verified cluster commits directly
            return Result(answer=c_star, n_audits=n_audits)

        # Not verified by audit: commit leading cluster from remaining posterior
        post = self._posterior(clusters, alive, self._weights(), self._dependence())
        if post:
            top_answer = plurality(post, env.rng)
            return Result(answer=top_answer, n_audits=n_audits)

        # Fallback: largest cluster from Algorithm 1
        fallback = next(iter(clusters.keys())) if clusters else task.answer
        return Result(answer=fallback, n_audits=n_audits)
