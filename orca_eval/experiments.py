"""Experiment suites for evaluating ORCA against baselines."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd

from .agents import ATTACKS, DEFAULT_MODEL, ClaudeHaikuClient, DiskCache
from .runner import Scenario, run_scenario
from .tasks import load_tasks

_TASKS = None
_CLIENT = None


def _init(path: str, model: str, api_key: Optional[str], mock: bool, cache_dir: str):
    global _TASKS, _CLIENT
    _TASKS = load_tasks(path)
    cache = DiskCache(os.path.join(cache_dir, "haiku_cache.json"))
    _CLIENT = ClaudeHaikuClient(model=model, api_key=api_key, cache=cache, mock=mock)


def _job(args):
    sc, seed = args
    return run_scenario(_TASKS, sc, seed, client=_CLIENT)


def scenarios_for(suite: str, n_agents: int = 10, budget_frac: float = 0.15,
                  f_main: float = 0.3, model: str = DEFAULT_MODEL,
                  mock: bool = False) -> List[Scenario]:
    """Generate scenario configurations for the specified experiment suite."""
    def S(n, f, a, b=budget_frac, L=40.0):
        return Scenario(n_agents=n, adv_frac=f, attack=a, budget_frac=b,
                        orca_loss=L, model=model, mock=mock)

    out = []
    if suite in ("main", "all"):
        out.append(S(n_agents, 0.0, "none"))
        out += [S(n_agents, f_main, a) for a in ATTACKS if a != "none"]

    if suite in ("fsweep", "all"):
        for a in ATTACKS[1:]:
            for f in (0.1, 0.2, 0.3, 0.4, 0.5):
                out.append(S(n_agents, f, a))

    if suite in ("nsweep", "all"):
        for n in (3, 5, 7, 10):
            out.append(S(n, 0.0, "none"))
            out.append(S(n, 0.3, "collusive"))

    if suite in ("budget", "all"):
        for b in (0.0, 0.03, 0.06, 0.1, 0.15, 0.25, 0.5, 1.0):
            out.append(S(n_agents, 0.0, "none", b=b))
            out.append(S(n_agents, f_main, "collusive", b=b))
            out.append(S(n_agents, f_main, "evasive", b=b))

    if suite in ("loss", "all"):
        for L in (10.0, 20.0, 40.0, 80.0, 160.0):
            out.append(S(n_agents, 0.0, "none", L=L))
            out.append(S(n_agents, f_main, "collusive", L=L))
            out.append(S(n_agents, f_main, "evasive", L=L))

    if suite == "quick":
        out = [S(n_agents, 0.0, "none"), S(n_agents, f_main, "collusive")]

    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def run_suite(tasks_path: str, suite: str, n_agents: int = 10, seeds: int = 10,
              budget_frac: float = 0.15, f_main: float = 0.3, workers: int = 1,
              model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
              mock: bool = False, cache_dir: str = "cache") -> pd.DataFrame:
    """Run an entire experiment suite with Claude Haiku 5.5 agents."""
    _init(tasks_path, model, api_key, mock, cache_dir)
    scenarios = scenarios_for(suite, n_agents, budget_frac, f_main, model, mock)
    jobs = [(sc, s) for sc in scenarios for s in range(seeds)]

    print(f"Starting suite '{suite}': {len(jobs)} total runs across {len(scenarios)} scenarios and {seeds} seeds.")
    print(f"Model: {model} (mock mode: {mock}), concurrency: {workers} workers.")

    results = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, res in enumerate(ex.map(_job, jobs)):
                results.append(res)
                if (k + 1) % 5 == 0 or (k + 1) == len(jobs):
                    print(f"  Completed {k + 1}/{len(jobs)} runs", flush=True)
    else:
        for k, j in enumerate(jobs):
            res = _job(j)
            results.append(res)
            if (k + 1) % 5 == 0 or (k + 1) == len(jobs):
                print(f"  Completed {k + 1}/{len(jobs)} runs", flush=True)

    flat = [r for rs in results for r in rs]
    return pd.DataFrame(flat)
