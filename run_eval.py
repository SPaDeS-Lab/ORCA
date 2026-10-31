#!/usr/bin/env python
"""CLI to run ORCA experiments with Claude Haiku 5.5.

Example usage:
  # Quick validation run (mock mode, no API key needed):
  python run_eval.py --suite quick --seeds 1 --mock

  # Run real Claude Haiku 5.5 experiments:
  python run_eval.py --suite main --seeds 10 --agents 10 --model claude-3-5-haiku-20241022

  # Rebuild report only from previous results:
  python run_eval.py --from-csv results/raw_results.csv
"""
import argparse
import os
import time

from orca_eval.agents import DEFAULT_MODEL
from orca_eval.experiments import run_suite
from orca_eval.report import write_report

ap = argparse.ArgumentParser(description="ORCA multi-agent consensus evaluation with Claude Haiku 5.5")
ap.add_argument("--tasks", default="data/tasks.jsonl", help="path to tasks.jsonl dataset")
ap.add_argument("--suite", default="main",
                choices=["quick", "main", "fsweep", "nsweep", "budget", "loss", "all"],
                help="experiment suite to execute")
ap.add_argument("--agents", type=int, default=10, help="number of agents n <= 10 (default 10)")
ap.add_argument("--seeds", type=int, default=10, help="number of independent seeds (default 10)")
ap.add_argument("--budget-frac", type=float, default=0.15,
                help="audit budget as fraction of full-audit cost (default 0.15)")
ap.add_argument("--adv-frac", type=float, default=0.3,
                help="compromised agent fraction f for main experiment (default 0.3)")
ap.add_argument("--model", default=DEFAULT_MODEL,
                help=f"Claude Haiku model identifier (default {DEFAULT_MODEL})")
ap.add_argument("--api-key", default=None,
                help="Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)")
ap.add_argument("--cache-dir", default="cache",
                help="directory to cache Claude Haiku responses (default 'cache')")
ap.add_argument("--mock", action="store_true",
                help="use deterministic offline mock responses for testing pipeline")
ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                help="parallel worker threads for LLM querying")
ap.add_argument("--out", default="results", help="directory for output CSV, report, and figures")
ap.add_argument("--from-csv", default=None,
                help="skip execution and rebuild report from existing raw_results.csv")

a = ap.parse_args()

if a.agents > 10:
    ap.error("--agents must be <= 10")
if a.agents < 1:
    ap.error("--agents must be >= 1")

os.makedirs(a.out, exist_ok=True)
t0 = time.time()

if a.from_csv:
    import pandas as pd
    print(f"Loading existing results from {a.from_csv}...")
    df = pd.read_csv(a.from_csv)
else:
    api_key = a.api_key or os.environ.get("ANTHROPIC_API_KEY")
    mock = a.mock
    if not mock and not api_key:
        print("\nNotice: No ANTHROPIC_API_KEY detected. Defaulting to offline mock mode.\n")
        mock = True

    print(f"Running ORCA evaluation with model: {a.model} (mock: {mock})...")
    df = run_suite(
        tasks_path=a.tasks,
        suite=a.suite,
        n_agents=a.agents,
        seeds=a.seeds,
        budget_frac=a.budget_frac,
        f_main=a.adv_frac,
        workers=a.workers,
        model=a.model,
        api_key=api_key,
        mock=mock,
        cache_dir=a.cache_dir,
    )
    raw_path = os.path.join(a.out, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw results -> {raw_path}")

print("Generating markdown report and figures...")
write_report(df, a.out, a.agents, a.budget_frac, a.adv_frac)
print(f"Done in {time.time() - t0:.1f}s -> {a.out}/")
