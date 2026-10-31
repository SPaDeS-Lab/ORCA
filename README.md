# ORCA: Secure Consensus for Adversarial Multi-Agent LLM Systems

Official codebase for **ORCA** (*Output and Risk-aware Consensus for Agents*), a security-oriented consensus mechanism designed to defend multi-agent LLM systems against compromised agents, coordinated collusion, and correlated errors.


## Baselines

ORCA is evaluated against five standard consensus mechanisms:

- **Majority Vote:** Standard plurality voting with uniform random tie-breaking.
- **Multi-Agent Debate:** Multi-round deliberation where honest agents can be persuaded by peer consensus, while adversaries remain stubborn.
- **Last-Writer-Wins:** Sequential overwrite of a shared memory slot where the final write stands.
- **Agreement-Weighted Vote:** Unsupervised weighting based on running agreement with peer pluralities without ground-truth verification.
- **Audit-Plurality:** Plurality voting that audits the leading candidate upon disagreement without reputation learning or cost pacing.

---

## Benchmark Tasks

The evaluation suite contains **1,000 tasks** across code generation, multi-hop question answering, and planning, categorized into four verification tiers:
- **Cheap ($c=1$):** Arithmetic, string counting (400 tasks).
- **Medium ($c=5$):** Multi-step math, date calculations, percentage analysis (300 tasks).
- **Expensive ($c=20$):** Knapsack optimization, schedule planning (200 tasks).
- **Unverifiable ($c=\infty$):** Open-domain knowledge base queries where automated audit is infeasible (100 tasks).

---

## Repository Structure

```
orca_eval/
├── data/
│   └── tasks.jsonl             # 1,000 evaluation tasks across 4 tiers
├── orca_eval/
│   ├── aggregators/
│   │   ├── base.py             # Abstract base aggregator and Environment
│   │   ├── baselines.py        # Baseline aggregation mechanisms
│   │   └── orca.py             # ORCA consensus engine (Algorithms 1 & 2)
│   ├── agents.py               # Agent representations & poisoned memory generation
│   ├── audit.py                # Budgeted audit oracle
│   ├── experiments.py          # Experiment suites and execution routines
│   ├── report.py               # Markdown and plot generation utilities
│   ├── runner.py               # Scenario simulation loop
│   └── tasks.py                # Task loader, answer extraction, canonicalization
├── scripts/
│   └── test_haiku_agent.py     # Diagnostic tool for inspecting agent memory files
├── tests/
│   └── test_core.py            # Unit tests for algorithms, agents, and baselines
├── requirements.txt            # Python dependencies
├── run_eval.py                 # Main CLI evaluation runner
└── README.md
```

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-org>/orca_eval.git
cd orca_eval
pip install -r requirements.txt
```

### 2. Run Tests
Verify algorithms, clustering, and audit budget enforcement:
```bash
pytest tests/
```

### 3. Run Evaluations
Run the evaluation suite using `run_eval.py`:

```bash
# Offline smoke test (mock mode, no API key required):
python run_eval.py --suite quick --mock

# Main evaluation across all threat models (10 agents, 10 seeds):
python run_eval.py --suite main --seeds 10 --agents 10

# Parameter sweeps:
python run_eval.py --suite fsweep   # Adversary fraction sweep (10% to 50%)
python run_eval.py --suite budget   # Audit budget sweep (0% to 100%)
python run_eval.py --suite nsweep   # Agent group size sweep (n=3, 5, 7, 10)
```

#### CLI Options
- `--suite`: Experiment suite (`quick`, `main`, `fsweep`, `nsweep`, `budget`, `loss`, `all`).
- `--agents`: Group size $n \le 10$ (default: `10`).
- `--seeds`: Number of independent random seeds (default: `10`).
- `--adv-frac`: Fraction of compromised agents $f \in [0.1, 0.5]$ (default: `0.3`).
- `--budget-frac`: Audit budget as a fraction of full audit cost (default: `0.15`).
- `--model`: Model identifier (default: `claude-3-5-haiku-20241022`).
- `--api-key`: Anthropic API key (defaults to `$ANTHROPIC_API_KEY`).
- `--mock`: Run in offline mock mode using deterministic outputs.
- `--out`: Output directory for generated CSV and reports (default: `results`).

