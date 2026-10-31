"""Unit tests for ORCA consensus engine, Claude Haiku agents, and baselines."""
import math
import numpy as np
import pytest

from orca_eval.tasks import canonical, extract_answer, load_tasks, full_audit_cost, Task
from orca_eval.audit import AuditOracle
from orca_eval.agents import (
    AgentPool, ClaudeHaikuClient, Proposal, PoisonedMemory, DiskCache
)
from orca_eval.aggregators import ORCA, MajorityVote, MultiAgentDebate, AuditPlurality, Env
from orca_eval.runner import Scenario, run_scenario

TASKS = load_tasks("data/tasks.jsonl")


def test_loading():
    """Verify benchmark tasks: 1,000 tasks across 4 tiers."""
    assert len(TASKS) == 1000
    assert sum(not t.verifiable for t in TASKS) == 100
    assert sum(t.tier == "cheap" for t in TASKS) == 400
    assert sum(t.tier == "medium" for t in TASKS) == 300
    assert sum(t.tier == "expensive" for t in TASKS) == 200
    assert sum(t.tier == "unverifiable" for t in TASKS) == 100
    assert all(t.answer not in t.wrong for t in TASKS)


def test_extract_and_canonical():
    """Algorithm 1: ExtractAnswer and Canonicalize."""
    # Test Answer extraction from reasoning outputs
    raw1 = "Step 1: calculate 5+5=10.\nTherefore, we conclude.\nFinal Answer: 10"
    assert extract_answer(raw1) == "10"

    raw2 = "Step 1: Check availability.\n**Final Answer:** B, C, D"
    assert extract_answer(raw2) == "B, C, D"

    raw3 = "The result is given by \\boxed{Saturday}."
    assert extract_answer(raw3) == "Saturday"

    raw4 = "Just the answer\n44.32"
    assert extract_answer(raw4) == "44.32"

    # Test Canonicalization
    assert canonical("46.10", "number") == canonical("46.1", "number")
    assert canonical("B, A", "set") == canonical("a,b", "set")
    assert canonical(" Saturday ", "text") == "saturday"


def test_poisoned_memory_files():
    """Verify poisoned memory generation for adversarial threat models."""
    t = TASKS[0]  # verifiable
    target = t.wrong[0]

    # Honest memory
    hon_mem = PoisonedMemory.honest_memory()
    assert "honest" in hon_mem.lower()

    # Random attack memory
    rand_mem = PoisonedMemory.adversarial_memory("random", t, 0, 100, target, t.wrong)
    assert "Random Sabotage" in rand_mem

    # Collusive attack memory
    col_mem = PoisonedMemory.adversarial_memory("collusive", t, 0, 100, target, t.wrong)
    assert f'Coordinated Coalition Target: "{target}"' in col_mem

    # Sleeper attack: dormant early, activated late
    dorm_mem = PoisonedMemory.adversarial_memory("sleeper", t, 5, 100, target, t.wrong, sleeper_frac=0.25)
    assert "Dormant Phase" in dorm_mem
    act_mem = PoisonedMemory.adversarial_memory("sleeper", t, 30, 100, target, t.wrong, sleeper_frac=0.25)
    assert "ACTIVATED" in act_mem

    # Evasive attack: honest on verifiable, collusive on unverifiable
    ev_ver = PoisonedMemory.adversarial_memory("evasive", t, 0, 100, target, t.wrong)
    assert "Evasion Mode" in ev_ver

    unv_task = next(x for x in TASKS if not x.verifiable)
    ev_unv = PoisonedMemory.adversarial_memory("evasive", unv_task, 0, 100, target, unv_task.wrong)
    assert "Attack Mode" in ev_unv


def test_agent_pool_and_limits():
    """Verify AgentPool respects n <= 10 and adversary assignment."""
    with pytest.raises(ValueError):
        AgentPool(11, 0.0, "none", 0)

    with pytest.raises(ValueError):
        AgentPool(0, 0.0, "none", 0)

    pool = AgentPool(10, adv_frac=0.3, attack="collusive", seed=42)
    assert len(pool) == 10
    assert len(pool.adversary_ids) == 3


def test_algorithm_1_semantic_clustering():
    """Verify Algorithm 1: Semantic Clustering of Proposals."""
    orca = ORCA()
    proposals = [
        Proposal(0, "sat", "Reasoning 1.\nFinal Answer: Saturday"),
        Proposal(1, "sat", "Reasoning 2.\nFinal Answer: saturday"),
        Proposal(2, "fri", "Reasoning 3.\nFinal Answer: Friday"),
        Proposal(3, "sat", "Reasoning 4.\nFinal Answer: **Saturday**"),
    ]
    clusters = orca.semantic_clustering(proposals, answer_type="text")
    assert "saturday" in clusters
    assert len(clusters["saturday"]) == 3
    assert len(clusters["friday"]) == 1
    # Check sorted by size
    assert list(clusters.keys())[0] == "saturday"


def test_algorithm_2_budgeted_audit():
    """Verify Algorithm 2: Budgeted Audit Planning & Verification."""
    orca = ORCA(loss=40.0, max_checks=3, theta_a=0.85)
    task = next(t for t in TASKS if t.verifiable and t.verify_cost == 1.0)
    pool = AgentPool(4, adv_frac=0.0, attack="none", seed=1)
    oracle = AuditOracle(budget=10.0)
    env = Env(rng=np.random.default_rng(42), pool=pool, oracle=oracle, t=0, T=100)
    orca.reset(env)

    # 2 honest, 2 wrong -> low posterior -> triggers audit
    proposals = [
        Proposal(0, task.answer, f"Final Answer: {task.answer}"),
        Proposal(1, task.answer, f"Final Answer: {task.answer}"),
        Proposal(2, task.wrong[0], f"Final Answer: {task.wrong[0]}"),
        Proposal(3, task.wrong[0], f"Final Answer: {task.wrong[0]}"),
    ]
    res = orca.aggregate(task, proposals, env)
    assert res.answer == task.answer
    assert res.n_audits >= 1
    assert oracle.spent >= task.verify_cost


def test_oracle_budget_enforced():
    """Exact audit verifier respects shared budget and unverifiable tasks."""
    t = next(t for t in TASKS if t.verifiable)
    o = AuditOracle(budget=t.verify_cost)
    assert o.check(t, t.answer)
    with pytest.raises(RuntimeError):
        o.check(t, t.answer)
    unv = next(t for t in TASKS if not t.verifiable)
    assert not AuditOracle(1e9).can_afford(unv)


def test_budget_respected_and_deterministic():
    """Verify paired execution determinism and budget enforcement."""
    sc = Scenario(adv_frac=0.3, attack="collusive", budget_frac=0.1, mock=True)
    r1 = run_scenario(TASKS[:20], sc, seed=3)
    r2 = run_scenario(TASKS[:20], sc, seed=3)
    assert [x["acc"] for x in r1] == [x["acc"] for x in r2]
    cap = 0.1 * full_audit_cost(TASKS[:20])
    for r in r1:
        assert r["audit_cost"] <= cap + 1e-6
        if r["method"] in ("Majority vote", "Last-writer-wins", "Multi-agent debate", "Agreement-weighted"):
            assert r["audit_cost"] == 0


def test_orca_reputation_and_collusion_resilience():
    """Verify ORCA learns reputations, discounts correlated error, and outperforms majority vote."""
    sc = Scenario(n_agents=10, adv_frac=0.4, attack="collusive", mock=True)
    results = run_scenario(TASKS[:50], sc, seed=7)
    res_map = {r["method"]: r for r in results}
    assert res_map["ORCA"]["acc"] >= res_map["Majority vote"]["acc"]
    assert "det_auc" in res_map["ORCA"]
    assert res_map["ORCA"]["det_auc"] >= 0.5
