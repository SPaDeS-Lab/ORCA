"""Tables (markdown) and figures from the raw results frame."""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

METHODS = ["Majority vote", "Multi-agent debate", "Last-writer-wins",
           "Agreement-weighted", "Audit-Plurality", "ORCA"]
SCEN_LABEL = {"none": "No adversary", "random": "Random", "collusive": "Collusive",
              "sleeper": "Sleeper", "evasive": "Evasive"}
ATTACK_ORDER = ["none", "random", "collusive", "sleeper", "evasive"]


def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def cell(x, scale=100, nd=1):
    return f"{np.mean(x) * scale:.{nd}f} ± {_ci(np.asarray(x) * scale if scale else x):.{nd}f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v)
    return df[m]


def main_table(df, n, b, f):
    cond = (df.n_agents == n) & np.isclose(df.budget_frac, b) & (df.orca_loss == 40.0)
    if "error_corr" in df.columns and (df.error_corr == 0.5).any():
        cond &= (df.error_corr == 0.5)
    d = df[cond]
    d = d[(d.attack == "none") | np.isclose(d.adv_frac, f)]
    if d.empty:
        return None
    atks = [a for a in ATTACK_ORDER if a in set(d.attack)]
    rows = []
    for m in METHODS:
        dm = d[d.method == m]
        cells = [cell(dm[dm.attack == a].acc) for a in atks]
        worst = min(dm[dm.attack == a].acc.mean() for a in atks) if atks else 0.0
        rows.append([f"**{m}**" if m == "ORCA" else m] + cells +
                    [f"{worst * 100:.1f}", f"{dm.audit_cost.mean():.0f}"])
    return md_table(["Method (accuracy %)"] + [SCEN_LABEL[a] for a in atks] + ["Worst-case", "Avg audit cost"], rows)


def tier_table(df, n, b, f, attack):
    cond = (df.n_agents == n) & np.isclose(df.budget_frac, b) & (df.orca_loss == 40.0)
    if "error_corr" in df.columns and (df.error_corr == 0.5).any():
        cond &= (df.error_corr == 0.5)
    d = df[cond]
    d = d[(d.attack == attack) & (np.isclose(d.adv_frac, f) | (attack == "none"))]
    if d.empty:
        return None
    rows = []
    for m in METHODS:
        dm = d[d.method == m]
        rows.append([m] + [cell(dm[f"acc_{t}"]) for t in ("cheap", "medium", "expensive", "unverifiable")])
    return md_table(["Method", "cheap (c=1)", "medium (c=5)", "expensive (c=20)", "unverifiable (c=∞)"], rows)


def sweep_table(d, xcol, xfmt, methods=METHODS):
    xs = sorted(d[xcol].unique())
    rows = [[m] + [cell(d[(d.method == m) & np.isclose(d[xcol], x)].acc) for x in xs] for m in methods]
    return md_table(["Method"] + [xfmt(x) for x in xs], rows)


def write_report(df: pd.DataFrame, out: str, n: int, b: float, f: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3})
    colors = dict(zip(METHODS, ["#7f7f7f", "#9467bd", "#bcbd22", "#17becf", "#ff7f0e", "#d62728"]))
    cond = (df.orca_loss == 40.0)
    if "error_corr" in df.columns and (df.error_corr == 0.5).any():
        cond &= (df.error_corr == 0.5)
    base = df[cond]
    md = [f"# ORCA evaluation report\n",
          f"Tasks: 1000 · agents: up to {n} · seeds: {df.seed.nunique()} · default audit budget: "
          f"{b:.0%} of the full-audit cost · main adversarial fraction: {f:.0%}. "
          "Cells are mean ± 95% CI over seeds.\n"]

    t = main_table(df, n, b, f)
    if t:
        md += [f"## 1. Main comparison (n={n}, adversaries={f:.0%})\n", t, ""]
        atks = [a for a in ATTACK_ORDER if a in set(base.attack)]
        d = base[(base.n_agents == n) & np.isclose(base.budget_frac, b) & ((base.attack == "none") | np.isclose(base.adv_frac, f))]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        w = 0.13
        for k, m in enumerate(METHODS):
            ys = [d[(d.method == m) & (d.attack == a)].acc.mean() * 100 for a in atks]
            es = [_ci(d[(d.method == m) & (d.attack == a)].acc * 100) for a in atks]
            ax.bar(np.arange(len(atks)) + (k - 2.5) * w, ys, w, yerr=es, label=m, color=colors[m])
        ax.set_xticks(range(len(atks))); ax.set_xticklabels([SCEN_LABEL[a] for a in atks])
        ax.set_ylabel("accuracy (%)"); ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout(); fig.savefig(f"{out}/fig_main.png", dpi=150); plt.close(fig)
        for atk in ("none", "collusive", "evasive"):
            tt = tier_table(df, n, b, f, atk)
            if tt:
                md += [f"### Accuracy by verification tier — {SCEN_LABEL[atk]}\n", tt, ""]
        # ORCA detection
        o = d[(d.method == "ORCA") & (d.attack != "none")]
        if "det_auc" in o:
            rows = [[SCEN_LABEL[a], cell(o[o.attack == a].det_auc, 1, 3),
                     cell(o[o.attack == a].rep_adv, 1, 2), cell(o[o.attack == a].rep_hon, 1, 2)]
                    for a in atks if a != "none"]
            md += ["### ORCA adversary detection\n",
                   "AUC = P(honest agent's final reputation > adversary's). Reputation is the audited "
                   "accuracy, biased downward because audits fire on contested tasks.\n",
                   md_table(["Attack", "AUC", "mean rep. (adv.)", "mean rep. (honest)"], rows), ""]

    fs = base[(base.n_agents == n) & np.isclose(base.budget_frac, b) & (base.attack != "none") & (base.adv_frac > 0)]
    if fs.adv_frac.nunique() > 1:
        md += ["## 2. Adversarial-fraction sweep\n"]
        atks = [a for a in ATTACK_ORDER[1:] if a in set(fs.attack)]
        fig, axs = plt.subplots(1, len(atks), figsize=(3.2 * len(atks), 3), sharey=True)
        for ax, a in zip(np.atleast_1d(axs), atks):
            da = fs[fs.attack == a]
            for m in METHODS:
                g = da[da.method == m].groupby("adv_frac").acc
                ax.errorbar(g.mean().index * 100, g.mean() * 100, g.apply(_ci) * 100, label=m, color=colors[m], marker="o", ms=3)
            ax.set_title(SCEN_LABEL[a]); ax.set_xlabel("adversaries (%)")
        np.atleast_1d(axs)[0].set_ylabel("accuracy (%)"); np.atleast_1d(axs)[-1].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{out}/fig_fsweep.png", dpi=150); plt.close(fig)
        for a in atks:
            md += [f"**{SCEN_LABEL[a]}**\n", sweep_table(fs[fs.attack == a], "adv_frac", lambda x: f"{x:.0%}"), ""]

    bs = base[(base.n_agents == n) & (base.budget_frac.nunique() > 0)]
    bs = bs[(bs.attack == "none") | ((bs.adv_frac == f) & bs.attack.isin(["collusive", "evasive"]))]
    if bs.budget_frac.nunique() > 2:
        md += ["## 3. Audit-budget sweep (budget as fraction of full-audit cost)\n"]
        atks = [a for a in ("none", "collusive", "evasive") if a in set(bs.attack)]
        fig, axs = plt.subplots(1, len(atks), figsize=(3.4 * len(atks), 3), sharey=True)
        for ax, a in zip(np.atleast_1d(axs), atks):
            da = bs[bs.attack == a]
            for m in METHODS:
                g = da[da.method == m].groupby("budget_frac").acc
                ax.errorbar(g.mean().index * 100, g.mean() * 100, g.apply(_ci) * 100, label=m, color=colors[m], marker="o", ms=3)
            ax.set_title(SCEN_LABEL[a]); ax.set_xlabel("budget (% of full audit)")
        np.atleast_1d(axs)[0].set_ylabel("accuracy (%)"); np.atleast_1d(axs)[-1].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{out}/fig_budget.png", dpi=150); plt.close(fig)
        for a in atks:
            md += [f"**{SCEN_LABEL[a]}**\n", sweep_table(bs[bs.attack == a], "budget_frac", lambda x: f"{x:.0%}",
                                                       ["Majority vote", "Audit-Plurality", "ORCA"]), ""]

    ns = base[np.isclose(base.budget_frac, b) & ((base.attack == "none") | ((base.attack == "collusive") & np.isclose(base.adv_frac, 0.3)))]
    if ns.n_agents.nunique() > 1:
        md += ["## 4. Number-of-agents sweep\n"]
        for a, ttl in (("none", "No adversary"), ("collusive", "Collusive, 30%")):
            md += [f"**{ttl}**\n", sweep_table(ns[ns.attack == a], "n_agents", lambda x: f"n={int(x)} ({int(round(0.3 * x)) if a != 'none' else 0} adv.)"), ""]

    cs = df[(df.orca_loss == 40.0) & np.isclose(df.budget_frac, b) & (df.n_agents == n)]
    cs = cs[(cs.attack == "none") | ((cs.attack == "collusive") & np.isclose(cs.adv_frac, f))]
    if "error_corr" in cs.columns and cs.error_corr.nunique() > 1:
        md += ["## 5. Honest-error correlation (shared misconceptions)\n"]
        for a, ttl in (("none", "No adversary"), ("collusive", f"Collusive, {f:.0%}")):
            md += [f"**{ttl}**\n", sweep_table(cs[cs.attack == a], "error_corr", lambda x: f"ρ={x:.2f}"), ""]

    cond_l = np.isclose(df.budget_frac, b) & (df.n_agents == n) & (df.method == "ORCA")
    if "error_corr" in df.columns and (df.error_corr == 0.5).any():
        cond_l &= (df.error_corr == 0.5)
    ls = df[cond_l]
    ls = ls[(ls.attack == "none") | (np.isclose(ls.adv_frac, f) & ls.attack.isin(["collusive", "evasive"]))]
    if ls.orca_loss.nunique() > 1:
        md += ["## 6. ORCA sensitivity to the loss parameter L (audit if (1−P(top))·L > cost)\n"]
        rows = [[SCEN_LABEL[a]] + [cell(ls[(ls.attack == a) & (ls.orca_loss == L)].acc) for L in sorted(ls.orca_loss.unique())]
                for a in ("none", "collusive", "evasive")]
        md += [md_table(["Scenario"] + [f"L={L:g}" for L in sorted(ls.orca_loss.unique())], rows), ""]

    with open(f"{out}/report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
