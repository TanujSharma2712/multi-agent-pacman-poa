"""
Experiment 2: DCOP message budget B AND number of regions m vs. selfish gap recovered.

Grid: m in {2, 3, 4} x algorithm in {DSA, Max-Sum} x B in {0, 6, 12, 24, 36, 48}.
For k = 3 one DSA round costs k(k-1) = 6 messages and one Max-Sum iteration costs
2k(k-1) = 12 messages, so the budget grid is aligned with those costs (a budget
below one round/iteration means no communication at all).

Gap Recovered = (J(DCOP) - J(selfish)) / (J(pi*) - J(selfish)), with the same
selfish reference as Experiment 4 (multi-seed independent Q-learning).
Also reports the messages ACTUALLY sent per step (metric from the proposal).

Outputs: results/experiment_2.json,
         plots/fig_e2_dcop_budget.png, plots/fig_e2_messages.png
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import COLORS, MARKERS, save_figure, save_json, style_axes  # noqa: E402
from src.metrics import compute_gap_recovered  # noqa: E402
from src.trainer import DEFAULT_TRAIN_SEEDS, evaluate_policy, train_and_evaluate_q  # noqa: E402

GHOSTS = 2
BETA = 0.5


def run_experiment_2(
    layout_name: str = "proposal_maze",
    k: int = 3,
    num_episodes: int = 20,
    budgets=(0, 6, 12, 24, 36, 48),
    region_counts=(2, 3, 4),
    train_seeds=DEFAULT_TRAIN_SEEDS,
    train_episodes: int = 40,
):
    print("==================================================")
    print("Experiment 2: DCOP message budget B and regions m (DSA vs Max-Sum)")
    print("==================================================")
    common = dict(layout_name=layout_name, num_pacman=k, num_ghosts=GHOSTS, ghost_beta=BETA)

    opt = evaluate_policy("centralized", num_episodes=num_episodes, seed=42, **common)
    selfish = train_and_evaluate_q("selfish", train_episodes=train_episodes,
                                   eval_episodes=num_episodes, train_seeds=train_seeds,
                                   base_seed=43, **common)
    j_opt, j_self = float(opt["team_return_mean"]), float(selfish["team_return_mean"])
    print(f"J(pi*)={j_opt:.1f}  J(selfish Q)={j_self:.1f}")

    cells = []
    for m in region_counts:
        for algo in ("dsa", "max_sum"):
            for b in budgets:
                # Common random numbers: every (algo, budget) for a given m is evaluated
                # on the SAME episodes, so differences come from the algorithm, not luck.
                r = evaluate_policy("dcop", dcop_algo=algo, dcop_budget=b, dcop_regions=m,
                                    num_episodes=num_episodes, seed=1000, **common)
                j = float(r["team_return_mean"])
                cells.append({
                    "m": m, "algo": algo, "budget": b,
                    "team_return": j, "team_return_ci": r["team_return_ci"],
                    "recovered_pct": 100.0 * compute_gap_recovered(j, j_self, j_opt),
                    "msgs_per_step": r["msgs_per_step_mean"],
                    "gini": r["gini_mean"], "captures": r["captures_mean"],
                })
                print(f"m={m} {algo:7s} B={b:2d}: J={j:7.1f} recovered="
                      f"{cells[-1]['recovered_pct']:5.1f}% msgs/step={r['msgs_per_step_mean']:.1f}")

    save_json({
        "layout": layout_name, "k": k, "ghosts": GHOSTS, "beta": BETA,
        "eval_episodes": num_episodes, "train_seeds": list(train_seeds),
        "j_opt": j_opt, "j_selfish": j_self,
        "j_selfish_seed_ci": selfish["team_return_seed_ci"],
        "budgets": list(budgets), "region_counts": list(region_counts),
        "cells": cells,
    }, "experiment_2.json")

    # Recovery vs budget, one panel per m
    fig, axes = plt.subplots(1, len(region_counts), figsize=(5 * len(region_counts), 4.5), sharey=True)
    if len(region_counts) == 1:
        axes = [axes]
    for ax, m in zip(axes, region_counts):
        for n, (algo, label) in enumerate((("dsa", "DSA"), ("max_sum", "Max-Sum"))):
            sel = [c for c in cells if c["m"] == m and c["algo"] == algo]
            ax.plot([c["budget"] for c in sel], [c["recovered_pct"] for c in sel],
                    marker=MARKERS[n], color=COLORS[n], linewidth=2.2, label=label)
        style_axes(ax, f"m = {m} regions", "Message budget B (per step)", "Selfish gap recovered (%)")
        ax.legend()
    fig.tight_layout()
    save_figure(fig, "fig_e2_dcop_budget.png")

    # Messages actually used
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = 0
    for m in region_counts:
        for algo, label in (("dsa", "DSA"), ("max_sum", "Max-Sum")):
            sel = [c for c in cells if c["m"] == m and c["algo"] == algo]
            ax.plot([c["budget"] for c in sel], [c["msgs_per_step"] for c in sel],
                    marker=MARKERS[n % len(MARKERS)], color=COLORS[n % len(COLORS)],
                    label=f"{label}, m={m}")
            n += 1
    ax.plot(budgets, budgets, linestyle=":", color="gray", label="budget limit")
    style_axes(ax, "Messages actually sent per step", "Message budget B", "Messages / step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_e2_messages.png")
    return cells


if __name__ == "__main__":
    run_experiment_2()
