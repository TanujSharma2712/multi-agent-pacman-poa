"""
Experiment 1: Effect of team size k, ghost aggression beta and PELLET SCARCITY on PoA.

Part A: k in {2,3,4} x beta in {0.0, 0.3, 0.6, 0.9}           (full pellets)
Part B: k in {2,3,4} x pellet density in {1.0, 0.5, 0.25}     (beta = 0.5)

PoA = J(pi*) / J(pi_selfish), where pi* is the centralized Hungarian planner and
pi_selfish is independent Q-learning trained with several seeds (pooled).
(The exact, equilibrium-based PoA on a toy maze is in run_experiment_toy_ne.py.)

Outputs: results/experiment_1.json,
         plots/fig_e1_poa_vs_k_beta.png, plots/fig_e1_poa_vs_scarcity.png
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import COLORS, MARKERS, save_figure, save_json, style_axes  # noqa: E402
from src.metrics import compute_poa  # noqa: E402
from src.trainer import DEFAULT_TRAIN_SEEDS, evaluate_policy, train_and_evaluate_q  # noqa: E402

GHOSTS = 2


def _poa_cell(layout_name, k, beta, density, num_episodes, train_seeds, train_episodes, seed):
    opt = evaluate_policy("centralized", layout_name=layout_name, num_pacman=k,
                          num_ghosts=GHOSTS, ghost_beta=beta, pellet_density=density,
                          num_episodes=num_episodes, seed=seed)
    selfish = train_and_evaluate_q("selfish", layout_name=layout_name, num_pacman=k,
                                   num_ghosts=GHOSTS, ghost_beta=beta, pellet_density=density,
                                   train_episodes=train_episodes, eval_episodes=num_episodes,
                                   train_seeds=train_seeds, base_seed=seed + 1)
    j_opt = float(opt["team_return_mean"])
    j_self = float(selfish["team_return_mean"])
    return {
        "k": k, "beta": beta, "pellet_density": density,
        "j_opt": j_opt, "j_opt_ci": opt["team_return_ci"],
        "j_selfish": j_self, "j_selfish_ci": selfish["team_return_ci"],
        "j_selfish_seed_ci": selfish["team_return_seed_ci"],
        "j_selfish_seed_means": selfish["seed_means"],
        "poa": compute_poa(j_opt, j_self),
    }


def run_experiment_1(
    layout_name: str = "proposal_maze",
    num_episodes: int = 20,
    train_seeds=DEFAULT_TRAIN_SEEDS,
    train_episodes: int = 40,
    k_values=(2, 3, 4),
    beta_values=(0.0, 0.3, 0.6, 0.9),
    densities=(1.0, 0.5, 0.25),
    scarcity_beta: float = 0.5,
):
    print("==================================================")
    print("Experiment 1: PoA vs team size k, ghost danger beta, pellet scarcity")
    print("==================================================")

    part_a, part_b = [], []
    for k in k_values:
        for beta in beta_values:
            cell = _poa_cell(layout_name, k, beta, 1.0, num_episodes, train_seeds,
                             train_episodes, seed=100 + 10 * k + int(beta * 10))
            part_a.append(cell)
            print(f"[A] k={k} beta={beta:.1f}: J*={cell['j_opt']:.1f} "
                  f"J_selfish={cell['j_selfish']:.1f} PoA={cell['poa']:.3f}")
    for k in k_values:
        for d in densities:
            cell = _poa_cell(layout_name, k, scarcity_beta, d, num_episodes, train_seeds,
                             train_episodes, seed=500 + 10 * k + int(d * 100))
            part_b.append(cell)
            print(f"[B] k={k} density={d:.2f}: J*={cell['j_opt']:.1f} "
                  f"J_selfish={cell['j_selfish']:.1f} PoA={cell['poa']:.3f}")

    save_json({
        "layout": layout_name, "ghosts": GHOSTS, "eval_episodes": num_episodes,
        "train_seeds": list(train_seeds), "train_episodes": train_episodes,
        "part_a_k_beta": part_a,
        "part_b_scarcity": {"beta": scarcity_beta, "cells": part_b},
    }, "experiment_1.json")

    # Part A figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n, k in enumerate(k_values):
        cells = [c for c in part_a if c["k"] == k]
        axes[0].plot([c["beta"] for c in cells], [c["poa"] for c in cells], marker=MARKERS[n],
                     color=COLORS[n], linewidth=2.2, label=f"k = {k}")
        axes[1].errorbar([c["beta"] for c in cells], [c["j_selfish"] for c in cells],
                         yerr=[c["j_selfish_seed_ci"] for c in cells], marker=MARKERS[n],
                         color=COLORS[n], linewidth=1.8, capsize=3, label=f"selfish k={k}")
        axes[1].plot([c["beta"] for c in cells], [c["j_opt"] for c in cells], linestyle=":",
                     color=COLORS[n], label=f"π* k={k}")
    axes[0].axhline(1.0, color="gray", linestyle=":")
    style_axes(axes[0], "PoA vs ghost aggression β", "Ghost aggression β", "Price of Anarchy")
    style_axes(axes[1], "Team return (error bars: 95% CI across seeds)", "Ghost aggression β",
               "Team return J")
    axes[0].legend()
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig_e1_poa_vs_k_beta.png")

    # Part B figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n, k in enumerate(k_values):
        cells = [c for c in part_b if c["k"] == k]
        axes[0].plot([c["pellet_density"] for c in cells], [c["poa"] for c in cells],
                     marker=MARKERS[n], color=COLORS[n], linewidth=2.2, label=f"k = {k}")
        axes[1].errorbar([c["pellet_density"] for c in cells], [c["j_selfish"] for c in cells],
                         yerr=[c["j_selfish_seed_ci"] for c in cells], marker=MARKERS[n],
                         color=COLORS[n], capsize=3, label=f"selfish k={k}")
        axes[1].plot([c["pellet_density"] for c in cells], [c["j_opt"] for c in cells],
                     linestyle=":", color=COLORS[n], label=f"π* k={k}")
    for ax in axes:
        ax.invert_xaxis()  # left = plentiful, right = scarce
    axes[0].axhline(1.0, color="gray", linestyle=":")
    style_axes(axes[0], f"PoA vs pellet scarcity (β={scarcity_beta})",
               "Fraction of pellets kept (→ scarcer)", "Price of Anarchy")
    style_axes(axes[1], "Team return under scarcity", "Fraction of pellets kept (→ scarcer)",
               "Team return J")
    axes[0].legend()
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig_e1_poa_vs_scarcity.png")
    return part_a, part_b


if __name__ == "__main__":
    run_experiment_1()
