"""
Experiment 4: Multi-method benchmark ACROSS conditions (ghost danger, scarcity, layouts).

Methods (8): Random, Independent Greedy, Selfish Q, Equal-Share Q, DCOP-DSA (B=12),
DCOP-Max-Sum (B=24), Shapley Q (lambda=0.5), Centralized planner (pi*).
Q-learning methods are trained with several seeds (pooled).

Conditions (k = 3, 2 ghosts):
  main         proposal_maze  beta=0.5  all pellets      <- headline table
  calm         proposal_maze  beta=0.0
  dangerous    proposal_maze  beta=0.9
  scarce       proposal_maze  beta=0.5  50% pellets
  small_grid / medium_classic / open_contention (Berkeley-style layouts), beta=0.5

Outputs: results/experiment_4_benchmark.csv / .json   (main condition)
         results/experiment_4_conditions.csv / .json  (all conditions)
         plots/fig_e4_method_comparison.png, plots/fig_e4_conditions.png
"""

import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import COLORS, save_figure, save_json, style_axes  # noqa: E402
from src.metrics import compute_gap_recovered, compute_poa  # noqa: E402
from src.trainer import DEFAULT_TRAIN_SEEDS, evaluate_policy, train_and_evaluate_q  # noqa: E402

GHOSTS = 2

CONDITIONS = [
    {"name": "main", "layout": "proposal_maze", "beta": 0.5, "density": 1.0},
    {"name": "calm (β=0)", "layout": "proposal_maze", "beta": 0.0, "density": 1.0},
    {"name": "dangerous (β=0.9)", "layout": "proposal_maze", "beta": 0.9, "density": 1.0},
    {"name": "scarce (50% pellets)", "layout": "proposal_maze", "beta": 0.5, "density": 0.5},
    {"name": "small_grid", "layout": "small_grid", "beta": 0.5, "density": 1.0},
    {"name": "medium_classic", "layout": "medium_classic", "beta": 0.5, "density": 1.0},
    {"name": "open_contention", "layout": "open_contention", "beta": 0.5, "density": 1.0},
]

METHODS = [
    "Random",
    "Independent Greedy",
    "Selfish Q-Learning",
    "Equal-Share Q-Learning",
    "DCOP-DSA (B=12)",
    "DCOP-Max-Sum (B=24)",
    "Shapley Q-Learning (λ=0.5)",
    "Centralized Planner (π*)",
]


def _benchmark(cond, k, num_episodes, train_seeds, train_episodes, seed0):
    common = dict(layout_name=cond["layout"], num_pacman=k, num_ghosts=GHOSTS,
                  ghost_beta=cond["beta"], pellet_density=cond["density"])
    qkw = dict(train_episodes=train_episodes, eval_episodes=num_episodes, train_seeds=train_seeds)
    res = [
        evaluate_policy("random", num_episodes=num_episodes, seed=seed0 + 10, **common),
        evaluate_policy("greedy", num_episodes=num_episodes, seed=seed0 + 20, **common),
        train_and_evaluate_q("selfish", base_seed=seed0 + 30, **qkw, **common),
        train_and_evaluate_q("equal_share", base_seed=seed0 + 40, **qkw, **common),
        evaluate_policy("dcop", dcop_algo="dsa", dcop_budget=12, num_episodes=num_episodes,
                        seed=seed0 + 50, **common),
        evaluate_policy("dcop", dcop_algo="max_sum", dcop_budget=24, num_episodes=num_episodes,
                        seed=seed0 + 60, **common),
        train_and_evaluate_q("shapley", shapley_lambda=0.5, base_seed=seed0 + 70, **qkw, **common),
        evaluate_policy("centralized", num_episodes=num_episodes, seed=seed0 + 80, **common),
    ]
    j_opt = float(res[7]["team_return_mean"])
    j_self = float(res[2]["team_return_mean"])
    rows = []
    for name, r in zip(METHODS, res):
        j = float(r["team_return_mean"])
        rows.append({
            "Condition": cond["name"],
            "Method": name,
            "Team Return": round(j, 2),
            "Return 95% CI": round(r["team_return_ci"], 2),
            "Seed CI": round(r.get("team_return_seed_ci", 0.0), 2),
            "PoA": round(compute_poa(j_opt, j), 3) if j > 0 else "n/a (J<=0)",
            "Gap Recovered (%)": round(100.0 * compute_gap_recovered(j, j_self, j_opt), 1),
            "Gini Index": round(r["gini_mean"], 3),
            "Captures": round(r["captures_mean"], 2),
            "Steps to Clear": round(r["steps_mean"], 2),
            "Msgs/Step": round(r["msgs_per_step_mean"], 2),
        })
    return rows


def _write_csv(rows, path):
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path}")


def run_experiment_4(
    layout_name: str = "proposal_maze",  # kept for backward compatibility (main condition)
    k: int = 3,
    num_episodes: int = 20,
    train_seeds=DEFAULT_TRAIN_SEEDS,
    train_episodes: int = 40,
    conditions=None,
):
    print("==================================================")
    print("Experiment 4: Multi-method benchmark across conditions")
    print("==================================================")
    conditions = conditions or CONDITIONS
    all_rows = []
    for ci, cond in enumerate(conditions):
        print(f"Condition '{cond['name']}' ({cond['layout']}, β={cond['beta']}, density={cond['density']})")
        rows = _benchmark(cond, k, num_episodes, train_seeds, train_episodes, seed0=10_000 * (ci + 1))
        for r in rows:
            print(f"  {r['Method']:28s} J={r['Team Return']:8.1f} recov={r['Gap Recovered (%)']:6.1f}% "
                  f"Gini={r['Gini Index']:.3f} msgs/step={r['Msgs/Step']}")
        all_rows += rows

    main_rows = [r for r in all_rows if r["Condition"] == conditions[0]["name"]]
    _write_csv(main_rows, "results/experiment_4_benchmark.csv")
    _write_csv(all_rows, "results/experiment_4_conditions.csv")
    save_json({"k": k, "ghosts": GHOSTS, "eval_episodes": num_episodes,
               "train_seeds": list(train_seeds), "condition": conditions[0],
               "rows": main_rows}, "experiment_4_benchmark.json")
    save_json({"k": k, "ghosts": GHOSTS, "eval_episodes": num_episodes,
               "train_seeds": list(train_seeds), "conditions": conditions,
               "rows": all_rows}, "experiment_4_conditions.json")

    # Main-condition 2x2 figure
    short = ["Random", "Greedy", "Selfish Q", "Equal-Share", "DSA", "Max-Sum", "Shapley", "π*"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [("Team Return", "Return 95% CI", "Team return J"),
              ("Gap Recovered (%)", None, "Selfish gap recovered (%)"),
              ("Gini Index", None, "Gini index (lower = fairer)"),
              ("Captures", None, "Ghost captures / episode")]
    for ax, (key, err, label) in zip(axes.flat, panels):
        vals = [r[key] for r in main_rows]
        yerr = [r[err] for r in main_rows] if err else None
        ax.bar(short, vals, yerr=yerr, color=COLORS, capsize=3)
        style_axes(ax, label, None, label)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Experiment 4 benchmark, condition: {conditions[0]['name']}", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_e4_method_comparison.png")

    # Heatmap: gap recovered per method x condition
    cond_names = [c["name"] for c in conditions]
    mat = np.array([[next(r["Gap Recovered (%)"] for r in all_rows
                          if r["Condition"] == c and r["Method"] == m) for c in cond_names]
                    for m in METHODS], dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 * len(cond_names) + 3, 6))
    # clip colours so very negative values (e.g. Random) do not wash out the rest;
    # the printed numbers stay exact
    im = ax.imshow(np.clip(mat, -100, 150), cmap="viridis", aspect="auto", vmin=-100, vmax=150)
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, rotation=30, ha="right")
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = "n/a" if np.isnan(mat[i, j]) else f"{mat[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if (np.isnan(mat[i, j]) or mat[i, j] >= 60) else "white",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="Selfish gap recovered (%)")
    ax.set_title("Gap recovered across conditions (selfish Q = 0%, π* = 100%)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_e4_conditions.png")
    return all_rows


if __name__ == "__main__":
    run_experiment_4()
