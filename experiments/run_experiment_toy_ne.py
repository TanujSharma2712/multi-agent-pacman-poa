"""
Experiment 1-T (proposal method M1): EXACT Price of Anarchy on a toy maze.

On the 5x5 ring maze 'toy_exact' (k=2 Pac-Men, 1 ghost, 7 pellets, horizon T=12)
the whole joint state space is enumerated, so we can compute:
- the EXACT team optimum J(pi*) by dynamic programming (P1),
- EXACT best responses, and Nash equilibria via iterated best response from
  several starting profiles (P2), giving the worst / best equilibrium found,
- PoA = J(pi*) / min_NE J  (P3) and Price of Stability = J(pi*) / max_NE J.
If IBR cycles (no pure NE in this policy class), the epsilon-NE with the
smallest Nash gap is reported and flagged as such.

It also validates the approximations used on large mazes:
- Hungarian centralized planner vs the exact optimum  (how close is our pi*?),
- independent Q-learning vs the exact equilibrium     (is it a fair 'selfish' proxy?),
- Monte-Carlo replay of the exact optimum inside the real simulator (model check).

Outputs: results/experiment_toy_ne.json, plots/fig_toy_exact_poa.png
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import COLORS, save_figure, save_json, strip_samples, style_axes  # noqa: E402
from src.exact_solver import ExactGame  # noqa: E402
from src.trainer import DEFAULT_TRAIN_SEEDS, evaluate_policy, train_and_evaluate_q  # noqa: E402

LAYOUT = "toy_exact"
K = 2
GHOSTS = 1


def run_experiment_toy_ne(
    betas=(0.0, 0.5, 0.9),
    horizon: int = 12,
    num_random_starts: int = 3,
    max_iters: int = 12,
    num_episodes: int = 200,
    train_seeds=DEFAULT_TRAIN_SEEDS,
    train_episodes: int = 60,
    mc_episodes: int = 1000,
):
    print("==================================================")
    print("Experiment 1-T: Exact PoA on toy maze (iterated best response)")
    print("==================================================")
    rows = []
    for beta in betas:
        t0 = time.time()
        game = ExactGame(LAYOUT, num_pacman=K, num_ghosts=GHOSTS, ghost_beta=beta, horizon=horizon)
        print(f"beta={beta}: {game.num_states()} reachable joint states")
        res = game.find_equilibria(num_random_starts=num_random_starts, max_iters=max_iters)
        res["mc_check_opt"] = game.simulate_profile(res["_opt_profile"], num_episodes=mc_episodes)

        common = dict(layout_name=LAYOUT, num_pacman=K, num_ghosts=GHOSTS, ghost_beta=beta,
                      max_steps=horizon)
        heur = evaluate_policy("centralized", num_episodes=num_episodes, seed=11, **common)
        greedy = evaluate_policy("greedy", num_episodes=num_episodes, seed=12, **common)
        q_self = train_and_evaluate_q("selfish", train_episodes=train_episodes,
                                      eval_episodes=max(20, num_episodes // len(train_seeds)),
                                      train_seeds=train_seeds, base_seed=13, **common)
        res["heuristic_planner"] = strip_samples(heur)
        res["greedy"] = strip_samples(greedy)
        res["q_selfish"] = strip_samples(q_self)
        res["heuristic_vs_exact_opt_pct"] = 100.0 * heur["team_return_mean"] / res["j_opt"]
        if "j_worst_ne" in res:
            res["poa_q_proxy"] = res["j_opt"] / q_self["team_return_mean"] if q_self["team_return_mean"] > 0 else float("inf")
        res["seconds"] = time.time() - t0

        print(f"  {res['equilibrium_type']}")
        print(f"  J*={res['j_opt']:.2f} (MC replay {res['mc_check_opt']:.2f}) | "
              f"worst NE={res.get('j_worst_ne')} best NE={res.get('j_best_ne')} | "
              f"PoA={res.get('poa', float('nan')):.3f} PoS={res.get('pos', float('nan')):.3f}")
        print(f"  heuristic planner={heur['team_return_mean']:.2f} "
              f"({res['heuristic_vs_exact_opt_pct']:.1f}% of exact optimum) | "
              f"Q selfish={q_self['team_return_mean']:.2f} | greedy={greedy['team_return_mean']:.2f} "
              f"| {res['seconds']:.0f}s")
        rows.append(res)

    save_json({"layout": LAYOUT, "k": K, "ghosts": GHOSTS, "horizon": horizon,
               "train_seeds": list(train_seeds), "results": rows}, "experiment_toy_ne.json")
    _plot(rows)
    return rows


def _plot(rows):
    betas = [r["beta"] for r in rows]
    series = [
        ("Exact optimum J(π*)", [r["j_opt"] for r in rows]),
        ("Best equilibrium", [r.get("j_best_ne", np.nan) for r in rows]),
        ("Worst equilibrium", [r.get("j_worst_ne", np.nan) for r in rows]),
        ("Hungarian planner", [r["heuristic_planner"]["team_return_mean"] for r in rows]),
        ("Independent Q (selfish)", [r["q_selfish"]["team_return_mean"] for r in rows]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(betas))
    w = 0.16
    for n, (label, vals) in enumerate(series):
        axes[0].bar(x + (n - 2) * w, vals, w, label=label, color=COLORS[n])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"β={b}" for b in betas])
    style_axes(axes[0], "Toy maze: exact optimum vs equilibria", None, "Expected discounted team return")
    axes[0].legend(fontsize=8, loc="lower right")

    poa = [r.get("poa", np.nan) for r in rows]
    pos = [r.get("pos", np.nan) for r in rows]
    exact = [("exact" in r.get("equilibrium_type", "")) and "no pure" not in r.get("equilibrium_type", "") for r in rows]
    axes[1].plot(betas, poa, marker="o", color=COLORS[2], linewidth=2.2, label="PoA (worst NE)")
    axes[1].plot(betas, pos, marker="s", color=COLORS[0], linewidth=2.2, label="Price of Stability (best NE)")
    for b, p, ex in zip(betas, poa, exact):
        axes[1].annotate("exact NE" if ex else "ε-NE", (b, p), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[1].axhline(1.0, color="gray", linestyle=":")
    style_axes(axes[1], "Exact PoA via iterated best response", "Ghost aggression β", "Ratio")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig_toy_exact_poa.png")


if __name__ == "__main__":
    run_experiment_toy_ne()
