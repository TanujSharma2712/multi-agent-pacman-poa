"""
Experiment 3: Shapley shaping parameter lambda ACROSS ghost danger beta (tests H3).

Grid: lambda in {0, 0.25, 0.5, 0.75, 1.0} x beta in {0.0, 0.5, 0.9}, plus the
equal-share baseline (r_i = r_team / k) at every beta. Every Q-learning setup is
trained with several seeds (pooled evaluation, CI across seeds reported).

H3 predicts the benefit of Shapley shaping over selfish learning (lambda = 0)
grows with ghost danger beta. We report, per beta, the gain of each lambda > 0
over lambda = 0 and the best gain.

Outputs: results/experiment_3.json,
         plots/fig_e3_shapley_lambda.png, plots/fig_e3_h3_gain_vs_beta.png
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import COLORS, MARKERS, save_figure, save_json, style_axes  # noqa: E402
from src.trainer import DEFAULT_TRAIN_SEEDS, train_and_evaluate_q  # noqa: E402

GHOSTS = 2


def run_experiment_3(
    layout_name: str = "proposal_maze",
    k: int = 3,
    num_episodes: int = 20,
    lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
    betas=(0.0, 0.5, 0.9),
    train_seeds=DEFAULT_TRAIN_SEEDS,
    train_episodes: int = 40,
):
    print("==================================================")
    print("Experiment 3: Shapley lambda x ghost danger beta (H3)")
    print("==================================================")
    per_beta = []
    for bi, beta in enumerate(betas):
        common = dict(layout_name=layout_name, num_pacman=k, num_ghosts=GHOSTS, ghost_beta=beta,
                      train_episodes=train_episodes, eval_episodes=num_episodes,
                      train_seeds=train_seeds)
        lam_rows = []
        for li, lam in enumerate(lambdas):
            mode = "selfish" if lam == 0.0 else "shapley"
            r = train_and_evaluate_q(mode, shapley_lambda=lam, base_seed=3000 + 100 * bi + 10 * li, **common)
            lam_rows.append({
                "lambda": lam,
                "team_return": r["team_return_mean"], "team_return_ci": r["team_return_ci"],
                "team_return_seed_ci": r["team_return_seed_ci"], "seed_means": r["seed_means"],
                "gini": r["gini_mean"], "gini_ci": r["gini_ci"], "captures": r["captures_mean"],
            })
            print(f"beta={beta:.1f} lambda={lam:.2f}: J={r['team_return_mean']:.1f} "
                  f"(seed CI ±{r['team_return_seed_ci']:.1f}) Gini={r['gini_mean']:.3f}")
        eq = train_and_evaluate_q("equal_share", base_seed=3900 + 100 * bi, **common)
        base = lam_rows[0]["team_return"]
        gains = {str(row["lambda"]): row["team_return"] - base for row in lam_rows[1:]}
        best_lam = max(lam_rows[1:], key=lambda row: row["team_return"])
        per_beta.append({
            "beta": beta,
            "lambdas": lam_rows,
            "equal_share": {"team_return": eq["team_return_mean"], "team_return_ci": eq["team_return_ci"],
                            "team_return_seed_ci": eq["team_return_seed_ci"], "gini": eq["gini_mean"]},
            "gain_over_selfish": gains,
            "best_lambda": best_lam["lambda"],
            "best_gain": best_lam["team_return"] - base,
            "best_gain_pct": 100.0 * (best_lam["team_return"] - base) / abs(base) if base else float("nan"),
        })
        print(f"  equal-share J={eq['team_return_mean']:.1f} Gini={eq['gini_mean']:.3f} | "
              f"best lambda={best_lam['lambda']} gain={per_beta[-1]['best_gain']:+.1f}")

    save_json({"layout": layout_name, "k": k, "ghosts": GHOSTS, "eval_episodes": num_episodes,
               "train_seeds": list(train_seeds), "train_episodes": train_episodes,
               "per_beta": per_beta}, "experiment_3.json")

    # Return and Gini vs lambda, one line per beta
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n, row in enumerate(per_beta):
        lam = [x["lambda"] for x in row["lambdas"]]
        axes[0].errorbar(lam, [x["team_return"] for x in row["lambdas"]],
                         yerr=[x["team_return_seed_ci"] for x in row["lambdas"]],
                         marker=MARKERS[n], color=COLORS[n], capsize=3, linewidth=2,
                         label=f"Shapley, β={row['beta']}")
        axes[0].axhline(row["equal_share"]["team_return"], color=COLORS[n], linestyle=":",
                        label=f"equal-share, β={row['beta']}")
        axes[1].plot(lam, [x["gini"] for x in row["lambdas"]], marker=MARKERS[n], color=COLORS[n],
                     linewidth=2, label=f"β={row['beta']}")
        axes[1].axhline(row["equal_share"]["gini"], color=COLORS[n], linestyle=":")
    style_axes(axes[0], "Team return vs λ (95% CI across seeds)", "Shapley weight λ", "Team return J")
    style_axes(axes[1], "Fairness vs λ (dotted: equal-share)", "Shapley weight λ", "Gini index (lower = fairer)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_e3_shapley_lambda.png")

    # H3: gain over selfish vs beta
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, lam in enumerate(lambdas[1:]):
        ax.plot([r["beta"] for r in per_beta], [r["gain_over_selfish"][str(lam)] for r in per_beta],
                marker=MARKERS[n], color=COLORS[n], linewidth=2, label=f"λ={lam}")
    ax.axhline(0.0, color="gray", linestyle=":")
    style_axes(ax, "H3: Shapley gain over selfish Q vs ghost danger", "Ghost aggression β",
               "J(λ) − J(λ=0)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "fig_e3_h3_gain_vs_beta.png")
    return per_beta


if __name__ == "__main__":
    run_experiment_3()
