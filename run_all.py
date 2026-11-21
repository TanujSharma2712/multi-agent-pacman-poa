"""
Master Runner & Reproducibility Suite:
1. Renders the maze visualization (Figure 1).
2. Generates animated simulation GIFs.
3. Experiment 1-T: EXACT PoA on a toy maze (exact optimum + iterated best response).
4. Experiment 1: PoA vs team size k, ghost aggression beta and pellet scarcity.
5. Experiment 2: DCOP message budget B and number of regions m (DSA vs Max-Sum).
6. Experiment 3: Shapley lambda across ghost danger beta (H3).
7. Experiment 4: 8-method benchmark across conditions and layouts.

Usage:
  python run_all.py            full run (multi-seed; roughly an hour on a laptop)
  python run_all.py --quick    smoke test with tiny settings (a few minutes)
  python run_all.py --only toy e1 e2 e3 e4 viz   run selected stages
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from experiments.run_experiment_1 import run_experiment_1  # noqa: E402
from experiments.run_experiment_2 import run_experiment_2  # noqa: E402
from experiments.run_experiment_3 import run_experiment_3  # noqa: E402
from experiments.run_experiment_4 import run_experiment_4  # noqa: E402
from experiments.run_experiment_toy_ne import run_experiment_toy_ne  # noqa: E402

STAGES = ["viz", "toy", "e1", "e2", "e3", "e4"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    parser.add_argument("--only", nargs="*", choices=STAGES, help="run only these stages")
    args = parser.parse_args()
    stages = args.only or STAGES

    if args.quick:
        ep, seeds, tr = 3, (0,), 4
        toy = dict(betas=(0.0,), horizon=8, num_random_starts=1, max_iters=4,
                   num_episodes=20, train_seeds=seeds, train_episodes=5, mc_episodes=50)
    else:
        ep, seeds, tr = 20, (0, 1, 2), 40
        toy = dict()
    q = dict(num_episodes=ep, train_seeds=seeds, train_episodes=tr)

    start = time.time()
    print("=" * 69)
    print(f"MULTI-AGENT PAC-MAN PoA TESTBED  ({'QUICK' if args.quick else 'FULL'} run, stages: {stages})")
    print("=" * 69)

    if "viz" in stages:
        from generate_animations import generate_side_by_side_gif, generate_single_policy_gif
        from visualize import generate_maze_visualization
        print("\n[viz] Maze figure and gameplay GIFs...")
        generate_maze_visualization(layout_name="proposal_maze", num_pacman=3, num_ghosts=2)
        if not args.quick:
            generate_side_by_side_gif(num_steps=45, out_path="plots/comparison_selfish_vs_dcop.gif")
            generate_single_policy_gif(policy_type="dcop", num_steps=40, out_path="plots/pacman_dcop_simulation.gif")
            generate_single_policy_gif(policy_type="shapley", num_steps=40, out_path="plots/pacman_shapley_simulation.gif")
    if "toy" in stages:
        print("\n[toy] Exact PoA on toy maze...")
        run_experiment_toy_ne(**toy)
    if "e1" in stages:
        print("\n[e1] PoA vs k, beta, scarcity...")
        if args.quick:
            run_experiment_1(k_values=(2, 3), beta_values=(0.0, 0.9), densities=(1.0, 0.25), **q)
        else:
            run_experiment_1(**q)
    if "e2" in stages:
        print("\n[e2] DCOP budget and regions...")
        if args.quick:
            run_experiment_2(budgets=(0, 12, 24), region_counts=(2, 3), **q)
        else:
            run_experiment_2(**q)
    if "e3" in stages:
        print("\n[e3] Shapley lambda x beta...")
        if args.quick:
            run_experiment_3(lambdas=(0.0, 0.5), betas=(0.0, 0.9), **q)
        else:
            run_experiment_3(**q)
    if "e4" in stages:
        print("\n[e4] Multi-method benchmark across conditions...")
        run_experiment_4(**q)

    print("\n" + "=" * 69)
    print(f"DONE in {time.time() - start:.1f}s. Results: results/  Figures: plots/")
    print("=" * 69)


if __name__ == "__main__":
    main()
