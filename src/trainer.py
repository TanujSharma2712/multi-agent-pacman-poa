"""
Training and Evaluation Framework for Multi-Agent Pac-Man:
Coordinates:
- Training loops for Selfish Q, Equal-Share Q, and Shapley-Shaped Q
- Multi-seed training + evaluation (train_and_evaluate_q)
- Evaluation loops across multiple random seeds
- Computation of returns, fairness, captures, completion steps and DCOP messages per step
"""

import copy
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.baselines import CentralizedTeamPlanner, IndependentGreedyAgent, RandomAgent
from src.dcop import DCOPAgent, DCOPSolver, RegionPartition
from src.environment import MultiAgentPacmanEnv
from src.metrics import compute_ci95, compute_gini_index
from src.q_learning import QLearningAgent
from src.shapley import StepwiseShapleyCredit

DEFAULT_TRAIN_SEEDS = (0, 1, 2)


def train_q_learning(
    layout_name: str = "small_grid",
    num_pacman: int = 2,
    num_ghosts: int = 1,
    ghost_beta: float = 0.5,
    mode: str = "selfish",  # 'selfish', 'equal_share', or 'shapley'
    shapley_lambda: float = 0.5,
    num_episodes: int = 100,
    alpha: float = 0.05,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.02,
    seed: int = 42,
    pellet_density: float = 1.0,
    max_steps: int = 100,
) -> List[QLearningAgent]:
    """
    Trains k Q-learning agents under the specified reward formulation.
    """
    env = MultiAgentPacmanEnv(
        layout_name=layout_name,
        num_pacman=num_pacman,
        num_ghosts=num_ghosts,
        ghost_beta=ghost_beta,
        seed=seed,
        pellet_density=pellet_density,
        max_steps=max_steps,
    )

    agents = [
        QLearningAgent(
            agent_id=i,
            alpha=alpha,
            gamma=env.gamma,
            epsilon=epsilon_start,
            rng=random.Random(seed + i * 17),
        )
        for i in range(num_pacman)
    ]

    shapley_credit = StepwiseShapleyCredit(num_pacman)

    for ep in range(num_episodes):
        # Epsilon decay
        cur_eps = epsilon_start - (epsilon_start - epsilon_end) * (ep / max(1, num_episodes))
        for ag in agents:
            ag.epsilon = cur_eps

        env.reset(seed=seed + ep * 101)
        done = False
        prev_captures = [0] * num_pacman

        while not done:
            # Snapshot environment for feature calculation
            env_before = copy.deepcopy(env)

            # Choose actions
            joint_action = [ag.get_action(env_before, explore=True) for ag in agents]

            # Collect contender information for Shapley credit before step
            next_positions = []
            for i, a in enumerate(joint_action):
                dx, dy = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)][a]
                cand = (env.pacman_positions[i][0] + dx, env.pacman_positions[i][1] + dy)
                next_positions.append(cand if cand in env.open_cells else env.pacman_positions[i])

            pellet_eaten_by: Dict = {}
            for i, p_pos in enumerate(next_positions):
                if p_pos in env.pellets:
                    pellet_eaten_by.setdefault(p_pos, []).append(i)

            # Step environment
            _, rewards, done, info = env.step(joint_action)

            # Agents caught at THIS step only (captures are cumulative in info).
            caught_now = {
                i for i in range(num_pacman) if info["agent_captures"][i] > prev_captures[i]
            }
            prev_captures = list(info["agent_captures"])

            # Determine reward shaping
            if mode == "selfish":
                effective_rewards = list(rewards)
            elif mode == "equal_share":
                team_r = sum(rewards)
                effective_rewards = [team_r / num_pacman] * num_pacman
            elif mode == "shapley":
                # A caught agent is respawned, so it cannot also have eaten the pellet
                # at its pre-capture target: drop it from the contender lists.
                pellet_eaten_by = {
                    p: [i for i in who if i not in caught_now]
                    for p, who in pellet_eaten_by.items()
                }
                pellet_eaten_by = {p: who for p, who in pellet_eaten_by.items() if who}
                phi = shapley_credit.compute_step_shapley(
                    individual_rewards=rewards,
                    pellet_eaten_by=pellet_eaten_by,
                    caught_agents=caught_now,
                    pellet_reward=env.pellet_reward,
                    ghost_penalty=env.ghost_penalty,
                    step_penalty=env.step_penalty,
                )
                effective_rewards = shapley_credit.shape_rewards(
                    rewards, phi, lmbda=shapley_lambda
                )
            else:
                effective_rewards = list(rewards)

            # Q-learning update
            for i in range(num_pacman):
                agents[i].update(
                    env=env_before,
                    action=joint_action[i],
                    reward=effective_rewards[i],
                    next_env=env,
                    done=done,
                )

    return agents


def evaluate_policy(
    policy_type: str,
    layout_name: str = "small_grid",
    num_pacman: int = 2,
    num_ghosts: int = 1,
    ghost_beta: float = 0.5,
    trained_agents: Optional[List[QLearningAgent]] = None,
    dcop_algo: str = "dsa",  # 'dsa' or 'max_sum'
    dcop_budget: int = 10,
    num_episodes: int = 30,
    seed: int = 1234,
    pellet_density: float = 1.0,
    max_steps: int = 100,
    dcop_regions: Optional[int] = None,
) -> Dict[str, object]:
    """
    Evaluates a specific policy type over num_episodes.
    Supported types:
    - 'random'
    - 'greedy'
    - 'centralized'
    - 'q_learning'
    - 'dcop'   (dcop_regions = number of regions m; default max(2, k))

    Also reports DCOP communication actually used: msgs_per_step_mean
    (0 for policies that do not communicate).
    """
    team_returns = []
    agent_returns_all = []
    captures_all = []
    steps_all = []
    ginis = []
    msgs_per_step_all = []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 73
        env = MultiAgentPacmanEnv(
            layout_name=layout_name,
            num_pacman=num_pacman,
            num_ghosts=num_ghosts,
            ghost_beta=ghost_beta,
            seed=ep_seed,
            pellet_density=pellet_density,
            max_steps=max_steps,
        )

        # Policy setup
        if policy_type == "random":
            agents = [RandomAgent(i, rng=random.Random(ep_seed + i)) for i in range(num_pacman)]
        elif policy_type == "greedy":
            agents = [
                IndependentGreedyAgent(i, rng=random.Random(ep_seed + i))
                for i in range(num_pacman)
            ]
        elif policy_type == "centralized":
            planner = CentralizedTeamPlanner(rng=random.Random(ep_seed))
        elif policy_type == "dcop":
            m = dcop_regions if dcop_regions is not None else max(2, num_pacman)
            partition = RegionPartition(env, num_regions=m)
            solver = DCOPSolver(partition, rng=random.Random(ep_seed))
            agents = [
                DCOPAgent(i, partition=partition, rng=random.Random(ep_seed + i))
                for i in range(num_pacman)
            ]
        elif policy_type == "q_learning":
            agents = trained_agents
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        env.reset(seed=ep_seed)
        done = False
        discounted_returns = [0.0] * num_pacman
        current_gamma = 1.0
        msgs_this_ep = 0

        while not done:
            if policy_type == "centralized":
                joint_action = planner.get_joint_action(env)
            elif policy_type == "dcop":
                # Solve DCOP region assignment under message budget
                if dcop_algo == "dsa":
                    z, used = solver.solve_dsa(message_budget=dcop_budget)
                else:
                    z, used = solver.solve_max_sum(message_budget=dcop_budget)
                msgs_this_ep += used
                for i, r in enumerate(z):
                    agents[i].set_region(r)
                joint_action = [agents[i].get_action(env) for i in range(num_pacman)]
            elif policy_type == "q_learning":
                joint_action = [agents[i].get_action(env, explore=False) for i in range(num_pacman)]
            else:  # random or greedy
                joint_action = [agents[i].get_action(env) for i in range(num_pacman)]

            _, rewards, done, info = env.step(joint_action)

            for i in range(num_pacman):
                discounted_returns[i] += current_gamma * rewards[i]
            current_gamma *= env.gamma

        team_ret = sum(discounted_returns)
        team_returns.append(team_ret)
        agent_returns_all.append(discounted_returns)
        captures_all.append(info["total_captures"])
        steps_all.append(info["step_count"])
        ginis.append(compute_gini_index(discounted_returns))
        msgs_per_step_all.append(msgs_this_ep / max(1, info["step_count"]))

    return _summarize(policy_type, team_returns, agent_returns_all, captures_all,
                      steps_all, ginis, msgs_per_step_all)


def _summarize(policy_type, team_returns, agent_returns_all, captures_all, steps_all,
               ginis, msgs_per_step_all) -> Dict[str, object]:
    mean_ret, ci_ret, _ = compute_ci95(team_returns)
    mean_captures, ci_captures, _ = compute_ci95(captures_all)
    mean_steps, ci_steps, _ = compute_ci95(steps_all)
    mean_gini, ci_gini, _ = compute_ci95(ginis)
    mean_msgs, ci_msgs, _ = compute_ci95(msgs_per_step_all)
    return {
        "policy_type": policy_type,
        "team_return_mean": mean_ret,
        "team_return_ci": ci_ret,
        "captures_mean": mean_captures,
        "captures_ci": ci_captures,
        "steps_mean": mean_steps,
        "steps_ci": ci_steps,
        "gini_mean": mean_gini,
        "gini_ci": ci_gini,
        "msgs_per_step_mean": mean_msgs,
        "msgs_per_step_ci": ci_msgs,
        "all_team_returns": team_returns,
        "all_agent_returns": agent_returns_all,
        "all_captures": list(captures_all),
        "all_steps": list(steps_all),
    }


def train_and_evaluate_q(
    mode: str,
    layout_name: str,
    num_pacman: int,
    num_ghosts: int,
    ghost_beta: float,
    shapley_lambda: float = 0.5,
    train_episodes: int = 40,
    eval_episodes: int = 20,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    base_seed: int = 0,
    pellet_density: float = 1.0,
    max_steps: int = 100,
) -> Dict[str, object]:
    """
    Trains one Q-learning team per training seed, evaluates each, and pools the
    results. Reported CI ('team_return_ci') is over pooled evaluation episodes;
    'team_return_seed_ci' is the 95% CI across per-seed means (seed-to-seed
    variability of the learned policy), which is the more conservative number.
    """
    team_returns, agent_returns, captures, steps, ginis, msgs = [], [], [], [], [], []
    seed_means, seed_ginis = [], []
    for ts in train_seeds:
        agents = train_q_learning(
            layout_name=layout_name, num_pacman=num_pacman, num_ghosts=num_ghosts,
            ghost_beta=ghost_beta, mode=mode, shapley_lambda=shapley_lambda,
            num_episodes=train_episodes, seed=base_seed + 1000 * ts + 7,
            pellet_density=pellet_density, max_steps=max_steps,
        )
        r = evaluate_policy(
            "q_learning", layout_name=layout_name, num_pacman=num_pacman,
            num_ghosts=num_ghosts, ghost_beta=ghost_beta, trained_agents=agents,
            num_episodes=eval_episodes, seed=base_seed + 1000 * ts + 500,
            pellet_density=pellet_density, max_steps=max_steps,
        )
        team_returns += r["all_team_returns"]
        agent_returns += r["all_agent_returns"]
        seed_means.append(r["team_return_mean"])
        seed_ginis.append(r["gini_mean"])
        captures += r["all_captures"]
        steps += r["all_steps"]
        ginis += [compute_gini_index(a) for a in r["all_agent_returns"]]
        msgs += [0.0] * len(r["all_team_returns"])

    out = _summarize(f"q_learning:{mode}", team_returns, agent_returns, captures, steps, ginis, msgs)
    _, seed_ci, _ = compute_ci95(seed_means)
    out["team_return_seed_ci"] = seed_ci
    out["seed_means"] = [float(x) for x in seed_means]
    out["seed_ginis"] = [float(x) for x in seed_ginis]
    out["num_train_seeds"] = len(list(train_seeds))
    return out
