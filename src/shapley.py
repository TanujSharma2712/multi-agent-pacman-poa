r"""
Shapley Value Computation Module:
Exact calculation of Shapley values for team credit assignment:
phi_i = sum_{S subseteq N \ {i}} [ |S|! (k - |S| - 1)! / k! ] * [ v(S cup {i}) - v(S) ]

Provides:
- Characteristic function evaluation v(S)
- Exact marginal contribution calculation for k <= 4
- Step-wise Shapley reward shaping:
  r_i^lambda = (1 - lambda) * r_i + lambda * phi_i
"""

import itertools
import math
from typing import Callable, Dict, List, Set, Tuple


def get_subsets(agents: List[int]) -> List[Tuple[int, ...]]:
    """Returns all 2^k subsets of agents."""
    subsets = []
    for r in range(len(agents) + 1):
        for combo in itertools.combinations(agents, r):
            subsets.append(combo)
    return subsets


def compute_shapley_values(
    num_agents: int,
    characteristic_fn: Callable[[Tuple[int, ...]], float],
) -> List[float]:
    """
    Computes exact Shapley values for num_agents given characteristic function v(S).
    S is passed as a sorted tuple of agent IDs.
    """
    k = num_agents
    agents = list(range(k))
    shapley_values = [0.0] * k

    # Precompute v(S) for all 2^k subsets
    v_cache: Dict[Tuple[int, ...], float] = {}
    for subset in get_subsets(agents):
        v_cache[subset] = characteristic_fn(subset)

    # Compute Shapley value for each agent
    for i in range(k):
        other_agents = [a for a in agents if a != i]
        phi_i = 0.0

        for subset in get_subsets(other_agents):
            s_size = len(subset)
            weight = (
                math.factorial(s_size)
                * math.factorial(k - s_size - 1)
                / math.factorial(k)
            )

            # Coalition with agent i
            with_i = tuple(sorted(list(subset) + [i]))
            marginal_contrib = v_cache[with_i] - v_cache[subset]
            phi_i += weight * marginal_contrib

        shapley_values[i] = phi_i

    return shapley_values


class StepwiseShapleyCredit:
    """
    Computes step-level Shapley value reward decomposition.
    Decomposes the team reward r_team into fair marginal shares phi_i,
    satisfying efficiency: sum_i phi_i = r_team.
    """
    def __init__(self, num_agents: int):
        self.k = num_agents

    def compute_step_shapley(
        self,
        individual_rewards: List[float],
        pellet_eaten_by: Dict[Tuple[int, int], List[int]],
        caught_agents: Set[int],
        pellet_reward: float = 10.0,
        ghost_penalty: float = 50.0,
        step_penalty: float = 1.0,
    ) -> List[float]:
        """
        Calculates exact step-wise Shapley credit for the transition.
        pellet_eaten_by: map of pellet position -> list of agents attempting to collect it.
        caught_agents: set of agent ids caught by ghosts at this step.
        """
        def characteristic_fn(coalition: Tuple[int, ...]) -> float:
            if not coalition:
                return 0.0
            coalition_set = set(coalition)
            c_reward = 0.0

            # Step penalty for agents in coalition
            c_reward -= step_penalty * len(coalition_set)

            # Ghost penalties
            for i in coalition_set:
                if i in caught_agents:
                    c_reward -= ghost_penalty

            # Pellets collected: if any agent in coalition reached the pellet,
            # the coalition gets the pellet reward!
            # If multiple agents in coalition contest the pellet, only ONE pellet reward is gained!
            for pos, contenders in pellet_eaten_by.items():
                if any(c in coalition_set for c in contenders):
                    c_reward += pellet_reward

            return float(c_reward)

        return compute_shapley_values(self.k, characteristic_fn)

    def shape_rewards(
        self, individual_rewards: List[float], shapley_rewards: List[float], lmbda: float
    ) -> List[float]:
        """
        Equation (P4): U_i^lambda = (1 - lambda) * J_i + lambda * phi_i
        """
        return [
            (1.0 - lmbda) * r_i + lmbda * phi_i
            for r_i, phi_i in zip(individual_rewards, shapley_rewards)
        ]
