"""
Multi-Agent Pac-Man Simulation Environment.
Supports:
- k Pac-Men (N = {0, ..., k-1})
- g Ghosts (M = {0, ..., g-1}) with aggression beta
- Contention resolution: random allocation when multiple Pac-Men reach the same pellet simultaneously
- Ghost collisions with penalty and respawn
- Fast precomputed graph distances (all-pairs shortest paths)
"""

import collections
import random
from typing import Dict, List, Optional, Set, Tuple

from src.ghosts import GhostPolicy
from src.layouts import get_layout

# Action constants
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3
ACTION_STAY = 4

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_RIGHT, ACTION_LEFT, ACTION_STAY]
ACTION_NAMES = ["North", "South", "East", "West", "Stay"]
ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_RIGHT: (1, 0),
    ACTION_LEFT: (-1, 0),
    ACTION_STAY: (0, 0),
}


class MultiAgentPacmanEnv:
    def __init__(
        self,
        layout_name: str = "small_grid",
        num_pacman: int = 2,
        num_ghosts: int = 1,
        ghost_beta: float = 0.5,
        pellet_reward: float = 10.0,
        ghost_penalty: float = 50.0,
        step_penalty: float = 1.0,
        max_steps: int = 100,
        discount_factor: float = 0.99,
        seed: Optional[int] = None,
        pellet_density: float = 1.0,
        pellet_seed: int = 0,
    ):
        """
        pellet_density: fraction of the layout's pellets kept (1.0 = all). Used to
            create pellet SCARCITY (Hypothesis H1). The kept subset is drawn with a
            dedicated RNG seeded by pellet_seed, so every method evaluated on the
            same (layout, density, pellet_seed) sees the identical pellet map.
        """
        self.layout_name = layout_name
        self.pellet_density = pellet_density
        self.pellet_seed = pellet_seed
        self.raw_grid = get_layout(layout_name)
        self.height = len(self.raw_grid)
        self.width = len(self.raw_grid[0])

        self.num_pacman = num_pacman
        self.num_ghosts = num_ghosts
        self.ghost_beta = ghost_beta
        self.pellet_reward = pellet_reward
        self.ghost_penalty = ghost_penalty
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        self.gamma = discount_factor

        self.rng = random.Random(seed)
        self.ghost_policy = GhostPolicy(beta=ghost_beta, rng=self.rng)

        # Parse static structure
        self.walls: Set[Tuple[int, int]] = set()
        self.open_cells: Set[Tuple[int, int]] = set()
        self.power_pellets: Set[Tuple[int, int]] = set()
        self.default_pacman_starts: List[Tuple[int, int]] = []
        self.default_ghost_starts: List[Tuple[int, int]] = []
        self.initial_pellet_set: Set[Tuple[int, int]] = set()

        self._parse_layout()
        self._apply_pellet_density()
        self._precompute_shortest_paths()

        # Dynamic state
        self.pacman_positions: List[Tuple[int, int]] = []
        self.ghost_positions: List[Tuple[int, int]] = []
        self.pacman_starts: List[Tuple[int, int]] = []
        self.ghost_starts: List[Tuple[int, int]] = []
        self.pellets: Set[Tuple[int, int]] = set()
        self.step_count = 0
        self.total_captures = 0
        self.agent_captures = [0] * self.num_pacman
        self.agent_pellets = [0] * self.num_pacman

        self.reset(seed=seed)

    def __deepcopy__(self, memo):
        """High-performance custom deepcopy sharing static graph structure."""
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("dist_matrix", "raw_grid", "walls", "open_cells", "power_pellets"):
                setattr(result, k, v)  # Shared immutable static graph data
            elif isinstance(v, list):
                setattr(result, k, list(v))
            elif isinstance(v, set):
                setattr(result, k, set(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _parse_layout(self):
        for y, row in enumerate(self.raw_grid):
            for x, char in enumerate(row):
                pos = (x, y)
                if char in ("#", "%"):
                    self.walls.add(pos)
                else:
                    self.open_cells.add(pos)
                    if char == ".":
                        self.initial_pellet_set.add(pos)
                    elif char == "o":
                        self.initial_pellet_set.add(pos)
                        self.power_pellets.add(pos)
                    elif char == "P":
                        self.default_pacman_starts.append(pos)
                        self.initial_pellet_set.add(pos)
                    elif char == "G":
                        self.default_ghost_starts.append(pos)

        # Sort starts consistently: Pac-Men left-to-right (Region A, B, C), Ghosts top-to-bottom
        self.default_pacman_starts.sort(key=lambda p: (p[0], p[1]))
        self.default_ghost_starts.sort(key=lambda g: (g[1], g[0]))

        # If not enough starts defined in layout, pick open cells
        available_open = sorted(list(self.open_cells))
        while len(self.default_pacman_starts) < self.num_pacman:
            cand = available_open[len(self.default_pacman_starts) % len(available_open)]
            if cand not in self.default_pacman_starts:
                self.default_pacman_starts.append(cand)

        while len(self.default_ghost_starts) < self.num_ghosts:
            cand = available_open[-(len(self.default_ghost_starts) + 1) % len(available_open)]
            if cand not in self.default_ghost_starts:
                self.default_ghost_starts.append(cand)

    def _apply_pellet_density(self):
        """Keep only a fixed random fraction of pellets (pellet scarcity)."""
        if self.pellet_density >= 1.0:
            return
        all_pellets = sorted(self.initial_pellet_set)
        n_keep = max(1, int(round(self.pellet_density * len(all_pellets))))
        kept = random.Random(self.pellet_seed).sample(all_pellets, n_keep)
        self.initial_pellet_set = set(kept)
        self.power_pellets &= self.initial_pellet_set

    def _precompute_shortest_paths(self):
        """BFS all-pairs shortest paths on open cells."""
        self.dist_matrix: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        for start in self.open_cells:
            distances = {start: 0}
            queue = collections.deque([start])
            while queue:
                curr = queue.popleft()
                cx, cy = curr
                curr_dist = distances[curr]
                for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                    nbr = (cx + dx, cy + dy)
                    if nbr in self.open_cells and nbr not in distances:
                        distances[nbr] = curr_dist + 1
                        queue.append(nbr)
            self.dist_matrix[start] = distances

    def get_distance(self, u: Tuple[int, int], v: Tuple[int, int]) -> int:
        """Returns shortest path distance between two cells, or large value if unreachable."""
        if u not in self.dist_matrix or v not in self.dist_matrix[u]:
            return 9999
        return self.dist_matrix[u][v]

    def get_valid_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        nbrs = []
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            np = (x + dx, y + dy)
            if np in self.open_cells:
                nbrs.append(np)
        return nbrs

    def get_valid_actions(self, pos: Tuple[int, int]) -> List[int]:
        valid = []
        x, y = pos
        for a, (dx, dy) in ACTION_DELTAS.items():
            np = (x + dx, y + dy)
            if a == ACTION_STAY or np in self.open_cells:
                valid.append(a)
        return valid

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.ghost_policy.rng = self.rng

        self.step_count = 0
        self.total_captures = 0
        self.agent_captures = [0] * self.num_pacman
        self.agent_pellets = [0] * self.num_pacman

        # Set starting positions
        self.pacman_starts = list(self.default_pacman_starts[: self.num_pacman])
        self.ghost_starts = list(self.default_ghost_starts[: self.num_ghosts])

        self.pacman_positions = list(self.pacman_starts)
        self.ghost_positions = list(self.ghost_starts)
        self.pellets = set(self.initial_pellet_set)

        return self.get_state(), {}

    def get_state(self) -> dict:
        return {
            "pacman_positions": list(self.pacman_positions),
            "ghost_positions": list(self.ghost_positions),
            "pellets": set(self.pellets),
            "step_count": self.step_count,
            "pellets_remaining": len(self.pellets),
        }

    def step(self, joint_action: List[int]) -> Tuple[dict, List[float], bool, dict]:
        """
        Execute one synchronous step in the environment.
        joint_action: list of actions for each Pac-Man [a_0, a_1, ..., a_{k-1}].
        Returns: next_state, rewards, done, info
        """
        self.step_count += 1
        rewards = [-self.step_penalty] * self.num_pacman

        old_pacman_pos = list(self.pacman_positions)
        new_pacman_pos = []

        # 1. Update Pac-Man positions
        for i, action in enumerate(joint_action):
            dx, dy = ACTION_DELTAS.get(action, (0, 0))
            cand = (self.pacman_positions[i][0] + dx, self.pacman_positions[i][1] + dy)
            if cand in self.open_cells:
                new_pacman_pos.append(cand)
            else:
                new_pacman_pos.append(self.pacman_positions[i])

        self.pacman_positions = new_pacman_pos

        # 2. Update Ghost positions
        old_ghost_pos = list(self.ghost_positions)
        new_ghost_pos = []
        for j in range(self.num_ghosts):
            g_pos = self.ghost_positions[j]
            valid_moves = self.get_valid_neighbors(g_pos) + [g_pos]
            next_g_pos = self.ghost_policy.choose_action(
                ghost_pos=g_pos,
                pacman_positions=self.pacman_positions,
                valid_moves=valid_moves,
                dist_fn=self.get_distance,
            )
            new_ghost_pos.append(next_g_pos)
        self.ghost_positions = new_ghost_pos

        # 3. Check Ghost Collisions
        # Caught if:
        # a) Ghost and Pac-Man occupy the same cell
        # b) Ghost and Pac-Man crossed paths (swapped positions)
        for i in range(self.num_pacman):
            caught = False
            for j in range(self.num_ghosts):
                same_cell = self.pacman_positions[i] == self.ghost_positions[j]
                swapped = (
                    self.pacman_positions[i] == old_ghost_pos[j]
                    and self.ghost_positions[j] == old_pacman_pos[i]
                )
                if same_cell or swapped:
                    caught = True
                    break

            if caught:
                rewards[i] -= self.ghost_penalty
                self.agent_captures[i] += 1
                self.total_captures += 1
                # Respawn at start position
                self.pacman_positions[i] = self.pacman_starts[i]

        # 4. Pellet collection and contention resolution
        # Group agents by the pellet they landed on
        pellet_contenders: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
        for i, pos in enumerate(self.pacman_positions):
            if pos in self.pellets:
                pellet_contenders[pos].append(i)

        for pos, contenders in pellet_contenders.items():
            self.pellets.remove(pos)
            # Award to exactly one agent chosen uniformly at random if simultaneous
            winner = self.rng.choice(contenders) if len(contenders) > 1 else contenders[0]
            rewards[winner] += self.pellet_reward
            self.agent_pellets[winner] += 1

        # Check termination
        done = (len(self.pellets) == 0) or (self.step_count >= self.max_steps)

        info = {
            "pellets_remaining": len(self.pellets),
            "step_count": self.step_count,
            "total_captures": self.total_captures,
            "agent_captures": list(self.agent_captures),
            "agent_pellets": list(self.agent_pellets),
            "team_reward": sum(rewards),
        }

        return self.get_state(), rewards, done, info
