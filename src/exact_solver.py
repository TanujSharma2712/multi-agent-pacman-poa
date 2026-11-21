"""
Exact game-theoretic analysis on tiny mazes (proposal method M1).

On a maze small enough to enumerate every joint state, we solve the finite-horizon
stochastic game EXACTLY with dynamic programming, using the same rules as
MultiAgentPacmanEnv (synchronous moves, ghosts pursue the nearest Pac-Man with
probability beta, catch on same cell or swap, respawn, random winner on a contested
pellet, step penalty, discount gamma, episode ends when pellets are cleared or at
the horizon).

Provides:
- solve_team_optimum():     exact social optimum pi* (P1), maximizing E[sum_i J_i].
- best_response(i, profile): exact best response of Pac-Man i to fixed policies of
                             the others (single-agent MDP over the joint state).
- iterated_best_response():  IBR from a starting profile; a fixed point is a pure
                             Nash equilibrium (P2) in time-dependent Markov policies.
- find_equilibria():         runs IBR from many starting profiles to approximate the
                             set Pi_NE, returning the worst / best equilibrium found,
                             hence PoA (P3) = J(pi*) / min J(NE) and the Price of
                             Stability = J(pi*) / max J(NE).
- simulate_profile():        Monte-Carlo replay of a solved profile inside the real
                             environment, to verify the exact model matches the simulator.

State  s = (pacman cell indices, ghost cell indices, remaining-pellet bitmask).
Policy pi_i[t][s] = next cell index chosen by Pac-Man i at time t in state s.
Policies are defined on EVERY reachable state, so best responses stay well defined
whatever the opponents do.
"""

import itertools
import random
from typing import Dict, List, Optional, Tuple

from src.environment import ACTION_DELTAS, ACTION_STAY, MultiAgentPacmanEnv

State = Tuple[Tuple[int, ...], Tuple[int, ...], int]


class ExactGame:
    def __init__(
        self,
        layout_name: str = "toy_exact",
        num_pacman: int = 2,
        num_ghosts: int = 1,
        ghost_beta: float = 0.5,
        horizon: int = 12,
        pellet_density: float = 1.0,
    ):
        self.env = MultiAgentPacmanEnv(
            layout_name=layout_name,
            num_pacman=num_pacman,
            num_ghosts=num_ghosts,
            ghost_beta=ghost_beta,
            max_steps=horizon,
            seed=0,
            pellet_density=pellet_density,
        )
        env = self.env
        self.layout_name = layout_name
        self.k = num_pacman
        self.g = num_ghosts
        self.beta = ghost_beta
        self.T = horizon
        self.gamma = env.gamma
        self.R_pellet = env.pellet_reward
        self.R_ghost = env.ghost_penalty
        self.R_step = env.step_penalty

        # Cell indexing
        self.cells: List[Tuple[int, int]] = sorted(env.open_cells)
        self.idx: Dict[Tuple[int, int], int] = {c: n for n, c in enumerate(self.cells)}
        n = len(self.cells)
        self.nbrs: List[List[int]] = [
            [self.idx[q] for q in env.get_valid_neighbors(c)] for c in self.cells
        ]
        # Pac-Man moves: neighbours + stay (moving into a wall == stay)
        self.moves: List[List[int]] = [sorted(set(self.nbrs[c] + [c])) for c in range(n)]
        self.dist = [[env.get_distance(a, b) for b in self.cells] for a in self.cells]

        # Pellet bit for each cell (-1 if the cell never holds a pellet)
        self.pellet_bit = [-1] * n
        for b, c in enumerate(sorted(env.initial_pellet_set)):
            self.pellet_bit[self.idx[c]] = b
        full_mask = (1 << len(env.initial_pellet_set)) - 1

        self.pac_starts = tuple(self.idx[c] for c in env.pacman_starts)
        ghost_starts = tuple(self.idx[c] for c in env.ghost_starts)
        self.start: State = (self.pac_starts, ghost_starts, full_mask)

        self._trans: Dict[Tuple[State, Tuple[int, ...]], list] = {}
        self._ghost_cache: Dict[Tuple[int, Tuple[int, ...]], List[Tuple[int, float]]] = {}
        self.layers: List[List[State]] = self._enumerate_reachable()

    # ------------------------------------------------------------------ dynamics
    def _ghost_distribution(self, gpos: int, pac: Tuple[int, ...]) -> List[Tuple[int, float]]:
        """Exact next-cell distribution of one ghost (mirrors GhostPolicy)."""
        key = (gpos, pac)
        if key in self._ghost_cache:
            return self._ghost_cache[key]
        valid = self.nbrs[gpos] + [gpos]
        probs: Dict[int, float] = {m: (1.0 - self.beta) / len(valid) for m in valid}
        if pac:
            # nearest Pac-Man: first index with strictly smallest distance
            nearest = pac[0]
            best = self.dist[gpos][pac[0]]
            for p in pac[1:]:
                if self.dist[gpos][p] < best:
                    best, nearest = self.dist[gpos][p], p
            dmin = min(self.dist[m][nearest] for m in valid)
            cand = [m for m in valid if self.dist[m][nearest] == dmin]
            for m in cand:
                probs[m] += self.beta / len(cand)
        else:
            probs = {m: 1.0 / len(valid) for m in valid}
        out = [(m, p) for m, p in probs.items() if p > 0.0]
        self._ghost_cache[key] = out
        return out

    def transitions(self, s: State, ja: Tuple[int, ...]):
        """List of (prob, next_state, reward_vector) for joint move ja (next cells)."""
        key = (s, ja)
        cached = self._trans.get(key)
        if cached is not None:
            return cached
        old_pac, old_ghost, mask = s
        k = self.k
        ghost_dists = [self._ghost_distribution(gp, ja) for gp in old_ghost]
        out = []
        for combo in itertools.product(*ghost_dists) if ghost_dists else [()]:
            prob = 1.0
            new_ghost = []
            for cell, p in combo:
                prob *= p
                new_ghost.append(cell)
            pos = list(ja)
            rew = [-self.R_step] * k
            for i in range(k):
                for j in range(self.g):
                    if pos[i] == new_ghost[j] or (
                        pos[i] == old_ghost[j] and new_ghost[j] == old_pac[i]
                    ):
                        rew[i] -= self.R_ghost
                        pos[i] = self.pac_starts[i]
                        break
            contenders: Dict[int, List[int]] = {}
            for i, c in enumerate(pos):
                b = self.pellet_bit[c]
                if b >= 0 and (mask >> b) & 1:
                    contenders.setdefault(c, []).append(i)
            new_mask = mask
            for c, who in contenders.items():
                new_mask &= ~(1 << self.pellet_bit[c])
                share = self.R_pellet / len(who)  # expected value of the random winner
                for i in who:
                    rew[i] += share
            out.append((prob, (tuple(pos), tuple(new_ghost), new_mask), tuple(rew)))
        self._trans[key] = out
        return out

    def joint_moves(self, s: State):
        return itertools.product(*(self.moves[c] for c in s[0]))

    def _enumerate_reachable(self) -> List[List[State]]:
        """layers[t] = all non-terminal states reachable at time t under ANY play."""
        layers = [[self.start]]
        for t in range(self.T - 1):
            nxt = set()
            for s in layers[t]:
                for ja in self.joint_moves(s):
                    for _, ns, _ in self.transitions(s, ja):
                        if ns[2] != 0:
                            nxt.add(ns)
            layers.append(sorted(nxt))
        return layers

    def num_states(self) -> int:
        return sum(len(L) for L in self.layers)

    def _continuation(self, t: int, ns: State, V_next: Optional[dict], default):
        """Value after the transition (0 if the episode ended)."""
        if ns[2] == 0 or t + 1 >= self.T:
            return default
        return V_next[ns]

    # ------------------------------------------------------------------ solvers
    def solve_team_optimum(self):
        """Exact pi* maximizing the expected discounted TEAM return (P1)."""
        V_next: Optional[dict] = None
        policy: List[dict] = [dict() for _ in range(self.T)]
        V0 = None
        for t in range(self.T - 1, -1, -1):
            V_t = {}
            for s in self.layers[t]:
                best_v, best_ja = -float("inf"), None
                for ja in self.joint_moves(s):
                    v = 0.0
                    for p, ns, rew in self.transitions(s, ja):
                        v += p * (sum(rew) + self.gamma * self._continuation(t, ns, V_next, 0.0))
                    if v > best_v + 1e-12:
                        best_v, best_ja = v, ja
                V_t[s] = best_v
                policy[t][s] = best_ja
            V_next = V_t
            if t == 0:
                V0 = V_t
        profile = [[{s: ja[i] for s, ja in policy[t].items()} for t in range(self.T)] for i in range(self.k)]
        return V0[self.start], profile

    def evaluate_profile(self, profile) -> List[float]:
        """Exact expected discounted return J_i of each Pac-Man under a profile."""
        zero = tuple([0.0] * self.k)
        V_next: Optional[dict] = None
        for t in range(self.T - 1, -1, -1):
            V_t = {}
            for s in self.layers[t]:
                ja = tuple(profile[i][t][s] for i in range(self.k))
                acc = [0.0] * self.k
                for p, ns, rew in self.transitions(s, ja):
                    cont = self._continuation(t, ns, V_next, zero)
                    for i in range(self.k):
                        acc[i] += p * (rew[i] + self.gamma * cont[i])
                V_t[s] = tuple(acc)
            V_next = V_t
        return list(V_next[self.start])

    def best_response(self, i: int, profile):
        """Exact best response of Pac-Man i to the others' fixed policies.
        Ties are broken in favour of i's CURRENT action, so IBR does not churn
        between equally good policies."""
        V_next: Optional[dict] = None
        br = [dict() for _ in range(self.T)]
        for t in range(self.T - 1, -1, -1):
            V_t = {}
            cur_pol = profile[i][t]
            for s in self.layers[t]:
                others = [profile[j][t][s] if j != i else None for j in range(self.k)]
                cur_a = cur_pol[s]
                vals = {}
                for a in self.moves[s[0][i]]:
                    others[i] = a
                    v = 0.0
                    for p, ns, rew in self.transitions(s, tuple(others)):
                        v += p * (rew[i] + self.gamma * self._continuation(t, ns, V_next, 0.0))
                    vals[a] = v
                best_a = cur_a
                for a, v in vals.items():
                    if v > vals[best_a] + 1e-9:
                        best_a = a
                V_t[s] = vals[best_a]
                br[t][s] = best_a
            V_next = V_t
        return V_next[self.start], br

    # ------------------------------------------------------------------ profiles
    def greedy_profile(self):
        """Selfish nearest-pellet heuristic (ignores teammates and ghosts)."""
        prof = [[dict() for _ in range(self.T)] for _ in range(self.k)]
        for t in range(self.T):
            for s in self.layers[t]:
                mask = s[2]
                pellets = [c for c in range(len(self.cells))
                           if self.pellet_bit[c] >= 0 and (mask >> self.pellet_bit[c]) & 1]
                for i in range(self.k):
                    here = s[0][i]
                    prof[i][t][s] = min(
                        self.moves[here],
                        key=lambda m: min(self.dist[m][c] for c in pellets),
                    )
        return prof

    def random_profile(self, seed: int):
        rng = random.Random(seed)
        return [[{s: rng.choice(self.moves[s[0][i]]) for s in self.layers[t]}
                 for t in range(self.T)] for i in range(self.k)]

    def iterated_best_response(self, profile, max_iters: int = 20, tol: float = 1e-7):
        """Round-robin IBR.

        Returns dict with: profile (exact NE if converged, else the visited profile
        with the smallest Nash gap, i.e. the best epsilon-NE), converged flag,
        iterations, cycle flag, agent returns J_i and the Nash gap epsilon.
        Stops early when a round reproduces a previously seen return vector
        (best-response cycle: typical when no pure equilibrium exists).
        """
        profile = [list(p) for p in profile]
        seen = set()
        best_eps, best_prof = float("inf"), profile
        for it in range(1, max_iters + 1):
            changed = False
            round_gap = 0.0
            for i in range(self.k):
                cur = self.evaluate_profile(profile)[i]
                v_br, br = self.best_response(i, profile)
                round_gap = max(round_gap, v_br - cur)
                if v_br > cur + tol:
                    profile[i] = br
                    changed = True
            if not changed:
                J = self.evaluate_profile(profile)
                return {"profile": profile, "converged": True, "cycle": False,
                        "iterations": it, "J": J, "eps": 0.0}
            # epsilon of the profile we just left is round_gap only approximately;
            # measure the current profile exactly for the epsilon-NE fallback
            eps_now, J_now = self._gap_and_returns(profile)
            if eps_now < best_eps:
                best_eps, best_prof = eps_now, [list(p) for p in profile]
            J_key = tuple(round(x, 6) for x in J_now)
            if J_key in seen:
                return {"profile": best_prof, "converged": False, "cycle": True,
                        "iterations": it, "J": self.evaluate_profile(best_prof), "eps": best_eps}
            seen.add(J_key)
        return {"profile": best_prof, "converged": False, "cycle": False,
                "iterations": max_iters, "J": self.evaluate_profile(best_prof), "eps": best_eps}

    def nash_gap(self, profile) -> float:
        """max_i [J_i(BR_i, pi_-i) - J_i(pi)]; 0 means exact Nash equilibrium."""
        return self._gap_and_returns(profile)[0]

    def _gap_and_returns(self, profile):
        J = self.evaluate_profile(profile)
        gap = max(self.best_response(i, profile)[0] - J[i] for i in range(self.k))
        return max(0.0, gap), J

    def find_equilibria(self, num_random_starts: int = 4, seed: int = 0, max_iters: int = 20):
        """IBR from optimum, greedy and random starts; returns summary dict.

        Exact pure NE (Nash gap 0) are used for PoA / PoS. If IBR finds none
        (best-response cycles), the epsilon-NE with the smallest gap is reported
        under 'eps_ne_*' instead, and the summary says so explicitly."""
        j_opt, opt_profile = self.solve_team_optimum()
        starts = [("team_optimum", opt_profile), ("greedy", self.greedy_profile())]
        starts += [(f"random_{r}", self.random_profile(seed + r)) for r in range(num_random_starts)]

        equilibria = []
        for name, prof in starts:
            res = self.iterated_best_response(prof, max_iters=max_iters)
            J = res["J"]
            equilibria.append({
                "start": name,
                "converged": res["converged"],
                "cycle": res["cycle"],
                "iterations": res["iterations"],
                "agent_returns": [round(x, 4) for x in J],
                "team_return": round(sum(J), 4),
                "nash_gap": round(res["eps"], 6),
                "_profile": res["profile"],
            })

        ne = [e for e in equilibria if e["converged"]]
        approx = [e for e in equilibria if not e["converged"]]
        out = {
            "layout": self.layout_name,
            "k": self.k,
            "ghosts": self.g,
            "beta": self.beta,
            "horizon": self.T,
            "num_states": self.num_states(),
            "j_opt": round(j_opt, 4),
            "opt_agent_returns": [round(x, 4) for x in self.evaluate_profile(opt_profile)],
            "opt_nash_gap": self.nash_gap(opt_profile),
            "num_starts": len(starts),
            "num_converged": len(ne),
            "equilibria": equilibria,
            "_opt_profile": opt_profile,
        }
        if ne:
            worst = min(ne, key=lambda e: e["team_return"])
            best = max(ne, key=lambda e: e["team_return"])
            out["j_worst_ne"] = worst["team_return"]
            out["j_best_ne"] = best["team_return"]
            out["distinct_ne_values"] = sorted({e["team_return"] for e in ne})
            out["poa"] = ratio(j_opt, worst["team_return"])
            out["pos"] = ratio(j_opt, best["team_return"])
            out["_worst_ne_profile"] = worst["_profile"]
            out["equilibrium_type"] = "exact pure NE (Nash gap 0)"
        elif approx:
            best_eps = min(e["nash_gap"] for e in approx)
            # all epsilon-NE within the smallest gap found (+ small slack)
            pool = [e for e in approx if e["nash_gap"] <= best_eps + 1e-6]
            worst = min(pool, key=lambda e: e["team_return"])
            out["equilibrium_type"] = (
                "no pure NE found by IBR (best-response cycle); "
                f"reporting epsilon-NE with epsilon={best_eps:.4f}"
            )
            out["eps"] = best_eps
            out["j_worst_ne"] = worst["team_return"]
            out["j_best_ne"] = max(e["team_return"] for e in pool)
            out["poa"] = ratio(j_opt, worst["team_return"])
            out["pos"] = ratio(j_opt, out["j_best_ne"])
            out["_worst_ne_profile"] = worst["_profile"]
        return out

    # ------------------------------------------------------------------ check
    def simulate_profile(self, profile, num_episodes: int = 2000, seed: int = 7) -> float:
        """Monte-Carlo team return of a profile in the REAL simulator (model check)."""
        delta_to_action = {d: a for a, d in ACTION_DELTAS.items()}
        total = 0.0
        for ep in range(num_episodes):
            env = MultiAgentPacmanEnv(
                layout_name=self.layout_name, num_pacman=self.k, num_ghosts=self.g,
                ghost_beta=self.beta, max_steps=self.T, seed=seed + ep,
                pellet_density=self.env.pellet_density,
            )
            disc, ret, done, t = 1.0, 0.0, False, 0
            while not done:
                s = (tuple(self.idx[c] for c in env.pacman_positions),
                     tuple(self.idx[c] for c in env.ghost_positions),
                     self._mask_of(env.pellets))
                actions = []
                for i in range(self.k):
                    here = self.cells[s[0][i]]
                    there = self.cells[profile[i][t][s]]
                    d = (there[0] - here[0], there[1] - here[1])
                    actions.append(delta_to_action.get(d, ACTION_STAY))
                _, rewards, done, _ = env.step(actions)
                ret += disc * sum(rewards)
                disc *= env.gamma
                t += 1
            total += ret
        return total / num_episodes

    def _mask_of(self, pellets) -> int:
        m = 0
        for c in pellets:
            m |= 1 << self.pellet_bit[self.idx[c]]
        return m


def ratio(j_opt: float, j_ne: float) -> float:
    """PoA-style ratio; meaningful only when both returns are positive."""
    if j_ne <= 0:
        return float("inf")
    return j_opt / j_ne
