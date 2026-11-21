"""
Evaluation metrics for Multi-Agent Pac-Man:
- Price of Anarchy (PoA)
- Gap recovery percentage
- Gini coefficient for reward fairness
- Summary statistics with 95% confidence intervals
"""

import math
from typing import List, Tuple
import numpy as np


def compute_gini_index(returns: List[float]) -> float:
    """
    Computes the Gini coefficient of a list of agent returns.
    0 represents perfect equality, 1 represents maximal inequality.
    Handles negative or zero sums by shifting returns to non-negative range.
    """
    k = len(returns)
    if k <= 1:
        return 0.0

    arr = np.array(returns, dtype=float)
    # If there are negative values, shift by min value so values are non-negative
    min_val = np.min(arr)
    if min_val < 0:
        arr = arr - min_val

    total = np.sum(arr)
    if total == 0:
        return 0.0

    diff_sum = np.sum(np.abs(arr[:, None] - arr[None, :]))
    gini = diff_sum / (2.0 * k * total)
    return float(gini)


def compute_poa(optimal_team_return: float, selfish_team_return: float) -> float:
    """
    Price of Anarchy = J(pi*) / J(pi_selfish).
    By definition, J(pi*) is the social optimum >= J(pi_selfish), so PoA >= 1.0.
    """
    effective_opt = max(optimal_team_return, selfish_team_return)
    if selfish_team_return > 1e-5:
        return float(max(1.0, effective_opt / selfish_team_return))
    elif selfish_team_return == 0:
        return float("inf") if effective_opt > 0 else 1.0
    else:
        # If returns are negative, ratio relative to baseline
        return float(max(1.0, effective_opt / (selfish_team_return + 1e-8)))


def compute_gap_recovered(
    remedy_return: float, selfish_return: float, optimal_return: float
) -> float:
    """
    Fraction of selfish gap recovered:
    Recov = (J(remedy) - J(selfish)) / (J(opt) - J(selfish))
    0 = no better than selfish, 1 = matches the optimum. NOT clamped: a negative
    value means the remedy did worse than selfish play, > 1 means it beat the
    reference optimum. Returns NaN when there is no gap to recover
    (J(opt) <= J(selfish)), because the ratio is then meaningless.
    """
    denom = optimal_return - selfish_return
    if denom <= 1e-6:
        return float("nan")
    return float((remedy_return - selfish_return) / denom)


def compute_ci95(data: List[float]) -> Tuple[float, float, float]:
    """
    Returns (mean, margin_of_error_95, std_err).
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(data))
    if n == 1:
        return mean, 0.0, 0.0
    std_err = float(np.std(data, ddof=1) / math.sqrt(n))
    margin = 1.96 * std_err
    return mean, margin, std_err
