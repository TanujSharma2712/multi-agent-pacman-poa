"""Shared helpers for the experiment scripts (plot style, saving, JSON)."""

import json
import math
import os
import sys

import matplotlib

# Windows consoles default to cp1252 and crash on symbols like β, λ, π, ±.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#0284c7", "#d97706", "#dc2626", "#16a34a", "#7c3aed", "#db2777", "#475569", "#0d9488"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)


def save_figure(fig, filename):
    """Save a figure to plots/."""
    os.makedirs("plots", exist_ok=True)
    path = os.path.join("plots", filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/{filename}")


def _clean(obj):
    """Make results JSON-safe: drop private keys, convert inf/nan to strings."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return str(obj)
        return round(obj, 4)
    if hasattr(obj, "item"):  # numpy scalar
        return _clean(obj.item())
    return obj


def save_json(data, filename):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(data), f, indent=2, ensure_ascii=False)
    print(f"  saved {path}")


def strip_samples(res):
    """Drop per-episode arrays from an evaluate_policy result before saving."""
    return {k: v for k, v in res.items() if not k.startswith("all_")}
