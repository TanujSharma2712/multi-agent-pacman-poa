"""
Retro-Arcade Maze Visualization & Trajectory Renderer:
Generates publication-grade figures matching Figure 1 of the Project Proposal:
1. Environment state: neon blue walls, glowing pellets, corner power pellets.
2. DCOP Region partitioning: Region A, Region B, Region C vertical columns with centroids.
3. Authentic Pac-Man wedges ($P_1, P_2, P_3$) and Ghost sprites ($G_1, G_2$).
4. Trajectory comparisons: Selfish contention vs DCOP/Shapley coordinated paths.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.environment import MultiAgentPacmanEnv
from src.dcop import RegionPartition


def draw_ghost(ax, x, y, color="#ef4444", label="G", look_direction=(-1, 0), radius=0.42):
    """Draws an arcade-style ghost with dome head, wavy tentacles, and eyes."""
    # Ghost body: Upper semicircle + lower body
    head = patches.Wedge((x, y + 0.05), radius, 0, 180, facecolor=color, edgecolor="black", linewidth=1.2, zorder=5)
    ax.add_patch(head)

    # Lower rectangular skirt
    body = patches.Rectangle(
        (x - radius, y - 0.25), radius * 2, 0.30,
        facecolor=color, edgecolor="none", zorder=5
    )
    ax.add_patch(body)

    # Tentacles (3 wavy scallops)
    num_tentacles = 3
    tentacle_w = (radius * 2) / num_tentacles
    for t in range(num_tentacles):
        tx = x - radius + t * tentacle_w + tentacle_w / 2
        scallop = patches.Wedge(
            (tx, y - 0.25), tentacle_w / 2, 180, 360,
            facecolor=color, edgecolor="none", zorder=5
        )
        ax.add_patch(scallop)

    # Eyes (sclera + pupil)
    eye_offset_x = 0.13
    eye_y = y + 0.12
    dx, dy = look_direction
    pupil_offset_x = dx * 0.04
    pupil_offset_y = dy * 0.04

    for side in (-1, 1):
        ex = x + side * eye_offset_x
        sclera = patches.Ellipse((ex, eye_y), 0.18, 0.22, facecolor="white", edgecolor="none", zorder=6)
        pupil = patches.Circle((ex + pupil_offset_x, eye_y + pupil_offset_y), 0.06, facecolor="#1e3a8a", zorder=7)
        ax.add_patch(sclera)
        ax.add_patch(pupil)

    # Label text below
    ax.text(x, y - 0.52, f"${label}$", color="white", fontsize=9, ha="center", va="center", fontweight="bold", zorder=8)


def draw_pacman(ax, x, y, color="#00e5ff", label="P1", mouth_angle=45, facing=0, radius=0.40):
    """Draws an authentic Pac-Man with open mouth wedge and label."""
    # facing in degrees: 0=East, 90=North, 180=West, 270=South
    theta1 = facing + mouth_angle / 2
    theta2 = facing + 360 - mouth_angle / 2
    wedge = patches.Wedge(
        (x, y), radius, theta1, theta2,
        facecolor=color, edgecolor="black", linewidth=1.2, zorder=5
    )
    ax.add_patch(wedge)
    # Labeled text inside or below
    ax.text(x, y, f"${label}$", color="#050e24", fontsize=8.5, ha="center", va="center", fontweight="black", zorder=6)


def generate_maze_visualization(layout_name: str = "proposal_maze", num_pacman: int = 3, num_ghosts: int = 2):
    os.makedirs("plots", exist_ok=True)

    env = MultiAgentPacmanEnv(layout_name=layout_name, num_pacman=num_pacman, num_ghosts=num_ghosts, seed=42)
    partition = RegionPartition(env, num_regions=num_pacman)

    fig, ax = plt.subplots(figsize=(10.5, 7.8), dpi=300, facecolor="#000000")
    ax.set_facecolor("#000000")

    # 1. Draw Region vertical background tints
    region_tints = ["#00b4d8", "#eab308", "#f43f5e"]
    for r in range(partition.num_regions):
        for (cx, cy) in partition.region_cells[r]:
            rect = patches.Rectangle(
                (cx - 0.5, cy - 0.5), 1, 1,
                facecolor=region_tints[r % len(region_tints)],
                alpha=0.08,
                edgecolor="none",
                zorder=1
            )
            ax.add_patch(rect)

    # 2. Draw Vertical Region Partition Dividers
    if layout_name == "proposal_maze" and num_pacman == 3:
        ax.axvline(6.5, color="#38bdf8", linestyle="--", linewidth=1.8, alpha=0.65, zorder=3)
        ax.axvline(13.5, color="#38bdf8", linestyle="--", linewidth=1.8, alpha=0.65, zorder=3)

    # 3. Draw Maze Walls with authentic Retro Arcade double-line effect
    for (wx, wy) in env.walls:
        # Dark base block
        wall_block = patches.Rectangle(
            (wx - 0.5, wy - 0.5), 1, 1,
            facecolor="#050e24",
            edgecolor="#1e40af",
            linewidth=2.0,
            zorder=2
        )
        ax.add_patch(wall_block)

        # Neon cyan inner glow line
        inner_border = patches.Rectangle(
            (wx - 0.44, wy - 0.44), 0.88, 0.88,
            facecolor="none",
            edgecolor="#2563eb",
            linewidth=1.0,
            alpha=0.8,
            zorder=2
        )
        ax.add_patch(inner_border)

    # 4. Draw Food Pellets (Yellow dots with glow)
    for (px, py) in env.pellets:
        if (px, py) in env.power_pellets:
            # Power pellet (large glowing golden dot)
            glow = patches.Circle((px, py), 0.42, facecolor="#ffd700", alpha=0.35, zorder=3)
            ax.add_patch(glow)
            power_dot = patches.Circle((px, py), 0.28, facecolor="#fffbeb", edgecolor="#facc15", linewidth=1.5, zorder=4)
            ax.add_patch(power_dot)
        else:
            # Regular food pellet
            glow = patches.Circle((px, py), 0.16, facecolor="#facc15", alpha=0.25, zorder=3)
            ax.add_patch(glow)
            dot = patches.Circle((px, py), 0.10, facecolor="#fef08a", edgecolor="none", zorder=4)
            ax.add_patch(dot)

    # 5. Draw Pac-Man Agents
    # P1 (Cyan in Region A), P2 (Yellow in Region B), P3 (Pink in Region C)
    pacman_colors = ["#00e5ff", "#ffd600", "#ff4081", "#a3e635"]
    for i, (px, py) in enumerate(env.pacman_positions):
        color = pacman_colors[i % len(pacman_colors)]
        facing = 0 if i != 2 else 180  # P1, P2 face East; P3 faces West
        draw_pacman(ax, px, py, color=color, label=f"P_{i+1}", mouth_angle=40, facing=facing, radius=0.40)

    # 6. Draw Ghosts
    # G1 (Blinky Red in upper B), G2 (Clyde Orange in lower B)
    ghost_colors = ["#ef4444", "#f97316", "#a855f7", "#06b6d4"]
    for j, (gx, gy) in enumerate(env.ghost_positions):
        color = ghost_colors[j % len(ghost_colors)]
        draw_ghost(ax, gx, gy, color=color, label=f"G_{j+1}", look_direction=(-1, 0), radius=0.40)

    # 7. Axes and Margins
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Top-down arcade coordinate system
    ax.set_xticks([])
    ax.set_yticks([])

    # 8. Title Banner & Region Headers matching Figure 1 in Proposal
    ax.text(
        env.width / 2.0 - 0.5, -1.8,
        "MULTI-AGENT MAZE ENVIRONMENT",
        color="white", fontsize=14, fontweight="bold", ha="center", va="center"
    )

    if layout_name == "proposal_maze" and num_pacman == 3:
        ax.text(3.0, -0.85, "Region A", color="#38bdf8", fontsize=11, fontweight="bold", ha="center", va="center")
        ax.text(10.0, -0.85, "Region B", color="#facc15", fontsize=11, fontweight="bold", ha="center", va="center")
        ax.text(17.0, -0.85, "Region C", color="#f472b6", fontsize=11, fontweight="bold", ha="center", va="center")

    # 9. Custom Bottom Legend matching Figure 1
    legend_elements = [
        patches.Patch(facecolor="#00e5ff", edgecolor="black", label="Pac-Man agent ($P_i$)"),
        patches.Patch(facecolor="#ef4444", edgecolor="black", label="Ghost ($G_j$)"),
        patches.Circle((0, 0), 0.10, facecolor="#fef08a", label="Pellet"),
        patches.Circle((0, 0), 0.24, facecolor="#fffbeb", edgecolor="#facc15", linewidth=1.5, label="Power pellet"),
    ]
    leg = ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=4, fontsize=10.5, frameon=True, facecolor="#050e24", edgecolor="#1e40af", labelcolor="white"
    )
    for patch in leg.get_patches():
        patch.set_height(10)

    plt.subplots_adjust(top=0.88, bottom=0.12)

    out_plot = "plots/fig_maze_simulation.png"
    plt.savefig(out_plot, bbox_inches="tight", dpi=300, facecolor="#000000")
    plt.close()
    print(f"Maze figure saved to {out_plot}")


if __name__ == "__main__":
    generate_maze_visualization()
