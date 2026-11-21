"""
Simulation Animation Generator for Multi-Agent Pac-Man:
Generates smooth, retro-arcade animated GIFs:
1. comparison_selfish_vs_dcop.gif: Side-by-side comparison of Selfish play vs DCOP Coordinated play.
2. pacman_dcop_simulation.gif: High-definition animation of DCOP Region Partition clearing the maze.
3. pacman_shapley_simulation.gif: High-definition animation of Shapley-value cooperative learning.
"""

import math
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.baselines import CentralizedTeamPlanner, IndependentGreedyAgent
from src.dcop import DCOPAgent, DCOPSolver, RegionPartition
from src.environment import MultiAgentPacmanEnv
from src.trainer import train_q_learning


def draw_retro_maze_frame(
    draw: ImageDraw.ImageDraw,
    env: MultiAgentPacmanEnv,
    offset_x: int,
    offset_y: int,
    cell_size: int = 24,
    show_regions: bool = True,
    step_num: int = 0,
    title: str = "",
    score: float = 0.0,
    captures: int = 0,
):
    # Colors
    c_bg = (0, 0, 0)
    c_wall_fill = (5, 14, 36)
    c_wall_border = (30, 64, 175)
    c_wall_inner = (37, 99, 235)
    c_pellet = (254, 240, 138)
    c_power = (255, 234, 0)
    c_p1 = (0, 229, 255)
    c_p2 = (255, 214, 0)
    c_p3 = (255, 64, 129)
    c_g1 = (239, 68, 68)
    c_g2 = (249, 115, 22)

    # 1. Background
    maze_w = env.width * cell_size
    maze_h = env.height * cell_size
    draw.rectangle([offset_x, offset_y, offset_x + maze_w, offset_y + maze_h], fill=c_bg)

    # 2. Region Tint overlays if enabled
    if show_regions and env.layout_name == "proposal_maze":
        # Region A
        draw.rectangle([offset_x + cell_size, offset_y + cell_size, offset_x + 7 * cell_size, offset_y + maze_h - cell_size], fill=(0, 25, 45))
        # Region B
        draw.rectangle([offset_x + 7 * cell_size, offset_y + cell_size, offset_x + 14 * cell_size, offset_y + maze_h - cell_size], fill=(35, 30, 0))
        # Region C
        draw.rectangle([offset_x + 14 * cell_size, offset_y + cell_size, offset_x + 20 * cell_size, offset_y + maze_h - cell_size], fill=(40, 0, 25))

        # Dividing dashed lines
        for y_dash in range(offset_y + cell_size, offset_y + maze_h - cell_size, 10):
            draw.line([(offset_x + 7 * cell_size, y_dash), (offset_x + 7 * cell_size, y_dash + 5)], fill=(56, 189, 248), width=2)
            draw.line([(offset_x + 14 * cell_size, y_dash), (offset_x + 14 * cell_size, y_dash + 5)], fill=(56, 189, 248), width=2)

    # 3. Walls
    for (wx, wy) in env.walls:
        x0 = offset_x + wx * cell_size
        y0 = offset_y + wy * cell_size
        draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=c_wall_fill, outline=c_wall_border, width=2)
        draw.rectangle([x0 + 3, y0 + 3, x0 + cell_size - 3, y0 + cell_size - 3], outline=c_wall_inner, width=1)

    # 4. Pellets
    for (px, py) in env.pellets:
        cx = offset_x + px * cell_size + cell_size // 2
        cy = offset_y + py * cell_size + cell_size // 2
        if (px, py) in env.power_pellets:
            r = 7
            draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], fill=(120, 100, 0))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c_power, outline=(255, 255, 255), width=1)
        else:
            r = 3
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c_pellet)

    # 5. Pac-Man agents
    pacman_colors = [c_p1, c_p2, c_p3]
    mouth_open = (step_num % 2 == 0)
    for i, (px, py) in enumerate(env.pacman_positions):
        cx = offset_x + px * cell_size + cell_size // 2
        cy = offset_y + py * cell_size + cell_size // 2
        col = pacman_colors[i % len(pacman_colors)]
        r = cell_size // 2 - 2

        if mouth_open:
            # Draw wedge (facing right for P1/P2, left for P3)
            start_ang = 30 if i != 2 else 210
            end_ang = 330 if i != 2 else 150
            draw.pieslice([cx - r, cy - r, cx + r, cy + r], start=start_ang, end=end_ang, fill=col, outline=(0, 0, 0))
        else:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(0, 0, 0))

        # Label inside
        draw.text((cx - 4, cy - 6), f"P{i+1}", fill=(0, 0, 0))

    # 6. Ghosts
    ghost_colors = [c_g1, c_g2]
    for j, (gx, gy) in enumerate(env.ghost_positions):
        cx = offset_x + gx * cell_size + cell_size // 2
        cy = offset_y + gy * cell_size + cell_size // 2
        gcol = ghost_colors[j % len(ghost_colors)]
        r = cell_size // 2 - 2

        # Dome head
        draw.pieslice([cx - r, cy - r, cx + r, cy + r], start=180, end=360, fill=gcol, outline=(0, 0, 0))
        # Skirt
        draw.rectangle([cx - r, cy, cx + r, cy + r - 2], fill=gcol)
        # Scallop tentacles
        tw = (2 * r) // 3
        for t in range(3):
            tx = cx - r + t * tw
            draw.pieslice([tx, cy + r - 5, tx + tw, cy + r + 3], start=0, end=180, fill=gcol)

        # White eyes + blue pupils
        for side in (-4, 4):
            draw.ellipse([cx + side - 3, cy - 3, cx + side + 3, cy + 3], fill=(255, 255, 255))
            draw.ellipse([cx + side - 2, cy - 1, cx + side + 1, cy + 2], fill=(15, 23, 42))

        draw.text((cx - 4, cy + r - 2), f"G{j+1}", fill=(255, 255, 255))

    # 7. Card Header / HUD Info
    draw.rectangle([offset_x, offset_y - 42, offset_x + maze_w, offset_y - 4], fill=(15, 23, 42), outline=(30, 41, 59))
    draw.text((offset_x + 12, offset_y - 34), title, fill=(255, 255, 255))
    hud = f"Step: {step_num:02d} | Pellets: {len(env.pellets):03d} | Captures: {captures} | Return: {score:.1f}"
    draw.text((offset_x + 12, offset_y - 18), hud, fill=(226, 232, 240))


def generate_side_by_side_gif(num_steps: int = 50, out_path: str = "plots/comparison_selfish_vs_dcop.gif"):
    print("Generating side-by-side comparison GIF (Selfish vs DCOP)...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Initialize two environments with identical seeds
    seed = 42
    layout = "proposal_maze"
    env_selfish = MultiAgentPacmanEnv(layout_name=layout, num_pacman=3, num_ghosts=2, seed=seed)
    env_dcop = MultiAgentPacmanEnv(layout_name=layout, num_pacman=3, num_ghosts=2, seed=seed)

    # Selfish agents (Independent Greedy with pellet collision)
    selfish_agents = [IndependentGreedyAgent(i, rng=random.Random(100 + i)) for i in range(3)]

    # DCOP agents (Region Partitioning)
    partition = RegionPartition(env_dcop, num_regions=3)
    solver = DCOPSolver(partition, rng=random.Random(200))
    dcop_agents = [DCOPAgent(i, partition=partition, rng=random.Random(200 + i)) for i in range(3)]

    cell_sz = 20
    panel_w = env_selfish.width * cell_sz
    panel_h = env_selfish.height * cell_sz
    margin_x = 24
    margin_top = 65
    margin_bot = 30
    total_w = panel_w * 2 + margin_x * 3
    total_h = panel_h + margin_top + margin_bot

    frames = []
    score_s = 0.0
    score_d = 0.0
    gamma_s = 1.0
    gamma_d = 1.0

    done_s = False
    done_d = False

    for t in range(num_steps):
        # Step Selfish
        if not done_s:
            act_s = [ag.get_action(env_selfish) for ag in selfish_agents]
            _, r_s, done_s, _ = env_selfish.step(act_s)
            score_s += gamma_s * sum(r_s)
            gamma_s *= env_selfish.gamma

        # Step DCOP
        if not done_d:
            z, _ = solver.solve_max_sum(message_budget=24)
            for i, r in enumerate(z):
                dcop_agents[i].set_region(r)
            act_d = [ag.get_action(env_dcop) for ag in dcop_agents]
            _, r_d, done_d, _ = env_dcop.step(act_d)
            score_d += gamma_d * sum(r_d)
            gamma_d *= env_dcop.gamma

        # Render combined frame
        img = Image.new("RGB", (total_w, total_h), color=(10, 15, 28))
        draw = ImageDraw.Draw(img)

        # Global Title Banner
        draw.text((total_w // 2 - 220, 12), "MULTI-AGENT PAC-MAN: SELFISH VS. DCOP COORDINATION", fill=(255, 255, 255))
        draw.text((total_w // 2 - 180, 30), "Emergence of Price of Anarchy and Recovery via Max-Sum", fill=(148, 163, 184))

        # Left Panel: Selfish
        off_x1 = margin_x
        off_y1 = margin_top
        draw_retro_maze_frame(
            draw, env_selfish, off_x1, off_y1, cell_size=cell_sz, show_regions=False,
            step_num=t, title="SELFISH PLAY (High Contention & Ghost Collisions)",
            score=score_s, captures=env_selfish.total_captures
        )

        # Right Panel: DCOP
        off_x2 = margin_x * 2 + panel_w
        off_y2 = margin_top
        draw_retro_maze_frame(
            draw, env_dcop, off_x2, off_y2, cell_size=cell_sz, show_regions=True,
            step_num=t, title="DCOP COORDINATED (Region A/B/C Partitioning)",
            score=score_d, captures=env_dcop.total_captures
        )

        frames.append(img)
        if done_s and done_d:
            break

    # Save animated GIF
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=220,  # ~4.5 FPS
        loop=0,
    )
    print(f"Animated comparison GIF saved to {out_path} ({len(frames)} frames)")


def generate_single_policy_gif(policy_type: str = "dcop", num_steps: int = 40, out_path: str = "plots/pacman_dcop_simulation.gif"):
    print(f"Generating single policy GIF for {policy_type}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seed = 42
    layout = "proposal_maze"
    env = MultiAgentPacmanEnv(layout_name=layout, num_pacman=3, num_ghosts=2, seed=seed)

    if policy_type == "dcop":
        partition = RegionPartition(env, num_regions=3)
        solver = DCOPSolver(partition, rng=random.Random(200))
        agents = [DCOPAgent(i, partition=partition, rng=random.Random(200 + i)) for i in range(3)]
        title_str = "DCOP MAX-SUM REGION ASSIGNMENT (B=24)"
    elif policy_type == "shapley":
        trained = train_q_learning(layout_name=layout, num_pacman=3, num_ghosts=2, mode="shapley", shapley_lambda=0.5, num_episodes=50, seed=55)
        agents = trained
        title_str = "SHAPLEY REWARD SHAPING (λ=0.5 Cooperative Credit)"
    else:
        agents = [IndependentGreedyAgent(i, rng=random.Random(100 + i)) for i in range(3)]
        title_str = "SELFISH GREEDY BASELINE"

    cell_sz = 24
    maze_w = env.width * cell_sz
    maze_h = env.height * cell_sz
    margin = 30
    header_h = 70
    total_w = maze_w + margin * 2
    total_h = maze_h + margin * 2 + header_h

    frames = []
    score = 0.0
    gamma = 1.0

    for t in range(num_steps):
        if policy_type == "dcop":
            z, _ = solver.solve_max_sum(message_budget=24)
            for i, r in enumerate(z):
                agents[i].set_region(r)
            joint_action = [ag.get_action(env) for ag in agents]
        elif policy_type == "shapley":
            joint_action = [ag.get_action(env, explore=False) for ag in agents]
        else:
            joint_action = [ag.get_action(env) for ag in agents]

        _, rewards, done, _ = env.step(joint_action)
        score += gamma * sum(rewards)
        gamma *= env.gamma

        img = Image.new("RGB", (total_w, total_h), color=(6, 11, 24))
        draw = ImageDraw.Draw(img)

        draw_retro_maze_frame(
            draw, env, margin, margin + header_h, cell_size=cell_sz,
            show_regions=(policy_type == "dcop"), step_num=t,
            title=title_str, score=score, captures=env.total_captures
        )

        frames.append(img)
        if done:
            break

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
    )
    print(f"Animated GIF saved to {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    generate_side_by_side_gif(num_steps=45, out_path="plots/comparison_selfish_vs_dcop.gif")
    generate_single_policy_gif(policy_type="dcop", num_steps=40, out_path="plots/pacman_dcop_simulation.gif")
    generate_single_policy_gif(policy_type="shapley", num_steps=40, out_path="plots/pacman_shapley_simulation.gif")
