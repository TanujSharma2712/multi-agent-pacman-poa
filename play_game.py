"""
Interactive Multi-Agent Pac-Man: Desktop GUI Game & Simulation Testbed
Refactored Architecture:
- 75% Board Canvas: Dynamically scales with window size so the maze is large, sharp, and arcade-grade.
- 25% Operating Panel: Complete command station with playback controls, on-screen D-Pad,
  and interactive experimental parameters from Experiments 1, 2, 3, and 4:
    * Algorithm selection (DCOP Max-Sum, DCOP DSA, Shapley Q-Learning, Equal-Share, Selfish Greedy, Centralized Optimum, Random)
    * Team Size k in [2, 3, 4, 5] (Exp 1)
    * Ghost count M in [0, 1, 2, 3, 4] and Ghost Aggression beta in [0.0, 1.0] (Exp 1 & Exp 3)
    * DCOP Message Budget B in [0, 48] (Exp 2)
    * Shapley Mixing Parameter lambda in [0.0, 1.0] (Exp 3)
    * Layout selection (proposal_maze, medium_classic, open_contention, small_grid)
- Clear distinction: Human Player mode (with YOU on P1) vs. Autonomous AI mode (P1 is a normal AI agent).
Built with Python's standard tkinter - zero external dependencies required!
"""

import os
import random
import sys
import tkinter as tk
from tkinter import ttk

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.baselines import CentralizedTeamPlanner, IndependentGreedyAgent
from src.dcop import DCOPAgent, DCOPSolver, RegionPartition
from src.environment import (
    ACTION_DELTAS,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    ACTIONS,
    MultiAgentPacmanEnv,
)
from src.metrics import compute_gini_index, compute_poa
from src.trainer import train_q_learning


class PacmanGameApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multi-Agent Pac-Man: Interactive Play & PoA Testbed (CS 329: Foundations of AI: Multi-Agent Systems)")
        self.root.configure(bg="#030712")

        # Game state parameters
        self.layout_name = "proposal_maze"
        self.num_pacman = 3
        self.num_ghosts = 2
        self.ghost_beta = 0.5
        self.dcop_budget = 24
        self.shapley_lambda = 0.5
        self.cell_size = 34
        self.is_running = False
        self.step_delay = 140  # ms per step
        self.current_action = ACTION_RIGHT
        self.queued_action = ACTION_RIGHT
        self.score = 0.0
        self.gamma = 1.0
        self.agent_rewards = [0.0] * self.num_pacman
        self.messages_used = 0

        # Mode Variables
        self.primary_mode = tk.StringVar(value="human")  # "human" or "autonomous"
        self.human_teammate_mode = tk.StringVar(value="DCOP Max-Sum (B=24)")
        self.auto_team_mode = tk.StringVar(value="DCOP Max-Sum (B=24)")

        # Experimental Param UI Variables
        self.var_layout = tk.StringVar(value="proposal_maze")
        self.var_k = tk.IntVar(value=3)
        self.var_ghosts = tk.IntVar(value=2)
        self.var_beta = tk.DoubleVar(value=0.5)
        self.var_budget = tk.IntVar(value=24)
        self.var_lambda = tk.DoubleVar(value=0.5)

        # Policy Models
        self.shapley_agents = None
        self.equal_share_agents = None
        self.selfish_agents = None

        self._init_environment()
        self._build_ui()
        self._bind_keys()
        self.reset_game()

    def _init_environment(self):
        """Build or re-initialize environment and solvers based on experimental settings."""
        self.env = MultiAgentPacmanEnv(
            layout_name=self.layout_name,
            num_pacman=self.num_pacman,
            num_ghosts=self.num_ghosts,
            ghost_beta=self.ghost_beta,
            seed=random.randint(1, 10000),
        )
        self.partition = RegionPartition(self.env, num_regions=max(2, self.num_pacman))
        self.dcop_solver = DCOPSolver(self.partition)
        self.dcop_agents = [DCOPAgent(i, partition=self.partition) for i in range(self.num_pacman)]
        self.greedy_agents = [IndependentGreedyAgent(i) for i in range(self.num_pacman)]
        self.centralized_planner = CentralizedTeamPlanner()
        self.agent_rewards = [0.0] * self.num_pacman

    def is_human_mode(self) -> bool:
        return self.primary_mode.get() == "human"

    def _build_ui(self):
        # Configure root grid expansion
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # -------------------------------------------------------------
        # 1. Top Header Frame
        # -------------------------------------------------------------
        top_bar = tk.Frame(self.root, bg="#0b1739", padx=12, pady=6, highlightbackground="#1e293b", highlightthickness=1)
        top_bar.grid(row=0, column=0, sticky="ew")

        top_left = tk.Frame(top_bar, bg="#0b1739")
        top_left.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            top_left,
            text="MULTI-AGENT PAC-MAN: PRICE OF ANARCHY (PoA) TESTBED",
            font=("Arial", 12, "bold"),
            fg="#38bdf8",
            bg="#0b1739",
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(
            top_left,
            text="CS 329: Foundations of AI: Multi-Agent Systems | IIT Gandhinagar",
            font=("Arial", 9),
            fg="#94a3b8",
            bg="#0b1739",
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(1, 0))

        # Mode Selection Radio Buttons in top header
        mode_box = tk.Frame(top_bar, bg="#0f172a", padx=10, pady=4, highlightbackground="#334155", highlightthickness=1)
        mode_box.pack(side=tk.RIGHT, padx=6)

        tk.Label(mode_box, text="Mode:", font=("Arial", 9, "bold"), fg="#f8fafc", bg="#0f172a").pack(side=tk.LEFT, padx=(2, 6))

        self.rb_human = tk.Radiobutton(
            mode_box,
            text="🎮 Human (Control P1)",
            variable=self.primary_mode,
            value="human",
            command=self.on_primary_mode_changed,
            font=("Arial", 9, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            activebackground="#0f172a",
            activeforeground="#38bdf8",
            selectcolor="#0284c7",
            cursor="hand2",
        )
        self.rb_human.pack(side=tk.LEFT, padx=6)

        self.rb_auto = tk.Radiobutton(
            mode_box,
            text="🤖 Autonomous AI",
            variable=self.primary_mode,
            value="autonomous",
            command=self.on_primary_mode_changed,
            font=("Arial", 9, "bold"),
            fg="#c084fc",
            bg="#0f172a",
            activebackground="#0f172a",
            activeforeground="#c084fc",
            selectcolor="#7e22ce",
            cursor="hand2",
        )
        self.rb_auto.pack(side=tk.LEFT, padx=6)

        # -------------------------------------------------------------
        # 2. Main Content Split: 75% Board (Left) vs 25% Operating Panel (Right)
        # -------------------------------------------------------------
        content_frame = tk.Frame(self.root, bg="#030712")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        content_frame.columnconfigure(0, weight=3)  # 75% weight
        content_frame.columnconfigure(1, weight=1)  # 25% weight
        content_frame.rowconfigure(0, weight=1)

        # LEFT CONTAINER (75%): Board, HUD, Legend
        left_container = tk.Frame(content_frame, bg="#050e24", padx=8, pady=6, highlightbackground="#1e293b", highlightthickness=1)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_container.rowconfigure(1, weight=1)
        left_container.columnconfigure(0, weight=1)

        # HUD Top Bar inside Left Container
        hud_frame = tk.Frame(left_container, bg="#0b1739", padx=8, pady=5, highlightbackground="#1e40af", highlightthickness=1)
        hud_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.lbl_score = tk.Label(hud_frame, text="Score: 0.0", font=("Arial", 10, "bold"), fg="#facc15", bg="#0b1739")
        self.lbl_score.pack(side=tk.LEFT, padx=6)

        self.lbl_pellets = tk.Label(hud_frame, text="Pellets: 149", font=("Arial", 10, "bold"), fg="#ffffff", bg="#0b1739")
        self.lbl_pellets.pack(side=tk.LEFT, padx=6)

        self.lbl_captures = tk.Label(hud_frame, text="Captures: 0", font=("Arial", 10, "bold"), fg="#ef4444", bg="#0b1739")
        self.lbl_captures.pack(side=tk.LEFT, padx=6)

        self.lbl_steps = tk.Label(hud_frame, text="Steps: 0", font=("Arial", 10, "bold"), fg="#a3e635", bg="#0b1739")
        self.lbl_steps.pack(side=tk.LEFT, padx=6)

        self.lbl_gini = tk.Label(hud_frame, text="Gini: 0.000", font=("Arial", 10, "bold"), fg="#38bdf8", bg="#0b1739")
        self.lbl_gini.pack(side=tk.LEFT, padx=6)

        self.lbl_poa = tk.Label(hud_frame, text="Live PoA: 1.000", font=("Arial", 11, "bold"), fg="#ec4899", bg="#0b1739")
        self.lbl_poa.pack(side=tk.RIGHT, padx=6)

        # Game Board Canvas Frame (Centers canvas inside 75% space)
        self.canvas_container = tk.Frame(left_container, bg="#020617")
        self.canvas_container.grid(row=1, column=0, sticky="nsew")
        self.canvas_container.bind("<Configure>", self.on_canvas_container_resize)

        self.canvas = tk.Canvas(
            self.canvas_container,
            bg="#000000",
            highlightthickness=2,
            highlightbackground="#1e40af",
        )
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Legend / Guide inside Left Container
        legend_frame = tk.Frame(left_container, bg="#050e24", pady=4)
        legend_frame.grid(row=2, column=0, sticky="ew")

        legend_text = "Legend:  Cyan = P1  |  Yellow = P2  |  Pink = P3  |  Green/Purple = P4, P5  |  Red/Orange = Ghosts  |  Dashed Lines = Region Sectors"
        self.lbl_legend = tk.Label(legend_frame, text=legend_text, font=("Arial", 8), fg="#64748b", bg="#050e24")
        self.lbl_legend.pack(side=tk.TOP)

        # RIGHT CONTAINER (25%): Operating & Control Station
        right_container = tk.Frame(content_frame, bg="#0f172a", padx=10, pady=8, highlightbackground="#1e293b", highlightthickness=1)
        right_container.grid(row=0, column=1, sticky="nsew")
        right_container.rowconfigure(0, weight=1)
        right_container.columnconfigure(0, weight=1)

        # Scrollable Frame for 25% Control Station
        ctrl_canvas = tk.Canvas(right_container, bg="#0f172a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=ctrl_canvas.yview)
        self.ctrl_frame = tk.Frame(ctrl_canvas, bg="#0f172a")

        self.ctrl_frame.bind(
            "<Configure>",
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
        )
        ctrl_window = ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        ctrl_canvas.bind(
            "<Configure>",
            lambda e: ctrl_canvas.itemconfig(ctrl_window, width=e.width),
        )
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)

        ctrl_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self._build_control_station()

    def _build_control_station(self):
        """Construct the 25% sidebar control station with playback, D-Pad, and experimental parameters."""
        f = self.ctrl_frame

        # Section 1: Playback Controls Card
        card_play = tk.LabelFrame(f, text="  1. Playback & Speed  ", font=("Arial", 9, "bold"), fg="#38bdf8", bg="#0f172a", padx=8, pady=6)
        card_play.pack(fill=tk.X, pady=(0, 6))

        btn_row = tk.Frame(card_play, bg="#0f172a")
        btn_row.pack(fill=tk.X, pady=2)

        self.btn_run = tk.Button(
            btn_row,
            text="▶ PLAY",
            font=("Arial", 9, "bold"),
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            command=self.toggle_run,
            padx=8,
            pady=3,
            cursor="hand2",
            relief=tk.RAISED,
        )
        self.btn_run.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_step = tk.Button(
            btn_row,
            text="⏭ Step",
            font=("Arial", 9, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            command=self.single_step,
            padx=6,
            pady=3,
            cursor="hand2",
        )
        self.btn_step.pack(side=tk.LEFT, padx=2)

        self.btn_reset = tk.Button(
            btn_row,
            text="↺ Reset",
            font=("Arial", 9),
            bg="#475569",
            fg="#ffffff",
            activebackground="#334155",
            activeforeground="#ffffff",
            command=self.reset_game,
            padx=6,
            pady=3,
            cursor="hand2",
        )
        self.btn_reset.pack(side=tk.LEFT, padx=2)

        speed_row = tk.Frame(card_play, bg="#0f172a")
        speed_row.pack(fill=tk.X, pady=(6, 2))
        tk.Label(speed_row, text="Speed / Delay:", font=("Arial", 8), fg="#94a3b8", bg="#0f172a").pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(
            speed_row,
            from_=40,
            to=280,
            orient=tk.HORIZONTAL,
            bg="#0f172a",
            fg="#38bdf8",
            troughcolor="#1e293b",
            highlightthickness=0,
            showvalue=False,
            command=self.on_speed_change,
        )
        self.speed_slider.set(140)
        self.speed_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

        # Section 2: Method / Strategy Card
        card_method = tk.LabelFrame(f, text="  2. Multi-Agent Strategy  ", font=("Arial", 9, "bold"), fg="#a855f7", bg="#0f172a", padx=8, pady=6)
        card_method.pack(fill=tk.X, pady=(0, 6))

        self.lbl_method = tk.Label(card_method, text="Teammate Coordination:", font=("Arial", 8, "bold"), fg="#cbd5e1", bg="#0f172a")
        self.lbl_method.pack(anchor=tk.W)

        self.combo_method = ttk.Combobox(card_method, state="readonly")
        self.combo_method.pack(fill=tk.X, pady=(2, 6))
        self.combo_method.bind("<<ComboboxSelected>>", lambda e: self.on_method_selected())

        # Section 3: Human Controller (D-Pad) OR Autonomous AI Status
        self.card_interactive = tk.LabelFrame(f, text="  3. Agent Interaction  ", font=("Arial", 9, "bold"), fg="#00e5ff", bg="#0f172a", padx=8, pady=6)
        self.card_interactive.pack(fill=tk.X, pady=(0, 6))

        # Human D-Pad Frame
        self.human_ctrl_box = tk.Frame(self.card_interactive, bg="#0f172a")

        self.lbl_heading = tk.Label(self.human_ctrl_box, text="Facing: Right ➔", font=("Arial", 9, "bold"), fg="#facc15", bg="#0f172a")
        self.lbl_heading.pack(side=tk.TOP, pady=(0, 4))

        dpad_grid = tk.Frame(self.human_ctrl_box, bg="#0f172a")
        dpad_grid.pack(side=tk.TOP, pady=2)

        self.btn_up = tk.Button(
            dpad_grid, text="▲\nUP", font=("Arial", 8, "bold"), bg="#1e293b", fg="#ffffff", width=5, height=2,
            command=lambda: self.set_human_direction(ACTION_UP, "Up ▲")
        )
        self.btn_up.grid(row=0, column=1, padx=2, pady=2)

        self.btn_left = tk.Button(
            dpad_grid, text="◄\nLEFT", font=("Arial", 8, "bold"), bg="#1e293b", fg="#ffffff", width=5, height=2,
            command=lambda: self.set_human_direction(ACTION_LEFT, "Left ◄")
        )
        self.btn_left.grid(row=1, column=0, padx=2, pady=2)

        self.btn_stay = tk.Button(
            dpad_grid, text="⏹\nSTAY", font=("Arial", 8, "bold"), bg="#334155", fg="#ffffff", width=5, height=2,
            command=lambda: self.set_human_direction(ACTION_STAY, "Stay ⏹")
        )
        self.btn_stay.grid(row=1, column=1, padx=2, pady=2)

        self.btn_right = tk.Button(
            dpad_grid, text="►\nRIGHT", font=("Arial", 8, "bold"), bg="#0284c7", fg="#ffffff", width=5, height=2,
            command=lambda: self.set_human_direction(ACTION_RIGHT, "Right ►")
        )
        self.btn_right.grid(row=1, column=2, padx=2, pady=2)

        self.btn_down = tk.Button(
            dpad_grid, text="▼\nDOWN", font=("Arial", 8, "bold"), bg="#1e293b", fg="#ffffff", width=5, height=2,
            command=lambda: self.set_human_direction(ACTION_DOWN, "Down ▼")
        )
        self.btn_down.grid(row=2, column=1, padx=2, pady=2)

        tk.Label(
            self.human_ctrl_box,
            text="Use Arrow Keys, WASD, or Click D-Pad\nSpace = Play/Pause",
            font=("Arial", 8),
            fg="#94a3b8",
            bg="#0f172a",
            justify=tk.CENTER,
        ).pack(side=tk.TOP, pady=(4, 0))

        # Autonomous AI Status Box
        self.auto_status_box = tk.Frame(self.card_interactive, bg="#0f172a")
        tk.Label(
            self.auto_status_box,
            text="🤖 All Pac-Men Autonomous",
            font=("Arial", 9, "bold"),
            fg="#c084fc",
            bg="#0f172a",
        ).pack(anchor=tk.W)

        self.lbl_auto_desc = tk.Label(
            self.auto_status_box,
            text="AI agents communicate under DCOP / Shapley rewards to clear the maze without contention.",
            font=("Arial", 8),
            fg="#94a3b8",
            bg="#0f172a",
            justify=tk.LEFT,
            wraplength=190,
        )
        self.lbl_auto_desc.pack(anchor=tk.W, pady=(2, 4))

        # Section 4: Experimental Controls (from Experiments 1, 2, 3, 4)
        card_exp = tk.LabelFrame(f, text="  4. Experimental Controls  ", font=("Arial", 9, "bold"), fg="#10b981", bg="#0f172a", padx=8, pady=6)
        card_exp.pack(fill=tk.X, pady=(0, 6))

        # Maze Layout Selector
        tk.Label(card_exp, text="Maze Layout:", font=("Arial", 8), fg="#cbd5e1", bg="#0f172a").pack(anchor=tk.W)
        self.combo_layout = ttk.Combobox(card_exp, textvariable=self.var_layout, values=["proposal_maze", "medium_classic", "open_contention", "small_grid"], state="readonly")
        self.combo_layout.pack(fill=tk.X, pady=(1, 4))

        # Team Size k (Exp 1)
        k_row = tk.Frame(card_exp, bg="#0f172a")
        k_row.pack(fill=tk.X, pady=2)
        tk.Label(k_row, text="Team Size k (Exp 1):", font=("Arial", 8), fg="#cbd5e1", bg="#0f172a").pack(side=tk.LEFT)
        self.lbl_k_val = tk.Label(k_row, text="3", font=("Arial", 8, "bold"), fg="#38bdf8", bg="#0f172a")
        self.lbl_k_val.pack(side=tk.RIGHT)
        self.slider_k = tk.Scale(card_exp, from_=2, to=5, orient=tk.HORIZONTAL, variable=self.var_k, bg="#0f172a", fg="#38bdf8", troughcolor="#1e293b", highlightthickness=0, showvalue=False, command=lambda v: self.lbl_k_val.config(text=str(v)))
        self.slider_k.pack(fill=tk.X, pady=(0, 4))

        # Ghost Count M & Danger Beta (Exp 1 & 3)
        g_row = tk.Frame(card_exp, bg="#0f172a")
        g_row.pack(fill=tk.X, pady=2)
        tk.Label(g_row, text="Ghosts M (Exp 3):", font=("Arial", 8), fg="#cbd5e1", bg="#0f172a").pack(side=tk.LEFT)
        self.lbl_g_val = tk.Label(g_row, text="2", font=("Arial", 8, "bold"), fg="#ef4444", bg="#0f172a")
        self.lbl_g_val.pack(side=tk.RIGHT)
        self.slider_ghosts = tk.Scale(card_exp, from_=0, to=4, orient=tk.HORIZONTAL, variable=self.var_ghosts, bg="#0f172a", fg="#ef4444", troughcolor="#1e293b", highlightthickness=0, showvalue=False, command=lambda v: self.lbl_g_val.config(text=str(v)))
        self.slider_ghosts.pack(fill=tk.X, pady=(0, 4))

        beta_row = tk.Frame(card_exp, bg="#0f172a")
        beta_row.pack(fill=tk.X, pady=2)
        tk.Label(beta_row, text="Ghost Danger β (Exp 1):", font=("Arial", 8), fg="#cbd5e1", bg="#0f172a").pack(side=tk.LEFT)
        self.lbl_beta_val = tk.Label(beta_row, text="0.5", font=("Arial", 8, "bold"), fg="#f97316", bg="#0f172a")
        self.lbl_beta_val.pack(side=tk.RIGHT)
        self.slider_beta = tk.Scale(card_exp, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.var_beta, bg="#0f172a", fg="#f97316", troughcolor="#1e293b", highlightthickness=0, showvalue=False, command=lambda v: self.lbl_beta_val.config(text=f"{float(v):.1f}"))
        self.slider_beta.pack(fill=tk.X, pady=(0, 4))

        # DCOP Budget B (Exp 2)
        b_row = tk.Frame(card_exp, bg="#0f172a")
        b_row.pack(fill=tk.X, pady=2)
        tk.Label(b_row, text="DCOP Budget B (Exp 2):", font=("Arial", 8), fg="#cbd5e1", bg="#0f172a").pack(side=tk.LEFT)
        self.lbl_b_val = tk.Label(b_row, text="24", font=("Arial", 8, "bold"), fg="#a855f7", bg="#0f172a")
        self.lbl_b_val.pack(side=tk.RIGHT)
        self.slider_budget = tk.Scale(card_exp, from_=0, to=48, orient=tk.HORIZONTAL, variable=self.var_budget, bg="#0f172a", fg="#a855f7", troughcolor="#1e293b", highlightthickness=0, showvalue=False, command=lambda v: self.lbl_b_val.config(text=str(v)))
        self.slider_budget.pack(fill=tk.X, pady=(0, 4))

        # Shapley Lambda (Exp 3)
        lmb_row = tk.Frame(card_exp, bg="#0f172a")
        lmb_row.pack(fill=tk.X, pady=2)
        tk.Label(lmb_row, text="Shapley λ (Exp 3):", font=("Arial", 8), fg="#cbd5e1", bg="#0f172a").pack(side=tk.LEFT)
        self.lbl_lmb_val = tk.Label(lmb_row, text="0.50", font=("Arial", 8, "bold"), fg="#facc15", bg="#0f172a")
        self.lbl_lmb_val.pack(side=tk.RIGHT)
        self.slider_lambda = tk.Scale(card_exp, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.var_lambda, bg="#0f172a", fg="#facc15", troughcolor="#1e293b", highlightthickness=0, showvalue=False, command=lambda v: self.lbl_lmb_val.config(text=f"{float(v):.2f}"))
        self.slider_lambda.pack(fill=tk.X, pady=(0, 6))

        # Apply Experimental Setup Button
        btn_apply = tk.Button(
            card_exp,
            text="⚙ Apply Scenario Settings",
            font=("Arial", 9, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            command=self.apply_experimental_settings,
            pady=3,
            cursor="hand2",
        )
        btn_apply.pack(fill=tk.X)

        self._sync_mode_ui()

    def _sync_mode_ui(self):
        """Synchronize the 25% sidebar widgets with the selected mode."""
        if self.is_human_mode():
            self.card_interactive.config(text="  3. Human Controller (P1)  ", fg="#00e5ff")
            self.lbl_method.config(text="Teammates Coordination (P2..Pk):")
            self.combo_method.config(
                values=[
                    "DCOP Max-Sum (B=24)",
                    "DCOP DSA (B=12)",
                    "Shapley Q-Learning (λ=0.5)",
                    "Equal-Share Q-Learning",
                    "Selfish Greedy",
                ]
            )
            self.combo_method.set(self.human_teammate_mode.get())
            self.auto_status_box.pack_forget()
            self.human_ctrl_box.pack(fill=tk.BOTH, expand=True)
            self.lbl_legend.config(
                text="Legend:  Cyan = YOU (P1)  |  Yellow = P2  |  Pink = P3  |  Red/Orange = Ghosts  |  Dashed Lines = Region Sectors"
            )
        else:
            self.card_interactive.config(text="  3. Autonomous Simulation  ", fg="#c084fc")
            self.lbl_method.config(text="Team Joint Algorithm (All Agents):")
            self.combo_method.config(
                values=[
                    "DCOP Max-Sum (B=24)",
                    "DCOP DSA (B=12)",
                    "Shapley Q-Learning (λ=0.5)",
                    "Equal-Share Q-Learning",
                    "Selfish Greedy",
                    "Centralized Social Optimum (π*)",
                    "Random Baseline",
                ]
            )
            self.combo_method.set(self.auto_team_mode.get())
            self.human_ctrl_box.pack_forget()
            self.auto_status_box.pack(fill=tk.BOTH, expand=True)
            self.lbl_legend.config(
                text="Legend:  Cyan = P1  |  Yellow = P2  |  Pink = P3  |  Red/Orange = Ghosts  |  Dashed Lines = Region Sectors"
            )

        self.recompute_board_dimensions()

    def on_primary_mode_changed(self):
        self._sync_mode_ui()
        self.reset_game()

    def on_method_selected(self):
        selected = self.combo_method.get()
        if self.is_human_mode():
            self.human_teammate_mode.set(selected)
        else:
            self.auto_team_mode.set(selected)

        # Lazy load trained RL policies if selected
        if "Shapley" in selected and self.shapley_agents is None:
            self.lbl_score.config(text="Training Shapley Agents...")
            self.root.update()
            self.shapley_agents = train_q_learning(
                layout_name=self.layout_name, num_pacman=self.num_pacman, num_ghosts=self.num_ghosts,
                ghost_beta=self.ghost_beta, mode="shapley", shapley_lambda=self.shapley_lambda,
                num_episodes=30, seed=42
            )
        elif "Equal-Share" in selected and self.equal_share_agents is None:
            self.lbl_score.config(text="Training Equal-Share Agents...")
            self.root.update()
            self.equal_share_agents = train_q_learning(
                layout_name=self.layout_name, num_pacman=self.num_pacman, num_ghosts=self.num_ghosts,
                ghost_beta=self.ghost_beta, mode="equal_share",
                num_episodes=30, seed=42
            )

        self.reset_game()

    def apply_experimental_settings(self):
        """Apply all settings from the Experimental Controls card and restart environment."""
        self.layout_name = self.var_layout.get()
        self.num_pacman = int(self.var_k.get())
        self.num_ghosts = int(self.var_ghosts.get())
        self.ghost_beta = float(self.var_beta.get())
        self.dcop_budget = int(self.var_budget.get())
        self.shapley_lambda = float(self.var_lambda.get())

        # Reset trained caches
        self.shapley_agents = None
        self.equal_share_agents = None

        self._init_environment()
        self.recompute_board_dimensions()
        self.reset_game()

    def on_canvas_container_resize(self, event):
        """Dynamically scale cell_size to fill the 75% board space."""
        if not hasattr(self, "env") or self.env.width == 0 or self.env.height == 0:
            return
        w = event.width - 24
        h = event.height - 24
        if w <= 50 or h <= 50:
            return

        calc_cell = min(w // self.env.width, h // self.env.height)
        new_cell_size = max(20, min(50, calc_cell))
        if new_cell_size != self.cell_size:
            self.cell_size = new_cell_size
            self.recompute_board_dimensions()
            self.draw_maze()
            if getattr(self, "game_over", False):
                self.draw_game_over()

    def recompute_board_dimensions(self):
        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size
        self.canvas.config(width=canvas_w, height=canvas_h)
        self.draw_maze()

    def _bind_keys(self):
        self.root.bind("<Up>", lambda e: self.set_human_direction(ACTION_UP, "Up ▲"))
        self.root.bind("<Down>", lambda e: self.set_human_direction(ACTION_DOWN, "Down ▼"))
        self.root.bind("<Left>", lambda e: self.set_human_direction(ACTION_LEFT, "Left ◄"))
        self.root.bind("<Right>", lambda e: self.set_human_direction(ACTION_RIGHT, "Right ►"))
        self.root.bind("<w>", lambda e: self.set_human_direction(ACTION_UP, "Up ▲"))
        self.root.bind("<W>", lambda e: self.set_human_direction(ACTION_UP, "Up ▲"))
        self.root.bind("<s>", lambda e: self.set_human_direction(ACTION_DOWN, "Down ▼"))
        self.root.bind("<S>", lambda e: self.set_human_direction(ACTION_DOWN, "Down ▼"))
        self.root.bind("<a>", lambda e: self.set_human_direction(ACTION_LEFT, "Left ◄"))
        self.root.bind("<A>", lambda e: self.set_human_direction(ACTION_LEFT, "Left ◄"))
        self.root.bind("<d>", lambda e: self.set_human_direction(ACTION_RIGHT, "Right ►"))
        self.root.bind("<D>", lambda e: self.set_human_direction(ACTION_RIGHT, "Right ►"))
        self.root.bind("<space>", lambda e: self.toggle_run())

    def set_human_direction(self, action: int, label_str: str):
        if not self.is_human_mode():
            return
        self.queued_action = action
        self.current_action = action
        if hasattr(self, "lbl_heading") and self.lbl_heading.winfo_exists():
            self.lbl_heading.config(text=f"Facing: {label_str}")

        if hasattr(self, "btn_up") and self.btn_up.winfo_exists():
            for b in (self.btn_up, self.btn_down, self.btn_left, self.btn_right, self.btn_stay):
                b.config(bg="#1e293b")
            if action == ACTION_UP: self.btn_up.config(bg="#0284c7")
            elif action == ACTION_DOWN: self.btn_down.config(bg="#0284c7")
            elif action == ACTION_LEFT: self.btn_left.config(bg="#0284c7")
            elif action == ACTION_RIGHT: self.btn_right.config(bg="#0284c7")
            elif action == ACTION_STAY: self.btn_stay.config(bg="#0284c7")

        # While paused, each move key advances one step; stop and show the
        # game-over panel when the episode ends (never step past the end).
        if not self.is_running and not getattr(self, "game_over", False):
            if self.step_environment():
                self._finish_game()

    def on_speed_change(self, val):
        self.step_delay = int(val)

    def reset_game(self):
        self.is_running = False
        self.btn_run.config(text="▶ PLAY", bg="#10b981")
        self.env.reset(seed=random.randint(1, 10000))
        self.score = 0.0
        self.gamma = 1.0
        self.agent_rewards = [0.0] * self.num_pacman
        self.messages_used = 0
        self.game_over = False
        self.initial_pellets = len(self.env.pellets)
        self.facing = self._initial_facing()
        self.current_action = ACTION_RIGHT
        self.queued_action = ACTION_RIGHT
        if hasattr(self, "lbl_heading") and self.lbl_heading.winfo_exists():
            self.lbl_heading.config(text="Facing: Right ➔")
        self.draw_maze()
        self.update_hud()

    def _initial_facing(self):
        """Start facing right, except P3 who starts on the right side facing left."""
        return [ACTION_LEFT if i == 2 else ACTION_RIGHT for i in range(self.num_pacman)]

    def toggle_run(self):
        if getattr(self, "game_over", False):
            self.reset_game()  # "↺ RESTART" really restarts
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_run.config(text="⏸ PAUSE", bg="#f59e0b")
            self._game_loop()
        else:
            self.btn_run.config(text="▶ PLAY", bg="#10b981")

    def _game_loop(self):
        if not self.is_running:
            return
        done = self.step_environment()
        if not done:
            self.root.after(self.step_delay, self._game_loop)
        else:
            self._finish_game()

    def single_step(self):
        if not self.is_running and not getattr(self, "game_over", False):
            if self.step_environment():
                self._finish_game()

    def _finish_game(self):
        self.is_running = False
        self.game_over = True
        self.btn_run.config(text="↺ RESTART", bg="#10b981")
        self.draw_game_over()

    def draw_game_over(self):
        """Game-over panel drawn on top of the final board."""
        cv = self.canvas
        w = max(cv.winfo_width(), int(cv.cget("width") or 0))
        h = max(cv.winfo_height(), int(cv.cget("height") or 0))
        cleared = len(self.env.pellets) == 0
        k = self.num_pacman
        eaten = self.initial_pellets - len(self.env.pellets)
        gini = compute_gini_index(self.agent_rewards)

        # dimmed background and card
        cv.create_rectangle(0, 0, w, h, fill="#020617", stipple="gray50", outline="", tags="gameover")
        cw, ch = min(440, w - 40), min(150 + 26 * k + 70, h - 30)
        x0, y0 = (w - cw) / 2, (h - ch) / 2
        cv.create_rectangle(x0, y0, x0 + cw, y0 + ch, fill="#0b1430", outline="#facc15", width=3, tags="gameover")

        cx = w / 2
        y = y0 + 28
        title = "🏆 MAZE CLEARED!" if cleared else "⏱ TIME UP!"
        cv.create_text(cx, y, text=title, fill="#4ade80" if cleared else "#facc15",
                       font=("Arial", 22, "bold"), tags="gameover")
        y += 28
        sub = (f"All {self.initial_pellets} pellets eaten in {self.env.step_count} steps."
               if cleared else
               f"Step limit of {self.env.max_steps} reached, {len(self.env.pellets)} pellets left.")
        cv.create_text(cx, y, text=sub, fill="#94a3b8", font=("Arial", 10), tags="gameover")
        y += 26
        cv.create_text(cx, y, fill="#f8fafc", font=("Arial", 11, "bold"), tags="gameover",
                       text=(f"Score {self.score:.0f}   |   Pellets {eaten}/{self.initial_pellets}   |   "
                             f"Captures {self.env.total_captures}   |   Gini {gini:.3f}"))
        y += 28

        # per-agent table
        cols = [x0 + cw * f for f in (0.18, 0.43, 0.63, 0.83)]
        for c, head in zip(cols, ("Pac-Man", "Pellets", "Captured", "Return")):
            cv.create_text(c, y, text=head, fill="#94a3b8", font=("Arial", 10, "bold"), tags="gameover")
        y += 20
        colors = ["#00e5ff", "#ffd600", "#ff4081", "#4ade80", "#c084fc"]
        pellets_by = list(self.env.agent_pellets)
        for i in range(k):
            name = "P1 (YOU)" if (self.is_human_mode() and i == 0) else f"P{i + 1}"
            vals = (name, pellets_by[i], self.env.agent_captures[i], f"{self.agent_rewards[i]:.0f}")
            for j, (c, v) in enumerate(zip(cols, vals)):
                cv.create_text(c, y, text=str(v), fill=colors[i % len(colors)] if j == 0 else "#e2e8f0",
                               font=("Arial", 10, "bold" if j == 0 else "normal"), tags="gameover")
            y += 22

        best = max(pellets_by) if pellets_by else 0
        leaders = [i for i, v in enumerate(pellets_by) if v == best]
        if best == 0:
            msg = "Nobody ate a pellet this round."
        elif len(leaders) > 1:
            msg = "Tie for top harvester: " + ", ".join(f"P{i + 1}" for i in leaders) + f" ({best} each)"
        else:
            i = leaders[0]
            who = "YOU (P1)" if (self.is_human_mode() and i == 0) else f"P{i + 1}"
            msg = f"Top harvester: {who} with {best} pellets"
        y += 6
        cv.create_text(cx, y, text=msg, fill="#facc15", font=("Arial", 10, "bold"), tags="gameover")
        y += 22
        cv.create_text(cx, y, text="Press ↺ RESTART to play again", fill="#94a3b8",
                       font=("Arial", 9, "italic"), tags="gameover")

    def step_environment(self) -> bool:
        joint_action = [ACTION_STAY] * self.num_pacman

        if self.is_human_mode():
            # Pac-Man 1: Human Player
            my_pos = self.env.pacman_positions[0]
            valid_p1 = self.env.get_valid_actions(my_pos)

            if self.queued_action in valid_p1:
                joint_action[0] = self.queued_action
                self.current_action = self.queued_action
            elif self.current_action in valid_p1:
                joint_action[0] = self.current_action
            else:
                joint_action[0] = ACTION_STAY

            # Teammates P2..Pk
            teammate_algo = self.human_teammate_mode.get()
            if "Max-Sum" in teammate_algo:
                z, msgs = self.dcop_solver.solve_max_sum(message_budget=self.dcop_budget)
                self.messages_used += msgs
                for i in range(1, self.num_pacman):
                    self.dcop_agents[i].set_region(z[i])
                    joint_action[i] = self.dcop_agents[i].get_action(self.env)
            elif "DSA" in teammate_algo:
                z, msgs = self.dcop_solver.solve_dsa(message_budget=self.dcop_budget)
                self.messages_used += msgs
                for i in range(1, self.num_pacman):
                    self.dcop_agents[i].set_region(z[i])
                    joint_action[i] = self.dcop_agents[i].get_action(self.env)
            elif "Shapley" in teammate_algo and self.shapley_agents:
                for i in range(1, self.num_pacman):
                    joint_action[i] = self.shapley_agents[i].get_action(self.env, explore=False)
            elif "Equal-Share" in teammate_algo and self.equal_share_agents:
                for i in range(1, self.num_pacman):
                    joint_action[i] = self.equal_share_agents[i].get_action(self.env, explore=False)
            else:  # Selfish Greedy
                for i in range(1, self.num_pacman):
                    joint_action[i] = self.greedy_agents[i].get_action(self.env)

        else:
            # Autonomous AI controls all k Pac-Men
            auto_algo = self.auto_team_mode.get()
            if "Max-Sum" in auto_algo:
                z, msgs = self.dcop_solver.solve_max_sum(message_budget=self.dcop_budget)
                self.messages_used += msgs
                for i in range(self.num_pacman):
                    self.dcop_agents[i].set_region(z[i])
                    joint_action[i] = self.dcop_agents[i].get_action(self.env)
            elif "DSA" in auto_algo:
                z, msgs = self.dcop_solver.solve_dsa(message_budget=self.dcop_budget)
                self.messages_used += msgs
                for i in range(self.num_pacman):
                    self.dcop_agents[i].set_region(z[i])
                    joint_action[i] = self.dcop_agents[i].get_action(self.env)
            elif "Shapley" in auto_algo and self.shapley_agents:
                joint_action = [self.shapley_agents[i].get_action(self.env, explore=False) for i in range(self.num_pacman)]
            elif "Equal-Share" in auto_algo and self.equal_share_agents:
                joint_action = [self.equal_share_agents[i].get_action(self.env, explore=False) for i in range(self.num_pacman)]
            elif "Centralized" in auto_algo:
                joint_action = self.centralized_planner.get_joint_action(self.env)
            elif "Random" in auto_algo:
                joint_action = [random.choice(self.env.get_valid_actions(self.env.pacman_positions[i])) for i in range(self.num_pacman)]
            else:  # Selfish Greedy
                joint_action = [self.greedy_agents[i].get_action(self.env) for i in range(self.num_pacman)]

        # Each Pac-Man faces the way it actually moves (blocked moves and Stay keep the old facing)
        if not hasattr(self, "facing") or len(self.facing) != self.num_pacman:
            self.facing = self._initial_facing()
        for i, a in enumerate(joint_action):
            if a == ACTION_STAY:
                continue
            dx, dy = ACTION_DELTAS[a]
            x, y = self.env.pacman_positions[i]
            if (x + dx, y + dy) in self.env.open_cells:
                self.facing[i] = a

        _, rewards, done, info = self.env.step(joint_action)
        for i in range(self.num_pacman):
            self.agent_rewards[i] += rewards[i]
        self.score += self.gamma * sum(rewards)
        self.gamma *= self.env.gamma

        self.draw_maze()
        self.update_hud()
        return done

    def update_hud(self):
        self.lbl_score.config(text=f"Score: {self.score:.1f}")
        self.lbl_pellets.config(text=f"Pellets: {len(self.env.pellets)}")
        self.lbl_captures.config(text=f"Captures: {self.env.total_captures}")
        self.lbl_steps.config(text=f"Steps: {self.env.step_count}")

        # Live Gini index across agents (same formula as src/metrics.py)
        gini = compute_gini_index(self.agent_rewards)
        self.lbl_gini.config(text=f"Gini: {gini:.3f}")

        # Benchmark PoA: measured in the Python experiments (report), not computed live
        poa = self.benchmark_poa()
        self.lbl_poa.config(text=f"Benchmark PoA: {poa:.3f}" if poa is not None else "Benchmark PoA: n/a")

    # Measured PoA from results/experiment_4_conditions.json (k=3, 2 ghosts, beta=0.5)
    # and results/experiment_2.json (Max-Sum budget sweep, m=3).
    BENCH_POA = {
        "proposal_maze": {"Max-Sum": 1.101, "DSA": 1.189, "Shapley": 1.193, "Equal-Share": 1.501,
                          "Centralized": 1.000, "Greedy": 1.019},
        "small_grid": {"Max-Sum": 1.398, "DSA": 1.414, "Shapley": 1.387, "Equal-Share": 1.275,
                       "Centralized": 1.000, "Greedy": 1.569},
        "medium_classic": {"Max-Sum": 1.182, "DSA": 1.093, "Shapley": 1.181, "Equal-Share": 1.449,
                           "Centralized": 1.000, "Greedy": 1.081},
        "open_contention": {"Max-Sum": 1.031, "DSA": 1.165, "Shapley": 1.211, "Equal-Share": 2.379,
                            "Centralized": 1.000, "Greedy": 1.049},
    }
    EXP2_MAXSUM_J = {0: 735.3, 6: 735.3, 12: 805.6, 24: 826.2, 36: 799.2, 48: 776.7}
    EXP2_OPT_J = 905.1

    def benchmark_poa(self):
        """Measured PoA for the current setting, or None when it was not measured
        (human mode, team size other than 3, ghost count other than 2, Random)."""
        if (self.is_human_mode() or self.num_pacman != 3 or self.env.num_ghosts != 2
                or abs(self.ghost_beta - 0.5) > 1e-9):
            return None
        algo = self.auto_team_mode.get()
        table = self.BENCH_POA.get(self.layout_name)
        if table is None:
            return None
        if "Max-Sum" in algo:
            if self.layout_name == "proposal_maze" and self.dcop_budget in self.EXP2_MAXSUM_J:
                return self.EXP2_OPT_J / self.EXP2_MAXSUM_J[self.dcop_budget]
            return table["Max-Sum"] if self.dcop_budget == 24 else None
        if "DSA" in algo:
            return table["DSA"] if self.dcop_budget == 12 else None  # measured at B=12 only
        for key, val in table.items():
            if key in algo:
                return val
        return None

    def draw_maze(self):
        self.canvas.delete("all")
        cs = self.cell_size

        # Region Shading for proposal_maze
        if self.layout_name == "proposal_maze":
            self.canvas.create_rectangle(cs, cs, 7 * cs, 14 * cs, fill="#041b33", outline="")
            self.canvas.create_rectangle(7 * cs, cs, 14 * cs, 14 * cs, fill="#262104", outline="")
            self.canvas.create_rectangle(14 * cs, cs, 20 * cs, 14 * cs, fill="#2e051d", outline="")

            # Region Divider dashed lines
            self.canvas.create_line(7 * cs, cs, 7 * cs, 14 * cs, fill="#38bdf8", dash=(4, 4), width=2)
            self.canvas.create_line(14 * cs, cs, 14 * cs, 14 * cs, fill="#38bdf8", dash=(4, 4), width=2)

        # Draw Walls
        for (wx, wy) in self.env.walls:
            x0, y0 = wx * cs, wy * cs
            self.canvas.create_rectangle(x0, y0, x0 + cs, y0 + cs, fill="#050e24", outline="#1e40af", width=2)
            self.canvas.create_rectangle(x0 + 3, y0 + 3, x0 + cs - 3, y0 + cs - 3, outline="#2563eb", width=1)

        # Draw Pellets
        pellet_r = max(2, cs // 8)
        power_r = max(4, cs // 3 - 2)
        for (px, py) in self.env.pellets:
            cx, cy = px * cs + cs // 2, py * cs + cs // 2
            if (px, py) in self.env.power_pellets:
                self.canvas.create_oval(cx - power_r - 2, cy - power_r - 2, cx + power_r + 2, cy + power_r + 2, fill="#786400", outline="")
                self.canvas.create_oval(cx - power_r, cy - power_r, cx + power_r, cy + power_r, fill="#facc15", outline="#ffffff", width=1)
            else:
                self.canvas.create_oval(cx - pellet_r, cy - pellet_r, cx + pellet_r, cy + pellet_r, fill="#fef08a", outline="")

        # Draw Pac-Man Agents
        pacman_colors = ["#00e5ff", "#ffd600", "#ff4081", "#4ade80", "#c084fc"]
        is_human = self.is_human_mode()
        r = cs // 2 - 2

        for i, (px, py) in enumerate(self.env.pacman_positions):
            cx, cy = px * cs + cs // 2, py * cs + cs // 2
            col = pacman_colors[i % len(pacman_colors)]

            # ONLY draw glowing halo and "YOU" badge if in HUMAN mode on Pac-Man 1 (i == 0)
            if is_human and i == 0:
                self.canvas.create_oval(cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4, outline="#00e5ff", width=2)
                self.canvas.create_text(cx, cy - r - 8, text="YOU", fill="#00e5ff", font=("Arial", max(7, cs // 4), "bold"))

            # Draw Pac-Man wedge with the mouth pointing where it is moving.
            # Tk angles: 0 = right, 90 = up, 180 = left, 270 = down (counter-clockwise).
            facing = self.facing[i] if hasattr(self, "facing") and i < len(self.facing) else ACTION_RIGHT
            mouth_center = {ACTION_RIGHT: 0, ACTION_UP: 90, ACTION_LEFT: 180, ACTION_DOWN: 270}.get(facing, 0)
            start_ang = mouth_center + 30
            extent_ang = 300
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=start_ang, extent=extent_ang,
                fill=col, outline="#000000", width=1.2
            )
            self.canvas.create_text(cx, cy, text=f"P{i+1}", fill="#050e24", font=("Arial", max(8, cs // 3), "bold"))

        # Draw Ghosts
        ghost_colors = ["#ef4444", "#f97316", "#06b6d4", "#a855f7"]
        for j, (gx, gy) in enumerate(self.env.ghost_positions):
            cx, cy = gx * cs + cs // 2, gy * cs + cs // 2
            gcol = ghost_colors[j % len(ghost_colors)]
            # Ghost dome
            self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180, fill=gcol, outline="#000000")
            self.canvas.create_rectangle(cx - r, cy, cx + r, cy + r - 2, fill=gcol, outline="")
            # Eyes
            eye_off = max(2, cs // 7)
            self.canvas.create_oval(cx - eye_off - 3, cy - 4, cx - eye_off + 2, cy + 2, fill="#ffffff", outline="")
            self.canvas.create_oval(cx + eye_off - 2, cy - 4, cx + eye_off + 3, cy + 2, fill="#ffffff", outline="")
            self.canvas.create_oval(cx - eye_off - 1, cy - 2, cx - eye_off + 1, cy + 1, fill="#0f172a", outline="")
            self.canvas.create_oval(cx + eye_off, cy - 2, cx + eye_off + 2, cy + 1, fill="#0f172a", outline="")
            self.canvas.create_text(cx, cy + r - 3, text=f"G{j+1}", fill="#ffffff", font=("Arial", max(7, cs // 4), "bold"))


def main():
    root = tk.Tk()
    root.geometry("1280x780")
    root.minsize(1050, 680)
    root.resizable(True, True)
    app = PacmanGameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
