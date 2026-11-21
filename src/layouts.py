"""
Maze layouts for Multi-Agent Pac-Man.
Representation:
  '%' or '#' : Wall
  '.' : Pellet
  ' ' : Empty open space
  'P' : Pac-Man spawn position placeholder
  'G' : Ghost spawn position placeholder
"""

LAYOUTS = {
    # 21x15 Multi-Agent Proposal Maze: exact layout from Project Proposal Figure 1
    # Features 3 balanced vertical sectors (Region A, Region B, Region C),
    # central ghost house, symmetric corridors, and 4 corner power pellets ('o')
    "proposal_maze": [
        "#####################",
        "#o........#........o#",
        "#.###.###.#.###.###.#",
        "#.#...#.......#.P.#.#",
        "#.#.#.#.#####.#.#.#.#",
        "#...#.....G.....#...#",
        "###.###.#   #.###.###",
        "#.......# G #.......#",
        "###.###.#####.###.###",
        "#...#.....P.....#...#",
        "#.#.#.#.#####.#.#.#.#",
        "#.#...#.......#...#.#",
        "#.###.###.#.###.###.#",
        "#oP.......#........o#",
        "#####################"
    ],

    # 5x5 Ring Toy Maze: tiny enough for EXACT analysis (proposal method M1).
    # 8 open cells, 7 pellets, 2 Pac-Men, 1 ghost -> the full joint state space
    # can be enumerated, so the team optimum pi* and best responses are solved exactly
    # by dynamic programming (see src/exact_solver.py). The top-middle pellet is
    # contested by both Pac-Men.
    "toy_exact": [
        "#####",
        "#P.P#",
        "#.#.#",
        "#.G.#",
        "#####"
    ],

    # 6x6 Toy Maze: compact, great for exact analysis and equilibrium verification
    "toy": [
        "######",
        "#P..P#",
        "#.##.#",
        "#.GG.#",
        "#P..P#",
        "######"
    ],

    # 9x7 Small Grid: classic loops and multiple corridors
    "small_grid": [
        "#########",
        "#P.....P#",
        "#.###.#.#",
        "#...G...#",
        "#.#.###.#",
        "#P.....P#",
        "#########"
    ],

    # 13x9 Medium Classic: standard CS188-style multi-room environment
    "medium_classic": [
        "#############",
        "#P....#....P#",
        "#.###.#.###.#",
        "#...........#",
        "#.###.G.###.#",
        "#.....#.....#",
        "#.###.#.###.#",
        "#P....#....P#",
        "#############"
    ],

    # 11x11 Open Contention Maze: high contention area in the center
    "open_contention": [
        "###########",
        "#P.......P#",
        "#.##...##.#",
        "#.........#",
        "#..##G##..#",
        "#....G....#",
        "#..## ##..#",
        "#.........#",
        "#.##...##.#",
        "#P.......P#",
        "###########"
    ]
}


def get_layout(name: str):
    """Retrieve layout by name."""
    if name not in LAYOUTS:
        raise ValueError(f"Unknown layout: {name}. Available: {list(LAYOUTS.keys())}")
    return LAYOUTS[name]
