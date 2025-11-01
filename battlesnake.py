"""
Battlesnake Duel A* — smart pathfinding & safety-aware move selection
--------------------------------------------------------------------
One-file Flask app implementing an A*-driven strategy tailored for 1v1 (duel)
Battlesnake. Designed to be readable, hackable, and strong out-of-the-box.

Key features:
- A* pathfinding with multi-goal support (e.g., all food squares)
- Dynamic danger map (enemy head reach, bodies, future neck zones)
- Head-to-head avoidance when out-sized or equal
- Tail-chasing & food targeting with health-aware policy
- Space (flood-fill) viability checks to avoid self-traps
- Soft scoring that blends distance, risk, space, and center bias
- Solid fallbacks when no A*-safe path exists

No external dependencies besides Flask (which Battlesnake quickstarts already use).

Run locally:
  export PORT=8000
  python app.py

Deploy on Replit/Heroku/Render/Fly the same way as the official quickstarts.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
import math
import os

from flask import Flask, request, jsonify

# -------------------------------
# Data structures
# -------------------------------

Coord = Tuple[int, int]

@dataclass
class Snake:
    id: str
    health: int
    body: List[Coord]  # head first
    length: int

    @property
    def head(self) -> Coord:
        return self.body[0]

    @property
    def tail(self) -> Coord:
        return self.body[-1]

@dataclass
class Board:
    width: int
    height: int
    food: Set[Coord]
    hazards: Set[Coord]
    you: Snake
    opponent: Optional[Snake]

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height

# -------------------------------
# Helpers
# -------------------------------

DIRS: Dict[str, Coord] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

INV_DIR: Dict[Coord, str] = {v: k for k, v in DIRS.items()}


def add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1])


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(board: Board, c: Coord) -> List[Coord]:
    res = []
    for d in DIRS.values():
        n = add(c, d)
        if board.in_bounds(n):
            res.append(n)
    return res

# -------------------------------
# Board parsing & occupancy
# -------------------------------

def parse_board(data: Dict) -> Board:
    width = data["board"]["width"]
    height = data["board"]["height"]
    you_raw = data["you"]
    you = Snake(
        id=you_raw["id"],
        health=you_raw["health"],
        body=[(p["x"], p["y"]) for p in you_raw["body"]["data"] if isinstance(you_raw["body"], dict)]
        if isinstance(you_raw.get("body"), dict)
        else [(p["x"], p["y"]) for p in you_raw["body"]],
        length=you_raw["length"],
    )

    # Battlesnake API v1 vs v2 compat for bodies
    snakes = data["board"].get("snakes", [])
    opp: Optional[Snake] = None
    for s in snakes:
        if s["id"] == you.id:
            continue
        body = [
            (p["x"], p["y"]) for p in s["body"]["data"] if isinstance(s["body"], dict)
        ] if isinstance(s.get("body"), dict) else [(p["x"], p["y"]) for p in s["body"]]
        opp = Snake(id=s["id"], health=s["health"], body=body, length=s["length"])
        break  # duel: at most one opponent matters

    food = set((f["x"], f["y"]) for f in data["board"].get("food", []))
    hazards = set((h["x"], h["y"]) for h in data["board"].get("hazards", []))

    return Board(width=width, height=height, food=food, hazards=hazards, you=you, opponent=opp)


def occupied_squares(board: Board) -> Set[Coord]:
    occ = set(board.you.body)
    if board.opponent:
        occ.update(board.opponent.body)
    return occ

# -------------------------------
# Danger map & safety
# -------------------------------

def future_enemy_reach(board: Board) -> Set[Coord]:
    """Squares the enemy head could move into next turn."""
    reach = set()
    if not board.opponent:
        return reach
    for n in neighbors(board, board.opponent.head):
        reach.add(n)
    return reach


def danger_map(board: Board) -> Dict[Coord, float]:
    """Return a risk weight per cell. Higher is worse."""
    danger: Dict[Coord, float] = {}
    occ = occupied_squares(board)

    # Hard walls & bodies are near-infinite cost (except tails that may move)
    for x in range(board.width):
        for y in range(board.height):
            c = (x, y)
            if c in occ and c != board.you.tail:  # allow stepping onto own tail (likely moves)
                danger[c] = 1e6
            else:
                danger[c] = 0.0

    # Hazards cost (royale) — tune as needed
    for c in board.hazards:
        if c in danger:
            danger[c] += 5.0

    # Enemy head proximity
    enemy_reach = future_enemy_reach(board)
    if board.opponent:
        # If enemy is same or larger, avoid their next squares hard
        scale = 30.0 if board.opponent.length >= board.you.length else 8.0
        for c in enemy_reach:
            if c in danger:
                danger[c] += scale
        # Add a gradient around enemy head
        for n in neighbors(board, board.opponent.head):
            if n in danger:
                danger[n] += 4.0

    return danger

# -------------------------------
# Flood fill to measure free space from a candidate cell
# -------------------------------

def flood_area(board: Board, start: Coord, blocked: Set[Coord], limit: int = 250) -> int:
    if not board.in_bounds(start) or start in blocked:
        return 0
    q = deque([start])
    seen = {start}
    count = 0
    while q and count < limit:
        c = q.popleft()
        count += 1
        for n in neighbors(board, c):
            if n in seen or n in blocked or not board.in_bounds(n):
                continue
            seen.add(n)
            q.append(n)
    return count

# -------------------------------
# A* pathfinding
# -------------------------------

def astar(board: Board, start: Coord, goals: Set[Coord], danger: Dict[Coord, float]) -> Optional[List[Coord]]:
    if not goals:
        return None

    def heuristic(c: Coord) -> float:
        # Nearest-goal Manhattan + a light center bias
        dmin = min(manhattan(c, g) for g in goals)
        cx = board.width / 2.0
        cy = board.height / 2.0
        center = abs(c[0] - cx) + abs(c[1] - cy)
        return dmin + 0.05 * center

    open_set: Set[Coord] = {start}
    came_from: Dict[Coord, Coord] = {}
    g: Dict[Coord, float] = {start: 0.0}
    f: Dict[Coord, float] = {start: heuristic(start)}

    # Use a simple loop to select min f (board sizes are small)
    while open_set:
        current = min(open_set, key=lambda c: f.get(c, math.inf))
        if current in goals:
            # Reconstruct path (excluding start, including current)
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        open_set.remove(current)

        for n in neighbors(board, current):
            # Cost to enter neighbor includes danger
            base = 1.0
            cost = base + danger.get(n, 0.0)
            tentative_g = g[current] + cost
            if tentative_g < g.get(n, math.inf):
                came_from[n] = current
                g[n] = tentative_g
                f[n] = tentative_g + heuristic(n)
                if n not in open_set:
                    open_set.add(n)

    return None

# -------------------------------
# Move scoring / policy
# -------------------------------

def choose_move(board: Board) -> str:
    occ = occupied_squares(board)
    danger = danger_map(board)

    head = board.you.head

    # Candidate next squares by direction
    candidates: List[Tuple[str, Coord]] = []
    for dname, delta in DIRS.items():
        nxt = add(head, delta)
        if board.in_bounds(nxt):
            candidates.append((dname, nxt))

    # Filter out immediate death (hard walls/bodies)
    safe: List[Tuple[str, Coord]] = []
    for dname, nxt in candidates:
        if danger.get(nxt, 0.0) >= 1e6:  # hard block
            continue
        safe.append((dname, nxt))

    if not safe:
        # No safe squares; choose the least bad to survive potential tail move
        least = min(candidates, key=lambda kv: danger.get(kv[1], 1e9))
        return least[0]

    # Health-driven goal selection
    go_for_food = board.you.health <= 35 or (board.opponent and board.you.length <= board.opponent.length - 2)

    goals: Set[Coord] = set()
    if go_for_food and board.food:
        goals = set(board.food)
    else:
        # Chase space/center or enemy tail (safer than head)
        goals = set()
        if board.opponent:
            goals.add(board.opponent.tail)
        # Add soft-center pseudo-goals to encourage space
        center_targets = [
            (board.width // 2, board.height // 2),
            (max(0, board.width // 2 - 1), board.height // 2),
            (board.width // 2, max(0, board.height // 2 - 1)),
        ]
        goals.update(c for c in center_targets if board.in_bounds(c))

    # A* towards goals, then evaluate each candidate by resulting path, danger, and area
    scored: List[Tuple[float, str]] = []
    blocked = set(occ)
    # Allow our tail to vacate
    blocked.discard(board.you.tail)

    for dname, nxt in safe:
        # Quick area check to avoid tiny pockets
        area = flood_area(board, nxt, blocked)
        area_score = 0.0
        if area < max(5, board.you.length // 2):
            area_score += 50.0  # very cramped
        elif area < board.you.length:
            area_score += 12.0

        # Temporarily mark our next as occupied for path calc realism
        temp_block = set(blocked)
        temp_block.add(nxt)

        # Build a temporary danger overlay that discourages stepping into traps
        temp_danger = dict(danger)
        # Mild penalty for hugging walls (discourage corridor traps)
        x, y = nxt
        if x == 0 or y == 0 or x == board.width - 1 or y == board.height - 1:
            temp_danger[nxt] = temp_danger.get(nxt, 0.0) + 2.0

        # Compute a path from NEXT square to goals
        path = astar(board, nxt, goals, temp_danger)

        # Path length & risk
        if path:
            # Sum risk along path (skip the start cell nxt, already counted)
            risk_along = sum(temp_danger.get(c, 0.0) for c in path[1:])
            dist = len(path) - 1
        else:
            # No path — heavily penalize unless we are not pursuing any hard goal
            risk_along = 80.0
            dist = 20

        # Enemy head-to-head risk if moving adjacent to enemy head
        h2h_penalty = 0.0
        if board.opponent:
            if manhattan(nxt, board.opponent.head) == 1 and board.opponent.length >= board.you.length:
                h2h_penalty += 40.0

        # Base move risk from danger map
        base_risk = danger.get(nxt, 0.0)

        # Score = lower is better
        score = (
            base_risk
            + h2h_penalty
            + area_score
            + 0.8 * dist
            + 0.15 * risk_along
        )
        # Prefer food square lightly when we need food
        if go_for_food and nxt in board.food:
            score -= 5.0

        # Light bias to not reverse into our neck if it reduces options
        if len(board.you.body) >= 2:
            neck = board.you.body[1]
            if nxt == neck:
                score += 8.0

        scored.append((score, dname))

    scored.sort()
    best_move = scored[0][1]
    return best_move

# -------------------------------
# Flask server
# -------------------------------

app = Flask(__name__)

@app.get("/")
def index():
    return jsonify({
        "apiversion": "1",
        "author": "gpt5-thinking",
        "color": "#4f46e5",
        "head": "smart-caterpillar",
        "tail": "pixel",
        "version": "duel-astar-1.0",
    })

@app.post("/move")
def move():
    data = request.get_json()
    board = parse_board(data)
    move_dir = choose_move(board)
    return jsonify({"move": move_dir, "shout": "A* online"})

@app.post("/start")
def start():
    return ("", 200)

@app.post("/end")
def end():
    return ("", 200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
