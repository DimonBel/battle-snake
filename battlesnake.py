"""
Battlesnake Duel A*+ — anticipatory pathfinding & robust safety checks
---------------------------------------------------------------------
One-file Flask app implementing an A*-driven strategy tailored for 1v1 (duel)
Battlesnake. Designed to be readable, hackable, and strong out-of-the-box.

Key upgrades vs baseline:
- Enemy-aware lookahead (1-ply) with worst-case safety scoring
- "Reachable food" filter: ignore food the enemy can claim strictly sooner
- Smarter danger model (wall/corner bias, squeeze corridors, head-on rules)
- Future-tail modeling (you & enemy) so stepping on a tail is allowed if it moves
- True space viability: BFS area + path-to-tail/escape test
- Risk-tuned soft scoring with smaller magic numbers collected in one config
- Deterministic tie-breaking via seeded RNG from game state
- Cleaner A* with heapq and closed set for speed

No external dependencies besides Flask (which Battlesnake quickstarts already use).

Run locally:
  export PORT=8000
  python app.py

Deploy on Replit/Heroku/Render/Fly the same way as the official quickstarts.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set, Iterable
from dataclasses import dataclass
from collections import deque
import math
import os
import random
import heapq

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

# Handles both legacy {body: {data: [...]}} and modern {body: [...]} formats

def _extract_body(raw) -> List[Coord]:
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return [(p["x"], p["y"]) for p in raw["data"]]
    return [(p["x"], p["y"]) for p in raw]


def parse_board(data: Dict) -> Board:
    width = data["board"]["width"]
    height = data["board"]["height"]

    you_raw = data["you"]
    you = Snake(
        id=you_raw["id"],
        health=you_raw["health"],
        body=_extract_body(you_raw["body"]),
        length=you_raw["length"],
    )

    opp: Optional[Snake] = None
    for s in data["board"].get("snakes", []):
        if s["id"] == you.id:
            continue
        opp = Snake(
            id=s["id"],
            health=s["health"],
            body=_extract_body(s["body"]),
            length=s["length"],
        )
        break  # duel only

    food = set((f["x"], f["y"]) for f in data["board"].get("food", []))
    hazards = set((h["x"], h["y"]) for h in data["board"].get("hazards", []))

    return Board(width=width, height=height, food=food, hazards=hazards, you=you, opponent=opp)


def occupied_squares(board: Board, *, include_tails: bool = True) -> Set[Coord]:
    occ = set(board.you.body)
    if board.opponent:
        occ.update(board.opponent.body)
    if not include_tails:
        occ.discard(board.you.tail)
        if board.opponent:
            occ.discard(board.opponent.tail)
    return occ

# -------------------------------
# Distance maps & food reachability
# -------------------------------

def multi_source_bfs(board: Board, starts: Iterable[Coord], blocked: Set[Coord]) -> Dict[Coord, int]:
    q = deque()
    dist: Dict[Coord, int] = {}
    for s in starts:
        if s in blocked or not board.in_bounds(s):
            continue
        dist[s] = 0
        q.append(s)
    while q:
        c = q.popleft()
        d = dist[c]
        for n in neighbors(board, c):
            if n in blocked or n in dist:
                continue
            dist[n] = d + 1
            q.append(n)
    return dist


def reachable_food(board: Board, blocked: Set[Coord]) -> Set[Coord]:
    if not board.food:
        return set()
    mydist = multi_source_bfs(board, [board.you.head], blocked)
    if board.opponent:
        oddist = multi_source_bfs(board, [board.opponent.head], blocked)
    else:
        oddist = {}
    good = set()
    for f in board.food:
        md = mydist.get(f, math.inf)
        od = oddist.get(f, math.inf)
        # Claim only if we arrive strictly sooner, or tie while longer (we can win head-to-head on food tile)
        if md < od or (md == od and (not board.opponent or board.you.length > board.opponent.length)):
            good.add(f)
    return good

# -------------------------------
# Danger map & safety
# -------------------------------

class Weights:
    HARD_BLOCK = 1e6
    HAZARD = 5.0
    ENEMY_REACH_SAME_OR_LONGER = 30.0
    ENEMY_REACH_SMALLER = 8.0
    ENEMY_HEAD_RING = 4.0
    WALL_HUG = 1.0
    CORNER = 1.5
    CORRIDOR_SQUEEZE = 3.5


def future_head_reach(board: Board, head: Coord) -> Set[Coord]:
    return {n for n in neighbors(board, head)}


def is_corner(board: Board, c: Coord) -> bool:
    x, y = c
    return (x in (0, board.width - 1)) and (y in (0, board.height - 1))


def corridor_sides(board: Board, c: Coord, blocked: Set[Coord]) -> int:
    # Count blocked/out-of-bounds orthogonal neighbors — high means corridor
    count = 0
    for d in DIRS.values():
        n = add(c, d)
        if (not board.in_bounds(n)) or (n in blocked):
            count += 1
    return count


def danger_map(board: Board) -> Dict[Coord, float]:
    danger: Dict[Coord, float] = {}
    # We allow stepping on tails if they are likely to move next turn
    occ = occupied_squares(board, include_tails=False)

    for x in range(board.width):
        for y in range(board.height):
            c = (x, y)
            if c in occ:
                danger[c] = Weights.HARD_BLOCK
            else:
                danger[c] = 0.0

    for c in board.hazards:
        danger[c] = danger.get(c, 0.0) + Weights.HAZARD

    # Wall/corner soft bias
    for x in range(board.width):
        for y in range(board.height):
            c = (x, y)
            if x in (0, board.width - 1) or y in (0, board.height - 1):
                danger[c] += Weights.WALL_HUG
            if is_corner(board, c):
                danger[c] += Weights.CORNER

    # Enemy head proximity and next-reach tiles
    if board.opponent:
        ereach = future_head_reach(board, board.opponent.head)
        scale = Weights.ENEMY_REACH_SAME_OR_LONGER if board.opponent.length >= board.you.length else Weights.ENEMY_REACH_SMALLER
        for c in ereach:
            danger[c] = danger.get(c, 0.0) + scale
        for n in neighbors(board, board.opponent.head):
            danger[n] = danger.get(n, 0.0) + Weights.ENEMY_HEAD_RING

    return danger

# -------------------------------
# Flood fill & escape tests
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


def has_escape(board: Board, start: Coord, blocked: Set[Coord]) -> bool:
    """Cheap check: from start, can we reach our (future) tail OR have area >= length?"""
    area = flood_area(board, start, blocked)
    if area >= board.you.length:
        return True
    # Allow stepping on our own tail if it moves (simulate that next turn tail pops)
    tail = board.you.tail
    if tail not in blocked:
        # already free
        if _reachable(board, start, tail, blocked):
            return True
    else:
        # assume tail vacates next turn
        blocked2 = set(blocked)
        blocked2.discard(tail)
        if _reachable(board, start, tail, blocked2):
            return True
    return False


def _reachable(board: Board, a: Coord, b: Coord, blocked: Set[Coord]) -> bool:
    q = deque([a])
    seen = {a}
    while q:
        c = q.popleft()
        if c == b:
            return True
        for n in neighbors(board, c):
            if n in seen or n in blocked:
                continue
            seen.add(n)
            q.append(n)
    return False

# -------------------------------
# A* pathfinding (heapq)
# -------------------------------

def astar(board: Board, start: Coord, goals: Set[Coord], danger: Dict[Coord, float]) -> Optional[List[Coord]]:
    if not goals:
        return None

    def heuristic(c: Coord) -> float:
        dmin = min(manhattan(c, g) for g in goals)
        cx = board.width / 2.0
        cy = board.height / 2.0
        center = abs(c[0] - cx) + abs(c[1] - cy)
        return dmin + 0.05 * center

    open_heap: List[Tuple[float, Coord]] = []
    heapq.heappush(open_heap, (heuristic(start), start))
    came_from: Dict[Coord, Coord] = {}
    g: Dict[Coord, float] = {start: 0.0}
    closed: Set[Coord] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current in goals:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)
        for n in neighbors(board, current):
            # hard blocks are already in high danger; skip to prune
            if danger.get(n, 0.0) >= Weights.HARD_BLOCK:
                continue
            tentative_g = g[current] + 1.0 + danger.get(n, 0.0)
            if tentative_g < g.get(n, math.inf):
                came_from[n] = current
                g[n] = tentative_g
                f = tentative_g + heuristic(n)
                heapq.heappush(open_heap, (f, n))
    return None

# -------------------------------
# Enemy lookahead (1-ply worst-case)
# -------------------------------

def enemy_best_reply_penalty(board: Board, my_next: Coord, blocked: Set[Coord]) -> float:
    """Estimate extra risk if opponent chooses an adversarial move next.
    Penalize if they can move adjacent/head-on to us while >= length.
    """
    if not board.opponent:
        return 0.0
    my_len = board.you.length
    opp_len = board.opponent.length
    worst = 0.0

    # Possible enemy next positions (avoid hard walls)
    ereach = [n for n in neighbors(board, board.opponent.head) if n not in blocked]
    for epos in ereach:
        # If enemy reaches our my_next simultaneously and is >= length -> huge penalty
        if epos == my_next and opp_len >= my_len:
            worst = max(worst, 60.0)
        # If enemy becomes adjacent to our head after our move, add some risk
        if manhattan(epos, my_next) == 1 and opp_len >= my_len:
            worst = max(worst, 25.0)
    return worst

# -------------------------------
# Move scoring / policy
# -------------------------------

def choose_move(board: Board, seed: Optional[int] = None) -> str:
    # Deterministic randomness for tie-breaking
    rnd = random.Random(seed)

    occ_with_tails = occupied_squares(board, include_tails=True)
    occ_no_tails = occupied_squares(board, include_tails=False)
    danger = danger_map(board)

    head = board.you.head

    candidates: List[Tuple[str, Coord]] = []
    for dname, delta in DIRS.items():
        nxt = add(head, delta)
        if board.in_bounds(nxt):
            candidates.append((dname, nxt))

    # Filter immediate death
    safe: List[Tuple[str, Coord]] = []
    for dname, nxt in candidates:
        if danger.get(nxt, 0.0) >= Weights.HARD_BLOCK:
            continue
        safe.append((dname, nxt))

    if not safe:
        # All bad — pick least terrible (prefer open squares)
        least = min(candidates, key=lambda kv: (danger.get(kv[1], 1e9), -len([n for n in neighbors(board, kv[1]) if n not in occ_with_tails])))
        return least[0]

    # Health-driven goal selection with reachable food filter
    need_food = board.you.health <= 35 or (board.opponent and board.you.length <= board.opponent.length - 2)

    # Build blocked map for reachability calcs (allow tails to move next turn)
    blocked_for_paths = set(occ_no_tails)

    goals: Set[Coord] = set()
    if need_food and board.food:
        good_food = reachable_food(board, blocked_for_paths)
        if good_food:
            goals = good_food
    if not goals:
        # Chase center & enemy tail (safer than head)
        if board.opponent:
            goals.add(board.opponent.tail)
        cx, cy = board.width // 2, board.height // 2
        center_targets = [(cx, cy), (max(0, cx - 1), cy), (cx, max(0, cy - 1))]
        goals.update(c for c in center_targets if board.in_bounds(c))

    # Score each candidate
    scored: List[Tuple[float, str]] = []

    for dname, nxt in safe:
        # Corridor squeeze penalty
        squeeze = corridor_sides(board, nxt, occ_with_tails)
        corridor_pen = Weights.CORRIDOR_SQUEEZE if squeeze >= 3 else 0.0

        # Soft bias away from stepping into our neck
        neck_pen = 0.0
        if len(board.you.body) >= 2 and nxt == board.you.body[1]:
            neck_pen = 8.0

        # Temporarily mark our next as occupied for realism
        temp_block = set(occ_no_tails)
        temp_block.add(nxt)

        # Path from NEXT to goals
        path = astar(board, nxt, goals, danger)
        if path:
            risk_along = sum(danger.get(c, 0.0) for c in path[1:])
            dist = len(path) - 1
        else:
            risk_along = 80.0
            dist = 20

        # Area/escape checks
        area = flood_area(board, nxt, occ_with_tails)
        area_pen = 0.0
        if area < max(5, board.you.length // 2):
            area_pen += 50.0
        elif area < board.you.length:
            area_pen += 12.0
        if not has_escape(board, nxt, occ_with_tails):
            area_pen += 25.0

        # Enemy lookahead penalty
        reply_pen = enemy_best_reply_penalty(board, nxt, occ_with_tails)

        # Base risk from static danger
        base_risk = danger.get(nxt, 0.0)

        # Compose score
        score = (
            base_risk
            + corridor_pen
            + neck_pen
            + area_pen
            + 0.8 * dist
            + 0.12 * risk_along
            + reply_pen
        )

        # Prefer food tile if we truly need food
        if need_food and nxt in board.food:
            score -= 6.0

        scored.append((score, dname))

    # Tie-break: shuffle tiny jitters for stability
    jittered = [(s + 1e-6 * rnd.random(), m) for s, m in scored]
    jittered.sort()
    best_move = jittered[0][1]
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
        "version": "duel-astar-1.1",
    })

@app.post("/move")
def move():
    data = request.get_json()
    board = parse_board(data)
    # Seed randomness from board state for deterministic but varied tie-breaks
    seed = (
        hash((board.width, board.height, board.you.head, tuple(board.food)))
        ^ (board.opponent.length if board.opponent else 0)
    ) & 0xFFFFFFFF
    move_dir = choose_move(board, seed=seed)
    return jsonify({"move": move_dir, "shout": "A*+ online"})

@app.post("/start")
def start():
    return ("", 200)

@app.post("/end")
def end():
    return ("", 200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
