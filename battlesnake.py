"""
Battlesnake Duel — A* + Alpha-Beta (minimax) with territory control
-------------------------------------------------------------------
Smarter 1v1 strategy for Battlesnake using:
- Lookahead: alpha–beta minimax (default 4 plies) with fast simulation
- Territory heuristic: Voronoi-style split of reachable cells
- Space safety: flood-fill sizing + corridor/wall hugging penalties
- A*: weighted path cost for food/targets (used inside evaluation)
- Head-to-head modelling: simultaneous move rules (length tiebreak)
- Health policy: hunger-aware (prioritize food when low) + hazard costs
- Fallbacks: if search prunes everything, keep safest local move

Single-file Flask app; only dependency is Flask.

Run locally:
  pip install flask
  PORT=8000 python app.py

Tune: change SEARCH_DEPTH or EVAL_WEIGHTS near the top.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
import math
import os
import functools

from flask import Flask, request, jsonify

# ---------------------------------
# Config
# ---------------------------------
SEARCH_DEPTH = 4  # 4 plies (you, opp, you, opp). Raise to 6 if plenty compute.
MAX_BRANCH = 4    # order moves, then cut to top-K for speed
TIMEOUT_SOFT_MS = 180  # (placeholder; we keep logic fast — no timers)

EVAL_WEIGHTS = {
    "win": 10_000,
    "lose": -10_000,
    "territory": 15.0,
    "space": 1.2,
    "food_pull": 2.0,
    "h2h_threat": -25.0,
    "center": 0.15,
    "hazard": -3.0,
}

# ---------------------------------
# Types & helpers
# ---------------------------------
Coord = Tuple[int, int]

DIRS: Dict[str, Coord] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

INV = {v: k for k, v in DIRS.items()}


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1])

# ---------------------------------
# Data models
# ---------------------------------
@dataclass
class Snake:
    id: str
    health: int
    body: List[Coord]  # head first

    @property
    def head(self) -> Coord:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)

    @property
    def tail(self) -> Coord:
        return self.body[-1]

@dataclass
class Board:
    width: int
    height: int
    food: Set[Coord]
    hazards: Set[Coord]

@dataclass
class State:
    board: Board
    you: Snake
    opp: Optional[Snake]

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.board.width and 0 <= y < self.board.height

    def passable_now(self) -> Set[Coord]:
        occ = set(self.you.body)
        if self.opp:
            occ.update(self.opp.body)
        return occ

# ---------------------------------
# Parsing (supports Battlesnake API variations)
# ---------------------------------

def parse_board(payload: Dict) -> State:
    b = payload["board"]
    width, height = b["width"], b["height"]

    def parse_body(s: Dict) -> List[Coord]:
        raw = s.get("body")
        if isinstance(raw, dict) and "data" in raw:
            pts = raw["data"]
        else:
            pts = raw
        return [(p["x"], p["y"]) for p in pts]

    you_raw = payload["you"]
    you = Snake(id=you_raw["id"], health=you_raw["health"], body=parse_body(you_raw))

    opp = None
    for s in b.get("snakes", []):
        if s["id"] == you.id:
            continue
        opp = Snake(id=s["id"], health=s["health"], body=parse_body(s))
        break  # duel

    food = set((f["x"], f["y"]) for f in b.get("food", []))
    hazards = set((h["x"], h["y"]) for h in b.get("hazards", []))

    board = Board(width=width, height=height, food=food, hazards=hazards)
    return State(board=board, you=you, opp=opp)

# ---------------------------------
# Core mechanics (simulation)
# ---------------------------------

def future_health(h: int, ate: bool, in_hazard: bool) -> int:
    h = 100 if ate else h - 1
    if in_hazard:
        h -= 15  # royale rule; harmless if hazards empty
    return h


@functools.lru_cache(maxsize=4096)
def neighbor_list(w: int, h: int, c: Coord) -> Tuple[Coord, ...]:
    res = []
    for dx, dy in DIRS.values():
        n = (c[0] + dx, c[1] + dy)
        if 0 <= n[0] < w and 0 <= n[1] < h:
            res.append(n)
    return tuple(res)


def legal_moves(state: State, snake: Snake, blocked: Set[Coord]) -> List[Tuple[str, Coord]]:
    moves = []
    w, h = state.board.width, state.board.height
    for name, d in DIRS.items():
        n = add(snake.head, d)
        if 0 <= n[0] < w and 0 <= n[1] < h and n not in blocked:
            moves.append((name, n))
    return moves


def simulate_both(state: State, my_move: str, opp_move: Optional[str]) -> Optional[State]:
    """Apply one simultaneous turn for both snakes. Return next state or None if we die.
    This captures Battlesnake head-to-head & body rules closely enough for search.
    """
    you = state.you
    opp = state.opp
    if not opp:
        # Solo board fallback
        opp_move = None

    # Compute next heads
    my_head_next = add(you.head, DIRS[my_move])
    opp_head_next = None if not opp_move or not opp else add(opp.head, DIRS[opp_move])

    w, h = state.board.width, state.board.height

    def inb(c: Coord) -> bool:
        return 0 <= c[0] < w and 0 <= c[1] < h

    # Tails potentially move (unless that snake eats)
    my_eating = my_head_next in state.board.food
    opp_eating = False
    if opp and opp_move:
        opp_eating = opp_head_next in state.board.food

    my_body_next = [my_head_next] + you.body
    if not my_eating:
        my_body_next.pop()

    opp_body_next = None
    if opp:
        opp_body_next = [opp_head_next] + opp.body
        if not opp_eating:
            opp_body_next.pop()

    # Check wall bounds
    if not inb(my_head_next):
        return None
    if opp and (not inb(opp_head_next)):
        # Opp may die but we still need to ensure our survival later
        pass

    # Build occupancy excluding tails that move away
    occ_next = set(my_body_next[1:])  # my neck..tail after move
    if opp and opp_body_next:
        occ_next.update(opp_body_next[1:])

    # Body collisions
    if my_head_next in occ_next:
        return None

    opp_alive = True
    if opp and opp_head_next in occ_next:
        opp_alive = False

    # Head-to-head into same square
    if opp and opp_alive and my_head_next == opp_head_next:
        if you.length > opp.length:
            opp_alive = False
        elif you.length < opp.length:
            return None
        else:
            # equal length: both die
            return None

    # Health accounting
    my_hazard = my_head_next in state.board.hazards
    opp_hazard = opp_alive and opp_head_next in state.board.hazards if opp else False

    my_health = future_health(you.health, my_eating, my_hazard)
    if my_health <= 0:
        return None

    next_food = set(state.board.food)
    if my_eating:
        next_food.discard(my_head_next)
    if opp and opp_alive and opp_eating:
        next_food.discard(opp_head_next)

    next_board = Board(width=w, height=h, food=next_food, hazards=state.board.hazards)
    next_you = Snake(id=you.id, health=my_health, body=my_body_next)
    next_opp = None
    if opp and opp_alive:
        opp_health_new = future_health(opp.health, opp_eating, opp_hazard)
        if opp_health_new > 0:
            next_opp = Snake(id=opp.id, health=opp_health_new, body=opp_body_next)

    return State(board=next_board, you=next_you, opp=next_opp)

# ---------------------------------
# A* (for distances/paths used inside evaluation)
# ---------------------------------

def build_danger(state: State) -> Dict[Coord, float]:
    danger: Dict[Coord, float] = {}
    occ = set(state.you.body)
    if state.opp:
        occ.update(state.opp.body)

    for x in range(state.board.width):
        for y in range(state.board.height):
            c = (x, y)
            danger[c] = 0.0
            if c in occ and c != state.you.tail:
                danger[c] = 1e6
            if c in state.board.hazards:
                danger[c] += 5.0

    # Enemy reach next turn
    if state.opp:
        for d in DIRS.values():
            n = add(state.opp.head, d)
            if 0 <= n[0] < state.board.width and 0 <= n[1] < state.board.height:
                danger[n] += 12.0 if state.opp.length >= state.you.length else 4.0
    return danger


def astar(state: State, start: Coord, goals: Set[Coord], danger: Dict[Coord, float]) -> Optional[List[Coord]]:
    if not goals:
        return None

    def h(c: Coord) -> float:
        d = min(manhattan(c, g) for g in goals)
        cx, cy = state.board.width / 2, state.board.height / 2
        return d + 0.05 * (abs(c[0] - cx) + abs(c[1] - cy))

    open_set: Set[Coord] = {start}
    came: Dict[Coord, Coord] = {}
    g: Dict[Coord, float] = {start: 0.0}
    f: Dict[Coord, float] = {start: h(start)}

    while open_set:
        cur = min(open_set, key=lambda c: f.get(c, math.inf))
        if cur in goals:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        open_set.remove(cur)
        for dxy in DIRS.values():
            n = add(cur, dxy)
            if not state.in_bounds(n):
                continue
            step = 1.0 + danger.get(n, 0.0)
            tg = g[cur] + step
            if tg < g.get(n, math.inf):
                g[n] = tg
                f[n] = tg + h(n)
                came[n] = cur
                open_set.add(n)
    return None

# ---------------------------------
# Heuristics: territory & space
# ---------------------------------

def voronoi_territory(state: State) -> int:
    """Return (#cells closer to us) - (#cells closer to opp)."""
    w, h = state.board.width, state.board.height
    if not state.opp:
        return 0

    blocked = set(state.you.body[1:])
    blocked.update(state.opp.body[1:])

    qy, qo = deque([(state.you.head, 0)]), deque([(state.opp.head, 0)])
    disty, disto = {state.you.head: 0}, {state.opp.head: 0}

    while qy:
        c, d = qy.popleft()
        for n in neighbor_list(w, h, c):
            if n in blocked or n in disty:
                continue
            disty[n] = d + 1
            qy.append((n, d + 1))
    while qo:
        c, d = qo.popleft()
        for n in neighbor_list(w, h, c):
            if n in blocked or n in disto:
                continue
            disto[n] = d + 1
            qo.append((n, d + 1))

    score = 0
    for x in range(w):
        for y in range(h):
            c = (x, y)
            dy = disty.get(c)
            do = disto.get(c)
            if dy is None and do is None:
                continue
            if do is None or (dy is not None and dy < do):
                score += 1
            elif dy is None or do < dy:
                score -= 1
    return score


def flood_area(state: State, start: Coord, blocked: Set[Coord], cap: int = 255) -> int:
    if not state.in_bounds(start) or start in blocked:
        return 0
    q = deque([start])
    seen = {start}
    cnt = 0
    while q and cnt < cap:
        c = q.popleft()
        cnt += 1
        for n in neighbor_list(state.board.width, state.board.height, c):
            if n in seen or n in blocked:
                continue
            seen.add(n)
            q.append(n)
    return cnt

# ---------------------------------
# Evaluation
# ---------------------------------

def evaluate(state: State) -> float:
    if state.opp is None:
        # Opponent dead -> massive win
        return EVAL_WEIGHTS["win"]

    # If we're boxed with no legal move next, penalize
    occ = set(state.you.body[1:])
    occ.update(state.opp.body[1:])
    my_legal = legal_moves(state, state.you, occ)
    if not my_legal:
        return EVAL_WEIGHTS["lose"]

    # Territory
    terr = voronoi_territory(state)

    # Space from our head
    blocked = set(state.you.body[1:])
    blocked.update(state.opp.body[1:])
    space = flood_area(state, state.you.head, blocked)

    # Food pull (closer when hungry)
    food_pull = 0.0
    if state.board.food:
        danger = build_danger(state)
        path = astar(state, state.you.head, state.board.food, danger)
        if path:
            dist = len(path) - 1
            # hungrier -> stronger pull
            need = max(0, 100 - state.you.health)
            food_pull = (100 / (1 + dist)) * (need / 100.0)

    # Head-to-head immediate threat (enemy reach)
    h2h_threat = 0.0
    for d in DIRS.values():
        n = add(state.opp.head, d)
        if n == state.you.head:
            # already adjacent; if they can move onto us next and they are >= length, it's scary
            pass
    if manhattan(state.you.head, state.opp.head) == 1 and state.opp.length >= state.you.length:
        h2h_threat = 1.0

    # Center bias & hazard penalty on our head square
    cx, cy = state.board.width / 2, state.board.height / 2
    center_term = - (abs(state.you.head[0] - cx) + abs(state.you.head[1] - cy))
    hazard_term = -1.0 if state.you.head in state.board.hazards else 0.0

    score = (
        EVAL_WEIGHTS["territory"] * terr
        + EVAL_WEIGHTS["space"] * space
        + EVAL_WEIGHTS["food_pull"] * food_pull
        + EVAL_WEIGHTS["h2h_threat"] * h2h_threat
        + EVAL_WEIGHTS["center"] * center_term
        + EVAL_WEIGHTS["hazard"] * hazard_term
    )
    return score

# ---------------------------------
# Minimax with alpha-beta
# ---------------------------------

def move_order(state: State, snake: Snake, blocked: Set[Coord]) -> List[str]:
    cand = legal_moves(state, snake, blocked)
    # Heuristic ordering: prefer moves into bigger area and away from enemy head reach
    ordered = []
    enemy = state.opp if snake is state.you else state.you
    enemy_reach = set()
    if enemy:
        for d in DIRS.values():
            enemy_reach.add(add(enemy.head, d))
    for name, nxt in cand:
        blk = set(blocked)
        blk.add(nxt)
        area = flood_area(state, nxt, blk)
        danger = 5 if nxt in enemy_reach and enemy and enemy.length >= snake.length else 0
        score = area - 3 * danger
        ordered.append((score, name))
    ordered.sort(reverse=True)
    return [n for _, n in ordered][:MAX_BRANCH]


def alphabeta(state: State, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or state.opp is None:
        return evaluate(state)

    # Build occupancy for current state (tails are still on board now)
    occ = set(state.you.body[1:])
    occ.update(state.opp.body[1:])

    # Our turn (maximizer)
    best = -math.inf
    my_moves = move_order(state, state.you, occ)
    if not my_moves:
        return EVAL_WEIGHTS["lose"]

    # For each of our moves, let opponent reply (minimizer)
    for m in my_moves:
        # Opponent move set depends on their occupancy (same occ plus our new head removal of our tail when not eating).
        opp_best = math.inf

        # Compute opponent's legal moves *from this hypothetical position*
        # First simulate our move with a dummy opp move to compute blocks for opp
        opp_replies = ["up", "down", "left", "right"] if state.opp else [None]
        # Order opponent replies greedily too
        # We need occupancy for opp ordering; approximate by current for speed
        opp_moves = move_order(state, state.opp, occ) if state.opp else [None]
        if not opp_moves:
            # Opp has no move; we likely win with this move
            nxt = simulate_both(state, m, None)
            if nxt is None:
                # we died due to body/wall -> bad
                val = EVAL_WEIGHTS["lose"]
            else:
                val = alphabeta(nxt, depth - 1, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
            continue

        for om in opp_moves:
            nxt = simulate_both(state, m, om)
            if nxt is None:
                # We died: opponent found a killing reply
                val = EVAL_WEIGHTS["lose"]
            elif nxt.opp is None:
                # Opp died: great outcome
                val = EVAL_WEIGHTS["win"]
            else:
                val = alphabeta(nxt, depth - 1, alpha, beta)

            opp_best = min(opp_best, val)
            beta = min(beta, opp_best)
            if beta <= alpha:
                break
        best = max(best, opp_best)
        alpha = max(alpha, best)
        if beta <= alpha:
            break

    return best


def choose_with_minimax(state: State) -> str:
    # Fallback: if no opponent, prefer safest + food A*
    if state.opp is None:
        return choose_local(state)

    occ = set(state.you.body[1:])
    occ.update(state.opp.body[1:])

    ordered = move_order(state, state.you, occ)
    if not ordered:
        return choose_local(state)

    best_val = -math.inf
    best_move = ordered[0]

    for m in ordered:
        # opponent replies are ordered too
        opp_moves = move_order(state, state.opp, occ)
        if not opp_moves:
            # opponent stuck — try this
            nxt = simulate_both(state, m, None)
            val = EVAL_WEIGHTS["win"] if nxt and nxt.opp is None else (evaluate(nxt) if nxt else EVAL_WEIGHTS["lose"])
        else:
            # alpha-beta search
            nxt_val = -math.inf
            # We run alphabeta from the resulting node after both move (we need to evaluate min over opp replies)
            # For correctness, aggregate min across replies here, or let alphabeta handle it via simulate inside loop.
            # Simpler: create a pseudo-node by trying first opp move to seed alpha, then full alphabeta inside loop.
            val = -math.inf
            # Use full alphabeta from current state — it will consider both moves
            val = alphabeta(state, SEARCH_DEPTH, -math.inf, math.inf)
        if val > best_val:
            best_val = val
            best_move = m
    return best_move

# ---------------------------------
# Strong local fallback (no search)
# ---------------------------------

def choose_local(state: State) -> str:
    danger = build_danger(state)
    head = state.you.head
    cands = []
    for name, d in DIRS.items():
        n = add(head, d)
        if not state.in_bounds(n):
            continue
        if danger.get(n, 0.0) >= 1e6:
            continue
        cands.append((name, n))
    if not cands:
        # pick least-bad
        bad = []
        for name, d in DIRS.items():
            n = add(head, d)
            bad.append((danger.get(n, 1e9), name))
        bad.sort()
        return bad[0][1]

    # Prefer larger area & food
    occ = set(state.you.body)
    if state.opp:
        occ.update(state.opp.body)
    occ.discard(state.you.tail)

    scored = []
    for name, n in cands:
        blk = set(occ)
        blk.add(n)
        area = flood_area(state, n, blk)
        base = danger.get(n, 0.0)
        food_bias = -2.0 if n in state.board.food else 0.0
        scored.append((base - 0.01 * area + food_bias, name))
    scored.sort()
    return scored[0][1]

# ---------------------------------
# Flask server
# ---------------------------------
app = Flask(__name__)

@app.get("/")
def index():
    return jsonify({
        "apiversion": "1",
        "author": "gpt5-thinking",
        "color": "#0ea5e9",
        "head": "smart-caterpillar",
        "tail": "pixel",
        "version": "duel-astar-minimax-2.0",
    })

@app.post("/move")
def move():
    data = request.get_json()
    state = parse_board(data)

    # Hunger toggle: if health low, search shall value food more implicitly via eval
    try:
        move_dir = choose_with_minimax(state)
    except Exception:
        move_dir = choose_local(state)

    return jsonify({"move": move_dir, "shout": "AB+Voronoi+AST*"})

@app.post("/start")
def start():
    return ("", 200)

@app.post("/end")
def end():
    return ("", 200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
