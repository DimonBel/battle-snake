"""
Battlesnake A*+ - Refactored Implementation
A clean, modular implementation of A*-based pathfinding for 1v1 Battlesnake duels.

Key features:
- A* pathfinding with danger-weighted nodes
- Multi-source BFS for competitive food reachability
- Future-tail modeling (tails are not permanent obstacles)
- 1-ply enemy lookahead with worst-case safety scoring
- Flood fill escape validation
- Deterministic tie-breaking via seeded RNG
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


# ==================== Configuration ====================

class Config:
    """Centralized configuration for all algorithm weights and constants."""
    
    # Danger weights
    HARD_BLOCK = 1e6
    HAZARD = 5.0
    ENEMY_REACH_SAME_OR_LONGER = 30.0
    ENEMY_REACH_SMALLER = 8.0
    ENEMY_HEAD_RING = 4.0
    WALL_HUG = 1.0
    CORNER = 1.5
    CORRIDOR_SQUEEZE = 3.5
    
    # Safety thresholds
    FLOOD_LIMIT = 250
    HEALTH_THRESHOLD = 35
    LENGTH_ADVANTAGE_THRESHOLD = 2
    
    # Penalties
    HEAD_ON_COLLISION_PENALTY = 60.0
    ADJACENT_THREAT_PENALTY = 25.0
    NECK_PENALTY = 8.0
    
    # Area penalties
    TRAPPED_PENALTY = 50.0
    TIGHT_SPACE_PENALTY = 12.0
    NO_ESCAPE_PENALTY = 25.0
    
    # Scoring weights
    DISTANCE_WEIGHT = 0.8
    RISK_ALONG_WEIGHT = 0.12
    FOOD_BONUS = 6.0
    
    # A* parameters
    CENTER_BIAS = 0.05


# ==================== Data Structures ====================

Coord = Tuple[int, int]


@dataclass
class Snake:
    """Represents a snake on the board."""
    id: str
    health: int
    body: List[Coord]  # head at index 0
    length: int

    @property
    def head(self) -> Coord:
        """Get the head position."""
        return self.body[0]

    @property
    def tail(self) -> Coord:
        """Get the tail position."""
        return self.body[-1]

    @property
    def neck(self) -> Optional[Coord]:
        """Get the neck position (body segment just behind head), if exists."""
        return self.body[1] if len(self.body) > 1 else None


@dataclass
class Board:
    """Represents the game board state."""
    width: int
    height: int
    food: Set[Coord]
    hazards: Set[Coord]
    you: Snake
    opponent: Optional[Snake]

    def in_bounds(self, c: Coord) -> bool:
        """Check if a coordinate is within board bounds."""
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height


# ==================== Utilities ====================

# Direction vectors
DIRECTIONS: Dict[str, Coord] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Reverse mapping from vector to direction name
VECTOR_TO_DIR: Dict[Coord, str] = {v: k for k, v in DIRECTIONS.items()}


def add_coords(a: Coord, b: Coord) -> Coord:
    """Add two coordinate vectors."""
    return (a[0] + b[0], a[1] + b[1])


def manhattan_distance(a: Coord, b: Coord) -> int:
    """Calculate Manhattan distance between two coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(board: Board, c: Coord) -> List[Coord]:
    """Get all valid neighboring coordinates."""
    neighbors = []
    for direction in DIRECTIONS.values():
        neighbor = add_coords(c, direction)
        if board.in_bounds(neighbor):
            neighbors.append(neighbor)
    return neighbors


# ==================== Board Parsing ====================

def _extract_body(raw_body) -> List[Coord]:
    """Extract body coordinates from raw JSON data."""
    if isinstance(raw_body, dict) and isinstance(raw_body.get("data"), list):
        return [(p["x"], p["y"]) for p in raw_body["data"]]
    return [(p["x"], p["y"]) for p in raw_body]


def parse_board(data: Dict) -> Board:
    """Parse game state from JSON data."""
    board_data = data["board"]
    
    # Parse your snake
    you_raw = data["you"]
    you = Snake(
        id=you_raw["id"],
        health=you_raw["health"],
        body=_extract_body(you_raw["body"]),
        length=you_raw["length"],
    )
    
    # Parse opponent (duel mode - first non-you snake)
    opponent = None
    for snake_raw in board_data.get("snakes", []):
        if snake_raw["id"] != you.id:
            opponent = Snake(
                id=snake_raw["id"],
                health=snake_raw["health"],
                body=_extract_body(snake_raw["body"]),
                length=snake_raw["length"],
            )
            break
    
    # Parse food and hazards
    food = set((f["x"], f["y"]) for f in board_data.get("food", []))
    hazards = set((h["x"], h["y"]) for h in board_data.get("hazards", []))
    
    return Board(
        width=board_data["width"],
        height=board_data["height"],
        food=food,
        hazards=hazards,
        you=you,
        opponent=opponent,
    )


def get_occupied_squares(board: Board, include_tails: bool = True) -> Set[Coord]:
    """Get all squares currently occupied by snakes."""
    occupied = set(board.you.body)
    
    if board.opponent:
        occupied.update(board.opponent.body)
    
    if not include_tails:
        occupied.discard(board.you.tail)
        if board.opponent:
            occupied.discard(board.opponent.tail)
    
    return occupied


# ==================== Danger System ====================

def build_danger_map(board: Board) -> Dict[Coord, float]:
    """
    Build a danger map assigning risk scores to each board cell.
    
    Higher values indicate more dangerous positions.
    """
    danger: Dict[Coord, float] = {}
    
    # Initialize danger map with occupied squares as hard blocks
    occupied = get_occupied_squares(board, include_tails=False)
    
    for x in range(board.width):
        for y in range(board.height):
            coord = (x, y)
            if coord in occupied:
                danger[coord] = Config.HARD_BLOCK
            else:
                danger[coord] = 0.0
    
    # Add hazard tile danger
    for coord in board.hazards:
        danger[coord] = danger.get(coord, 0.0) + Config.HAZARD
    
    # Add wall and corner biases
    for x in range(board.width):
        for y in range(board.height):
            coord = (x, y)
            
            # Wall bias (edge tiles)
            if x == 0 or x == board.width - 1 or y == 0 or y == board.height - 1:
                danger[coord] += Config.WALL_HUG
            
            # Corner bias (highest risk)
            if (x == 0 or x == board.width - 1) and (y == 0 or y == board.height - 1):
                danger[coord] += Config.CORNER
    
    # Add enemy proximity danger
    if board.opponent:
        _add_enemy_danger(board, danger)
    
    return danger


def _add_enemy_danger(board: Board, danger: Dict[Coord, float]) -> None:
    """Add danger based on enemy head position and future reach."""
    opponent = board.opponent
    if not opponent:
        return
    
    # Enemy head ring (tiles adjacent to enemy head)
    for neighbor in get_neighbors(board, opponent.head):
        danger[neighbor] = danger.get(neighbor, 0.0) + Config.ENEMY_HEAD_RING
    
    # Future head reach (tiles enemy can move to next turn)
    reach_scale = (
        Config.ENEMY_REACH_SAME_OR_LONGER 
        if opponent.length >= board.you.length 
        else Config.ENEMY_REACH_SMALLER
    )
    
    for neighbor in get_neighbors(board, opponent.head):
        danger[neighbor] = danger.get(neighbor, 0.0) + reach_scale


def count_blocked_neighbors(board: Board, coord: Coord, blocked: Set[Coord]) -> int:
    """
    Count how many orthogonal neighbors are blocked or out of bounds.
    Used for corridor detection.
    """
    count = 0
    for direction in DIRECTIONS.values():
        neighbor = add_coords(coord, direction)
        if not board.in_bounds(neighbor) or neighbor in blocked:
            count += 1
    return count


def is_corner(board: Board, coord: Coord) -> bool:
    """Check if a coordinate is in a corner of the board."""
    x, y = coord
    return (x in (0, board.width - 1)) and (y in (0, board.height - 1))


# ==================== Pathfinding (A*) ====================

def find_path_astar(
    board: Board,
    start: Coord,
    goals: Set[Coord],
    danger: Dict[Coord, float]
) -> Optional[List[Coord]]:
    """
    Find path from start to any goal using A* algorithm.
    
    Returns the path (list of coordinates) if found, None otherwise.
    """
    if not goals:
        return None
    
    def heuristic(coord: Coord) -> float:
        """Admissible heuristic: min Manhattan distance to any goal + slight center bias."""
        min_dist = min(manhattan_distance(coord, goal) for goal in goals)
        
        # Add slight bias toward board center
        center_x = board.width / 2.0
        center_y = board.height / 2.0
        center_dist = abs(coord[0] - center_x) + abs(coord[1] - center_y)
        
        return min_dist + Config.CENTER_BIAS * center_dist
    
    # Initialize open set with start node
    open_heap: List[Tuple[float, Coord]] = [(heuristic(start), start)]
    
    # Track path and costs
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    closed: Set[Coord] = set()
    
    while open_heap:
        _, current = heapq.heappop(open_heap)
        
        # Skip if already processed
        if current in closed:
            continue
        
        # Check if we reached a goal
        if current in goals:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed.add(current)
        
        # Explore neighbors
        for neighbor in get_neighbors(board, current):
            # Skip hard blocks
            if danger.get(neighbor, 0.0) >= Config.HARD_BLOCK:
                continue
            
            # Calculate tentative g-score (cost from start to neighbor through current)
            movement_cost = 1.0 + danger.get(neighbor, 0.0)
            tentative_g = g_score[current] + movement_cost
            
            # If this path is better, update
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, neighbor))
    
    return None


# ==================== Food Analysis ====================

def multi_source_bfs(
    board: Board,
    starts: Iterable[Coord],
    blocked: Set[Coord]
) -> Dict[Coord, int]:
    """
    Run BFS from multiple start positions simultaneously.
    
    Returns a dictionary mapping each reachable coordinate to its distance.
    """
    queue = deque()
    distances: Dict[Coord, int] = {}
    
    # Initialize queue with all start positions
    for start in starts:
        if start not in blocked and board.in_bounds(start):
            distances[start] = 0
            queue.append(start)
    
    # BFS traversal
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        
        for neighbor in get_neighbors(board, current):
            if neighbor in blocked or neighbor in distances:
                continue
            distances[neighbor] = current_dist + 1
            queue.append(neighbor)
    
    return distances


def get_reachable_food(board: Board, blocked: Set[Coord]) -> Set[Coord]:
    """
    Find food that our snake can reach before or simultaneously with opponent.
    
    Uses multi-source BFS to compute distances from both heads and filters
    food based on competitive reachability.
    """
    if not board.food:
        return set()
    
    # Compute distances from our head
    my_distances = multi_source_bfs(board, [board.you.head], blocked)
    
    # Compute distances from opponent head (if exists)
    if board.opponent:
        opponent_distances = multi_source_bfs(board, [board.opponent.head], blocked)
    else:
        opponent_distances = {}
    
    # Filter food based on competitive reachability
    reachable = set()
    for food_coord in board.food:
        my_dist = my_distances.get(food_coord, math.inf)
        opponent_dist = opponent_distances.get(food_coord, math.inf)
        
        # Food is claimable if:
        # 1. We arrive strictly sooner, OR
        # 2. We arrive simultaneously and are longer (win head-to-head)
        if my_dist < opponent_dist:
            reachable.add(food_coord)
        elif my_dist == opponent_dist and board.opponent and board.you.length > board.opponent.length:
            reachable.add(food_coord)
    
    return reachable


# ==================== Safety Analysis ====================

def flood_fill_area(
    board: Board,
    start: Coord,
    blocked: Set[Coord],
    limit: int = Config.FLOOD_LIMIT
) -> int:
    """
    Count reachable tiles from start position using BFS.
    
    Used to evaluate available space and detect trapped positions.
    """
    if not board.in_bounds(start) or start in blocked:
        return 0
    
    queue = deque([start])
    visited = {start}
    count = 0
    
    while queue and count < limit:
        current = queue.popleft()
        count += 1
        
        for neighbor in get_neighbors(board, current):
            if neighbor in visited or neighbor in blocked:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    
    return count


def has_escape_path(
    board: Board,
    start: Coord,
    blocked: Set[Coord]
) -> bool:
    """
    Check if a position is viable by verifying escape conditions.
    
    A position is viable if:
    1. Area >= snake length (enough space to survive), OR
    2. Can reach own tail position (assuming tail will move)
    """
    area = flood_fill_area(board, start, blocked)
    
    # Condition 1: Enough space
    if area >= board.you.length:
        return True
    
    # Condition 2: Can reach tail
    tail = board.you.tail
    
    if tail not in blocked:
        # Tail is already free
        return _bfs_reachable(board, start, tail, blocked)
    else:
        # Assume tail vacates next turn
        blocked_without_tail = set(blocked)
        blocked_without_tail.discard(tail)
        return _bfs_reachable(board, start, tail, blocked_without_tail)


def _bfs_reachable(
    board: Board,
    start: Coord,
    target: Coord,
    blocked: Set[Coord]
) -> bool:
    """Check if target is reachable from start using BFS."""
    if start == target:
        return True
    
    queue = deque([start])
    visited = {start}
    
    while queue:
        current = queue.popleft()
        
        if current == target:
            return True
        
        for neighbor in get_neighbors(board, current):
            if neighbor in visited or neighbor in blocked:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    
    return False


# ==================== Enemy Analysis ====================

def calculate_enemy_reply_penalty(
    board: Board,
    my_next_position: Coord,
    blocked: Set[Coord]
) -> float:
    """
    Estimate extra risk if opponent plays optimally next turn.
    
    Penalizes positions where opponent can:
    1. Reach the same tile simultaneously (head-on collision)
    2. Move adjacent to our position with equal or greater length
    """
    if not board.opponent:
        return 0.0
    
    my_length = board.you.length
    opponent_length = board.opponent.length
    worst_penalty = 0.0
    
    # Get opponent's possible next positions
    opponent_moves = [
        neighbor for neighbor in get_neighbors(board, board.opponent.head)
        if neighbor not in blocked
    ]
    
    for opponent_next in opponent_moves:
        # Head-on collision: opponent reaches same tile with >= length
        if opponent_next == my_next_position and opponent_length >= my_length:
            worst_penalty = max(worst_penalty, Config.HEAD_ON_COLLISION_PENALTY)
        
        # Adjacent threat: opponent becomes adjacent with >= length
        if manhattan_distance(opponent_next, my_next_position) == 1 and opponent_length >= my_length:
            worst_penalty = max(worst_penalty, Config.ADJACENT_THREAT_PENALTY)
    
    return worst_penalty


# ==================== Decision Making ====================

def select_move(board: Board, seed: Optional[int] = None) -> str:
    """
    Select the best move based on comprehensive safety and strategy analysis.
    """
    # Initialize deterministic random for tie-breaking
    rng = random.Random(seed)
    
    # Precompute board analysis
    occupied_with_tails = get_occupied_squares(board, include_tails=True)
    occupied_without_tails = get_occupied_squares(board, include_tails=False)
    danger = build_danger_map(board)
    
    head = board.you.head
    
    # Generate all possible moves
    candidates = []
    for direction_name, direction_vec in DIRECTIONS.items():
        next_pos = add_coords(head, direction_vec)
        if board.in_bounds(next_pos):
            candidates.append((direction_name, next_pos))
    
    # Filter out immediate death (hard blocks)
    safe_moves = []
    for direction_name, next_pos in candidates:
        if danger.get(next_pos, 0.0) < Config.HARD_BLOCK:
            safe_moves.append((direction_name, next_pos))
    
    # If no safe moves, pick least terrible option
    if not safe_moves:
        return _select_least_dangerous(board, candidates, occupied_with_tails, danger)
    
    # Determine goals based on game state
    goals = _select_goals(board, occupied_without_tails)
    
    # Score each safe move
    scored_moves = []
    for direction_name, next_pos in safe_moves:
        score = _evaluate_move(
            board, next_pos, goals, danger,
            occupied_with_tails, occupied_without_tails
        )
        scored_moves.append((score, direction_name))
    
    # Tie-breaking with small random jitter
    jittered_moves = [
        (score + 1e-6 * rng.random(), direction)
        for score, direction in scored_moves
    ]
    jittered_moves.sort()
    
    return jittered_moves[0][1]


def _select_goals(board: Board, blocked: Set[Coord]) -> Set[Coord]:
    """Select appropriate target goals based on game state."""
    goals = set()
    
    # Determine if we need food
    need_food = (
        board.you.health <= Config.HEALTH_THRESHOLD or
        (board.opponent and board.you.length <= board.opponent.length - Config.LENGTH_ADVANTAGE_THRESHOLD)
    )
    
    # If we need food, target reachable food
    if need_food and board.food:
        reachable = get_reachable_food(board, blocked)
        if reachable:
            goals = reachable
    
    # If no food goals, target strategic positions
    if not goals:
        # Target enemy tail (safer than head)
        if board.opponent:
            goals.add(board.opponent.tail)
        
        # Target board center area
        center_x = board.width // 2
        center_y = board.height // 2
        center_candidates = [
            (center_x, center_y),
            (max(0, center_x - 1), center_y),
            (center_x, max(0, center_y - 1))
        ]
        goals.update(c for c in center_candidates if board.in_bounds(c))
    
    return goals


def _evaluate_move(
    board: Board,
    next_pos: Coord,
    goals: Set[Coord],
    danger: Dict[Coord, float],
    occupied_with_tails: Set[Coord],
    occupied_without_tails: Set[Coord]
) -> float:
    """
    Evaluate a move by computing a comprehensive safety score.
    
    Lower score indicates a better move.
    """
    score = 0.0
    
    # 1. Base danger from static danger map
    base_danger = danger.get(next_pos, 0.0)
    score += base_danger
    
    # 2. Corridor squeeze penalty
    blocked_neighbors = count_blocked_neighbors(board, next_pos, occupied_with_tails)
    if blocked_neighbors >= 3:
        score += Config.CORRIDOR_SQUEEZE
    
    # 3. Neck penalty (avoid stepping into own neck)
    if board.you.neck and next_pos == board.you.neck:
        score += Config.NECK_PENALTY
    
    # 4. Path evaluation to goals
    # Temporarily mark next position as occupied for realistic pathfinding
    temp_blocked = set(occupied_without_tails)
    temp_blocked.add(next_pos)
    
    path = find_path_astar(board, next_pos, goals, danger)
    
    if path:
        # Calculate risk along the path
        path_risk = sum(danger.get(coord, 0.0) for coord in path[1:])
        path_length = len(path) - 1
    else:
        # No path found - high penalty
        path_risk = 80.0
        path_length = 20
    
    score += Config.DISTANCE_WEIGHT * path_length
    score += Config.RISK_ALONG_WEIGHT * path_risk
    
    # 5. Area and escape penalties
    area = flood_fill_area(board, next_pos, occupied_with_tails)
    
    # Penalize trapped positions
    min_area = max(5, board.you.length // 2)
    if area < min_area:
        score += Config.TRAPPED_PENALTY
    elif area < board.you.length:
        score += Config.TIGHT_SPACE_PENALTY
    
    # Penalize positions without escape path
    if not has_escape_path(board, next_pos, occupied_with_tails):
        score += Config.NO_ESCAPE_PENALTY
    
    # 6. Enemy lookahead penalty
    reply_penalty = calculate_enemy_reply_penalty(board, next_pos, occupied_with_tails)
    score += reply_penalty
    
    # 7. Food bonus (if we need food)
    need_food = (
        board.you.health <= Config.HEALTH_THRESHOLD or
        (board.opponent and board.you.length <= board.opponent.length - Config.LENGTH_ADVANTAGE_THRESHOLD)
    )
    if need_food and next_pos in board.food:
        score -= Config.FOOD_BONUS
    
    return score


def _select_least_dangerous(
    board: Board,
    candidates: List[Tuple[str, Coord]],
    occupied: Set[Coord],
    danger: Dict[Coord, float]
) -> str:
    """Select the least dangerous move when no safe moves exist."""
    def sort_key(item):
        direction_name, pos = item
        pos_danger = danger.get(pos, 1e9)
        # Prefer positions with more open neighbors
        open_neighbors = len([n for n in get_neighbors(board, pos) if n not in occupied])
        return (pos_danger, -open_neighbors)
    
    return min(candidates, key=sort_key)[0]


# ==================== Flask Server ====================

app = Flask(__name__)


@app.route("/")
def index():
    """Handle index endpoint - return snake configuration."""
    return jsonify({
        "apiversion": "1",
        "author": "refactored-astar",
        "color": "#4f46e5",
        "head": "smart-caterpillar",
        "tail": "pixel",
        "version": "refactored-1.0",
    })


@app.route("/move", methods=["POST"])
def move():
    """Handle move endpoint - select and return next move."""
    data = request.get_json()
    board = parse_board(data)
    
    # Generate seed for deterministic tie-breaking
    seed = (
        hash((
            board.width,
            board.height,
            board.you.head,
            tuple(sorted(board.food))
        )) ^ (board.opponent.length if board.opponent else 0)
    ) & 0xFFFFFFFF
    
    selected_move = select_move(board, seed=seed)
    
    return jsonify({
        "move": selected_move,
        "shout": "A*+ refactored"
    })


@app.route("/start", methods=["POST"])
def start():
    """Handle start endpoint - game initialization."""
    return "", 200


@app.route("/end", methods=["POST"])
def end():
    """Handle end endpoint - game cleanup."""
    return "", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)