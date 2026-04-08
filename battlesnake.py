from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
import heapq

from flask import Flask, request, jsonify


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
        return self.body[0]

    @property
    def tail(self) -> Coord:
        return self.body[-1]

    @property
    def neck(self) -> Optional[Coord]:
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
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height


# ==================== Configuration ====================

class Config:
    # Danger scores
    DANGER_SNAKE = 1000
    DANGER_HAZARD = 50
    DANGER_WALL = 10
    DANGER_ENEMY_HEAD = 80
    
    # Food priorities
    FOOD_DISTANCE_WEIGHT = 1.0
    FOOD_SAFETY_WEIGHT = 2.0
    FOOD_COMPETITIVE_WEIGHT = 1.5
    
    # Health thresholds
    HEALTH_CRITICAL = 20
    HEALTH_LOW = 40
    
    # Move evaluation
    PATH_LENGTH_WEIGHT = 1.0
    PATH_DANGER_WEIGHT = 0.3
    ESCAPE_SPACE_WEIGHT = 0.5


# ==================== Utilities ====================

DIRECTIONS: Dict[str, Coord] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

VECTOR_TO_DIR: Dict[Coord, str] = {v: k for k, v in DIRECTIONS.items()}


def add_coords(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1])


def manhattan_distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(board: Board, c: Coord) -> List[Coord]:
    neighbors = []
    for direction in DIRECTIONS.values():
        neighbor = add_coords(c, direction)
        if board.in_bounds(neighbor):
            neighbors.append(neighbor)
    return neighbors


# ==================== Board Parsing ====================

def parse_board(data: Dict) -> Board:
    """Parse game state from JSON data."""
    board_data = data["board"]
    
    # Parse your snake
    you_raw = data["you"]
    you = Snake(
        id=you_raw["id"],
        health=you_raw["health"],
        body=[(p["x"], p["y"]) for p in you_raw["body"]],
        length=you_raw["length"],
    )
    
    # Parse opponent (first non-you snake)
    opponent = None
    for snake_raw in board_data.get("snakes", []):
        if snake_raw["id"] != you.id:
            opponent = Snake(
                id=snake_raw["id"],
                health=snake_raw["health"],
                body=[(p["x"], p["y"]) for p in snake_raw["body"]],
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


# ==================== Danger Assessment ====================

def build_danger_map(board: Board) -> Dict[Coord, float]:
    """Build a danger map with risk scores for each cell."""
    danger: Dict[Coord, float] = {}
    
    # Initialize with zero danger
    for x in range(board.width):
        for y in range(board.height):
            danger[(x, y)] = 0.0
    
    # Add danger from occupied squares
    occupied = get_occupied_squares(board, include_tails=False)
    for coord in occupied:
        danger[coord] = Config.DANGER_SNAKE
    
    # Add danger from hazards
    for coord in board.hazards:
        danger[coord] = max(danger.get(coord, 0), Config.DANGER_HAZARD)
    
    # Add wall danger (edges are slightly dangerous)
    for x in range(board.width):
        danger[(x, 0)] += Config.DANGER_WALL
        danger[(x, board.height - 1)] += Config.DANGER_WALL
    
    for y in range(board.height):
        danger[(0, y)] += Config.DANGER_WALL
        danger[(board.width - 1, y)] += Config.DANGER_WALL
    
    # Add enemy head danger
    if board.opponent:
        for neighbor in get_neighbors(board, board.opponent.head):
            danger[neighbor] = max(danger.get(neighbor, 0), Config.DANGER_ENEMY_HEAD)
    
    return danger


def is_safe(board: Board, coord: Coord, danger: Dict[Coord, float]) -> bool:
    """Check if a coordinate is safe to move to."""
    if not board.in_bounds(coord):
        return False
    if danger.get(coord, 0) >= Config.DANGER_SNAKE:
        return False
    return True


# ==================== A* Pathfinding ====================

def find_path_astar(
    board: Board,
    start: Coord,
    goal: Coord,
    danger: Dict[Coord, float]
) -> Optional[List[Coord]]:
    """
    Find path from start to goal using A* algorithm.
    Returns the path (list of coordinates) if found, None otherwise.
    """
    if not is_safe(board, goal, danger) and goal not in board.food:
        return None
    
    def heuristic(a: Coord, b: Coord) -> float:
        return manhattan_distance(a, b)
    
    # Priority queue: (f_score, coord)
    open_heap: List[Tuple[float, Coord]] = [(heuristic(start, goal), start)]
    
    # Track costs and path
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    closed: Set[Coord] = set()
    
    while open_heap:
        _, current = heapq.heappop(open_heap)
        
        if current in closed:
            continue
        
        if current == goal:
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
            if neighbor in closed:
                continue
            
            # Calculate movement cost (1 + danger at target)
            move_cost = 1.0 + (danger.get(neighbor, 0) / 100.0)
            tentative_g = g_score[current] + move_cost
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))
    
    return None


# ==================== Food Selection ====================

def bfs_distance(board: Board, start: Coord, blocked: Set[Coord]) -> Dict[Coord, int]:
    """Calculate BFS distances from start to all reachable cells."""
    distances: Dict[Coord, int] = {}
    queue = deque([start])
    distances[start] = 0
    
    while queue:
        current = queue.popleft()
        
        for neighbor in get_neighbors(board, current):
            if neighbor in distances or neighbor in blocked:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    
    return distances


def select_best_food(board: Board, danger: Dict[Coord, float]) -> Optional[Coord]:
    """Select the best food to target based on distance, safety, and competition."""
    if not board.food:
        return None
    
    # Get blocked squares for BFS
    occupied = get_occupied_squares(board, include_tails=False)
    
    # Calculate distances from our head
    my_distances = bfs_distance(board, board.you.head, occupied)
    
    # Calculate distances from opponent head (if exists)
    if board.opponent:
        opponent_distances = bfs_distance(board, board.opponent.head, occupied)
    else:
        opponent_distances = {}
    
    # Score each food
    best_food = None
    best_score = float('inf')
    
    for food in board.food:
        my_dist = my_distances.get(food, float('inf'))
        
        if my_dist == float('inf'):
            continue  # Can't reach this food
        
        # Base score: distance
        score = my_dist * Config.FOOD_DISTANCE_WEIGHT
        
        # Safety bonus: lower danger at food location
        food_danger = danger.get(food, 0)
        score += food_danger * Config.FOOD_SAFETY_WEIGHT
        
        # Competition: penalize if opponent can reach it first or same time when shorter
        if board.opponent:
            opp_dist = opponent_distances.get(food, float('inf'))
            if opp_dist < my_dist:
                score += 100  # Opponent gets there first
            elif opp_dist == my_dist and board.opponent.length >= board.you.length:
                score += 50  # Tie but opponent wins
            else:
                score -= Config.FOOD_COMPETITIVE_WEIGHT  # We can get there first
        
        # Urgency bonus if health is low
        if board.you.health <= Config.HEALTH_LOW:
            score -= 5  # Prioritize food more when hungry
        
        if score < best_score:
            best_score = score
            best_food = food
    
    return best_food


# ==================== Escape Analysis ====================

def count_escape_space(board: Board, start: Coord, blocked: Set[Coord]) -> int:
    """Count reachable space from a position using BFS."""
    if not board.in_bounds(start) or start in blocked:
        return 0
    
    visited = {start}
    queue = deque([start])
    count = 0
    
    while queue and count < 100:  # Limit for performance
        current = queue.popleft()
        count += 1
        
        for neighbor in get_neighbors(board, current):
            if neighbor in visited or neighbor in blocked:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    
    return count


def can_reach_tail(board: Board, start: Coord, blocked: Set[Coord]) -> bool:
    """Check if we can reach our own tail from a position."""
    tail = board.you.tail
    
    # Tail might be free (not blocked)
    if tail not in blocked:
        blocked_without_tail = set(blocked)
    else:
        # Assume tail will move, so it becomes free
        blocked_without_tail = set(blocked)
        blocked_without_tail.discard(tail)
    
    # BFS to tail
    if start == tail:
        return True
    
    visited = {start}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        
        if current == tail:
            return True
        
        for neighbor in get_neighbors(board, current):
            if neighbor in visited or neighbor in blocked_without_tail:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    
    return False


# ==================== Move Evaluation ====================

def evaluate_move(
    board: Board,
    next_pos: Coord,
    danger: Dict[Coord, float],
    occupied: Set[Coord]
) -> float:
    """Evaluate a move and return a score (lower is better)."""
    score = 0.0
    
    # Base danger
    score += danger.get(next_pos, 0)
    
    # Avoid neck
    if board.you.neck and next_pos == board.you.neck:
        score += 50
    
    # Path to food
    best_food = select_best_food(board, danger)
    if best_food:
        path = find_path_astar(board, next_pos, best_food, danger)
        if path:
            # Good: have path
            path_length = len(path) - 1
            path_danger = sum(danger.get(coord, 0) for coord in path[1:])
            
            score += path_length * Config.PATH_LENGTH_WEIGHT
            score += path_danger * Config.PATH_DANGER_WEIGHT
        else:
            # Bad: no path to food
            score += 100
    
    # Escape space
    escape_space = count_escape_space(board, next_pos, occupied)
    score -= escape_space * Config.ESCAPE_SPACE_WEIGHT
    
    # Can reach tail?
    if not can_reach_tail(board, next_pos, occupied):
        score += 30
    
    # Avoid enemy head-on collision
    if board.opponent:
        for neighbor in get_neighbors(board, board.opponent.head):
            if neighbor == next_pos:
                if board.opponent.length >= board.you.length:
                    score += 200  # Dangerous
                else:
                    score -= 20  # We win this
    
    return score


def select_best_move(board: Board) -> str:
    """Select the best move for the current board state."""
    # Build danger map
    danger = build_danger_map(board)
    
    # Get occupied squares
    occupied = get_occupied_squares(board, include_tails=True)
    
    # Get our head
    head = board.you.head
    
    # Evaluate all possible moves
    best_move = None
    best_score = float('inf')
    
    for direction_name, direction_vec in DIRECTIONS.items():
        next_pos = add_coords(head, direction_vec)
        
        # Skip if not safe
        if not is_safe(board, next_pos, danger):
            continue
        
        # Evaluate this move
        score = evaluate_move(board, next_pos, danger, occupied)
        
        if score < best_score:
            best_score = score
            best_move = direction_name
    
    # Fallback: pick any safe move
    if best_move is None:
        for direction_name, direction_vec in DIRECTIONS.items():
            next_pos = add_coords(head, direction_vec)
            if board.in_bounds(next_pos) and danger.get(next_pos, 0) < Config.DANGER_SNAKE:
                best_move = direction_name
                break
    
    # Last resort: pick any valid direction
    if best_move is None:
        best_move = list(DIRECTIONS.keys())[0]
    
    return best_move


# ==================== Flask Server ====================

app = Flask(__name__)


@app.route("/")
def index():
    """Handle index endpoint - return snake configuration."""
    return jsonify({
        "apiversion": "1",
        "author": "balanced-astar",
        "color": "#3b82f6",  # Blue for balanced
        "head": "bento",
        "tail": "hook",
        "version": "1.0",
    })


@app.route("/move", methods=["POST"])
def move():
    """Handle move endpoint - select and return next move."""
    data = request.get_json()
    board = parse_board(data)
    
    selected_move = select_best_move(board)
    
    # Generate shout based on health
    if board.you.health <= Config.HEALTH_CRITICAL:
        shout = "HUNGRY!"
    elif board.you.health <= Config.HEALTH_LOW:
        shout = "Need food..."
    else:
        shout = "Tasty"
    
    return jsonify({
        "move": selected_move,
        "shout": shout
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