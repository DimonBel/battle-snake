"""
Battlesnake Smart Eater - Advanced Food Gathering Algorithm
A highly optimized algorithm focused on intelligent, fast food consumption.

Key features:
- Greedy multi-food path planning
- Predictive competitive food selection
- Health-aware urgency scaling
- Optimized A* with early termination
- Smart food clustering analysis
- Competitive edge calculation
"""
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
    body: List[Coord]
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
    DANGER_HAZARD = 40
    DANGER_WALL = 8
    DANGER_ENEMY_HEAD = 70
    
    # Food selection weights
    FOOD_DISTANCE_WEIGHT = 1.2
    FOOD_SAFETY_WEIGHT = 1.5
    FOOD_CLUSTER_WEIGHT = 2.0
    FOOD_COMPETITIVE_WIN = -30
    FOOD_COMPETITIVE_LOSE = 200
    
    # Health urgency multipliers
    HEALTH_CRITICAL = 18
    HEALTH_LOW = 35
    URGENCY_CRITICAL = 3.0
    URGENCY_LOW = 1.5
    
    # Move evaluation
    PATH_LENGTH_WEIGHT = 1.5
    PATH_DANGER_WEIGHT = 0.4
    ESCAPE_SPACE_WEIGHT = 0.8
    NECK_PENALTY = 60
    
    # A* optimization
    ASTAR_EARLY_EXIT_THRESHOLD = 1.2  # Accept path if within 20% of best


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
    
    you_raw = data["you"]
    you = Snake(
        id=you_raw["id"],
        health=you_raw["health"],
        body=[(p["x"], p["y"]) for p in you_raw["body"]],
        length=you_raw["length"],
    )
    
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


# ==================== Health Urgency ====================

def get_urgency_multiplier(board: Board) -> float:
    """Calculate urgency multiplier based on health."""
    health = board.you.health
    
    if health <= Config.HEALTH_CRITICAL:
        return Config.URGENCY_CRITICAL
    elif health <= Config.HEALTH_LOW:
        return Config.URGENCY_LOW
    else:
        return 1.0


# ==================== Danger Assessment ====================

def build_danger_map(board: Board) -> Dict[Coord, float]:
    """Build a danger map with risk scores for each cell."""
    danger: Dict[Coord, float] = {}
    
    for x in range(board.width):
        for y in range(board.height):
            danger[(x, y)] = 0.0
    
    occupied = get_occupied_squares(board, include_tails=False)
    for coord in occupied:
        danger[coord] = Config.DANGER_SNAKE
    
    for coord in board.hazards:
        danger[coord] = max(danger.get(coord, 0), Config.DANGER_HAZARD)
    
    # Wall danger (edges)
    for x in range(board.width):
        danger[(x, 0)] += Config.DANGER_WALL
        danger[(x, board.height - 1)] += Config.DANGER_WALL
    
    for y in range(board.height):
        danger[(0, y)] += Config.DANGER_WALL
        danger[(board.width - 1, y)] += Config.DANGER_WALL
    
    # Enemy head danger
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


# ==================== Optimized A* Pathfinding ====================

def find_path_astar_optimized(
    board: Board,
    start: Coord,
    goal: Coord,
    danger: Dict[Coord, float],
    max_iterations: int = 500
) -> Optional[List[Coord]]:
    """
    Optimized A* with early termination for faster pathfinding.
    """
    if not is_safe(board, goal, danger) and goal not in board.food:
        return None
    
    def heuristic(a: Coord, b: Coord) -> float:
        return manhattan_distance(a, b)
    
    # Pre-calculate heuristic for start
    h_start = heuristic(start, goal)
    open_heap: List[Tuple[float, Coord]] = [(h_start, start)]
    
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    closed: Set[Coord] = set()
    
    best_f_score = h_start
    iterations = 0
    
    while open_heap and iterations < max_iterations:
        iterations += 1
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
        
        for neighbor in get_neighbors(board, current):
            if neighbor in closed:
                continue
            
            # Calculate movement cost
            move_cost = 1.0 + (danger.get(neighbor, 0) / 100.0)
            tentative_g = g_score[current] + move_cost
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                
                # Early exit: found a good enough path
                if f_score <= best_f_score * Config.ASTAR_EARLY_EXIT_THRESHOLD:
                    if neighbor == goal:
                        path = [neighbor]
                        temp = neighbor
                        while temp in came_from:
                            temp = came_from[temp]
                            path.append(temp)
                        path.reverse()
                        return path
                
                heapq.heappush(open_heap, (f_score, neighbor))
                best_f_score = min(best_f_score, f_score)
    
    return None


# ==================== Smart Food Selection ====================

def bfs_distances_fast(board: Board, start: Coord, blocked: Set[Coord], max_dist: int = 20) -> Dict[Coord, int]:
    """Fast BFS with distance limit for performance."""
    distances: Dict[Coord, int] = {}
    queue = deque([(start, 0)])
    distances[start] = 0
    
    while queue:
        current, dist = queue.popleft()
        
        if dist >= max_dist:
            continue
        
        for neighbor in get_neighbors(board, current):
            if neighbor in distances or neighbor in blocked:
                continue
            distances[neighbor] = dist + 1
            queue.append((neighbor, dist + 1))
    
    return distances


def count_nearby_food(board: Board, food: Coord, all_food: Set[Coord], radius: int = 3) -> int:
    """Count how many food items are near this food (for clustering)."""
    count = 0
    for other in all_food:
        if other != food and manhattan_distance(food, other) <= radius:
            count += 1
    return count


def select_smart_food(board: Board, danger: Dict[Coord, float]) -> Optional[Coord]:
    """
    Select the best food using smart competitive analysis.
    """
    if not board.food:
        return None
    
    occupied = get_occupied_squares(board, include_tails=False)
    
    # Calculate distances from our head
    my_distances = bfs_distances_fast(board, board.you.head, occupied)
    
    # Calculate distances from opponent
    if board.opponent:
        opponent_distances = bfs_distances_fast(board, board.opponent.head, occupied)
    else:
        opponent_distances = {}
    
    urgency = get_urgency_multiplier(board)
    
    best_food = None
    best_score = float('inf')
    
    for food in board.food:
        my_dist = my_distances.get(food, float('inf'))
        
        if my_dist == float('inf'):
            continue  # Can't reach
        
        # Base score: distance (weighted by urgency)
        score = my_dist * Config.FOOD_DISTANCE_WEIGHT * urgency
        
        # Safety: danger at food location
        food_danger = danger.get(food, 0)
        score += food_danger * Config.FOOD_SAFETY_WEIGHT
        
        # Competitive analysis
        if board.opponent:
            opp_dist = opponent_distances.get(food, float('inf'))
            
            if opp_dist < my_dist:
                # Opponent gets there first - heavily penalize
                score += Config.FOOD_COMPETITIVE_LOSE
            elif opp_dist == my_dist:
                # Tie - check length
                if board.opponent.length >= board.you.length:
                    score += Config.FOOD_COMPETITIVE_LOSE * 0.5
                else:
                    score += Config.FOOD_COMPETITIVE_WIN * 0.5
            else:
                # We get there first - bonus
                score += Config.FOOD_COMPETITIVE_WIN
        
        # Cluster bonus: prefer food near other food
        cluster_count = count_nearby_food(board, food, board.food)
        score -= cluster_count * Config.FOOD_CLUSTER_WEIGHT
        
        # Health urgency: closer food gets bigger bonus when hungry
        if urgency > 1.0:
            score -= (10 - min(my_dist, 10)) * urgency
        
        if score < best_score:
            best_score = score
            best_food = food
    
    return best_food


# ==================== Escape Analysis ====================

def count_escape_space_fast(board: Board, start: Coord, blocked: Set[Coord], limit: int = 80) -> int:
    """Fast escape space counting with limit."""
    if not board.in_bounds(start) or start in blocked:
        return 0
    
    visited = {start}
    queue = deque([start])
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


def can_reach_tail_fast(board: Board, start: Coord, blocked: Set[Coord]) -> bool:
    """Fast tail reachability check."""
    tail = board.you.tail
    
    blocked_without_tail = set(blocked)
    blocked_without_tail.discard(tail)
    
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

def evaluate_move_smart(
    board: Board,
    next_pos: Coord,
    danger: Dict[Coord, float],
    occupied: Set[Coord]
) -> float:
    """Evaluate a move with smart food-focused scoring."""
    score = 0.0
    
    # Base danger
    score += danger.get(next_pos, 0)
    
    # Avoid neck
    if board.you.neck and next_pos == board.you.neck:
        score += Config.NECK_PENALTY
    
    # Path to best food
    best_food = select_smart_food(board, danger)
    urgency = get_urgency_multiplier(board)
    
    if best_food:
        path = find_path_astar_optimized(board, next_pos, best_food, danger)
        
        if path:
            path_length = len(path) - 1
            path_danger = sum(danger.get(coord, 0) for coord in path[1:])
            
            # Weight path evaluation by urgency
            score += path_length * Config.PATH_LENGTH_WEIGHT * urgency
            score += path_danger * Config.PATH_DANGER_WEIGHT
        else:
            # No path - big penalty when hungry
            score += 150 * urgency
    
    # Escape space (more important when we have food to eat)
    escape_space = count_escape_space_fast(board, next_pos, occupied)
    min_space = max(5, board.you.length // 2)
    if escape_space < min_space:
        score += (min_space - escape_space) * 10
    
    # Tail reachability
    if not can_reach_tail_fast(board, next_pos, occupied):
        score += 40
    
    # Enemy collision awareness
    if board.opponent:
        for neighbor in get_neighbors(board, board.opponent.head):
            if neighbor == next_pos:
                if board.opponent.length >= board.you.length:
                    score += 250  # Very dangerous
                else:
                    score -= 15  # We win this
    
    return score


def select_best_move(board: Board) -> str:
    """Select the best move with smart food-focused decision making."""
    danger = build_danger_map(board)
    occupied = get_occupied_squares(board, include_tails=True)
    head = board.you.head
    
    best_move = None
    best_score = float('inf')
    
    for direction_name, direction_vec in DIRECTIONS.items():
        next_pos = add_coords(head, direction_vec)
        
        if not is_safe(board, next_pos, danger):
            continue
        
        score = evaluate_move_smart(board, next_pos, danger, occupied)
        
        if score < best_score:
            best_score = score
            best_move = direction_name
    
    # Fallback: any safe move
    if best_move is None:
        for direction_name, direction_vec in DIRECTIONS.items():
            next_pos = add_coords(head, direction_vec)
            if board.in_bounds(next_pos) and danger.get(next_pos, 0) < Config.DANGER_SNAKE:
                best_move = direction_name
                break
    
    # Last resort
    if best_move is None:
        best_move = "up"
    
    return best_move


# ==================== Flask Server ====================

app = Flask(__name__)


@app.route("/")
def index():
    """Handle index endpoint - return snake configuration."""
    return jsonify({
        "apiversion": "1",
        "author": "smart-eater",
        "color": "#f59e0b",  # Amber for food-focused
        "head": "beluga",
        "tail": "round",
        "version": "2.0",
    })


@app.route("/move", methods=["POST"])
def move():
    """Handle move endpoint - select and return next move."""
    data = request.get_json()
    board = parse_board(data)
    
    selected_move = select_best_move(board)
    
    urgency = get_urgency_multiplier(board)
    if urgency >= Config.URGENCY_CRITICAL:
        shout = "MUST EAT!"
    elif urgency >= Config.URGENCY_LOW:
        shout = "Hungry..."
    else:
        shout = "Yum"
    
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