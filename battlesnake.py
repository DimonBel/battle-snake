"""
Battlesnake Simple A* - Clean Food-Only Algorithm
A straightforward A* implementation focused solely on eating food.

Key features:
- Simple A* pathfinding to nearest food
- Basic safety checks
- Clean, minimal implementation
"""
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq

from flask import Flask, request, jsonify


# ==================== Data Structures ====================

Coord = Tuple[int, int]


@dataclass
class Snake:
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
    width: int
    height: int
    food: Set[Coord]
    hazards: Set[Coord]
    you: Snake
    opponent: Optional[Snake]

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height


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


def get_occupied_squares(board: Board) -> Set[Coord]:
    occupied = set(board.you.body)
    if board.opponent:
        occupied.update(board.opponent.body)
    return occupied


# ==================== A* Pathfinding ====================

def find_path_astar(board: Board, start: Coord, goal: Coord, blocked: Set[Coord]) -> Optional[List[Coord]]:
    """
    Simple A* pathfinding from start to goal.
    Returns path as list of coordinates, or None if no path exists.
    """
    def heuristic(a: Coord, b: Coord) -> float:
        return manhattan_distance(a, b)
    
    open_heap = [(heuristic(start, goal), start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    
    while open_heap:
        _, current = heapq.heappop(open_heap)
        
        if current in closed:
            continue
        
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed.add(current)
        
        for neighbor in get_neighbors(board, current):
            if neighbor in blocked or neighbor in closed:
                continue
            
            tentative_g = g_score[current] + 1.0
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))
    
    return None


# ==================== Move Selection ====================

def select_best_move(board: Board) -> str:
    """Select the best move using A* to nearest food."""
    blocked = get_occupied_squares(board)
    head = board.you.head
    
    # Find nearest food
    nearest_food = None
    nearest_dist = float('inf')
    
    for food in board.food:
        dist = manhattan_distance(head, food)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_food = food
    
    # If no food, just pick a safe direction
    if not nearest_food:
        for direction_name, direction_vec in DIRECTIONS.items():
            next_pos = add_coords(head, direction_vec)
            if board.in_bounds(next_pos) and next_pos not in blocked:
                return direction_name
        return "up"
    
    # Find path to nearest food
    path = find_path_astar(board, head, nearest_food, blocked)
    
    if path and len(path) > 1:
        next_pos = path[1]
        # Calculate direction vector from head to next_pos
        direction_vec = (next_pos[0] - head[0], next_pos[1] - head[1])
        return VECTOR_TO_DIR.get(direction_vec, "up")
    
    # If no path found, pick any safe move
    for direction_name, direction_vec in DIRECTIONS.items():
        next_pos = add_coords(head, direction_vec)
        if board.in_bounds(next_pos) and next_pos not in blocked:
            return direction_name
    
    return "up"


# ==================== Flask Server ====================

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({
        "apiversion": "1",
        "author": "simple-astar",
        "color": "#10b981",
        "head": "safe",
        "tail": "small",
        "version": "1.0",
    })


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    board = parse_board(data)
    
    selected_move = select_best_move(board)
    
    return jsonify({
        "move": selected_move,
        "shout": "Hungry"
    })


@app.route("/start", methods=["POST"])
def start():
    return "", 200


@app.route("/end", methods=["POST"])
def end():
    return "", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)