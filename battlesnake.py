"""
BattleSnake AI Bot inspired by geometric decision tree algorithms
Based on strategic spatial reasoning and efficient search strategies
"""

from typing import Dict, List, Tuple, Set
import random


class BattleSnakeBot:
    """
    BattleSnake bot using decision tree-inspired strategies
    Implements spatial analysis and strategic movement planning
    """
    
    def __init__(self):
        self.directions = ["up", "down", "left", "right"]
        self.move_vectors = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }
    
    def get_move(self, game_state: Dict) -> str:
        """
        Main decision function - returns the best move
        """
        board = game_state["board"]
        my_snake = game_state["you"]
        my_head = my_snake["head"]
        my_body = my_snake["body"]
        health = my_snake["health"]
        
        # Build occupancy map (inspired by battleship grid analysis)
        occupied = self._build_occupied_set(board, my_snake)
        
        # Calculate safe moves using decision tree approach
        safe_moves = self._get_safe_moves(my_head, occupied, board)
        
        if not safe_moves:
            # Last resort - try any move
            return random.choice(self.directions)
        
        # Strategic move selection (inspired by minimizing "misses")
        
        # Priority 1: If low health, aggressively seek food
        if health < 30:
            food_move = self._move_toward_closest_food(
                my_head, board["food"], safe_moves, occupied, board
            )
            if food_move:
                return food_move
        
        # Priority 2: Control space (maximize reachable area)
        space_scores = self._evaluate_space_control(
            my_head, safe_moves, occupied, board
        )
        
        # Priority 3: Avoid head-to-head with larger snakes
        space_scores = self._avoid_larger_snakes(
            my_head, safe_moves, space_scores, board, my_snake
        )
        
        # Priority 4: Seek food opportunistically
        if health < 70:
            food_bonus = self._calculate_food_bonus(
                my_head, board["food"], safe_moves
            )
            for move in safe_moves:
                space_scores[move] += food_bonus.get(move, 0)
        
        # Select move with highest score
        best_move = max(safe_moves, key=lambda m: space_scores[m])
        return best_move
    
    def _build_occupied_set(self, board: Dict, my_snake: Dict) -> Set[Tuple[int, int]]:
        """
        Build set of occupied cells (inspired by shape detection in battleship)
        """
        occupied = set()
        
        # Add all snake bodies
        for snake in board["snakes"]:
            for segment in snake["body"][:-1]:  # Exclude tail (it moves)
                occupied.add((segment["x"], segment["y"]))
        
        return occupied
    
    def _get_safe_moves(
        self, head: Dict, occupied: Set[Tuple[int, int]], board: Dict
    ) -> List[str]:
        """
        Get moves that don't immediately result in death
        Uses decision tree logic to prune unsafe branches
        """
        safe = []
        width = board["width"]
        height = board["height"]
        
        for direction in self.directions:
            dx, dy = self.move_vectors[direction]
            new_x = head["x"] + dx
            new_y = head["y"] + dy
            
            # Check bounds
            if not (0 <= new_x < width and 0 <= new_y < height):
                continue
            
            # Check collision
            if (new_x, new_y) in occupied:
                continue
            
            safe.append(direction)
        
        return safe
    
    def _move_toward_closest_food(
        self,
        head: Dict,
        food: List[Dict],
        safe_moves: List[str],
        occupied: Set[Tuple[int, int]],
        board: Dict
    ) -> str:
        """
        Move toward closest food using Manhattan distance
        Inspired by directional search in battleship algorithms
        """
        if not food:
            return None
        
        # Find closest food
        min_dist = float('inf')
        target = None
        for f in food:
            dist = abs(f["x"] - head["x"]) + abs(f["y"] - head["y"])
            if dist < min_dist:
                min_dist = dist
                target = f
        
        if not target:
            return None
        
        # Score moves by distance to target
        best_move = None
        best_score = float('inf')
        
        for move in safe_moves:
            dx, dy = self.move_vectors[move]
            new_x = head["x"] + dx
            new_y = head["y"] + dy
            
            # Check if path is somewhat clear
            if not self._has_escape_path((new_x, new_y), occupied, board):
                continue
            
            dist = abs(target["x"] - new_x) + abs(target["y"] - new_y)
            if dist < best_score:
                best_score = dist
                best_move = move
        
        return best_move
    
    def _evaluate_space_control(
        self,
        head: Dict,
        safe_moves: List[str],
        occupied: Set[Tuple[int, int]],
        board: Dict
    ) -> Dict[str, float]:
        """
        Evaluate reachable space for each move using flood fill
        Inspired by area coverage in battleship strategies
        """
        scores = {}
        
        for move in safe_moves:
            dx, dy = self.move_vectors[move]
            new_pos = (head["x"] + dx, head["y"] + dy)
            
            # Flood fill to count reachable squares
            reachable = self._flood_fill(new_pos, occupied, board, max_depth=10)
            scores[move] = len(reachable)
        
        return scores
    
    def _flood_fill(
        self,
        start: Tuple[int, int],
        occupied: Set[Tuple[int, int]],
        board: Dict,
        max_depth: int = 10
    ) -> Set[Tuple[int, int]]:
        """
        Flood fill to count reachable area (limited depth for performance)
        """
        width = board["width"]
        height = board["height"]
        visited = {start}
        queue = [(start, 0)]
        idx = 0
        
        while idx < len(queue) and len(visited) < 100:  # Limit for performance
            (x, y), depth = queue[idx]
            idx += 1
            
            if depth >= max_depth:
                continue
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in occupied or (nx, ny) in visited:
                    continue
                
                visited.add((nx, ny))
                queue.append(((nx, ny), depth + 1))
        
        return visited
    
    def _has_escape_path(
        self,
        pos: Tuple[int, int],
        occupied: Set[Tuple[int, int]],
        board: Dict
    ) -> bool:
        """
        Check if position has reasonable escape routes
        """
        reachable = self._flood_fill(pos, occupied, board, max_depth=5)
        return len(reachable) >= 4  # Need some room to maneuver
    
    def _avoid_larger_snakes(
        self,
        head: Dict,
        safe_moves: List[str],
        scores: Dict[str, float],
        board: Dict,
        my_snake: Dict
    ) -> Dict[str, float]:
        """
        Penalize moves that could lead to head-to-head with larger snakes
        """
        my_length = len(my_snake["body"])
        
        for move in safe_moves:
            dx, dy = self.move_vectors[move]
            new_x = head["x"] + dx
            new_y = head["y"] + dy
            
            # Check adjacent cells for enemy snake heads
            for snake in board["snakes"]:
                if snake["id"] == my_snake["id"]:
                    continue
                
                enemy_head = snake["head"]
                enemy_length = len(snake["body"])
                
                # Check if enemy could move to adjacent cell
                for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if (enemy_head["x"] + edx == new_x and 
                        enemy_head["y"] + edy == new_y):
                        
                        if enemy_length >= my_length:
                            # Heavily penalize head-to-head with larger/equal snake
                            scores[move] -= 100
        
        return scores
    
    def _calculate_food_bonus(
        self, head: Dict, food: List[Dict], safe_moves: List[str]
    ) -> Dict[str, float]:
        """
        Calculate bonus for moves toward food
        """
        bonus = {}
        if not food:
            return bonus
        
        for move in safe_moves:
            dx, dy = self.move_vectors[move]
            new_x = head["x"] + dx
            new_y = head["y"] + dy
            
            # Find closest food to new position
            min_dist = float('inf')
            for f in food:
                dist = abs(f["x"] - new_x) + abs(f["y"] - new_y)
                min_dist = min(min_dist, dist)
            
            # Bonus inversely proportional to distance
            if min_dist > 0:
                bonus[move] = 20.0 / min_dist
            else:
                bonus[move] = 100  # Food is on this square!
        
        return bonus


# BattleSnake API endpoints
def info() -> Dict:
    """
    Return snake appearance and metadata
    """
    return {
        "apiversion": "1",
        "author": "Strategic-AI",
        "color": "#00FF00",
        "head": "default",
        "tail": "default",
    }


def start(game_state: Dict):
    """
    Called when a new game starts
    """
    print(f"Game {game_state['game']['id']} started!")


def move(game_state: Dict) -> Dict:
    """
    Called on each turn - return the move decision
    """
    bot = BattleSnakeBot()
    chosen_move = bot.get_move(game_state)
    
    print(f"Turn {game_state['turn']}: Moving {chosen_move}")
    
    return {
        "move": chosen_move,
        "shout": "Strategizing!"
    }


def end(game_state: Dict):
    """
    Called when the game ends
    """
    print(f"Game {game_state['game']['id']} ended!")


# Flask server setup - app at module level for deployment
from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
def handle_info():
    return info()

@app.route("/start", methods=["POST"])
def handle_start():
    game_state = request.get_json()
    start(game_state)
    return "ok"

@app.route("/move", methods=["POST"])
def handle_move():
    game_state = request.get_json()
    return move(game_state)

@app.route("/end", methods=["POST"])
def handle_end():
    game_state = request.get_json()
    end(game_state)
    return "ok"

# For local development
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)