
import time
import math
import json
from flask import Flask, request, jsonify
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import deque
import copy

app = Flask(__name__)

class GameState:
    """Represents the complete state of the Battlesnake game"""

    def __init__(self, board_data: Dict, our_snake_id: str, turn: int):
        self.board = board_data
        self.width = board_data['width']
        self.height = board_data['height']
        self.food = [(f['x'], f['y']) for f in board_data['food']]
        self.hazards = [(h['x'], h['y']) for h in board_data.get('hazards', [])]
        self.turn = turn

        # Parse snakes
        self.snakes = {}
        self.our_snake_id = our_snake_id
        self.alive_snakes = []

        for snake in board_data['snakes']:
            snake_id = snake['id']
            snake_info = {
                'id': snake_id,
                'body': [(s['x'], s['y']) for s in snake['body']],
                'head': (snake['head']['x'], snake['head']['y']),
                'length': snake['length'],
                'health': snake['health']
            }

            self.snakes[snake_id] = snake_info
            self.alive_snakes.append(snake_id)

    def copy(self):
        """Create a deep copy of the game state"""
        new_state = copy.deepcopy(self)
        return new_state

    def is_valid_position(self, pos: Tuple[int, int], ignore_tails: bool = True) -> bool:
        """Check if position is valid (within bounds and not occupied)"""
        x, y = pos

        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        # Check snake bodies
        for snake_id in self.alive_snakes:
            snake = self.snakes[snake_id]
            body_to_check = snake['body'][:-1] if ignore_tails and len(snake['body']) > 1 else snake['body']
            if pos in body_to_check:
                return False

        return True

    def get_valid_moves(self, snake_id: str) -> List[str]:
        """Get all valid moves for a snake"""
        if snake_id not in self.alive_snakes:
            return []

        snake = self.snakes[snake_id]
        head_x, head_y = snake['head']
        valid_moves = []

        directions = [
            ("up", (0, 1)),
            ("down", (0, -1)),
            ("left", (-1, 0)),
            ("right", (1, 0))
        ]

        for move, (dx, dy) in directions:
            new_pos = (head_x + dx, head_y + dy)
            if self.is_valid_position(new_pos):
                valid_moves.append(move)

        return valid_moves if valid_moves else ["up"]  # Fallback to prevent crash

    def apply_moves(self, moves: Dict[str, str]) -> 'GameState':
        """Apply moves to create new game state"""
        new_state = self.copy()

        # Move each snake
        for snake_id in list(new_state.alive_snakes):
            if snake_id not in moves:
                continue

            snake = new_state.snakes[snake_id]
            move = moves[snake_id]

            # Calculate new head position
            head_x, head_y = snake['head']
            direction_map = {
                "up": (0, 1),
                "down": (0, -1),
                "left": (-1, 0),
                "right": (1, 0)
            }

            if move in direction_map:
                dx, dy = direction_map[move]
                new_head = (head_x + dx, head_y + dy)

                # Check if move is valid
                if not new_state.is_valid_position(new_head, ignore_tails=True):
                    # Snake dies
                    new_state.alive_snakes.remove(snake_id)
                    continue

                # Move snake
                new_body = [new_head] + snake['body']

                # Check if snake ate food
                ate_food = new_head in new_state.food
                if ate_food:
                    new_state.food.remove(new_head)
                    snake['health'] = 100
                else:
                    # Remove tail if didn't eat
                    new_body = new_body[:-1]
                    snake['health'] -= 1

                    # Snake dies if health reaches 0
                    if snake['health'] <= 0:
                        new_state.alive_snakes.remove(snake_id)
                        continue

                # Update snake
                snake['body'] = new_body
                snake['head'] = new_head
                snake['length'] = len(new_body)

        # Check for head-to-head collisions
        head_positions = {}
        for snake_id in new_state.alive_snakes:
            head = new_state.snakes[snake_id]['head']
            if head in head_positions:
                # Collision! Remove shorter snake(s)
                colliding_snakes = head_positions[head] + [snake_id]
                max_length = max(new_state.snakes[sid]['length'] for sid in colliding_snakes)

                for sid in colliding_snakes:
                    if new_state.snakes[sid]['length'] < max_length:
                        if sid in new_state.alive_snakes:
                            new_state.alive_snakes.remove(sid)
            else:
                head_positions[head] = [snake_id]

        return new_state

class IDAPOSBattlesnakeAI:
    """IDAPOS Algorithm Implementation for Battlesnake"""

    def __init__(self):
        self.directions = ["up", "down", "left", "right"]
        self.transposition_table = {}

    def select_played_out(self, state: GameState, depth: int) -> List[str]:
        """
        Select which snakes to play out in the search.
        Returns list of snake IDs to consider in search.
        """
        # Always include our snake
        played_out = [state.our_snake_id]

        # Add closest and most threatening enemies
        our_snake = state.snakes[state.our_snake_id]
        our_head = our_snake['head']
        our_length = our_snake['length']

        # Calculate threat scores for each enemy
        threats = []
        for snake_id in state.alive_snakes:
            if snake_id == state.our_snake_id:
                continue

            enemy = state.snakes[snake_id]
            enemy_head = enemy['head']
            enemy_length = enemy['length']

            # Distance threat
            distance = abs(our_head[0] - enemy_head[0]) + abs(our_head[1] - enemy_head[1])
            distance_threat = max(0, 10 - distance)

            # Size threat
            size_threat = max(0, enemy_length - our_length) * 2

            # Health threat (low health enemies are less threatening)
            health_threat = enemy['health'] / 100.0

            total_threat = distance_threat + size_threat + health_threat
            threats.append((snake_id, total_threat))

        # Sort by threat level
        threats.sort(key=lambda x: x[1], reverse=True)

        # Add most threatening enemies based on depth budget
        max_enemies = min(len(threats), max(1, 4 - depth // 2))
        for snake_id, _ in threats[:max_enemies]:
            played_out.append(snake_id)

        return played_out

    def mask(self, state: GameState, played_out: List[str]) -> GameState:
        """
        Create masked state with only selected snakes.
        Non-selected snakes are treated as static obstacles.
        """
        masked_state = state.copy()

        # Remove non-selected snakes from alive list but keep their bodies as obstacles
        masked_state.alive_snakes = [sid for sid in masked_state.alive_snakes if sid in played_out]

        return masked_state

    def evaluate_position(self, state: GameState) -> float:
        """
        Evaluate the current position from our snake's perspective.
        Higher values are better for us.
        """
        if state.our_snake_id not in state.alive_snakes:
            return -10000  # We died

        our_snake = state.snakes[state.our_snake_id]
        our_head = our_snake['head']
        our_length = our_snake['length']
        our_health = our_snake['health']

        score = 0

        # Survival bonus
        score += 1000

        # Length advantage
        enemy_lengths = [state.snakes[sid]['length'] for sid in state.alive_snakes if sid != state.our_snake_id]
        if enemy_lengths:
            avg_enemy_length = sum(enemy_lengths) / len(enemy_lengths)
            score += (our_length - avg_enemy_length) * 100

        # Health score
        score += our_health * 2

        # Space control (flood fill)
        controlled_space = self.flood_fill(our_head, state)
        score += len(controlled_space) * 5

        # Food accessibility
        if state.food:
            min_food_distance = min(abs(our_head[0] - fx) + abs(our_head[1] - fy)
                                  for fx, fy in state.food)
            score -= min_food_distance * 10

        # Center control bonus
        center_x, center_y = state.width // 2, state.height // 2
        center_distance = abs(our_head[0] - center_x) + abs(our_head[1] - center_y)
        score -= center_distance * 2

        # Penalty for being close to larger enemies
        for snake_id in state.alive_snakes:
            if snake_id == state.our_snake_id:
                continue

            enemy = state.snakes[snake_id]
            if enemy['length'] > our_length:
                enemy_head = enemy['head']
                distance = abs(our_head[0] - enemy_head[0]) + abs(our_head[1] - enemy_head[1])
                if distance <= 3:
                    score -= (4 - distance) * 50

        return score

    def flood_fill(self, start: Tuple[int, int], state: GameState, max_depth: int = 50) -> Set[Tuple[int, int]]:
        """Flood fill to calculate controlled space"""
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue and len(visited) < max_depth:
            x, y = queue.popleft()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (x + dx, y + dy)
                if new_pos not in visited and state.is_valid_position(new_pos):
                    visited.add(new_pos)
                    queue.append(new_pos)

        return visited

    def alpha_beta_search(self, state: GameState, depth: int, alpha: float = -math.inf,
                         beta: float = math.inf, maximizing: bool = True) -> Tuple[float, Optional[str]]:
        """
        Alpha-Beta search for 2-player scenarios.
        """
        if depth == 0 or len(state.alive_snakes) <= 1:
            return self.evaluate_position(state), None

        if maximizing:
            max_eval = -math.inf
            best_move = None

            our_moves = state.get_valid_moves(state.our_snake_id)

            for our_move in our_moves:
                # Find best enemy response
                enemy_id = [sid for sid in state.alive_snakes if sid != state.our_snake_id][0]
                enemy_moves = state.get_valid_moves(enemy_id)

                worst_eval = math.inf
                for enemy_move in enemy_moves:
                    moves = {state.our_snake_id: our_move, enemy_id: enemy_move}
                    new_state = state.apply_moves(moves)

                    eval_score, _ = self.alpha_beta_search(new_state, depth - 1, alpha, beta, False)
                    worst_eval = min(worst_eval, eval_score)

                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break

                if worst_eval > max_eval:
                    max_eval = worst_eval
                    best_move = our_move

                alpha = max(alpha, worst_eval)
                if beta <= alpha:
                    break

            return max_eval, best_move

        else:
            min_eval = math.inf

            enemy_id = [sid for sid in state.alive_snakes if sid != state.our_snake_id][0]
            enemy_moves = state.get_valid_moves(enemy_id)

            for enemy_move in enemy_moves:
                our_moves = state.get_valid_moves(state.our_snake_id)

                best_eval = -math.inf
                for our_move in our_moves:
                    moves = {state.our_snake_id: our_move, enemy_id: enemy_move}
                    new_state = state.apply_moves(moves)

                    eval_score, _ = self.alpha_beta_search(new_state, depth - 1, alpha, beta, True)
                    best_eval = max(best_eval, eval_score)

                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break

                min_eval = min(min_eval, best_eval)
                beta = min(beta, best_eval)
                if beta <= alpha:
                    break

            return min_eval, None

    def max_n_search(self, state: GameState, depth: int) -> Tuple[float, Optional[str]]:
        """
        Max-N search for multi-player scenarios (>2 players).
        """
        if depth == 0 or len(state.alive_snakes) <= 1:
            return self.evaluate_position(state), None

        best_move = None
        best_score = -math.inf

        our_moves = state.get_valid_moves(state.our_snake_id)

        for our_move in our_moves:
            # Generate all possible enemy move combinations
            enemy_snakes = [sid for sid in state.alive_snakes if sid != state.our_snake_id]
            enemy_move_combinations = self.generate_move_combinations(state, enemy_snakes)

            # Assume enemies play optimally for themselves (worst case for us)
            worst_score = math.inf

            for enemy_moves in enemy_move_combinations:
                moves = {state.our_snake_id: our_move}
                moves.update(enemy_moves)

                new_state = state.apply_moves(moves)
                score, _ = self.max_n_search(new_state, depth - 1)
                worst_score = min(worst_score, score)

            if worst_score > best_score:
                best_score = worst_score
                best_move = our_move

        return best_score, best_move

    def generate_move_combinations(self, state: GameState, snake_ids: List[str],
                                 max_combinations: int = 16) -> List[Dict[str, str]]:
        """Generate combinations of moves for multiple snakes (limited for performance)"""
        if not snake_ids:
            return [{}]

        combinations = []

        def generate_recursive(remaining_snakes: List[str], current_combo: Dict[str, str]):
            if len(combinations) >= max_combinations:
                return

            if not remaining_snakes:
                combinations.append(current_combo.copy())
                return

            snake_id = remaining_snakes[0]
            valid_moves = state.get_valid_moves(snake_id)

            for move in valid_moves[:2]:  # Limit to 2 best moves per snake for performance
                current_combo[snake_id] = move
                generate_recursive(remaining_snakes[1:], current_combo)
                del current_combo[snake_id]

        generate_recursive(snake_ids, {})
        return combinations if combinations else [{}]

    def idapos(self, state: GameState, time_cutoff: float) -> str:
        """
        Main IDAPOS algorithm implementation.
        """
        start_time = time.time()
        depth = 1
        best_move = "up"  # Fallback

        while (time.time() - start_time) < time_cutoff:
            try:
                # Select snakes to play out at current depth
                played_out = self.select_played_out(state, depth)

                # Create masked state
                masked_state = self.mask(state, played_out)

                # Choose search algorithm based on number of players
                if len(played_out) == 2:
                    # Use Alpha-Beta for 2-player scenario
                    score, move = self.alpha_beta_search(masked_state, depth)
                else:
                    # Use Max-N for multi-player scenario
                    score, move = self.max_n_search(masked_state, depth)

                if move:
                    best_move = move

                print(f"Depth {depth}: Best move = {best_move}, Score = {score:.2f}")

                depth += 1

                # Limit maximum depth to prevent infinite loops
                if depth > 8:
                    break

            except Exception as e:
                print(f"Error at depth {depth}: {e}")
                break

        elapsed_time = (time.time() - start_time) * 1000
        print(f"IDAPOS completed in {elapsed_time:.1f}ms, final depth: {depth-1}")

        return best_move

# Global AI instance
ai = IDAPOSBattlesnakeAI()

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "apiversion": "1",
        "author": "IDAPOS Algorithm",
        "color": "#2B06066E",
        "head": "default",
        "tail": "default",
        "version": "1.0"
    })

@app.route('/start', methods=['POST'])
def start():
    data = request.get_json()
    ai.transposition_table.clear()
    print(f"🧠 IDAPOS Game {data['game']['id']} started!")
    return jsonify({})

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()

    try:
        # Parse game state
        game_state = GameState(data['board'], data['you']['id'], data['turn'])

        # Use IDAPOS with time limit (400ms to be safe)
        chosen_move = ai.idapos(game_state, time_cutoff=0.4)

        print(f"Turn {data['turn']}: IDAPOS chose {chosen_move}")

        return jsonify({
            "move": chosen_move,
            "shout": "IDAPOS!"
        })

    except Exception as e:
        print(f"Error: {e}")
        # Fallback: choose any valid move
        try:
            game_state = GameState(data['board'], data['you']['id'], data['turn'])
            valid_moves = game_state.get_valid_moves(data['you']['id'])
            fallback_move = valid_moves[0] if valid_moves else "up"
            return jsonify({
                "move": fallback_move,
                "shout": "Fallback!"
            })
        except:
            return jsonify({
                "move": "up",
                "shout": "Emergency!"
            })

@app.route('/end', methods=['POST'])
def end():
    data = request.get_json()
    print(f"Game {data['game']['id']} ended!")
    return jsonify({})

if __name__ == '__main__':
    print("🧠 Starting IDAPOS Battlesnake AI...")
    print("📊 Features: Iterative Deepening, Alpha-Beta Pruning, Max-N Search")
    print("⚡ Adaptive opponent selection and time-bounded search")
    print("🎯 Server running on http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
