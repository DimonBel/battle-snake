import random
import typing


def info() -> typing.Dict:
    """
    This function is called when you create your Battlesnake on play.battlesnake.com
    It should return a personalized response for your snake.
    """
    return {
        "apiversion": "1",
        "author": "Dumas",
        "color": "#2B06066E",
        "head": "default",
        "tail": "default",
    }


def start(game_state: typing.Dict):
    """
    This function is called every time your snake enters a game.
    game_state is a dictionary containing information about the game.
    """
    print("GAME START")


def end(game_state: typing.Dict):
    """
    This function is called when your snake dies or the game ends.
    """
    print("GAME OVER")


def move(game_state: typing.Dict) -> typing.Dict:
    """
    This function is called on every turn and returns your move.
    Valid moves are "up", "down", "left", "right".
    """
    # Get game information
    my_snake = game_state["you"]
    my_head = my_snake["head"]
    my_body = my_snake["body"]
    my_health = my_snake["health"]
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    food = game_state["board"]["food"]
    snakes = game_state["board"]["snakes"]

    # Calculate safe moves
    safe_moves = get_safe_moves(my_head, my_body, board_width, board_height, snakes)

    if not safe_moves:
        print("No safe moves available!")
        return {"move": "up"}

    # Priority 1: Avoid immediate death
    if len(safe_moves) == 1:
        return {"move": safe_moves[0]}

    # Priority 2: Go for food if health is low
    if my_health < 30 and food:
        food_move = move_towards_food(my_head, food, safe_moves)
        if food_move:
            return {"move": food_move}

    # Priority 3: Control center and avoid other snakes
    best_move = choose_best_move(
        my_head, my_body, safe_moves, board_width, board_height, snakes, food
    )

    return {"move": best_move}


def get_safe_moves(head, body, board_width, board_height, snakes):
    """
    Calculate moves that won't immediately kill the snake
    """
    possible_moves = ["up", "down", "left", "right"]
    safe_moves = []

    for move in possible_moves:
        new_head = get_new_head_position(head, move)

        # Check if move is safe
        if is_safe_position(new_head, body, board_width, board_height, snakes):
            safe_moves.append(move)

    return safe_moves


def get_new_head_position(head, move):
    """
    Calculate the new head position for a given move
    """
    x, y = head["x"], head["y"]

    if move == "up":
        return {"x": x, "y": y + 1}
    elif move == "down":
        return {"x": x, "y": y - 1}
    elif move == "left":
        return {"x": x - 1, "y": y}
    elif move == "right":
        return {"x": x + 1, "y": y}


def is_safe_position(pos, my_body, board_width, board_height, snakes):
    """
    Check if a position is safe to move to
    """
    x, y = pos["x"], pos["y"]

    # Check board boundaries
    if x < 0 or x >= board_width or y < 0 or y >= board_height:
        return False

    # Check collision with own body (excluding tail since it moves)
    for i, body_part in enumerate(my_body[:-1]):  # Exclude tail
        if x == body_part["x"] and y == body_part["y"]:
            return False

    # Check collision with other snakes
    for snake in snakes:
        for body_part in snake["body"]:
            if x == body_part["x"] and y == body_part["y"]:
                return False

    return True


def move_towards_food(head, food, safe_moves):
    """
    Find the best move towards the nearest food
    """
    if not food or not safe_moves:
        return None

    # Find nearest food
    nearest_food = min(food, key=lambda f: manhattan_distance(head, f))

    best_move = None
    min_distance = float("inf")

    for move in safe_moves:
        new_head = get_new_head_position(head, move)
        distance = manhattan_distance(new_head, nearest_food)

        if distance < min_distance:
            min_distance = distance
            best_move = move

    return best_move


def manhattan_distance(pos1, pos2):
    """
    Calculate Manhattan distance between two positions
    """
    return abs(pos1["x"] - pos2["x"]) + abs(pos1["y"] - pos2["y"])


def choose_best_move(head, body, safe_moves, board_width, board_height, snakes, food):
    """
    Choose the best move based on multiple factors
    """
    if len(safe_moves) == 1:
        return safe_moves[0]

    move_scores = {}

    for move in safe_moves:
        new_head = get_new_head_position(head, move)
        score = 0

        # Prefer moves towards center
        center_x, center_y = board_width // 2, board_height // 2
        center_distance = manhattan_distance(new_head, {"x": center_x, "y": center_y})
        score -= center_distance * 0.1

        # Prefer moves with more space (flood fill)
        space_available = count_reachable_spaces(
            new_head, body, board_width, board_height, snakes
        )
        score += space_available * 2

        # Avoid moves towards larger snakes' heads
        for snake in snakes:
            if snake["id"] != body[0] and len(snake["body"]) >= len(
                body
            ):  # Larger or equal snake
                snake_head = snake["head"]
                if manhattan_distance(new_head, snake_head) <= 2:
                    score -= 50  # Heavy penalty for getting close to larger snakes

        # Slight preference for food if we're not in danger
        if food and len(body) < board_width * board_height // 4:  # Not too long yet
            nearest_food = min(food, key=lambda f: manhattan_distance(new_head, f))
            food_distance = manhattan_distance(new_head, nearest_food)
            score -= food_distance * 0.5

        move_scores[move] = score

    # Return move with highest score
    best_move = max(move_scores, key=move_scores.get)
    return best_move


def count_reachable_spaces(start_pos, my_body, board_width, board_height, snakes):
    """
    Count how many spaces are reachable from a given position using flood fill
    """
    visited = set()
    queue = [start_pos]
    count = 0

    while queue and count < 100:  # Limit search to avoid timeout
        pos = queue.pop(0)
        pos_key = f"{pos['x']},{pos['y']}"

        if pos_key in visited:
            continue

        visited.add(pos_key)
        count += 1

        # Check all adjacent positions
        for move in ["up", "down", "left", "right"]:
            new_pos = get_new_head_position(pos, move)
            new_key = f"{new_pos['x']},{new_pos['y']}"

            if new_key not in visited and is_safe_position(
                new_pos, my_body, board_width, board_height, snakes
            ):
                queue.append(new_pos)

    return count


# Server setup (Flask application)
from flask import Flask, request, jsonify
import os

app = Flask(__name__)


@app.route("/")
def handle_info():
    return jsonify(info())


@app.route("/start", methods=["POST"])
def handle_start():
    game_state = request.get_json()
    start(game_state)
    return "ok"


@app.route("/move", methods=["POST"])
def handle_move():
    game_state = request.get_json()
    return jsonify(move(game_state))


@app.route("/end", methods=["POST"])
def handle_end():
    game_state = request.get_json()
    end(game_state)
    return "ok"


@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)