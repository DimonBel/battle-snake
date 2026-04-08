from flask import Flask, request, jsonify
import heapq
import collections
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


class BattlesnakeAI:
    """
    A high-performance Battlesnake AI using A* Pathfinding integrated with
    an Influence Map (Analog potential field) and Flood Fill for survival.
    """

    def __init__(self, game_state):
        self.board = game_state["board"]
        self.width = self.board["width"]
        self.height = self.board["height"]
        self.you = game_state["you"]
        self.my_head = (self.you["head"]["x"], self.you["head"]["y"])
        self.my_body = [(p["x"], p["y"]) for p in self.you["body"]]
        self.snakes = self.board["snakes"]
        self.hazards = [(h["x"], h["y"]) for h in self.board["hazards"]]
        self.food = [(f["x"], f["y"]) for f in self.board["food"]]

        # Pre-calculate occupied tiles
        self.occupied = set()
        for snake in self.snakes:
            for part in snake["body"][:-1]:  # Tail might move
                self.occupied.add((part["x"], part["y"]))

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [
            n for n in neighbors if 0 <= n[0] < self.width and 0 <= n[1] < self.height
        ]

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_influence_cost(self, pos):
        """
        The 'Analog' part: Calculates a potential field cost for a tile.
        """
        cost = 1

        # 1. Proximity to other snake heads
        for snake in self.snakes:
            if snake["id"] == self.you["id"]:
                continue

            head = (snake["head"]["x"], snake["head"]["y"])
            dist = self.manhattan(pos, head)

            if dist == 1:
                # Potential head-on collision
                if snake["length"] >= self.you["length"]:
                    cost += 100  # Lethal
                else:
                    cost -= 10  # Advantage
            elif dist == 2:
                # Threat zone
                cost += 20

        # 2. Hazards
        if pos in self.hazards:
            cost += 15

        # 3. Edge/Corner penalty (staying central is usually better)
        dist_to_edge_x = min(pos[0], self.width - 1 - pos[0])
        dist_to_edge_y = min(pos[1], self.height - 1 - pos[1])
        if dist_to_edge_x == 0 or dist_to_edge_y == 0:
            cost += 2

        return cost

    def a_star(self, start, goal):
        """
        A* pathfinding with influence map heuristic.
        """
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next_pos in self.get_neighbors(current):
                if next_pos in self.occupied:
                    continue

                new_cost = cost_so_far[current] + self.get_influence_cost(next_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.manhattan(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        if goal not in came_from:
            return None

        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
        return path[::-1]

    def flood_fill(self, start):
        """
        Measures reachable area from 'start'.
        """
        if start in self.occupied:
            return 0

        visited = {start}
        queue = collections.deque([start])
        count = 0

        while queue:
            curr = queue.popleft()
            count += 1
            for n in self.get_neighbors(curr):
                if n not in visited and n not in self.occupied:
                    visited.add(n)
                    queue.append(n)
        return count

    def get_move(self):
        """
        Decision logic:
        1. If hungry or small, find path to food.
        2. If safe path to food exists, take it.
        3. Otherwise, move to the largest available space.
        """
        # Sort food by distance
        sorted_food = sorted(self.food, key=lambda f: self.manhattan(self.my_head, f))

        # Strategy A: Go for food if health is low or we are small
        if self.you["health"] < 50 or self.you["length"] < 10:
            for food_pos in sorted_food[:3]:
                path = self.a_star(self.my_head, food_pos)
                if path:
                    # Check if taking the first step is safe
                    if self.flood_fill(path[0]) >= self.you["length"]:
                        return self.to_direction(path[0])

        # Strategy B: Find any safe move that maximizes space
        best_move = None
        max_space = -1

        for n in self.get_neighbors(self.my_head):
            if n in self.occupied:
                continue

            space = self.flood_fill(n)
            # Add a bit of influence cost to the space score to prefer safer areas
            score = space - (self.get_influence_cost(n) * 0.5)

            if score > max_space:
                max_space = score
                best_move = n

        if best_move:
            return self.to_direction(best_move)

        # Last resort: just move anywhere not occupied
        for n in self.get_neighbors(self.my_head):
            if n not in self.occupied:
                return self.to_direction(n)

        return "up"  # Good luck!

    def to_direction(self, target):
        tx, ty = target
        hx, hy = self.my_head
        if tx > hx:
            return "right"
        if tx < hx:
            return "left"
        if ty > hy:
            return "up"
        if ty < hy:
            return "down"
        return "up"


# Battlesnake API Routes


@app.get("/")
def index():
    """
    Your Battlesnake API info endpoint.
    """
    return jsonify(
        {
            "apiversion": "1",
            "author": "Manus AI",
            "color": "#FF3CF5",
            "head": "default",
            "tail": "default",
            "version": "1.0.0",
        }
    )


@app.post("/start")
def start():
    """
    Called when your Battlesnake begins a game.
    """
    data = request.get_json()
    game_id = data["game"]["id"]
    board_width = data["board"]["width"]
    board_height = data["board"]["height"]

    app.logger.info(f"Game {game_id} started on {board_width}x{board_height} board")

    return "ok"


@app.post("/move")
def move():
    """
    Called for every turn of each game.
    Returns your next move.
    """
    data = request.get_json()

    try:
        ai = BattlesnakeAI(data)
        next_move = ai.get_move()

        app.logger.info(f"Move: {next_move}")

        return jsonify({"move": next_move})
    except Exception as e:
        app.logger.error(f"Error in move: {str(e)}")
        return jsonify({"move": "up"})


@app.post("/end")
def end():
    """
    Called when a game your Battlesnake was in ends.
    """
    data = request.get_json()
    game_id = data["game"]["id"]

    app.logger.info(f"Game {game_id} ended")

    return "ok"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
