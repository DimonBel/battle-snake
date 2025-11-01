# battle-snake

### High-level approach

- Parse the incoming game state and represent it with simple dataclasses (`Snake`, `Board`).
- Create an occupancy map (occupied squares) while allowing tail squares to be treated flexibly since tails often move.
- Build a danger map which assigns soft costs to squares (hazards, walls, enemy head proximity, etc.). Hard-blocked squares (occupied) are given a very large cost and treated as impassable by pathfinding.
- Use A* to search from each candidate next head position toward a set of goals (food when needed, otherwise center + enemy tail). The A* f-score uses path cost (including danger) plus a heuristic (Manhattan distance plus a small center bias).
- Score candidate moves combining immediate danger, path distance & risk, area/flood-fill escape measures, corridor squeezing, neck/backwards penalties, and a 1-ply enemy lookahead penalty.
- Return the direction with the lowest combined score. Deterministic jitter (seeded RNG) is used for stable tie-breaking.

## Occupancy and reachability

- `occupied_squares(board, include_tails=True)` returns the set of coordinates currently occupied by both snakes; tails can be optionally excluded to allow stepping into a tile that will vacate next turn.
- `multi_source_bfs()` computes distances from one or more starting points while honoring blocked cells. This is used to compute reachability and to decide which food items are contested or reachable before the opponent (`reachable_food`).

## Danger map and weights

The bot computes a floating-point danger value for every board tile. The `Weights` class centralizes the constants used. Examples:

- HARD_BLOCK (very large) — used for occupied squares (impenetrable for path planning).
- HAZARD — added for hazard tiles.
- ENEMY_REACH_SAME_OR_LONGER / ENEMY_REACH_SMALLER — penalties on tiles the enemy head can reach; scaled depending on relative lengths.
- ENEMY_HEAD_RING — extra cost for tiles surrounding the enemy head.
- WALL_HUG, CORNER — soft biases to discourage hugging walls/corners.
- CORRIDOR_SQUEEZE — used when a square has many blocked or out-of-bounds neighbors.

The danger map is added into A* path costs to bias routes away from risky tiles.

## Flood-fill & escape checks

- `flood_area()` performs a bounded flood-fill to estimate the available free area from a given tile. This helps detect traps and small open areas.
- `has_escape()` checks cheaply whether a move likely has an escape route by verifying flood area size and reachability to the current tail (with a heuristic that the tail may vacate next turn).

Edge cases handled:
- If the starting tile is out-of-bounds or blocked, flood_area returns 0.
- Tails are allowed as potential escape targets if they are expected to move.

## A* implementation

astar(board, start, goals, danger)

- Start: one coordinate (typically the candidate next head position).
- Goals: a set of coordinates to search toward (food, center, or opponent tail).
- Danger: a map from Coord -> float added to step costs.

Cost model inside A*:
- Moving to a neighbor adds 1.0 (base) plus the danger value of that neighbor.
- Nodes with danger >= HARD_BLOCK are skipped (treated as impassable).

Heuristic:
- The heuristic is the Manhattan distance to the nearest goal plus a small center-bias term: h(c) = min_g manhattan(c,g) + 0.05 * (distance to board center). The center bias encourages the snake to prefer more central positions when distances are equal.

Returned path:
- A* returns a list of coordinates from the start to the reached goal (inclusive). The path cost considers both distance and danger values.

Why A* here?
- A* gives a shortest-cost path under the chosen cost metric (distance + danger). It is efficient and flexible: changing the danger map or heuristic immediately changes route preference without changing the core search logic.

## Move scoring and policy

The main decision function `choose_move()` follows these steps for each legal candidate move:

1. Filter out moves that step on hard-blocked tiles (occupied squares treated as HARD_BLOCK by danger map).
2. Determine if food is needed (low health or size disadvantage). If so, compute `reachable_food` and use food tiles as goals when safe.
3. If no food goals, use a safer set of goals (opponent tail if present, and board center tiles).
4. For each remaining candidate:
	- Compute corridor squeeze penalty using `corridor_sides()` (counts blocked/out-of-bounds orthogonal neighbors).
	- Penalize stepping into the neck (moving backwards) mildly.
	- Run A* from the candidate (as the new head) to the set of goals using the current danger map. If no path is found, apply a large fallback penalty.
	- Compute area (flood-fill) penalty; small areas cause heavy penalties to avoid getting trapped.
	- Run `enemy_best_reply_penalty()` which does a 1-ply enemy lookahead: consider enemy next positions and penalize moves that allow the enemy to move into your head or adjacent to it when they're equal or longer.
	- Combine all components (base danger + corridor + neck + area + 0.8*dist + 0.12*risk_along + reply_pen) into a final numerical score. Lower is better.
5. Break ties using a small, deterministically-seeded random jitter so the behavior is repeatable but avoids consistent ties.

This combination attempts to trade off safety (avoid head-on collisions, hazardous tiles, and traps) with utility (reach food when needed and move toward useful targets otherwise).

## Enemy lookahead

The bot performs a simple 1-ply adversarial evaluation: it enumerates the opponent's possible next head squares and penalizes our candidate move if the enemy can move into the same square (head-on) or an adjacent square and has equal-or-greater length. This is a heuristic to avoid risky head-to-head confrontations.


3. The server implements three endpoints used by Battlesnake:
- GET `/` — metadata about the snake.
- POST `/start` — game start hook (no-op).
- POST `/move` — main move selection. Expects the Battlesnake game JSON and returns {"move": "up|down|left|right", "shout": string}.

Notes: the code seeds its deterministic jitter using a hash of the board state so repeated identical boards produce the same tie-breaking.

## Assumptions & limitations

- This bot is written for a one-versus-one duel. It picks the first non-self snake from the board's list as the opponent and ignores multiple opponents.
- The danger and scoring heuristics are hand-tuned constants; they can be adapted or learned.
- Only a shallow (1-ply) enemy lookahead is used; deeper search would be more costly but could be more robust in some tactical situations.
- A* uses a simple heuristic which is admissible for the base distance part but the addition of the center bias makes it inadmissible (intentionally) to prefer central positions.


## Summary

This Battlesnake agent couples an A* pathfinder with a soft danger map and several heuristics to select safe, useful moves. It favors safety and escape routes while pursuing food when necessary, and uses a deterministic jitter for stable tie-breaking.

If you'd like, I can also add a short example board and a small unit test that runs `choose_move()` on it and demonstrates the scoring breakdown.

[Battleship conference paper] https://pageperso.lis-lab.fr/guilherme.fonseca/battleship_conf.pdf

[Battlesnake useful algorithms] https://docs.battlesnake.com/guides/useful-algorithms

[Relevant arXiv paper] https://arxiv.org/pdf/2007.10504
