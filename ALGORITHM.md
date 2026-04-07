# A*+ Algorithm Implementation for Battle Snake Duel

## Overview

This document describes the A*-based pathfinding algorithm used in this Battle Snake implementation. The algorithm is specifically designed for 1v1 (duel) gameplay with anticipatory pathfinding and robust safety checks.

## Core Algorithm: A* Pathfinding

### Basic A* Concept

A* (A-star) is a graph traversal algorithm that finds the shortest path using:
- **g(n)**: Cost from start to current node
- **h(n)**: Heuristic estimate from current node to goal
- **f(n) = g(n) + h(n)**: Total estimated cost

The algorithm uses a priority queue (min-heap) to always expand the node with lowest f(n) value, guaranteeing optimal path when the heuristic is admissible.

### Adaptation for Battlesnake

In this implementation, A* is adapted with several enhancements:

1. **Danger-weighted nodes**: Instead of uniform movement cost, each cell has a danger cost derived from the danger map
2. **Multiple goal targets**: Can target multiple positions (food, center, enemy tail)
3. **Early exit**: Returns first goal reached for real-time performance

## Danger Map Construction

The danger map assigns a risk score to each board cell, computed from multiple factors:

### Hard Blocks (Weight: 1e6)
- Snake bodies (head, body, tail)
- Out-of-bounds positions

### Hazard Tiles (Weight: 5.0)
- Hazard zones that damage the snake

### Enemy Proximity (Weights: 4.0 - 30.0)
- **Enemy head ring (+4.0)**: Tiles adjacent to enemy head
- **Future head reach (+8.0 - 30.0)**: Tiles the enemy can reach in one move
  - Same or longer enemy: 30.0
  - Smaller enemy: 8.0

### Spatial Penalties
- **Wall hug (+1.0)**: Edge tiles
- **Corner (+1.5)**: Corner positions (highest risk)
- **Corridor squeeze (+3.5)**: Tiles with 3+ blocked neighbors

### Future Tail Modeling

Tails are not treated as permanent obstacles. The algorithm allows stepping on a tail if:
- It will move in the next turn (simulated by excluding tails from occupied set)
- There is an escape path from that position

## Flood Fill & Escape Tests

### Flood Area (BFS)

Counts reachable tiles from a position within a limit (250 cells). Used to:
- Evaluate available space after a move
- Detect trapped positions

### Escape Check

Determines if a position is viable by checking:
1. **Area >= snake length**: Enough space to survive
2. **Path to tail**: Can reach own tail position
3. **Tail vacated**: If blocked, assumes it moves next turn

This ensures the snake won't move into a dead-end.

## Multi-Source BFS for Food Reachability

### Purpose

Not all food is worth chasing. This module filters food based on competitive reachability.

### Algorithm

```
1. Run BFS from my head (blocked = occupied without tails)
2. Run BFS from enemy head (same blocked set)
3. For each food tile:
   - If my_dist < enemy_dist: claimable
   - If my_dist == enemy_dist AND my_length > enemy_length: claimable
   - Otherwise: ignore
```

This prevents chasing food the enemy will definitely get first.

## Enemy Lookahead (1-Ply Worst-Case)

### Purpose

Anticipate adversarial moves from the opponent.

### Implementation

For each candidate move, estimate extra risk if opponent responds optimally:
- **Head-on collision (+60.0)**: Enemy reaches same tile with equal/longer length
- **Adjacent threat (+25.0)**: Enemy moves adjacent to our next position with equal/larger length

### Usage in Scoring

The worst-case penalty is added to the move score, discouraging positions vulnerable to aggressive enemy responses.

## A* Implementation Details

### Heuristic Function

```python
def heuristic(c: Coord) -> float:
    dmin = min(manhattan(c, g) for g in goals)
    center = abs(c[0] - cx) + abs(c[1] - cy)  # center bias
    return dmin + 0.05 * center
```

- **Manhattan distance**: Admissible heuristic for grid movement
- **Center bias (0.05)**: Slight preference toward board center for strategic positioning

### Priority Queue (heapq)

```python
open_heap: List[Tuple[float, Coord]] = [(f, node)]
```

- Stores (f_score, coordinate) tuples
- Heapq ensures O(log n) pop of minimum f_score

### Closed Set

```python
closed: Set[Coord]
```

- Prevents revisiting nodes, improving efficiency
- Cleared each pathfinding call

### Danger-Aware Edge Cost

```python
tentative_g = g[current] + 1.0 + danger.get(n, 0.0)
```

Movement cost = 1 + danger penalty. This makes the pathfinder prefer safer routes even if slightly longer.

## Move Selection Policy

### Step 1: Generate Candidates

From head position, generate all 4 directional moves (up, down, left, right).

### Step 2: Filter Immediate Death

Remove moves where danger >= HARD_BLOCK (collision with snake body).

### Step 3: Goal Selection

Based on health and length:
- **Need food**: health <= 35 OR significantly shorter than enemy
  - Use reachable food as goals (filtered by competitive reachability)
- **Otherwise**:
  - Enemy tail (safer target than head)
  - Board center (strategic default)

### Step 4: Score Each Move

For each safe move, compute:

```
score = base_risk
      + corridor_pen          # 3.5 if squeezed
      + neck_pen             # 8.0 if stepping into own neck
      + area_pen             # 50 if trapped, 12 if tight, 25 if no escape
      + 0.8 * dist           # path length to goal
      + 0.12 * risk_along    # danger along path
      + reply_pen            # enemy lookahead penalty
```

**Bonuses**:
- -6.0 if moving onto food and need food

### Step 5: Tie-Breaking

Uses seeded RNG from game state for deterministic but varied tie-breaking:
```python
seed = hash((width, height, head, food) ^ opponent_length)
```

This ensures consistent behavior while avoiding deadlocks in symmetric positions.

## Weight Configuration Summary

| Constant | Value | Purpose |
|----------|-------|---------|
| HARD_BLOCK | 1e6 | Impossible moves |
| HAZARD | 5.0 | Damage tiles |
| ENEMY_REACH_SAME_OR_LONGER | 30.0 | High-risk enemy proximity |
| ENEMY_REACH_SMALLER | 8.0 | Low-risk enemy proximity |
| ENEMY_HEAD_RING | 4.0 | Adjacent to enemy head |
| WALL_HUG | 1.0 | Edge preference |
| CORNER | 1.5 | Corner penalty |
| CORRIDOR_SQUEEZE | 3.5 | Trapped position warning |

## Key Improvements Over Baseline A*

1. **Enemy-aware lookahead**: 1-ply anticipation with worst-case safety scoring
2. **Reachable food filter**: Ignore food enemy claims first
3. **Smarter danger model**: Wall/corner bias, corridor detection, head-on rules
4. **Future-tail modeling**: Allow stepping on tails that will move
5. **True space viability**: BFS area + escape test
6. **Risk-tuned scoring**: Configurable weights for behavior tuning

## Performance Considerations

- BFS for food: O(food_count × board_size)
- A* worst case: O(board_size × log(board_size))
- Flood fill: O(area_count)
- Typical frame: < 5ms on standard board sizes

## References

- A* Algorithm: https://en.wikipedia.org/wiki/A*_search_algorithm
- Battle Snake API: https://docs.battlesnake.com/