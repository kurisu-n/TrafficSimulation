# In Simulation/utilities/pathfinding/astar_jit.py (new or updated file)

import numpy as np
from numba import njit, types, int32
from numba.typed import List
from numba.core.types import UniTuple, int64
from Simulation.config import Defaults


# Predefine direction vectors and mapping for Numba
# Neighbor moves: N, E, S, W (in terms of (dx, dy) on grid)
NEIGHBOR_DELTAS = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=np.int32)  # N, E, S, W
# Map direction bit indices: 'N'=0, 'E'=1, 'S'=2, 'W'=3 (must match NEIGHBOR_DELTAS order)
# (This mapping should be consistent with how CellAgent.directions are defined in Defaults.AVAILABLE_DIRECTIONS)

# Cache penalty values for quick access (assumed to be int)
CONTRA_PENALTY = Defaults.VEHICLE_CONTRAFLOW_PENALTY           # cost for contraflow move
VEHICLE_PENALTY = Defaults.VEHICLE_OBSTACLE_PENALTY_VEHICLE    # cost for entering occupied cell (if soft obstacle)
STOP_PENALTY = Defaults.VEHICLE_OBSTACLE_PENALTY_STOP

NodeTuple = UniTuple(int64, 4)# cost for entering stop-sign cell (if soft obstacle)
CoordTuple = UniTuple(int64, 2)

DX_ARRAY = np.array([ 1,  1,  0, -1, -1, -1,  0,  1], dtype=np.int32)
DY_ARRAY = np.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=np.int32)

@njit(inline='always')
def compute_fov_numba(cx: int, cy: int, height: int, width: int,
                      awareness: int, is_road_map: np.ndarray,
                      dir_dx: np.ndarray, dir_dy: np.ndarray):
    """
    Return a typed-list of (x,y) positions visible from (cx,cy) within the awareness range,
    along the four cardinal directions, stopping when the cell is no longer road.
    """
    # Define element type as a 2-tuple of int32
    out = List.empty_list(CoordTuple)

    # Four cardinal directions: N, E, S, W
    for d in range(4):
        dx = dir_dx[d]
        dy = dir_dy[d]
        # perpendicular for sideways offsets
        px = -dy
        py = dx
        # for each parallel line
        for offset in range(-awareness + 1, awareness):
            x = cx + offset * px
            y = cy + offset * py
            # march along the direction until out of bounds or non-road
            step = 0
            while x >= 0 and x < width and y >= 0 and y < height and is_road_map[y, x] == 1:
                out.append((x, y))
                step += 1
                x = cx + offset * px + dx * step
                y = cy + offset * py + dy * step
    return out

@njit
def astar_numba(width: int, height: int, start_x: int, start_y: int, goal_x: int, goal_y: int,
                occupancy_map: np.ndarray, stop_map: np.ndarray,
                is_road_map: np.ndarray, allowed_dirs_map: np.ndarray,
                respect_awareness: bool, awareness_range: int,
                soft_obstacles: bool, ignore_flow: bool,
                maximum_steps: int = 0x7FFFFFFF) -> List[UniTuple(int32,2)]:
    """Numba-optimized A* pathfinder. Returns a list of (x,y) positions from start (exclusive) to goal (inclusive),
    or an empty list if no path is found under the given conditions."""
    # Initialize structures
    start_idx = start_y * width + start_x
    goal_idx  = goal_y * width + goal_x

    # Distance (g cost) array and came_from map
    INF = 0x3F3F3F3F  # a large number to represent "infinity" (here using 0x3F3F3F3F ~ 1e9)
    dist = np.full(width * height, INF, np.int32)
    dist[start_idx] = 0
    came_from = np.full(width * height, -1, np.int32)

    # Open set (list of nodes to explore) as a list of tuples: (f, g, steps, node_index)
    # Using a Python list (via numba.typed.List for mutability in Numba)
    open_set = List.empty_list(NodeTuple)
    # Push the start node
    # Heuristic (Manhattan distance from start to goal)
    h0 = abs(start_x - goal_x) + abs(start_y - goal_y)
    open_set.append((h0, 0, 0, start_y * width + start_x))

    fov_map = np.ones((height, width), np.int8)
    if respect_awareness:
        dir_dx = np.array([0, 1, 0, -1], np.int32)
        dir_dy = np.array([1, 0, -1, 0], np.int32)
        fov_positions = compute_fov_numba(start_x, start_y,
                                          height, width,
                                          awareness_range,
                                          is_road_map,
                                          dir_dx, dir_dy)
        # mark them in the mask
        for i in range(len(fov_positions)):
            fx, fy = fov_positions[i]
            fov_map[fy, fx] = 1

    # A* search loop
    while len(open_set) > 0:
        # Extract the node with minimum f from open_set (linear search for simplicity)
        best_i = 0
        best_f = open_set[0][0]
        for i in range(1, len(open_set)):
            if open_set[i][0] < best_f:
                best_f = open_set[i][0]
                best_i = i
        # Pop the best node (swap-and-pop technique to avoid shifting list elements)
        f, g, steps, current_idx = open_set[best_i]
        last_idx = len(open_set) - 1
        open_set[best_i] = open_set[last_idx]
        open_set.pop()  # remove the last element

        # If this node is the goal, reconstruct and return the path
        if current_idx == goal_idx:
            # Reconstruct path by backtracking from goal to start
            result_path = List.empty_list(CoordTuple)
            idx = current_idx
            while idx != start_idx:
                # Prepend current cell’s coordinates to path (will reverse later)
                x = idx % width
                y = idx // width
                result_path.append((x, y))
                idx = came_from[idx]
            # (Optionally, include the start cell if needed; here we exclude start to mimic original behavior)
            # Now result_path contains goal->...->neighbor of start (in reverse order). Reverse it:
            path = List.empty_list(CoordTuple)
            for j in range(len(result_path)-1, -1, -1):
                path.append(result_path[j])
            return path  # path from start (excluded) to goal (included)

        # If we already found a better way to this node, skip processing it
        if g > dist[current_idx]:
            continue

        # Get current cell coordinates
        cur_x = current_idx % width
        cur_y = current_idx // width

        # Explore all neighbors (N, E, S, W)
        for d in range(4):
            # Determine neighbor coordinates
            nx = cur_x + NEIGHBOR_DELTAS[d][0]
            ny = cur_y + NEIGHBOR_DELTAS[d][1]
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue  # neighbor is out of bounds

            new_steps = steps + 1
            if new_steps > maximum_steps:
                continue  # exceeds step limit for this search mode

            neighbor_idx = ny * width + nx
            # Base movement cost is 1 per step
            new_g = g + 1

            # Check road direction (flow) constraint
            # Determine if moving in direction d is contraflow for the current cell
            # (allowed_dirs_map holds bits for allowed directions from (cur_x,cur_y))
            allowed_bits = allowed_dirs_map[cur_y, cur_x]
            is_contraflow = (allowed_bits & (1 << d)) == 0
            if is_contraflow:
                if ignore_flow and is_road_map[ny, nx] == 1:
                    new_g += CONTRA_PENALTY  # allow wrong-way move with penalty
                else:
                    continue  # disallowed move (wrong way on one-way road)

            # Check if neighbor is occupied by a vehicle
            if occupancy_map[ny, nx] == 1:
                # Consider it an obstacle only if we are not ignoring FOV, or if it's within FOV
                if (not respect_awareness) or (fov_map[ny, nx] == 1):
                    if soft_obstacles:
                        new_g += VEHICLE_PENALTY  # treat occupied cell as a soft obstacle (add cost)
                    else:
                        continue  # hard obstacle: skip this neighbor

            # Check if neighbor has a stop sign/light
            if stop_map[ny, nx] == 1:
                if (not respect_awareness) or (fov_map[ny, nx] == 1):
                    if soft_obstacles:
                        new_g += STOP_PENALTY  # treat stop cell as soft obstacle (add cost)
                    else:
                        continue  # hard obstacle: cannot enter (except maybe as final step)

            # If we reach here, the neighbor is a valid move. Check if this path is better than any previously found.
            if new_g < dist[neighbor_idx]:
                dist[neighbor_idx] = new_g
                came_from[neighbor_idx] = current_idx
                # Push neighbor into open set with its f-score = g + heuristic
                # Manhattan heuristic from neighbor to goal:
                h = abs(nx - goal_x) + abs(ny - goal_y)
                open_set.append((new_g + h, new_g, new_steps, neighbor_idx))
    # If open_set is empty and goal was never reached, return an empty path (no route found)
    return List.empty_list(CoordTuple)