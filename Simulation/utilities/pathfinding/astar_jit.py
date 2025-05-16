import numpy as np
from numba import njit, int32, float32, boolean, types
from numba.typed import List

NEIGHBOR_VECTORS = np.array([
  (-1,  0),   # k = 0 → north (row–1, same col)
  ( 1,  0),   # k = 1 → south (row+1, same col)
  ( 0, -1),   # k = 2 → west  (same row, col–1)
  ( 0,  1),   # k = 3 → east  (same row, col+1)
], dtype=np.int32)

DIRECTION_VECTORS = {
  "N": (-1, 0),
  "S": ( 1, 0),
  "W": ( 0,-1),
  "E": ( 0, 1)
}
Point2 = types.UniTuple(int32, 2)

def astar_numba(
    start_x: int32,
    start_y: int32,
    goal_x:  int32,
    goal_y:  int32,
    grid:    np.ndarray,       # 0=road,1=vehicle,2=stop,3=blocked
    flow_mask: np.ndarray,     # shape (rows,cols,4)
    pen_stop:    float32,
    pen_vehicle: float32,
    pen_contraflow: float32,
    soft_obstacles: boolean,
    ignore_flow:    boolean,
    maximum_steps:   int32
) -> List[Point2]:
    rows, cols = grid.shape

    # Cost & state grids
    g_cost = np.full((rows, cols), np.inf, dtype=np.float32)
    f_cost = np.full((rows, cols), np.inf, dtype=np.float32)
    pred_x = np.full((rows, cols), -1, dtype=np.int32)
    pred_y = np.full((rows, cols), -1, dtype=np.int32)
    steps_arr = np.full((rows, cols), -1, dtype=np.int32)

    # OPEN list as arrays
    max_open = rows * cols + 4
    open_f = np.empty(max_open, dtype=np.float32)
    open_x = np.empty(max_open, dtype=np.int32)
    open_y = np.empty(max_open, dtype=np.int32)

    head = 0
    tail = 1

    # Initialize start
    g_cost[start_x, start_y] = 0.0
    h0 = abs(start_x - goal_x) + abs(start_y - goal_y)
    f_cost[start_x, start_y] = h0
    steps_arr[start_x, start_y] = 0
    open_f[0] = h0
    open_x[0] = start_x
    open_y[0] = start_y

    path_rev = List.empty_list(Point2)
    path = List.empty_list(Point2)

    # Main A* loop
    while head < tail:
        # Find best index by (f, g, steps) tie-break
        best = head
        bf = open_f[head]
        bx = open_x[head]
        by = open_y[head]
        bg = g_cost[bx, by]
        bs = steps_arr[bx, by]
        for i in range(head + 1, tail):
            fi = open_f[i]
            xi = open_x[i]
            yi = open_y[i]
            gi = g_cost[xi, yi]
            si = steps_arr[xi, yi]
            if fi < bf or (fi == bf and (gi < bg or (gi == bg and si < bs))):
                best, bf, bx, by, bg, bs = i, fi, xi, yi, gi, si
        # Swap head and best
        if best != head:
            open_f[head], open_f[best] = open_f[best], open_f[head]
            open_x[head], open_x[best] = open_x[best], open_x[head]
            open_y[head], open_y[best] = open_y[best], open_y[head]
        # Pop
        cx = open_x[head]
        cy = open_y[head]
        head += 1

        # Enforce step limit
        if steps_arr[cx, cy] > maximum_steps:
            continue

        # Goal reached?
        if cx == goal_x and cy == goal_y:
            # Backtrack
            while not (cx == start_x and cy == start_y):
                path_rev.append((cx, cy))
                px = pred_x[cx, cy]
                py = pred_y[cx, cy]
                cx, cy = px, py
            # Reverse path
            for i in range(len(path_rev) - 1, -1, -1):
                path.append(path_rev[i])
            return path

        # Expand neighbors
        for k in range(4):
            dx = NEIGHBOR_VECTORS[k, 0]
            dy = NEIGHBOR_VECTORS[k, 1]
            nx = cx + dx
            ny = cy + dy
            # Bounds check
            if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                continue

            # Steps
            new_steps = steps_arr[cx, cy] + 1
            if new_steps > maximum_steps:
                continue

            # Directional flow
            if flow_mask[cx, cy, k]:
                cost = 1.0
            else:
                if ignore_flow:
                    cost = 1.0 + pen_contraflow
                else:
                    continue

            cell = grid[nx, ny]
            # Impassable
            if cell == 3:
                continue

            ng = g_cost[cx, cy] + cost

            # Vehicle obstacle
            if cell == 1:
                if not soft_obstacles:
                    continue
                ng += pen_vehicle
            # Stop obstacle
            elif cell == 2:
                if not soft_obstacles:
                    continue
                ng += pen_stop

            # Record improved path
            if ng < g_cost[nx, ny] or steps_arr[nx, ny] < 0:
                g_cost[nx, ny] = ng
                steps_arr[nx, ny] = new_steps
                f_cost[nx, ny] = ng + abs(nx - goal_x) + abs(ny - goal_y)
                pred_x[nx, ny] = cx
                pred_y[nx, ny] = cy

                if tail < max_open:
                    open_f[tail] = f_cost[nx, ny]
                    open_x[tail] = nx
                    open_y[tail] = ny
                    tail += 1

    # No path found
    return List.empty_list(Point2)




@njit(cache=True)
def scan_ahead_numba(path_slice, veh_dict, stop_dict):
    idx_stop    = 1_000_000
    idx_vehicle = 1_000_000

    for i in range(len(path_slice)):
        xy = path_slice[i]

        if idx_vehicle == 1_000_000 and xy in veh_dict:
            idx_vehicle = i + 1
            if idx_stop != 1_000_000:
                break

        if idx_stop == 1_000_000 and xy in stop_dict:
            idx_stop = i + 1
            if idx_vehicle != 1_000_000:
                break

    return idx_stop, idx_vehicle