import numpy as np
from numba import njit, jit, int32, float32, types
from numba.typed import List

_NEIGH = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32)
Point2 = types.UniTuple(int32, 2)
BOOL   = types.boolean

@njit(cache=True)
def astar_numba(start_x: int32,
                start_y: int32,
                goal_x:  int32,
                goal_y:  int32,
                grid:    np.ndarray,                 # 0=road,1=veh,2=stop,3=block
                pen_stop:    float32,
                pen_vehicle: float32,
                max_steps:   int32):
    """
    Pure-Numba A* that returns a numba.typed.List[(int32, int32)].
    Cell codes:
        0 = free road / intersection
        1 = other vehicle (penalised)
        2 = stop / red-light   (penalised)
        3 = impassable cell (building, sidewalk, …)
    """

    rows, cols = grid.shape

    # ── cost & predecessor grids ────────────────────────────────────────
    g_cost  = np.full((rows, cols), np.inf, dtype=np.float32)
    f_cost  = np.full((rows, cols), np.inf, dtype=np.float32)
    pred_x  = np.full((rows, cols), -1,  dtype=np.int32)
    pred_y  = np.full((rows, cols), -1,  dtype=np.int32)

    # ── OPEN list (three parallel arrays) ───────────────────────────────
    max_open = rows * cols + 4                          # safety margin
    open_f   = np.empty(max_open, dtype=np.float32)
    open_x   = np.empty(max_open, dtype=np.int32)
    open_y   = np.empty(max_open, dtype=np.int32)
    head = tail = 0

    # push START
    g_cost[start_x, start_y] = 0.0
    f_cost[start_x, start_y] = abs(start_x - goal_x) + abs(start_y - goal_y)
    open_f[0], open_x[0], open_y[0] = f_cost[start_x, start_y], start_x, start_y
    tail = 1

    # containers for the resulting path
    path_rev = List.empty_list(Point2)      # filled backwards
    path     = List.empty_list(Point2)      # final, forward order

    # ── MAIN LOOP ───────────────────────────────────────────────────────
    while head < tail:

        # pop lowest-f node (small O(n) scan)
        best = head
        bf   = open_f[head]
        for i in range(head + 1, tail):
            if open_f[i] < bf:
                best, bf = i, open_f[i]
        if best != head:
            open_f[head], open_f[best] = open_f[best], open_f[head]
            open_x[head], open_x[best] = open_x[best], open_x[head]
            open_y[head], open_y[best] = open_y[best], open_y[head]

        cx = open_x[head]
        cy = open_y[head]
        head += 1

        # ── goal reached – back-track ──────────────────────────────────
        if cx == goal_x and cy == goal_y:
            while not (cx == start_x and cy == start_y):
                path_rev.append((np.int32(cx), np.int32(cy)))
                px, py = pred_x[cx, cy], pred_y[cx, cy]
                cx, cy = px, py
            for i in range(len(path_rev) - 1, -1, -1):
                path.append(path_rev[i])
            return path                                 # SUCCESS

        # ── expand neighbours ──────────────────────────────────────────
        for k in range(4):
            nx = cx + _NEIGH[k, 0]
            ny = cy + _NEIGH[k, 1]

            if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                continue

            cell = grid[nx, ny]
            if cell == 3:                               # impassable
                continue

            tentative = g_cost[cx, cy] + 1.0
            if cell == 1:
                tentative += pen_vehicle
            elif cell == 2:
                tentative += pen_stop

            if tentative < g_cost[nx, ny]:
                g_cost[nx, ny] = tentative
                f_cost[nx, ny] = tentative + abs(nx - goal_x) + abs(ny - goal_y)
                pred_x[nx, ny] = cx
                pred_y[nx, ny] = cy

                # ---- OPEN-list push (bounds check first) -------------
                if tail >= max_open:                    # hard guard
                    continue
                open_f[tail] = f_cost[nx, ny]
                open_x[tail] = nx
                open_y[tail] = ny
                tail += 1

                # optional step-limit
                if tail >= max_steps:
                    break

    # ── no path found ──────────────────────────────────────────────────
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