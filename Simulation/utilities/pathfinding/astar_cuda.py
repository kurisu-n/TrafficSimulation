from numba import cuda, int32, uint8, float32
from collections import namedtuple
import numpy as np
from Simulation.config import Defaults

PathRequest = namedtuple("PathRequest",
                         "start goal flags callback")

class PathPlanner:
    """Collects route requests & solves them on the GPU once per tick."""
    def __init__(self, model):
        self.model = model
        self.reqs: list[PathRequest] = []

    # ----------------------------------------------------------
    def request(self, start, goal, avoid_stops, allow_overtake, cb):
        flags = (1 if avoid_stops else 0) | (2 if allow_overtake else 0)
        self.reqs.append(PathRequest(start, goal, flags, cb))

    # ----------------------------------------------------------
    def solve_all(self):
        if not self.reqs:
            return

        n = len(self.reqs)
        # gather inputs
        starts = np.array([r.start for r in self.reqs], dtype=np.int32)
        goals = np.array([r.goal for r in self.reqs], dtype=np.int32)
        flags = np.array([r.flags for r in self.reqs], dtype=np.int32)
        grid = self.model.export_to_grid_cuda(dtype=np.uint8)

        # device buffers
        d_starts = cuda.to_device(starts)
        d_goals = cuda.to_device(goals)
        d_flags = cuda.to_device(flags)
        d_grid = cuda.to_device(grid)
        d_paths = cuda.device_array((n * MAX_PATH * 2,), dtype=np.int32)
        d_len = cuda.device_array(n, dtype=np.int32)

        # penalty parameters
        pen_stop = np.float32(Defaults.VEHICLE_OBSTACLE_PENALTY_STOP)
        pen_vehicle = np.float32(Defaults.VEHICLE_OBSTACLE_PENALTY_VEHICLE)

        # launch kernel with penalties
        batched_astar[n, 32](
            d_starts, d_goals, d_flags,
            d_grid, d_paths, d_len,
            pen_stop, pen_vehicle
        )
        cuda.synchronize()

        # pull results and callbacks
        paths = d_paths.copy_to_host()
        lens = d_len.copy_to_host()

        # --------- deliver each path to its requester --------
        for i, req in enumerate(self.reqs):
            path_len = int(lens[i])
            if path_len == 0:  # no route found
                req.callback([])
                continue

            base = i * MAX_PATH * 2
            flat = paths[base: base + path_len * 2]

            # kernel writes goal→start, flip to start→goal
            xy_coords = [(flat[j], flat[j + 1]) for j in range(0, path_len * 2, 2)][::-1]

            # translate coordinates to CellAgent objects
            cells = [self.model.get_cell_contents(x, y)[0] for (x, y) in xy_coords]

            req.callback(cells)

        # ready for next tick
        self.reqs.clear()




MAX_PATH      = 512          # longest path we actually return
MAX_FRONTIER  = 8192         # capacity of the OPEN list (was 512)

@cuda.jit
# Add pen_stop and pen_vehicle to signature
def batched_astar(starts, goals, flags,
                  grid,                # uint8[width, height]
                  paths,               # int32[n * MAX_PATH * 2]
                  path_len,            # int32[n]
                  pen_stop,            # float32 penalty for stop cells
                  pen_vehicle):        # float32 penalty for vehicle cells
    """
    GPU-backed A* with penalties for vehicles and stops,
    and skipping impassable cells (grid > 2).
    """
    bid = int(cuda.blockIdx.x)
    tid = int(cuda.threadIdx.x)
    n   = int(starts.shape[0])
    if bid >= n or tid != 0:
        return

    width, height = grid.shape
    # Heuristic: Manhattan distance
    def h(x, y, gx, gy):
        return abs(gx - x) + abs(gy - y)

    sx, sy = starts[bid][0], starts[bid][1]
    gx, gy = goals[bid][0], goals[bid][1]
    # blk_flags = flags[bid]  # reserved for further logic

    # Local OPEN list
    open_x = cuda.local.array(shape=MAX_FRONTIER, dtype=int32)
    open_y = cuda.local.array(shape=MAX_FRONTIER, dtype=int32)
    open_f = cuda.local.array(shape=MAX_FRONTIER, dtype=float32)
    parent = cuda.local.array(shape=MAX_FRONTIER, dtype=int32)

    # Seed with start node
    open_x[0], open_y[0] = sx, sy
    open_f[0] = float32(h(sx, sy, gx, gy))
    parent[0] = -1
    head = 0
    tail = 1

    found = -1
    # A* loop
    while head < tail and tail < MAX_FRONTIER:
        # 1) select lowest-f
        best = head
        best_f = open_f[head]
        for i in range(head + 1, tail):
            if open_f[i] < best_f:
                best, best_f = i, open_f[i]
        # swap-remove
        if best != head:
            open_x[head], open_x[best] = open_x[best], open_x[head]
            open_y[head], open_y[best] = open_y[best], open_y[head]
            open_f[head], open_f[best] = open_f[best], open_f[head]
            parent[head], parent[best] = parent[best], parent[head]

        cx, cy = open_x[head], open_y[head]
        # goal check
        if cx == gx and cy == gy:
            found = head
            break
        # cost so far
        g_here = best_f - h(cx, cy, gx, gy)
        head += 1

        # 3) expand neighbors
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx + dx, cy + dy
            # bounds
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            cell = grid[nx, ny]
            # skip walls / sidewalk / buildings (any >2)
            if cell > 2:
                continue
            # incremental cost
            g_new = g_here + float32(1.0)
            # add penalties
            if cell == 1:
                g_new += pen_vehicle
            elif cell == 2:
                g_new += pen_stop
            # push into OPEN
            if tail < MAX_FRONTIER:
                open_x[tail] = nx
                open_y[tail] = ny
                open_f[tail] = g_new + float32(h(nx, ny, gx, gy))
                parent[tail] = head - 1
                tail += 1

    # write path back
    base = bid * MAX_PATH * 2
    if found == -1:
        path_len[bid] = 0
        return
    length = 0
    i = found
    while i >= 0 and length < MAX_PATH:
        paths[base + length*2    ] = open_x[i]
        paths[base + length*2 + 1] = open_y[i]
        length += 1
        i = parent[i]
    path_len[bid] = length
