# path_planner.py  --------------------------------------------------------------
from collections import namedtuple
import numpy as np
from numba import cuda
from Simulation.utilities.pathfinding.astar_cuda import batched_astar, MAX_PATH

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

        # --------- gather inputs for the GPU kernel ----------
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

        # --------- launch batched A* (1 block per route) -----
        batched_astar[n, 32](d_starts, d_goals, d_flags, d_grid, d_paths, d_len)
        cuda.synchronize()  # catch any kernel errors

        # pull results back to host
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


