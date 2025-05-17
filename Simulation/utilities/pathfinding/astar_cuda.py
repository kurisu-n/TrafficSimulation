# Simulation/utilities/pathfinding/astar_cuda.py
import numpy as np
from numba import cuda, int32
from Simulation.config import Defaults

# Constants for pathfinding penalties (read from configuration)
CONTRA_PENALTY   = int(Defaults.VEHICLE_CONTRAFLOW_PENALTY)         # penalty cost for contraflow moves
VEHICLE_PENALTY = int(Defaults.VEHICLE_OBSTACLE_PENALTY_VEHICLE)   # penalty cost for treating occupied cell as soft obstacle
STOP_PENALTY    = int(Defaults.VEHICLE_OBSTACLE_PENALTY_STOP)      # penalty cost for treating stop-sign/light as soft obstacle
INF_COST       = np.int32(0x3F3F3F3F)  # "infinity" cost (a very large int)

DX = (0, 1, 0, -1)
DY = (1, 0, -1, 0)

@cuda.jit(device=True)
def _abs(val):
    """Device function for absolute value of an integer (since math.abs not directly available)."""
    return val if val >= 0 else -val

@cuda.jit(device=True, inline=True)
def in_fov(nx, ny, sx, sy, width, height, awareness, is_road_map):
    """
    Return True if (nx,ny) is visible from (sx,sy) within 'awareness' range,
    marching along each of the four cardinal directions until non-road.
    """
    for d in range(4):
        dx = DX[d]
        dy = DY[d]
        # perpendicular offsets
        px = -dy
        py = dx
        # parallel lines within awareness band
        for offset in range(-awareness + 1, awareness):
            x0 = sx + offset * px
            y0 = sy + offset * py
            step = 0
            # march until out of bounds or non-road
            while 0 <= x0 < width and 0 <= y0 < height and is_road_map[y0, x0] == 1:
                if x0 == nx and y0 == ny:
                    return True
                step += 1
                x0 = sx + offset * px + dx * step
                y0 = sy + offset * py + dy * step
    return False

@cuda.jit
def astar_kernel(
    width, height,
    start_xs, start_ys, goal_xs, goal_ys,
    occupancy_map, stop_map, is_road_map, allowed_dirs_map,
    dist_array, came_from_array, steps_array,
    output_paths, output_lengths,
    respect_awareness, awareness_range,
    soft_obstacles, ignore_flow, maximum_steps
):
    tid = cuda.grid(1)
    if tid >= start_xs.size:
        return

    wh     = width * height
    offset = tid * wh

    # load start/goal
    sx = start_xs[tid]
    sy = start_ys[tid]
    gx = goal_xs[tid]
    gy = goal_ys[tid]
    start_idx = sy * width + sx
    goal_idx  = gy * width + gx

    # 1) initialize
    for i in range(wh):
        dist_array[offset + i]      = INF_COST
        came_from_array[offset + i] = -1
        steps_array[offset + i]     = int32(0)
    dist_array[offset + start_idx]  = int32(0)
    steps_array[offset + start_idx] = int32(0)

    # 2) A* loop
    while True:
        # find open node with smallest f = g + h
        best_f = INF_COST
        current_idx = -1
        for i in range(wh):
            g = dist_array[offset + i]
            if g == INF_COST:
                continue
            x = i % width
            y = i // width
            # heuristic: Manhattan
            dx = gx - x
            if dx < 0: dx = -dx
            dy = gy - y
            if dy < 0: dy = -dy
            f = g + (dx + dy)
            if f < best_f:
                best_f = f
                current_idx = i

        # no node left ⇒ fail
        if current_idx == -1:
            output_lengths[tid] = 0
            return

        # goal reached ⇒ reconstruct
        if current_idx == goal_idx:
            length = 0
            node = current_idx
            while node != start_idx and node != -1:
                x = node % width
                y = node // width
                output_paths[tid, length, 0] = x
                output_paths[tid, length, 1] = y
                length += 1
                node = came_from_array[offset + node]
            # reverse in place
            for j in range(length // 2):
                opp = length - 1 - j
                tx = output_paths[tid, j, 0]
                ty = output_paths[tid, j, 1]
                output_paths[tid, j, 0] = output_paths[tid, opp, 0]
                output_paths[tid, j, 1] = output_paths[tid, opp, 1]
                output_paths[tid, opp, 0] = tx
                output_paths[tid, opp, 1] = ty
            output_lengths[tid] = length
            return

        # mark current closed
        current_g = dist_array[offset + current_idx]
        dist_array[offset + current_idx] = INF_COST

        cx = current_idx % width
        cy = current_idx // width
        allowed = allowed_dirs_map[cy, cx]

        # explore neighbors N, E, S, W
        for d in range(4):
            if d == 0:
                nx, ny = cx,     cy + 1
            elif d == 1:
                nx, ny = cx + 1, cy
            elif d == 2:
                nx, ny = cx,     cy - 1
            else:
                nx, ny = cx - 1, cy

            # bounds
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            nidx = ny * width + nx
            step = steps_array[offset + current_idx] + 1
            if step > maximum_steps:
                continue

            new_g = current_g + 1

            # one‐way / contraflow
            if (allowed & (1 << d)) == 0:
                if ignore_flow and is_road_map[ny, nx] == 1:
                    new_g += CONTRA_PENALTY
                else:
                    continue

            # occupancy obstacle
            if occupancy_map[ny, nx] == 1:
                if respect_awareness:
                    if in_fov(nx, ny, sx, sy, width, height, awareness_range, is_road_map):
                        if soft_obstacles:
                            new_g += VEHICLE_PENALTY
                        else:
                            continue
                    # else: obstacle outside FOV ⇒ ignore entirely
                else:
                    if soft_obstacles:
                        new_g += VEHICLE_PENALTY
                    else:
                        continue

            # stop‐sign obstacle
            if stop_map[ny, nx] == 1:
                if respect_awareness:
                    if in_fov(nx, ny, sx, sy, width, height, awareness_range, is_road_map):
                        if soft_obstacles:
                            new_g += STOP_PENALTY
                        else:
                            continue
                else:
                    if soft_obstacles:
                        new_g += STOP_PENALTY
                    else:
                        continue

            # record if better
            if new_g < dist_array[offset + nidx]:
                dist_array[offset + nidx]      = new_g
                came_from_array[offset + nidx] = current_idx
                steps_array[offset + nidx]     = step

class PathPlanner:
    """
    GPU pathfinding planner that manages CUDA data and computations.
    """

    def __init__(self, city_model):
        self.city_model = city_model
        self.width = city_model.get_width()
        self.height = city_model.get_height()
        # Copy static maps to device memory (directions and road layout)
        self.d_allowed_dirs = cuda.to_device(city_model.allowed_dirs_map.astype(np.uint8))
        self.d_is_road = cuda.to_device(city_model.is_road_map.astype(np.int8))
        # Prepare device arrays for dynamic maps (occupancy and stop), will update each tick
        self.d_occupancy = cuda.to_device(city_model.occupancy_map.astype(np.int8))
        self.d_stop = cuda.to_device(city_model.stop_map.astype(np.int8))
        # Initialize a list for pending pathfinding requests (for batched solving)
        self._pending_requests = []

        # Pre-allocate device arrays for batched pathfinding (max queries per batch can be adjusted)
        # For simplicity, we'll allocate based on a reasonable upper bound of queries to handle simultaneously.
        self.max_batch_size = 1024  # maximum number of concurrent path queries
        max_cells = self.width * self.height
        # Flattened arrays for cost, predecessor, steps (size = max_batch_size * max_cells)
        self.d_dist = cuda.device_array(shape=(self.max_batch_size * max_cells,), dtype=np.int32)
        self.d_came_from = cuda.device_array(shape=(self.max_batch_size * max_cells,), dtype=np.int32)
        self.d_steps = cuda.device_array(shape=(self.max_batch_size * max_cells,), dtype=np.int32)
        # Output path storage: to accommodate worst-case path length, allocate max_cells entries per query
        self.d_out_paths = cuda.device_array(shape=(self.max_batch_size, max_cells, 2), dtype=np.int32)
        self.d_out_lengths = cuda.device_array(shape=(self.max_batch_size,), dtype=np.int32)

    def update_dynamic_maps(self):
        """Refresh dynamic occupancy and stop maps on the GPU (call each tick before pathfinding)."""
        # Copy the latest occupancy_map and stop_map from the CityModel to device
        cuda.to_device(self.city_model.occupancy_map, to=self.d_occupancy)
        cuda.to_device(self.city_model.stop_map, to=self.d_stop)

    def request_path(self, vehicle_agent, start_pos: tuple[int, int], goal_pos: tuple[int, int],
                     soft_obstacles=False, ignore_flow=False, respect_fov=False, maximum_steps=0x7FFFFFFF):
        """
        Queue a pathfinding request to compute a route from start_pos to goal_pos for the given vehicle_agent.
        The actual computation will be performed when `solve_all()` is called.
        """
        self._pending_requests.append({
            "vehicle": vehicle_agent,
            "start": start_pos,
            "goal": goal_pos,
            "soft": soft_obstacles,
            "ignore_flow": ignore_flow,
            "respect_fov": respect_fov,
            "max_steps": maximum_steps
        })

    def solve_all(self):
        """
        Execute all queued pathfinding requests in a batch on the GPU.
        This will process the pending requests and assign the resulting paths to the respective vehicle agents.
        """
        if not self._pending_requests:
            return  # nothing to do

        # Update occupancy and stop maps on GPU to reflect current state
        self.update_dynamic_maps()

        # Prepare batched start/goal arrays for GPU
        num_queries = len(self._pending_requests)
        # Limit batch size to preallocated capacity
        if num_queries > self.max_batch_size:
            num_queries = self.max_batch_size  # (In a real scenario, handle the overflow or split into multiple batches)
        start_xs = np.zeros(num_queries, dtype=np.int32)
        start_ys = np.zeros(num_queries, dtype=np.int32)
        goal_xs = np.zeros(num_queries, dtype=np.int32)
        goal_ys = np.zeros(num_queries, dtype=np.int32)
        # Use flags from first request (assume all requests use same mode in current design)
        # (Alternatively, could run separate batches for different flag combinations if needed)
        soft_flag = self._pending_requests[0]["soft"]
        ignore_flow_flag = self._pending_requests[0]["ignore_flow"]
        respect_fov_flag = self._pending_requests[0]["respect_fov"]
        max_steps_val = self._pending_requests[0]["max_steps"]

        for i, req in enumerate(self._pending_requests[:num_queries]):
            sx, sy = req["start"]
            gx, gy = req["goal"]
            start_xs[i] = sx
            start_ys[i] = sy
            goal_xs[i] = gx
            goal_ys[i] = gy
            # If any request has different flags, we could handle separately.
            # (For simplicity, assume uniform flags for now or handle in separate solve_all calls.)
            soft_flag = soft_flag or req["soft"]
            ignore_flow_flag = ignore_flow_flag or req["ignore_flow"]
            respect_fov_flag = respect_fov_flag or req["respect_fov"]
            if req["max_steps"] < max_steps_val:
                max_steps_val = req[
                    "max_steps"]  # use minimum max_steps among requests (if contraflow queries specify a smaller limit)

        # Transfer start/goal data to device
        d_start_xs = cuda.to_device(start_xs)
        d_start_ys = cuda.to_device(start_ys)
        d_goal_xs = cuda.to_device(goal_xs)
        d_goal_ys = cuda.to_device(goal_ys)
        # Reset output length array
        self.d_out_lengths.copy_to_device(np.zeros(num_queries, dtype=np.int32))

        # Launch kernel: one thread per query
        threads_per_block = 64
        blocks_per_grid = (num_queries + threads_per_block - 1) // threads_per_block
        astar_kernel[blocks_per_grid, threads_per_block](
            self.width, self.height,
            d_start_xs, d_start_ys, d_goal_xs, d_goal_ys,
            self.d_occupancy, self.d_stop, self.d_is_road, self.d_allowed_dirs,
            self.d_dist, self.d_came_from, self.d_steps,
            self.d_out_paths, self.d_out_lengths,
            soft_flag, ignore_flow_flag, respect_fov_flag, max_steps_val
        )
        # Retrieve results from device
        out_lengths = np.empty(num_queries, dtype=np.int32)
        self.d_out_lengths.copy_to_host(out_lengths)
        out_paths = np.empty((num_queries, self.width * self.height, 2), dtype=np.int32)
        self.d_out_paths.copy_to_host(out_paths)

        # Assign paths to vehicles
        for i, req in enumerate(self._pending_requests[:num_queries]):
            vehicle = req["vehicle"]
            length = out_lengths[i]
            if length <= 0:
                # No path found
                vehicle.path = None
            else:
                # Reconstruct path as list of CellAgents
                coords = [tuple(out_paths[i, j]) for j in range(length)]
                # The path computed by the kernel excludes the starting cell (start position), matching the expected behavior.
                vehicle.path = [self.city_model.get_cell_contents(x, y)[0] for (x, y) in coords]
        # Clear the processed requests
        self._pending_requests = self._pending_requests[num_queries:]
        # If any leftover requests beyond max_batch_size, they will be processed in subsequent calls (or adjust max_batch_size).

    def find_path(self, start_pos: tuple[int, int], goal_pos: tuple[int, int],
                  soft_obstacles=False, ignore_flow=False, respect_awareness=False, maximum_steps=0x7FFFFFFF) -> list[
        tuple[int, int]]:
        """
        Compute a single path from start_pos to goal_pos on the GPU synchronously.
        Returns a list of (x,y) coordinates from the cell after start to the goal (inclusive), or an empty list if no path found.
        """
        sx, sy = start_pos
        gx, gy = goal_pos
        # Prepare single-query data
        start_x = np.array([sx], dtype=np.int32)
        start_y = np.array([sy], dtype=np.int32)
        goal_x = np.array([gx], dtype=np.int32)
        goal_y = np.array([gy], dtype=np.int32)
        d_sx = cuda.to_device(start_x)
        d_sy = cuda.to_device(start_y)
        d_gx = cuda.to_device(goal_x)
        d_gy = cuda.to_device(goal_y)
        # Reset/initialize working arrays for one query
        # (Reuse self.d_dist, etc., for the first segment of the array)
        max_cells = self.width * self.height
        self.d_out_lengths.copy_to_device(np.array([0], dtype=np.int32))

        # 🚩 **Explicit call matching the kernel signature** 🚩
        awareness_range = Defaults.VEHICLE_AWARENESS_RANGE
        astar_kernel[1, 1](
            # Core grid dims
            self.width, self.height,
            # Start / Goal
            d_sx, d_sy, d_gx, d_gy,
            # Maps
            self.d_occupancy, self.d_stop,
            self.d_is_road, self.d_allowed_dirs,
            # Working buffers
            self.d_dist, self.d_came_from, self.d_steps,
            # Outputs
            self.d_out_paths, self.d_out_lengths,
            # Flags & parameters (in exact kernel order):
            respect_awareness,          # respect FOV?
            awareness_range,            # how wide is FOV?
            soft_obstacles,             # soft vs hard obstacles
            ignore_flow,                # allow contraflow?
            maximum_steps               # step‐limit
        )
        # Get result
        out_len = np.zeros(1, dtype=np.int32)
        self.d_out_lengths.copy_to_host(out_len)
        length = int(out_len[0])
        if length <= 0:
            return []  # no path
        out_path = np.zeros((1, max_cells, 2), dtype=np.int32)
        self.d_out_paths.copy_to_host(out_path)
        # Extract the coordinates from the output array
        coords = [tuple(out_path[0, j]) for j in range(length)]
        return coords