# numba_utilities.py – Numba-accelerated helpers
import numpy as np
from numba import njit

DIRECTION_VECTORS = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0)
}

REVERSE_DIRECTION_VECTORS = {v: k for k, v in DIRECTION_VECTORS.items()}

@njit
def compute_direction(old_pos, new_pos):
    dx = new_pos[0] - old_pos[0]
    dy = new_pos[1] - old_pos[1]

    if dx == 0 and dy == 1:
        return 0  # N
    elif dx == 1 and dy == 0:
        return 1  # E
    elif dx == 0 and dy == -1:
        return 2  # S
    elif dx == -1 and dy == 0:
        return 3  # W
    else:
        return -1  # Unknown or diagonal

@njit
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


@njit
def scan_ahead_for_obstacles_jit(path_array: np.ndarray, stop_map: np.ndarray, occupancy_map: np.ndarray, awareness_range: int):
    """
    JIT-compiled version of VehicleAgent._scan_ahead_for_obstacles using a numpy array input.
    path_array: 2D np.ndarray of shape (N, 2) with dtype int64
    stop_map: 2D numpy array
    occupancy_map: 2D numpy array
    awareness_range: int
    Returns: (idx_stop, idx_vehicle) where -1 means None.
    """
    idx_stop = -1
    idx_vehicle = -1
    # limit lookahead
    n = path_array.shape[0]
    max_lookahead = awareness_range if n > awareness_range else n
    for idx in range(max_lookahead):
        x = path_array[idx, 0]
        y = path_array[idx, 1]
        # first stop cell
        if idx_stop < 0 and stop_map[y, x] == 1:
            idx_stop = idx
        # first vehicle obstacle
        if idx_vehicle < 0 and occupancy_map[y, x] == 1:
            idx_vehicle = idx
        # break early if immediate
        if idx_stop == 0 or idx_vehicle == 0:
            break
    return idx_stop, idx_vehicle


@njit
def compute_approach_queue(occupancy_map: np.ndarray, lane_cells: np.ndarray):
    queue = 0
    for i in range(lane_cells.shape[0]):
        x = lane_cells[i, 0]
        y = lane_cells[i, 1]
        queue += occupancy_map[y, x]
    return queue

@njit
def compute_max_pressure(ns_in: np.ndarray, ns_out: np.ndarray,
                         ew_in: np.ndarray, ew_out: np.ndarray,
                         occupancy_map: np.ndarray):
    ns_in_q = compute_approach_queue(occupancy_map, ns_in)
    ns_out_q = compute_approach_queue(occupancy_map, ns_out)
    ew_in_q = compute_approach_queue(occupancy_map, ew_in)
    ew_out_q = compute_approach_queue(occupancy_map, ew_out)
    ns_pressure = ns_in_q - ns_out_q
    ew_pressure = ew_in_q - ew_out_q
    return ns_pressure, ew_pressure

@njit
def compute_total_queue(lane_cells: np.ndarray, occupancy_map: np.ndarray):
    return compute_approach_queue(occupancy_map, lane_cells)

