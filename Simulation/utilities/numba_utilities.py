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
