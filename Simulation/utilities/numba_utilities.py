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

@njit
def compute_cross_pressure(ns_coords: np.ndarray,
                           ew_coords: np.ndarray,
                           occupancy: np.ndarray):
    """
    ns_coords: shape (N,2) array of (x,y) for north–south in‐flow
    ew_coords: shape (M,2) array of (x,y) for east–west in‐flow
    occupancy: 2D grid of vehicle counts

    Returns (p_ns, p_ew) = (#NS_in – #EW_in, #EW_in – #NS_in)
    """
    local_ns = 0
    # sum occupancy at each NS coord
    for i in range(ns_coords.shape[0]):
        x = ns_coords[i, 0]
        y = ns_coords[i, 1]
        local_ns += occupancy[y, x]

    local_ew = 0
    # sum occupancy at each EW coord
    for j in range(ew_coords.shape[0]):
        x = ew_coords[j, 0]
        y = ew_coords[j, 1]
        local_ew += occupancy[y, x]

    # cross-difference
    p_ns = local_ns - local_ew
    p_ew = local_ew - local_ns

    return p_ns, p_ew

@njit
def compute_local_and_cross_pressure(
    ns_x: np.ndarray, ns_y: np.ndarray,
    ew_x: np.ndarray, ew_y: np.ndarray,
    occupancy: np.ndarray
):
    """
    ns_x, ns_y: 1-D int64 arrays of the north–south in-flow coords
    ew_x, ew_y: 1-D int64 arrays of the east–west in-flow coords
    occupancy:  2-D int array of vehicle counts

    Returns:
      local_ns: int (# vehicles on NS in-flow)
      local_ew: int (# vehicles on EW in-flow)
      p_ns:     local_ns – local_ew
      p_ew:     local_ew – local_ns
    """
    # sum up NS in-flow
    local_ns = 0
    for i in range(ns_x.shape[0]):
        local_ns += occupancy[ns_y[i], ns_x[i]]

    # sum up EW in-flow
    local_ew = 0
    for i in range(ew_x.shape[0]):
        local_ew += occupancy[ew_y[i], ew_x[i]]

    # cross-difference pressures
    p_ns = local_ns - local_ew
    p_ew = local_ew - local_ns

    return local_ns, local_ew, p_ns, p_ew

@njit
def avg_pressures_in_neighbors(
    pressures_ns: np.ndarray,
    pressures_ew: np.ndarray,
    occupancies: np.ndarray
):
    """
    Compute (possibly weighted) average of two pressure series.
    If all occupancies are zero, falls back to unweighted mean.
    """
    total_occ = 0.0
    sum_ns = 0.0
    sum_ew = 0.0

    n = pressures_ns.shape[0]
    for i in range(n):
        w = occupancies[i]
        total_occ += w
        sum_ns += pressures_ns[i] * w
        sum_ew += pressures_ew[i] * w

    if total_occ > 0.0:
        return sum_ns / total_occ, sum_ew / total_occ
    # fallback to simple average
    if n == 0:
        return 0.0, 0.0
    return sum_ns / n, sum_ew / n
