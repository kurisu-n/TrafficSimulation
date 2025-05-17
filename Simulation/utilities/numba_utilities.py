# numba_utilities.py – Numba-accelerated helpers

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