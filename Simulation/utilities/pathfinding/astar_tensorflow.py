"""
astar_tensorflow.py ───────────────────────────────────────────────────────────
GPU-optional A* path-finding with identical behaviour to the original Numba
implementation.

▪ Uses TensorFlow 2.x  (GPU if available -- TF automatically places tensors on
  the first CUDA device; otherwise it runs on CPU).
▪ Keeps every feature:
      • contraflow / one-way enforcement
      • turn-penalty
      • soft vs. hard dynamic obstacles
      • density-scaled vehicle penalty
      • stop-sign / red-light penalty
      • road-type penalties (R1/R2/R3)
      • awareness-limited field-of-view
▪ Signature matches astar_numba.astar_numba so VehicleAgent code needs no edit.
"""

from __future__ import annotations
import heapq
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from Simulation.config import Defaults

# ──────────────────────────────────────────────────────────────────────────────
#  Constants and helpers
# ──────────────────────────────────────────────────────────────────────────────
NEIGHBOUR_DELTAS: list[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N,E,S,W
DIR_MASKS = [1 << 0, 1 << 1, 1 << 2, 1 << 3]                                   # N,E,S,W → bits

INF = 0x3F3F3F3F

# penalties
CONTRA_PENALTY = Defaults.VEHICLE_CONTRAFLOW_PENALTY
TURN_PENALTY_ENABLED = Defaults.VEHICLE_TURN_PENALTY_ENABLED
TURN_PENALTY = Defaults.VEHICLE_TURN_PENALTY

ROAD_PEN_ENABLED = Defaults.VEHICLE_ROAD_TYPES_PENALTIES_ENABLED
PEN_R1 = int(Defaults.VEHICLE_ROAD_TYPES_PENALTY_R1)
PEN_R2 = int(Defaults.VEHICLE_ROAD_TYPES_PENALTY_R2)
PEN_R3 = int(Defaults.VEHICLE_ROAD_TYPES_PENALTY_R3)

VEHICLE_PENALTY = Defaults.VEHICLE_OBSTACLE_PENALTY_VEHICLE
STOP_PENALTY    = Defaults.VEHICLE_OBSTACLE_PENALTY_STOP
DYN_ENABLED     = Defaults.VEHICLE_DYNAMIC_PENALTIES_ENABLED
DYN_SCALE       = Defaults.VEHICLE_DYNAMIC_PENALTY_SCALE


# ──────────────────────────────────────────────────────────────────────────────
#  Field-of-view mask  (NumPy → very small, no need for TF/GPU)
# ──────────────────────────────────────────────────────────────────────────────
def _compute_fov_mask(cx: int, cy: int,
                      awareness: int,
                      is_road: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask (1=within FOV) that covers the visible road cells
    along the four cardinal directions within ±awareness offsets.
    """
    h, w = is_road.shape
    mask = np.zeros_like(is_road, dtype=np.int8)

    for d, (dx, dy) in enumerate(NEIGHBOUR_DELTAS):
        px, py = -dy, dx                         # perpendicular for lateral offsets
        for offset in range(-awareness + 1, awareness):
            x0, y0 = cx + offset * px, cy + offset * py
            step = 0
            x, y = x0, y0
            while 0 <= x < w and 0 <= y < h and is_road[y, x]:
                mask[y, x] = 1
                step += 1
                x, y = x0 + dx * step, y0 + dy * step
    return mask


# ──────────────────────────────────────────────────────────────────────────────
#  Main A*  (heap-based, costs computed with *TensorFlow tensors*)
# ──────────────────────────────────────────────────────────────────────────────
def astar_tensorflow(
    width: int,
    height: int,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    occupancy_map: np.ndarray,
    stop_map: np.ndarray,
    is_road_map: np.ndarray,
    road_type_map: np.ndarray,
    allowed_dirs_map: np.ndarray,
    respect_awareness: bool,
    awareness_range: int,
    density_map: np.ndarray,
    soft_obstacles: bool,
    ignore_flow: bool,
    maximum_steps: int = INF,
) -> List[Tuple[int, int]]:
    """
    Identical call-signature to *astar_numba.astar_numba*.

    Returns a list of (x,y) path cells **excluding** the start cell.
    """

    # ── fast-exit ───────────────────────────────────────────────────────────
    if (start_x, start_y) == (goal_x, goal_y):
        return []

    # ── Awareness mask ─────────────────────────────────────────────────────
    if respect_awareness:
        fov_mask = _compute_fov_mask(start_x, start_y,
                                     awareness_range, is_road_map)
        occ_map = occupancy_map * fov_mask
        stp_map = stop_map * fov_mask
    else:
        occ_map = occupancy_map
        stp_map = stop_map

    # ── Convert every static map → tf.Tensor ONCE (placed on GPU if present) ─
    #    The tensors are kept read-only; only small scalars are updated in Python.
    occ_tf     = tf.constant(occ_map.reshape(-1),     dtype=tf.int32)
    stop_tf    = tf.constant(stp_map.reshape(-1),     dtype=tf.int32)
    road_tf    = tf.constant(is_road_map.reshape(-1), dtype=tf.int32)
    type_tf    = tf.constant(road_type_map.reshape(-1), dtype=tf.int32)
    dirs_tf    = tf.constant(allowed_dirs_map.reshape(-1), dtype=tf.int32)
    dens_tf    = tf.constant(density_map.reshape(-1), dtype=tf.float32)

    # ── Heuristic (Manhattan) on GPU - cheap to precompute ──────────────────
    xs = tf.range(width, dtype=tf.int32)
    ys = tf.range(height, dtype=tf.int32)
    manhat = tf.reshape(tf.abs(xs - goal_x)[None, :] +
                        tf.abs(ys[:, None] - goal_y), [-1])

    # ── host-side cost / parent / dir arrays (NumPy for easy mutability) ────
    n = width * height
    g_cost   = np.full(n, INF, dtype=np.int32)
    came_from = np.full(n, -1,  dtype=np.int32)
    dir_arr   = np.full(n, -1,  dtype=np.int8)      # direction *into* this node
    steps_arr = np.full(n, INF, dtype=np.int32)

    start_idx = start_y * width + start_x
    goal_idx  = goal_y * width + goal_x

    g_cost[start_idx]   = 0
    steps_arr[start_idx] = 0

    # ── open-set as heap of (f, g, idx) ─────────────────────────────────────
    open_heap: list[Tuple[int, int, int, int]] = []
    f_start = int(manhat.numpy()[start_idx])
    heapq.heappush(open_heap, (f_start, 0, 0, start_idx))   # (f, g, steps, idx)

    # ── Main loop ───────────────────────────────────────────────────────────
    while open_heap:
        f_curr, g_curr, s_curr, idx = heapq.heappop(open_heap)
        if idx == goal_idx:                               # goal reached
            break

        if g_curr > g_cost[idx]:                          # stale heap entry
            continue
        if s_curr >= maximum_steps:                       # step-limit
            continue

        cx, cy = idx % width, idx // width
        prev_dir = dir_arr[idx]

        # Expand 4 neighbours
        for d, (dx, dy) in enumerate(NEIGHBOUR_DELTAS):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nidx = ny * width + nx
            if steps_arr[idx] + 1 >= maximum_steps:
                continue

            # ── base move cost = 1
            ng = g_curr + 1

            # ── turn penalty
            if TURN_PENALTY_ENABLED and prev_dir != -1 and d != prev_dir:
                ng += TURN_PENALTY

            # ── flow / contraflow rules
            cell_bits = dirs_tf[idx].numpy()
            allowed = (cell_bits & DIR_MASKS[d]) != 0
            if not allowed:
                if ignore_flow and road_tf[nidx].numpy() == 1:
                    ng += CONTRA_PENALTY
                else:
                    continue  # blocked

            # ── dynamic occupancy / stop handling
            occ = occ_tf[nidx].numpy()
            stp = stop_tf[nidx].numpy()

            if occ == 1:                                  # occupied by vehicle
                if soft_obstacles:
                    pen = VEHICLE_PENALTY
                    if DYN_ENABLED:
                        dens = dens_tf[nidx].numpy()
                        pen = int(pen * (1.0 + DYN_SCALE * dens))
                    ng += pen
                else:
                    continue                              # hard obstacle

            if stp == 1:                                  # stop-sign / red
                if soft_obstacles:
                    ng += STOP_PENALTY
                else:
                    continue

            # ── road-type penalty
            if ROAD_PEN_ENABLED and road_tf[nidx].numpy() == 1:
                rt = type_tf[nidx].numpy()
                if rt == 1:   ng += PEN_R1
                elif rt == 2: ng += PEN_R2
                elif rt == 3: ng += PEN_R3

            # ── relax edge
            if ng < g_cost[nidx]:
                g_cost[nidx]   = ng
                came_from[nidx] = idx
                dir_arr[nidx]   = d
                steps_arr[nidx] = s_curr + 1
                f = ng + int(manhat.numpy()[nidx])
                heapq.heappush(open_heap, (f, ng, s_curr + 1, nidx))

    # ── reconstruct path ────────────────────────────────────────────────────
    path: list[Tuple[int, int]] = []
    if came_from[goal_idx] == -1:                 # goal unreachable
        return path

    idx = goal_idx
    while idx != start_idx and idx != -1:
        x, y = idx % width, idx // width
        path.append((x, y))
        idx = came_from[idx]
    path.reverse()
    return path
