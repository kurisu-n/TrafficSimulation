import numpy as np
from numba import njit, int32, int64, float64
from numba.core.types import UniTuple, int64
from numba.typed import List, Dict

from Simulation.config import Defaults

# Direction deltas for N, E, S, W
NEIGHBOR_DELTAS = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=np.int32)
# Penalty constants
CONTRA_PENALTY = Defaults.VEHICLE_CONTRAFLOW_PENALTY
VEHICLE_PENALTY = Defaults.VEHICLE_OBSTACLE_PENALTY_VEHICLE
STOP_PENALTY = Defaults.VEHICLE_OBSTACLE_PENALTY_STOP
ROAD_TYPES_PENALTY_ENABLED = Defaults.VEHICLE_ROAD_TYPES_PENALTIES_ENABLED

ROAD_TYPES_PENALTY_R1 = Defaults.VEHICLE_ROAD_TYPES_PENALTY_R1
ROAD_TYPES_PENALTY_R2 = Defaults.VEHICLE_ROAD_TYPES_PENALTY_R2
ROAD_TYPES_PENALTY_R3 = Defaults.VEHICLE_ROAD_TYPES_PENALTY_R3

VEHICLE_TURN_PENALTY = Defaults.VEHICLE_TURN_PENALTY
VEHICLE_TURN_PENALTY_ENABLED = Defaults.VEHICLE_TURN_PENALTY_ENABLED

VEHICLE_DYNAMIC_PENALTIES_ENABLED = Defaults.VEHICLE_DYNAMIC_PENALTIES_ENABLED
VEHICLE_DYNAMIC_PENALTY_SCALE = Defaults.VEHICLE_DYNAMIC_PENALTY_SCALE

NodeTuple = UniTuple(int64, 4)
CoordTuple = UniTuple(int64, 2)

@njit(inline='always')
def compute_fov_inplace(cx: int, cy: int, height: int, width: int,
                        awareness: int, is_road_map: np.ndarray, fov_map: np.ndarray):
    # reset FOV mask
    for yy in range(height):
        for xx in range(width):
            fov_map[yy, xx] = 0
    # four cardinal directions
    for d in range(4):
        dx = NEIGHBOR_DELTAS[d, 0]
        dy = NEIGHBOR_DELTAS[d, 1]
        px = -dy; py = dx
        for offset in range(-awareness + 1, awareness):
            x0 = cx + offset * px
            y0 = cy + offset * py
            step = 0
            x = x0; y = y0
            while 0 <= x < width and 0 <= y < height and is_road_map[y, x] == 1:
                fov_map[y, x] = 1
                step += 1
                x = x0 + dx * step
                y = y0 + dy * step

@njit(inline='always')
def heap_sift_up(f_arr: np.ndarray, g_arr: np.ndarray, s_arr: np.ndarray, i_arr: np.ndarray, i: int):
    while i > 0:
        parent = (i - 1) // 2
        if f_arr[i] < f_arr[parent]:
            # swap elements
            f_arr[i], f_arr[parent] = f_arr[parent], f_arr[i]
            g_arr[i], g_arr[parent] = g_arr[parent], g_arr[i]
            s_arr[i], s_arr[parent] = s_arr[parent], s_arr[i]
            i_arr[i], i_arr[parent] = i_arr[parent], i_arr[i]
            i = parent
        else:
            break

@njit(inline='always')
def heap_sift_down(f_arr: np.ndarray, g_arr: np.ndarray, s_arr: np.ndarray, i_arr: np.ndarray, size: int):
    idx = 0
    while True:
        left = 2 * idx + 1
        right = left + 1
        smallest = idx
        if left < size and f_arr[left] < f_arr[smallest]:
            smallest = left
        if right < size and f_arr[right] < f_arr[smallest]:
            smallest = right
        if smallest != idx:
            # swap
            f_arr[idx], f_arr[smallest] = f_arr[smallest], f_arr[idx]
            g_arr[idx], g_arr[smallest] = g_arr[smallest], g_arr[idx]
            s_arr[idx], s_arr[smallest] = s_arr[smallest], s_arr[idx]
            i_arr[idx], i_arr[smallest] = i_arr[smallest], i_arr[idx]
            idx = smallest
        else:
            break

@njit
def astar_core(width: int, height: int,
               start_x: int, start_y: int,
               goal_x: int, goal_y: int,
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
               maximum_steps: int,
               fov_map: np.ndarray,
               f_arr: np.ndarray,
               g_arr: np.ndarray,
               s_arr: np.ndarray,
               i_arr: np.ndarray,
               dir_arr: np.ndarray,
               dist: np.ndarray,
               came_from: np.ndarray) -> List[CoordTuple]:
    """
    A* με penalties ανά τύπο δρόμου και επιπλέον TURN_PENALTY.
    """
    INF = 0x3F3F3F3F
    n = width * height
    start_idx = start_y * width + start_x
    goal_idx  = goal_y * width + goal_x

    # αρχικοποίηση
    for i in range(n):
        dist[i]      = INF
        came_from[i] = -1
        dir_arr[i]   = -1      # -1: χωρίς προηγούμενη κατεύθυνση
    dist[start_idx] = 0

    # heap
    heap_size = 1
    h0 = abs(start_x - goal_x) + abs(start_y - goal_y)
    f_arr[0] = h0; g_arr[0] = 0; s_arr[0] = 0; i_arr[0] = start_idx; dir_arr[0] = -1

    # FOV
    if respect_awareness:
        compute_fov_inplace(start_x, start_y, height, width,
                            awareness_range, is_road_map, fov_map)

    # main loop
    while heap_size > 0:
        # pop
        f = f_arr[0]; g = g_arr[0]; steps = s_arr[0]; current_idx = i_arr[0]
        prev_dir = dir_arr[0]
        heap_size -= 1
        if heap_size > 0:
            # replace root και sift down
            f_arr[0] = f_arr[heap_size]
            g_arr[0] = g_arr[heap_size]
            s_arr[0] = s_arr[heap_size]
            i_arr[0] = i_arr[heap_size]
            dir_arr[0] = dir_arr[heap_size]
            heap_sift_down(f_arr, g_arr, s_arr, i_arr, heap_size)

        # αν φτάσαμε
        if current_idx == goal_idx:
            path = List.empty_list(CoordTuple)
            idx = current_idx
            while idx != start_idx:
                x = idx % width; y = idx // width
                path.append((x, y))
                idx = came_from[idx]
            # αντίστροφη
            result = List.empty_list(CoordTuple)
            for j in range(len(path)-1, -1, -1):
                result.append(path[j])
            return result

        # skip αν worse
        if g > dist[current_idx]:
            continue

        cx = current_idx % width; cy = current_idx // width

        # γείτονες
        for d in range(4):
            nx = cx + NEIGHBOR_DELTAS[d,0]
            ny = cy + NEIGHBOR_DELTAS[d,1]
            if nx<0 or nx>=width or ny<0 or ny>=height:
                continue
            ns = steps + 1
            if ns > maximum_steps:
                continue

            nidx = ny*width + nx
            ng = g + 1

            # TURN_PENALTY
            if VEHICLE_TURN_PENALTY_ENABLED and prev_dir != -1 and d != prev_dir:
                ng += VEHICLE_TURN_PENALTY

            # flow check
            bits = allowed_dirs_map[cy, cx]
            if (bits & (1<<d)) == 0:
                if ignore_flow and is_road_map[ny, nx]==1:
                    ng += CONTRA_PENALTY
                else:
                    continue

            # occupancy
            if occupancy_map[ny, nx]==1 and (not respect_awareness or fov_map[ny, nx]==1):
                if soft_obstacles and VEHICLE_DYNAMIC_PENALTIES_ENABLED:
                    p = VEHICLE_PENALTY
                    local_density = density_map[ny, nx]
                    p = int(p * (1.0 + VEHICLE_DYNAMIC_PENALTY_SCALE * local_density))
                    ng += p
                elif soft_obstacles:
                    ng += VEHICLE_PENALTY
                else:
                    continue

            # stop
            if stop_map[ny, nx]==1 and (not respect_awareness or fov_map[ny, nx]==1):
                if soft_obstacles:
                    ng += STOP_PENALTY
                else:
                    continue

            # ROAD TYPES penalty
            if ROAD_TYPES_PENALTY_ENABLED and is_road_map[ny, nx] == 1:
                rt = road_type_map[ny, nx]
                if rt == 1:
                    ng += ROAD_TYPES_PENALTY_R1
                elif rt == 2:
                    ng += ROAD_TYPES_PENALTY_R2
                elif rt == 3:
                    ng += ROAD_TYPES_PENALTY_R3

            # relax
            if ng < dist[nidx]:
                dist[nidx]      = ng
                came_from[nidx] = current_idx
                # heap push
                h = abs(nx - goal_x) + abs(ny - goal_y)
                i = heap_size
                f_arr[i]   = ng + h
                g_arr[i]   = ng
                s_arr[i]   = ns
                i_arr[i]   = nidx
                dir_arr[i] = d
                heap_sift_up(f_arr, g_arr, s_arr, i_arr, i)
                heap_size += 1

    return List.empty_list(CoordTuple)

# Python wrapper to pre-allocate buffers

def astar_numba(width: int, height: int,
                start_x: int, start_y: int,
                goal_x: int, goal_y: int,
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
                maximum_steps: int = 0x7FFFFFFF) -> List[CoordTuple]:
    """
    Wrapper για astar_core με TURN_PENALTY και ROAD_TYPES_PENALTY.
    """
    size     = width * height
    fov_map  = np.zeros((height, width), np.int8)
    f_arr    = np.empty(size, np.int32)
    g_arr    = np.empty(size, np.int32)
    s_arr    = np.empty(size, np.int32)
    i_arr    = np.empty(size, np.int32)
    dir_arr  = np.empty(size, np.int8)
    dist     = np.empty(size, np.int32)
    came_from= np.empty(size, np.int32)

    return astar_core(width, height,
                      start_x, start_y,
                      goal_x, goal_y,
                      occupancy_map, stop_map,
                      is_road_map, road_type_map,
                      allowed_dirs_map,
                      respect_awareness, awareness_range,
                      density_map,
                      soft_obstacles, ignore_flow,
                      maximum_steps,
                      fov_map, f_arr, g_arr, s_arr, i_arr,
                      dir_arr, dist, came_from)
