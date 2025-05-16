import heapq
from typing import cast

from numexpr.expressions import max_int32

from Simulation.config import Defaults


def _path_from_positions(city, positions: list[tuple[int, int]]):
    return [city.get_cell_contents(pos[0], pos[1])[0] for pos in positions]

def _is_a_stop_cell(cell) -> bool:
    """Return *True* if *cell* carries a traffic‑control status set to "Stop"."""
    return getattr(cell, "status", "Pass") == "Stop"

def astar_python(city, start, goal, fov_positions, respect_fov: bool = False, soft_obstacles: bool = False,
                 ignore_flow: bool = False, maximum_steps: int = max_int32) -> list["CellAgent"]:
    """Inner A*; soft_obstacles=True ⇒ obstacles add cost."""


    def h(a: tuple[int, int]) -> int:
        # Manhattan heuristic
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    # open_set entry: (f, g, steps, pos, path)
    open_set: list[tuple[int, int, int, tuple[int, int], list[tuple[int, int]]]] = []
    # start with cost=0, steps=0
    heapq.heappush(open_set, (h(start), 0, 0, start, []))
    seen: dict[tuple[int, int], int] = {start: 0}

    while open_set:
        f, g, steps, pos, path = heapq.heappop(open_set)
        if pos == goal:
            return _path_from_positions(city, path + [pos])[1:]

        x, y = pos
        cell = city.get_cell_contents(x, y)[0]
        directions = Defaults.AVAILABLE_DIRECTIONS

        for d in directions:
            nx, ny = city.next_cell_in_direction(x, y, d)
            if not city.in_bounds(nx, ny):
                continue

            # pure step count
            current_steps = steps + 1
            if current_steps > maximum_steps:
                continue

            npos = (nx, ny)
            ncell = city.get_cell_contents(nx, ny)[0]

            # base cost: +1 for the move
            ng = g + 1

            contraflow_penalty = Defaults.VEHICLE_CONTRAFLOW_PENALTY
            vehicle_penalty = Defaults.VEHICLE_OBSTACLE_PENALTY_VEHICLE
            stop_penalty = Defaults.VEHICLE_OBSTACLE_PENALTY_STOP

            # ─ Obstacles inside FOV ──────────────────────
            is_in_fov = npos in fov_positions
            is_occupied = ncell.occupied
            is_stop = _is_a_stop_cell(ncell)
            is_contraflow = d not in cell.directions

            if is_contraflow:
                if ignore_flow and ncell.cell_type in Defaults.ROADS:
                    ng += contraflow_penalty
                else:
                    continue

            if is_occupied and (not respect_fov or is_in_fov):
                if soft_obstacles:
                    ng += vehicle_penalty
                else:
                    continue

            if is_stop and (not respect_fov or is_in_fov):
                if soft_obstacles:
                    ng += stop_penalty
                else:
                    continue

            # record if this path is better cost-wise
            if npos not in seen or ng < seen[npos]:
                seen[npos] = ng
                heapq.heappush(
                    open_set,
                    (ng + h(npos),  # f = cost + heuristic
                     ng,  # new cost
                     current_steps,  # pure steps so far
                     npos,
                     path + [pos])
                )

    return []  # goal unreachable in this mode