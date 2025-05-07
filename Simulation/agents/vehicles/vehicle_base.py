# vehicle_base.py – refactored step logic & vehicle‑block speed reset
from mesa import Agent
import random
import heapq
from typing import TYPE_CHECKING, cast

from numexpr.expressions import max_int32

from Simulation.config import Defaults
from Simulation.agents.cell import CellAgent
from Simulation.utilities import *

if TYPE_CHECKING:
    from Simulation.city_model import CityModel


class VehicleAgent(Agent):
    """
    A vehicle agent that navigates from a start to a destination using A* path‑finding.
    """

    # ════════════════════════════════════════════════════════════
    #  INIT / HELPERS
    # ════════════════════════════════════════════════════════════
    def __init__(self, custom_id, model, start_cell: CellAgent, target_cell: CellAgent):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.start_cell = start_cell
        self.target = target_cell
        self.previous_cell = None
        self.current_cell = None

        self.base_speed = 0      # «cruising» speed valid until a full stop
        self.current_speed = 0   # speed granted *this* tick
        self.direction: str | None = None

        self.is_overtaking = False
        self.overtake_path: list[CellAgent] | None = None

        self.is_parked = False
        self.is_in_collision = False
        self.is_in_malfunction = False

        self.remove_on_arrival = True

        self.base_color = Defaults.VEHICLE_BASE_COLOR

        self.city_model = cast("CityModel", self.model)

        self.city_model.place_vehicle(self,self.start_cell.get_position())

        # initial path (ignoring Stop‑blocks so we can park *on* them if needed)
        self.path: list[CellAgent] = self._compute_path(avoid_stops=True, avoid_vehicles=True)

    # ------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------
    def _compute_speed(self) -> int:
        """Return the speed allowed this tick, weather‑adjusted if needed."""
        # Establish / keep the persistent cruising speed.
        if self.base_speed == 0:
            self.base_speed = self._choose_new_speed()

        speed = self.base_speed

        # --- Weather adjustment (rain) --------------------------------
        if self.current_cell.is_raining:
            speed = max(1, speed - Defaults.RAIN_SPEED_REDUCTION)

        return speed

    def _choose_new_speed(self) -> int:
        """Pick a persistent cruising speed until the next full stop."""
        return random.randint(*(Defaults.VEHICLE_MIN_SPEED, Defaults.VEHICLE_MAX_SPEED))


    def _is_a_stop_cell(self, cell: CellAgent) -> bool:
        """Return *True* if *cell* carries a traffic‑control status set to "Stop"."""
        return getattr(cell, "status", "Pass") == "Stop"

    def _is_at_stopped_cell(self) -> bool:
        """Return *True* when the agent must wait because it sits on a red‑light cell."""
        if self._is_a_stop_cell(self.current_cell):
            return True
        return False

    def _path_from_positions(self, positions: list[tuple[int, int]]):
        return [self.city_model.get_cell_contents(pos[0], pos[1])[0] for pos in positions]

    def _neighbor_positions(self, pos: tuple[int, int]):
        x, y = pos
        cell = self.city_model.get_cell_contents(x, y)[0]
        for d in cell.directions:
            dx, dy = Defaults.DIRECTION_VECTORS[d]
            nx, ny = x + dx, y + dy
            if self.city_model.in_bounds(nx, ny):
                yield (nx, ny)

    def _positions_in_fov(self) -> list[tuple[int, int]]:
        """
        Return two lists of positions of stopped and occupied cells along lines in each of the four cardinal directions
        from the current position, stopping when the block type is no longer Defaults.ROADS.
        """
        city = cast("CityModel", self.model)
        cx, cy = self.current_cell.get_position()
        width = Defaults.VEHICLE_AWARENESS_WIDTH

        fov_positions: set[tuple[int, int]] = set()

        # Iterate over the four cardinal directions
        for dir_name in ['N', 'E', 'S', 'W']:
            dx, dy = Defaults.DIRECTION_VECTORS[dir_name]
            # Perpendicular vector for sideways offsets
            px, py = -dy, dx

            # Draw parallel lines based on width
            for offset in range(-width + 1, width):
                x = cx + offset * px
                y = cy + offset * py

                # Step along the direction until a non-road block is encountered
                while city.in_bounds(x, y):
                    fov_positions.add((x, y))
                    x += dx
                    y += dy

        return list(fov_positions)


    # ------------------------------------------------------------
    #  A* that first tries a *clean* path; if none exists, falls
    #  back to “least‑obstructed” where obstacles just add cost.
    # ------------------------------------------------------------
    def _compute_path(
        self,
        *,
        avoid_stops: bool = True,
        avoid_vehicles: bool = True,
        allow_contraflow_overtake: bool = False,
    ) -> list["CellAgent"]:
        """
        Plan a route to `self.target`.

        Phase1 – strict:
            • Vehicles inside FOV are impassable if `avoid_vehicles`.
            • Stops inside FOV are impassable if `avoid_stops`.

        Phase2 – penalty:
            • If Phase1 fails, the exact same search is rerun but
              obstacles are **passable** and add a large cost.
            • The resulting path therefore minimises
                  (# obstacles crossed, then length).

        Returns an empty list only when even penalty mode cannot
        reach the goal from the current cell.
        """

        def _astar(start=None, goal=None, soft_obstacles: bool = False, respect_fov: bool = False,
                   ignore_flow: bool = False, maximum_steps: int = max_int32) -> list["CellAgent"]:
            """Inner A*; soft_obstacles=True ⇒ obstacles add cost."""
            city = cast("CityModel", self.model)
            start = cast(tuple[int, int], self.current_cell.get_position()) if start is None else start
            goal = cast(tuple[int, int], self.target.get_position()) if goal is None else goal
            fov_positions = self._positions_in_fov() if avoid_vehicles else set()

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
                    return self._path_from_positions(path + [pos])[1:]

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
                    is_stop = self._is_a_stop_cell(ncell)
                    is_contraflow = d not in cell.directions

                    if is_contraflow:
                        if ignore_flow and ncell.cell_type in Defaults.ROADS:
                            ng += contraflow_penalty
                        else:
                            continue

                    if is_occupied and avoid_vehicles and (not respect_fov or is_in_fov):
                        if soft_obstacles:
                            ng += vehicle_penalty
                        else:
                            continue

                    if is_stop and avoid_stops and (not respect_fov or is_in_fov):
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

        # ─────────  Phase 1: strict avoidance  ─────────
        path = _astar(ignore_flow=False, soft_obstacles=False)
        if path is None or len(path) == 0:
            path = _astar(ignore_flow=False, soft_obstacles=True)

        if allow_contraflow_overtake:
            idx_stop, idx_vehicle = self._scan_ahead_for_obstacles(path=path)
            if idx_vehicle == 0:
                blocker = self._vehicle_on_cell(path[0])
                if blocker and (blocker.is_stranded() or blocker.is_parked):
                    # find first free cell past blocker
                    bypass_position = None
                    for cell in path:
                        if not cell.occupied:
                            bypass_position = cell.get_position()
                            break
                    if bypass_position is not None:
                        # compute bypass path ignoring vehicles and flow
                        bypass_path = _astar(
                            ignore_flow=True,
                            soft_obstacles=False,
                            goal=bypass_position,
                            maximum_steps=Defaults.VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS
                        )
                        # validate bypass_path
                        if bypass_path and len(bypass_path) > 1:
                            # ensure bypass_path starts at current and ends at bypass_position
                            if bypass_path[-1].get_position() == bypass_position:
                                # patch original path: replace segment up to bypass with bypass_path
                                idx_bp = next(
                                    i for i, c in enumerate(path)
                                    if c.get_position() == bypass_position
                                )
                                patched = bypass_path + path[idx_bp+1:]
                                self.overtake_path = bypass_path
                                self.is_overtaking = True
                                return patched

        return path


    # ────────────────────────────────────────────────────────────────

    def _scan_ahead_for_obstacles(self, path: list["CellAgent"] = None):
        """Return indices of the first Stop‑cell or occupied cell on the path."""
        if path is None:
            path = self.path

        idx_stop = idx_vehicle = None
        rng = min(Defaults.VEHICLE_AWARENESS_RANGE, len(path))
        for idx in range(rng):
            cell = path[idx]
            if idx_stop is None and self._is_a_stop_cell(cell):
                idx_stop = idx  # inclusive obstacle
            if idx_vehicle is None and cell.occupied:
                idx_vehicle = idx  # exclusive obstacle
            if (idx_stop == 0) or (idx_vehicle == 0):
                break
        return idx_stop, idx_vehicle

    def _recompute_path_on_obstacle(self):
        """
        Normal re‑route ➜ if that yields *zero* moves *and* the first
        obstacle is a stranded/parked car, do an expensive contraflow search.
        """
        if self.is_overtaking and (self.overtake_path is None or self.current_cell not in self.overtake_path):
            self.overtake_path = None
            self.is_overtaking = False

        idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()

        if self.is_overtaking:
            return idx_stop, idx_vehicle

        # —— First try the ordinary planner (same as before) ——
        if (idx_stop is not None) or (idx_vehicle is not None):
            path = self._compute_path(avoid_stops=True, avoid_vehicles=True)
            if path:                     # success → adopt and leave
                self.path = path
                idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()

        # —— We’re boxed in… is it a stranded / parked car? ——
        if idx_vehicle == 0 and Defaults.VEHICLE_CONTRAFLOW_OVERTAKE_ACTIVE: # obstacle is *next* cell
            veh = self._vehicle_on_cell(self.path[0])
            if veh and (veh.is_stranded() or veh.is_parked):
                path = self._compute_path(
                    avoid_stops=True,
                    avoid_vehicles=True,
                    allow_contraflow_overtake=True,
                )
                if path: # found a way round
                    self.path = path
                    idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()

        return idx_stop, idx_vehicle

    def _move_to(self, cell: CellAgent, new_pos: tuple[int, int], old_pos: tuple[int, int] = None):
        self.previous_cell = self.current_cell
        self.city_model.move_vehicle(self, cell, new_pos, old_pos)
        self.direction = self._compute_direction(old_pos, new_pos)

    def _compute_direction(self, old_pos: tuple[int, int], new_pos: tuple[int, int]):
        vector = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
        inv = {v: k for k, v in Defaults.DIRECTION_VECTORS.items()}
        return inv.get(vector, self.direction)

    def _set_collision(self, ticks: int):
        """Mark the vehicle stranded due to a collision."""
        self.is_in_collision = True
        self.is_in_malfunction = False
        self._stranded_ticks_remaining = ticks
        self.base_speed = 0

    def _set_malfunction(self, ticks: int):
        """Mark the vehicle stranded due to a malfunction."""
        self.is_in_malfunction = True
        self.is_in_collision = False
        self._stranded_ticks_remaining = ticks
        self.base_speed = 0

    def _tick_stranded(self) -> bool:
        if not self.is_stranded():
            return False
        self._stranded_ticks_remaining -= 1
        if self._stranded_ticks_remaining <= 0:
            self.is_in_collision = False
            self.is_in_malfunction = False
            self._stranded_ticks_remaining = 0
        return self.is_stranded()

    def _check_sideswipe_collision(self):
        if not Defaults.VEHICLE_SIDESWIPE_COLLISION_ACTIVE or not self.direction:
            return

        # build a correct “to my left” lookup
        DIRECTION_TO_THE_LEFT = {
            v: k for k, v in Defaults.DIRECTION_TO_THE_RIGHT.items()
        }

        right_dir = Defaults.DIRECTION_TO_THE_RIGHT[self.direction]
        left_dir  = DIRECTION_TO_THE_LEFT[self.direction]

        # look for same-direction traffic immediately to my left and right
        for lat_dir in (left_dir, right_dir):
            dx, dy = Defaults.DIRECTION_VECTORS[lat_dir]
            x, y   = self.current_cell.get_position()
            nx, ny = x + dx, y + dy

            if not self.city_model.in_bounds(nx, ny):
                continue

            for ag in self.city_model.get_cell_contents(nx, ny):
                # only sideswipe if they’re moving alongside me
                if isinstance(ag, VehicleAgent) and ag.direction is Defaults.DIRECTION_OPPOSITES[self.get_current_direction()]:
                    if random.random() >= Defaults.VEHICLE_SIDESWIPE_COLLISION_CHANCE:
                        return
                    self._set_collision(Defaults.VEHICLE_SIDESWIPE_COLLISION_DURATION)
                    ag._set_collision(Defaults.VEHICLE_SIDESWIPE_COLLISION_DURATION)
                    return


    def _check_malfunction(self):
        if not Defaults.VEHICLE_MALFUNCTION_ACTIVE or random.random() < Defaults.VEHICLE_MALFUNCTION_CHANCE:
            self._set_malfunction(Defaults.VEHICLE_MALFUNCTION_DURATION)

    # ════════════════════════════════════════════════════════════
    #  STEP  (public) – orchestrates one simulation tick
    # ════════════════════════════════════════════════════════════
    def step(self):
        """Advance the vehicle for one tick, obeying lights and traffic."""

        if self._tick_stranded():
            self.base_speed = 0
            return

        self._check_malfunction()
        if self.is_stranded():
            self.base_speed = 0
            return

        self._check_sideswipe_collision()
        if self.is_stranded():
            self.base_speed = 0
            return

        if self._is_at_stopped_cell():
            self.base_speed = 0
            return


        self.current_speed = self._compute_speed()

        idx_stop, idx_vehicle = self._recompute_path_on_obstacle()

        max_steps, blocked_by_vehicle = self._determine_max_steps(idx_stop, idx_vehicle)

        if max_steps <= 0:
            self.base_speed = 0
            return  # no movement possible this tick

        self._execute_movement(max_steps)

        if not self.path or len(self.path) == 0:
            self.path = self._compute_path(avoid_vehicles=True, avoid_stops=True)

        if self.current_cell is self.target:
            self.on_target_reached()

    # ------------------------------------------------------------
    #  STEP helper methods (private)
    # ------------------------------------------------------------

    def _vehicle_on_cell(self, cell: "CellAgent"):
        """Return the *VehicleAgent* occupying *cell* or *None*."""
        for ag in self.city_model.get_cell_contents(*cell.get_position()):
            if isinstance(ag, VehicleAgent):
                return ag
        return None



    def _determine_max_steps(self, idx_stop, idx_vehicle):
        """Return a tuple *(max_steps, blocked_by_vehicle)* for this tick."""
        max_steps = self.current_speed
        blocked_by_vehicle = False

        if idx_stop is not None:
            max_steps = min(max_steps, idx_stop + 1)  # allowed to *enter* Stop cell

        if idx_vehicle is not None:
            if idx_vehicle == 0:
                blocked_by_vehicle = True        # cannot even enter the next cell
            max_steps = min(max_steps, idx_vehicle)  # stop *before* occupied cell
        return max_steps, blocked_by_vehicle

    def _execute_movement(self, max_steps: int):
        """Move up to *max_steps* cells along `self.path`"""
        old_pos = self.current_cell.get_position()

        for step_idx in range(max_steps):
            if not self.path:
                break  # already at destination
            next_cell = self.path[0]

            # Safety re‑checks at runtime
            if next_cell.occupied and next_cell is not self.current_cell:
                break
            if self._is_a_stop_cell(next_cell):
                # May *enter* the Stop cell only as the very last permitted step
                if next_cell is not self.current_cell and step_idx != max_steps - 1:
                    break

            new_pos = next_cell.get_position()
            self._move_to(next_cell, new_pos, old_pos)
            old_pos = new_pos
            self.path.pop(0)

    def on_target_reached(self) -> None:
        """Invoked exactly once when the vehicle enters *target*.

        If *remove_on_arrival* is *True*, the vehicle is despawned; otherwise
        it simply stops in its target cell.
        """
        if self.remove_on_arrival:
            self._despawn()
        else:
            self._park()

    def _despawn(self) -> None:
        """Remove this agent from the grid and scheduler."""
        self.city_model.remove_vehicle(self)

    def _park(self) -> None:
        """Leave the vehicle in place but mark it as inactive."""
        self.is_parked = True
        pass

    def _get_base_color(self):
        if self.is_overtaking:
            return Defaults.VEHICLE_CONTRAFLOW_OVERTAKE_COLOR
        else:
            return self.base_color

    def get_vehicle_type_name(self):
        return "Vehicle"

    def is_stranded(self):
        return self.is_in_collision or self.is_in_malfunction

    def get_current_direction(self):
        return self.direction

    def get_current_path(self):
        return [agent.get_position() for agent in self.path]

    def get_target(self):
        return self.target.get_position()

    def get_description(self):
        return f"A Base {self.get_vehicle_type_name()}"

    def get_portrayal(self):

        direction_arrow = Defaults.DIRECTION_ICONS.get(self.get_current_direction(), '?')

        portrayal = {
            "Shape": "circle", "r": 0.66, "Filled": True,
            "Color": self._get_base_color(),
            "Layer": 1,
            "Type": self.get_vehicle_type_name(),
            "Identifier": self.unique_id,
            "Position": self.current_cell.get_position(),
            "Description": self.get_description(),

            "Direction": direction_arrow,
            "Speed": self.current_speed,
            "Overtaking": self.is_overtaking,
        }

        flash_on = (self.city_model.step_count % 2) == 0
        if self.is_in_collision:
            col_alt = Defaults.VEHICLE_COLLISION_COLOR
            portrayal["Color"] = self._get_base_color() if flash_on else col_alt
        elif self.is_in_malfunction:
            mal_alt = Defaults.VEHICLE_MALFUNCTION_COLOR
            portrayal["Color"] = self._get_base_color() if flash_on else mal_alt
        elif self.is_parked:
            park_alt = Defaults.VEHICLE_PARKED_COLOR
            portrayal["Color"] = self._get_base_color() if flash_on else park_alt
        else:
            portrayal["Color"] = self._get_base_color()

        return portrayal
