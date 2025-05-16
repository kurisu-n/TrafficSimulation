# vehicle_base.py – refactored step logic & vehicle‑block speed reset
from mesa import Agent
import random
from typing import TYPE_CHECKING, cast

import numpy as np
from numba import float32, int32
from numba.typed import List, Dict

from Simulation.config import Defaults
from Simulation.agents.city_structure_entities.cell import CellAgent
from Simulation.utilities.pathfinding.astar_jit import astar_numba
from Simulation.utilities.general import *
from Simulation.utilities.pathfinding.astar_python import astar_python

if TYPE_CHECKING:
    from Simulation.city_model import CityModel

class VehicleAgent(Agent):
    """
    A vehicle agent that navigates from a start to a destination using A* path‑finding.
    """

    # ════════════════════════════════════════════════════════════
    #  INIT / HELPERS
    # ════════════════════════════════════════════════════════════
    def __init__(self, custom_id, model, start_cell: CellAgent, target_cell: CellAgent, population_type = None):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.start_cell = start_cell
        self.target = target_cell
        self.previous_cell = None
        self.current_cell = None
        self.population_type = population_type

        self.base_speed = 0      # «cruising» speed valid until a full stop
        self.current_speed = 0   # speed granted *this* tick
        self.direction: str | None = None

        self.is_overtaking = False
        self.overtake_path: list[CellAgent] | None = None
        self.overtaking_duration: int = -1

        self.is_parked = False
        self.is_in_collision = False
        self.is_in_malfunction = False

        self.remove_on_arrival = True

        self.base_color = Defaults.VEHICLE_BASE_COLOR

        self.city_model = cast("CityModel", self.model)


        if Defaults.ENABLE_TRAFFIC:
            self.depart_time: float = getattr(self.city_model.dynamic_traffic_generator, "elapsed", 0.0)
        else:
            self.depart_time: float = 0.0

        self.steps_traveled: int = 0

        self.city_model.place_vehicle(self,self.start_cell.get_position())

        self.path: list[CellAgent] = []  # ❶ always present
        self.path = self._compute_path()

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


    # --------------------------------------------------------------------------- #
    #  compute_path wrapper – *inside* VehicleAgent
    # --------------------------------------------------------------------------- #

    def _compute_path(self,* args, **kwargs) -> list[CellAgent] | None | list[Agent | None]:
        """
        Wrapper for the path planner. Calls the JIT-compiled A* algorithm.
        """

        if Defaults.PATHFINDING_METHOD == "CUDA":
            return self._compute_path_cuda(*args, **kwargs)
        elif Defaults.PATHFINDING_METHOD == "JIT":
            return self._compute_path_jit()
        else:
            return self._compute_path_python()


    def _compute_path_python(
        self
    ) -> list["CellAgent"]:

        city = cast("CityModel", self.model)
        default_start = cast(tuple[int, int], self.current_cell.get_position())
        default_goal = cast(tuple[int, int], self.target.get_position())
        fov_positions = self._positions_in_fov()

        # ─────────  Phase 1: strict avoidance  ─────────
        path = astar_python(city, default_start, default_goal, fov_positions,
                            soft_obstacles= False, ignore_flow = False,)

        # ─────────  Phase 2: allow soft obstacles if no path  ─────────
        if path is None or len(path) == 0:
            path = astar_python(city, default_start, default_goal, fov_positions,
                            soft_obstacles=True, ignore_flow=False)

        # ─────────  Phase 3: Contraflow‐overtake bypass (ignore flow)  ─────────
        if Defaults.VEHICLE_CONTRAFLOW_OVERTAKE_ACTIVE:
            idx_stop, idx_vehicle = self._scan_ahead_for_obstacles(path)
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
                        bypass_path = astar_python(
                            city, default_start, bypass_position, fov_positions,
                            soft_obstacles=False, ignore_flow=True,
                            maximum_steps=Defaults.VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS
                        )
                        # validate bypass_path
                        if (bypass_path and len(bypass_path) > 1 and bypass_path[-1].get_position() == bypass_position):
                                # patch original path: replace segment up to bypass with bypass_path
                                idx_bp = next(
                                    i for i, c in enumerate(path)
                                    if c.get_position() == bypass_position)

                                patched = bypass_path + path[idx_bp+1:]
                                self.overtake_path = bypass_path
                                self.is_overtaking = True
                                self.overtaking_duration = 0
                                return patched

        return path

    def _compute_path_jit(self) -> list[CellAgent]:
        """
        JIT wrapper using cached flow_mask from CityModel:
          1. Phase 1: strict avoidance (hard‐block vehicles & stops, respect flow)
          2. Phase 2: allow soft obstacles (penalty) if Phase 1 fails
          3. Optional contraflow‐overtake bypass: ignore flow, allow soft obstacles
        """
        # ── Data Prep ───────────────────────────────────────────────────────
        sx, sy = self.current_cell.get_position()
        gx, gy = self.target.get_position()

        grid = self.city_model.export_to_grid(dtype=np.int8)
        for vx, vy in self.city_model.occupied_vehicle_positions():
            grid[vx, vy] = 1
        for tx, ty in self.city_model.stop_cell_positions():
            grid[tx, ty] = 2

        flow_mask = self.city_model.get_flow_mask()

        penalty_vehicle = float32(Defaults.VEHICLE_OBSTACLE_PENALTY_VEHICLE)
        penalty_stop = float32(Defaults.VEHICLE_OBSTACLE_PENALTY_STOP)
        penalty_contraflow = float32(Defaults.VEHICLE_CONTRAFLOW_PENALTY)
        steps_contraflow = int32(Defaults.VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS)
        steps_default = int32(grid.size)

        # ─────────  Phase 1: strict avoidance  ─────────
        raw_path = astar_numba(sx, sy, gx, gy, grid, flow_mask, penalty_stop, penalty_vehicle, penalty_contraflow,
                           soft_obstacles= False, ignore_flow = True,
                               maximum_steps = steps_default)

        # ─────────  Phase 2: allow soft obstacles if no path  ─────────
        if not raw_path:
            raw_path = astar_numba(sx, sy, gx, gy, grid, flow_mask, penalty_stop, penalty_vehicle, penalty_contraflow,
                           soft_obstacles= True, ignore_flow = False,
                                   maximum_steps = steps_default)

        path = [self.city_model.get_cell_contents(x, y)[0] for x, y in raw_path]

        # ─────────  Phase 3: Contraflow‐overtake bypass (ignore flow)  ─────────
        if Defaults.VEHICLE_CONTRAFLOW_OVERTAKE_ACTIVE:
            idx_stop, idx_vehicle = self._scan_ahead_for_obstacles(path)
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
                        raw_path = astar_numba(sx, sy, bypass_position[0], bypass_position[1], grid, flow_mask, penalty_stop, penalty_vehicle, penalty_contraflow,
                                               soft_obstacles = False, ignore_flow=False,
                                               maximum_steps=steps_contraflow)
                        bypass_path = [self.city_model.get_cell_contents(x, y)[0] for x, y in raw_path]
                        # validate bypass_path
                        if (bypass_path and len(bypass_path) > 1 and bypass_path[-1].get_position() == bypass_position):
                            # patch original path: replace segment up to bypass with bypass_path
                            i_bp = next(
                                i for i, cell in enumerate(path)
                                if cell.get_position() == bypass_position)

                            patched = bypass_path + path[i_bp + 1:]
                            self.overtake_path = bypass_path
                            self.is_overtaking = True
                            self.overtaking_duration = 0
                            return patched
        return path

    def _compute_path_cuda(self,
                      *,
                      avoid_stops: bool = True,
                      avoid_vehicles: bool = True,
                      allow_contraflow_overtake: bool = False) -> list[CellAgent] | None:
        """
        Thin Python wrapper: gather data ➜ call Numba A* ➜ translate back.
        """
        model = self.city_model
        if avoid_vehicles:
            model.path_planner.request(
                self.current_cell.position,
                self.target.position,
                avoid_stops,
                allow_contraflow_overtake,
                lambda p: setattr(self, "path", p),
            )
            return self.path

        return None


    def _scan_ahead_for_obstacles(self, path: list["CellAgent"]) -> tuple[
        None | int, None | int ]:
        """Return indices of the first Stop‑cell or occupied cell on the path."""
        rng = min(Defaults.VEHICLE_AWARENESS_RANGE, len(path))

        idx_stop: int | None = None
        idx_vehicle: int | None = None

        stop_cells: set[tuple[int, int]] = self.city_model.stop_cells
        vehicle_cells: set[tuple[int, int]] = self.city_model.vehicle_cells

        for idx in range(rng):
            cell = path[idx]
            pos = cell.position
            if idx_stop is None and pos in stop_cells:
                idx_stop = idx  # inclusive obstacle
            if idx_vehicle is None and pos in vehicle_cells:
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

        idx_stop, idx_vehicle = self._scan_ahead_for_obstacles(self.path)

        if self.is_overtaking:
            self.overtaking_duration += 1
            if (idx_stop is None) and (idx_vehicle is None) and self.overtaking_duration <= Defaults.VEHICLE_CONTRAFLOW_OVERTAKE_DURATION:
                return idx_stop, idx_vehicle

        # —— First try the ordinary planner (same as before) ——
        if (idx_stop is not None) or (idx_vehicle is not None):
            path = self._compute_path()
            if path:                     # success → adopt and leave
                self.path = path
                idx_stop, idx_vehicle = self._scan_ahead_for_obstacles(path)

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
            self.path = self._compute_path()

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
            max_steps = min(max_steps, idx_stop)  # allowed to *enter* Stop cell

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
            self.steps_traveled += 1

            old_pos = new_pos
            self.path.pop(0)

    def on_target_reached(self) -> None:
        """Invoked exactly once when the vehicle enters *target*.

        If *remove_on_arrival* is *True*, the vehicle is despawned; otherwise
        it simply stops in its target cell.
        """
        if Defaults.ENABLE_TRAFFIC:
            gen = self.city_model.dynamic_traffic_generator
            duration = gen.elapsed - self.depart_time

            distance = self.steps_traveled

            if self.population_type == "internal":
                gen.record_internal_trip(duration, distance)
            elif self.population_type == "through":
                gen.record_through_trip(duration, distance)

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
        return f"{self.population_type} Citizen"

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
            "Speed": self.current_speed
        }

        flags = []
        for flag, label in [
            ("is_overtaking", "Overtaking"),
            ("is_malfunctioning", "Malfunctioning"),
            ("in_collision", "InCollision"),
            ("is_parked", "Parked"),
        ]:
            if getattr(self, flag, False):
                flags.append(label)
        if flags and len(flags) > 0:
            portrayal["Status"] = ", ".join(flags)
        else:
            portrayal["Status"] = "Ok"

        portrayal["Current Destination"] = self.target.id if self.target else "None"

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
