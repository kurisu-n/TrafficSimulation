# vehicle_base.py – refactored step logic & vehicle‑block speed reset
from mesa import Agent
import random
from typing import TYPE_CHECKING, cast

from Simulation.config import Defaults
from Simulation.agents.city_structure_entities.cell import CellAgent
from Simulation.utilities.general import *
from Simulation.utilities.numba_utilities import compute_direction

if Defaults.PATHFINDING_METHOD in ("NUMBA", "CYTHON"):
    from Simulation.utilities.pathfinding.astar_numba import astar_numba as astar
elif Defaults.PATHFINDING_METHOD == "TENSORFLOW":
    from Simulation.utilities.pathfinding.astar_tensorflow import astar_tensorflow as astar
elif Defaults.PATHFINDING_METHOD == "TENSORFLOW_VEC":      # ← add
    from Simulation.utilities.pathfinding.astar_tensorflow_batch import astar_tensorflow_vectorised as astar

if TYPE_CHECKING:
    from Simulation.city_model import CityModel

class VehicleAgent(Agent):
    """
    A vehicle agent that navigates from a start to a destination using A* path‑finding.
    """

    # ════════════════════════════════════════════════════════════
    #  INIT / HELPERS
    # ════════════════════════════════════════════════════════════
    def __init__(self, custom_id, model, start_cell: CellAgent, target_cell: CellAgent, population_type = "undefined", vehicle_type = "undefined"):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.start_cell = start_cell
        self.target = target_cell
        self.previous_pos = None
        self.population_type = population_type
        self.vehicle_type = vehicle_type

        self.base_speed = 0      # «cruising» speed valid until a full stop
        self.current_speed = 0   # speed granted *this* tick
        self.max_steps = 0
        self.blocked_by_vehicle: bool = False
        self.direction: str | None = None
        self.path_retry_cooldown = 0

        self.is_overtaking = False
        self.overtake_path: list[tuple[int, int]] = []
        self.pre_overtake_path: list[tuple[int, int]] = []
        self.overtaking_duration: int = -1

        self.is_in_stuck_detour: bool = False
        self.pre_stuck_detour_path: list[tuple[int, int]] = []
        self.stuck_detour_path: list[tuple[int, int]] = []
        self.stuck_detour_target: tuple[int, int] | None = None
        self.stuck_detour_duration: int = -1

        self.stuck_ticks = 0

        self._early_exit = False
        self.is_stuck = False
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

        self.city_model.place_vehicle(self, self.start_cell.get_position(), population_type=population_type, vehicle_type=vehicle_type)

        self.path: list[tuple[int, int]] = []
        self.path = self._compute_path()

        self._base_portrayal = {
            "Shape":      "circle",
            "r":          0.66,
            "Filled":     True,
            "Layer":      1,
            "Color":      self.base_color
        }

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
        x, y = self.pos
        if Defaults.RAIN_ENABLED and self.city_model.rain_map[y, x] == 1:
            speed = max(1, speed - Defaults.RAIN_SPEED_REDUCTION)

        return speed

    @staticmethod
    def _choose_new_speed() -> int:
        """Pick a persistent cruising speed until the next full stop."""
        return random.randint(*(Defaults.VEHICLE_MIN_SPEED, Defaults.VEHICLE_MAX_SPEED))

    def _is_a_stop_cell_position(self, pos: tuple[int, int]) -> bool:
        if pos is None:
            return False

        x, y = pos
        return self.city_model.stop_map[y, x] == 1

    def _is_at_stopped_cell(self) -> bool:
        if self.pos is None:
            return False

        if self._is_a_stop_cell_position(self.pos):
            return True
        return False

    def _is_at_intersection(self) -> bool:
        if self.pos is None:
            return False

        x0, y0 = self.pos
        if self.city_model.intersection_map[y0, x0] == 1:
            return True
        else:
            return False

    # --------------------------------------------------------------------------- #
    #  compute_path wrapper – *inside* VehicleAgent
    # --------------------------------------------------------------------------- #

    def _compute_path(self, use_cache: bool = True) -> list[tuple[int, int]]:
        """
        Wrapper for the path planner. Calls the JIT-compiled A* algorithm.
        """
        self.path_retry_cooldown = Defaults.PATHFINDING_COOLDOWN

        # Normalize positions (optional: round to improve cache hit rate)
        start = self.pos
        goal = self.target.get_position()
        cache_key = (start, goal)

        if use_cache and Defaults.PATHFINDING_CACHE:
            # Look up cache
            if hasattr(self.city_model, "_path_cache") and cache_key in self.city_model._path_cache:
                return list(self.city_model._path_cache[cache_key])

        path = self._compute_path_internal()

        # Cache path if valid and not a contraflow overtaking result
        if  use_cache and Defaults.PATHFINDING_CACHE and path and not self.is_overtaking and not self.is_in_stuck_detour:
            if not hasattr(self.city_model, "_path_cache"):
                self.city_model._path_cache = {}
            self.city_model._path_cache[cache_key] = list(path)

        return path

    def _positions_in_fov_python(self) -> list[tuple[int, int]]:
        """
        Return two lists of positions of stopped and occupied cells along lines in each of the four cardinal directions
        from the current position, stopping when the block type is no longer Defaults.ROADS.
        """
        city = cast("CityModel", self.model)
        cx, cy = self.pos
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

    def _compute_path_internal(self) -> list[tuple[int, int]]:
        """
        Find a path to the target using the optimized JIT-compiled A*.
        Uses static maps (allowed_dirs_map, is_road_map) cached in CityModel,
        and dynamic maps (occupancy_map, stop_map) maintained per tick.
        """
        city = self.city_model
        width, height = city.get_width(), city.get_height()
        start_x, start_y = self.pos
        goal_x, goal_y = self.target.get_position()

        # 2. Grab precomputed maps from CityModel
        occupancy_map    = city.occupancy_map      # 1 = occupied by vehicle
        stop_map         = city.stop_map           # 1 = stop-sign / red-light
        allowed_dirs_map = city.allowed_dirs_map   # bitmask of allowed flows
        is_road_map      = city.is_road_map        # 1 = this is a road cell
        road_type_map    = city.road_type_map      # 1 = this is a road cell
        density_map      = city.density_map

        # ─────────  Phase 0: back to the original path if overtaking or patching  ─────────
        if self.is_overtaking and self.pre_overtake_path:
            city = self.city_model
            # 1. Find first free merge cell on the saved original path
            merge_idx = None
            for i, cell in enumerate(self.pre_overtake_path):
                x_m, y_m = cell
                if occupancy_map[y_m, x_m] == 0:
                    merge_idx = i
                    bx, by = x_m, y_m
                    break
            # 2. If found, build FOV map and run contraflow A* to merge point
            if merge_idx is not None:
                bypass_coords = astar(
                    width, height,
                    *self.pos, bx, by,
                    occupancy_map, stop_map,
                    is_road_map, road_type_map, allowed_dirs_map,
                    respect_awareness=Defaults.VEHICLE_RESPECT_AWARENESS,
                    awareness_range=Defaults.VEHICLE_AWARENESS_RANGE,
                    density_map = density_map,
                    soft_obstacles=False,
                    ignore_flow=True,
                    maximum_steps=Defaults.VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS
                )
                bypass_path = list(bypass_coords)
                # 3. If bypass reaches the merge point, splice and return immediately
                if bypass_path and bypass_path[-1] == (bx, by):
                    self.overtake_path = bypass_path
                    return bypass_path + self.pre_overtake_path[merge_idx + 1:]

        if self.is_in_stuck_detour and self.pre_stuck_detour_path:
            city = self.city_model
            # 1. Find first free merge cell on the saved original path
            merge_idx = None
            for i, cell in enumerate(self.pre_stuck_detour_path):
                x_m, y_m = cell
                if occupancy_map[y_m, x_m] == 0:
                    merge_idx = i
                    bx, by = x_m, y_m
                    break
            # 2. If found, build FOV map and run contraflow A* to merge point
            if merge_idx is not None:
                bypass_coords = astar(
                    width, height,
                    *self.pos, bx, by,
                    occupancy_map, stop_map,
                    is_road_map, road_type_map, allowed_dirs_map,
                    respect_awareness=Defaults.VEHICLE_RESPECT_AWARENESS,
                    awareness_range=Defaults.VEHICLE_AWARENESS_RANGE,
                    density_map=density_map,
                    soft_obstacles=False,
                    ignore_flow=True,
                    maximum_steps=Defaults.VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS
                )
                bypass_path = list(bypass_coords)
                # 3. If bypass reaches the merge point, splice and return immediately
                if bypass_path and bypass_path[-1] == (bx, by):
                    self.stuck_detour_path = bypass_path
                    return bypass_path + self.pre_stuck_detour_path[merge_idx + 1:]

        # ───────── Phase 1: strict avoidance ─────────
        path_coords = astar(
            width, height,
            start_x, start_y, goal_x, goal_y,
            occupancy_map, stop_map,
            is_road_map, road_type_map, allowed_dirs_map,
            respect_awareness=Defaults.VEHICLE_RESPECT_AWARENESS,
            awareness_range=Defaults.VEHICLE_AWARENESS_RANGE,
            density_map=density_map,
            soft_obstacles=False,
            ignore_flow=False,
        )
        path = list(path_coords)

        # ───────── Phase 2: allow soft obstacles ─────────
        if not path or len(path) == 0:
            path_coords = astar(
                width, height,
                start_x, start_y, goal_x, goal_y,
                occupancy_map, stop_map,
                is_road_map, road_type_map, allowed_dirs_map,
                respect_awareness=Defaults.VEHICLE_RESPECT_AWARENESS,
                awareness_range=Defaults.VEHICLE_AWARENESS_RANGE,
                density_map=density_map,
                soft_obstacles=True,
                ignore_flow=False,
            )
            path = list(path_coords)

        # ───────── Phase 3: contraflow overtaking bypass ─────────
        if Defaults.VEHICLE_CONTRAFLOW_OVERTAKE_ACTIVE and path:
            # scan ahead for first stop or vehicle obstacle
            aw = Defaults.VEHICLE_AWARENESS_RANGE
            idx_stop = idx_vehicle = None
            for i in range(min(aw, len(path))):
                x_i, y_i = path[i]
                if idx_stop is None and stop_map[y_i, x_i] == 1:
                    idx_stop = i
                if idx_vehicle is None and occupancy_map[y_i, x_i] == 1:
                    idx_vehicle = i
                if idx_stop is not None and idx_vehicle is not None:
                    break

            if idx_vehicle == 0:
                blocker = self._vehicle_on_cell(path[0])
                if blocker and (blocker.is_stranded() or blocker.is_parked):
                    # find first free cell down the planned route
                    bypass_target = None
                    for cell in path:
                        bx, by = cell
                        if occupancy_map[by, bx] == 0:
                            bypass_target = (bx, by)
                            break

                    if bypass_target:
                        bx, by = bypass_target
                        # compute contraflow bypass (ignore_flow=True)
                        bypass_coords = astar(
                            width, height,
                            start_x, start_y, bx, by,
                            occupancy_map, stop_map,
                            is_road_map, road_type_map, allowed_dirs_map,
                            respect_awareness=Defaults.VEHICLE_RESPECT_AWARENESS,
                            awareness_range=Defaults.VEHICLE_AWARENESS_RANGE,
                            density_map=density_map,
                            soft_obstacles=False,
                            ignore_flow=True,
                            maximum_steps=Defaults.VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS
                        )
                        bypass_path = list(bypass_coords)
                        if (bypass_path
                                and bypass_path[-1] == (bx, by)
                                and len(bypass_path) > 1):
                            # splice bypass into original path

                            try:
                                idx_bp = next(i for i, c in enumerate(path) if c == (bx, by))
                            except StopIteration:
                                idx_bp = None

                            if idx_bp is not None:
                                self.pre_overtake_path = path
                                self.overtake_path = bypass_path
                                patched = bypass_path + path[idx_bp+1:]
                                self.is_overtaking = True
                                self.city_model.dynamic_traffic_generator.overtaking += 1
                                self.overtaking_duration = 0
                                return patched

        # ───────── Phase 4: generic patch-on-block obstacle ─────────
        if Defaults.VEHICLE_STUCK_CONTRAFLOW_ENABLED and path:
            if self._is_at_intersection():
                threshold = Defaults.VEHICLE_STUCK_CONTRAFLOW_THRESHOLD_INTERSECTION
            else:
                threshold = Defaults.VEHICLE_STUCK_CONTRAFLOW_THRESHOLD

            stuck_generic = self.stuck_ticks >= threshold
            if stuck_generic and path:
                # Find next free cell down the planned route
                bypass_target = None
                for cell in path:
                    tx, ty = cell
                    if self.city_model.occupancy_map[ty, tx] == 0:
                        bypass_target = (tx, ty)
                        break

                if bypass_target:
                    tx, ty = bypass_target
                    # compute bypass path ignoring dynamic obstacles
                    bypass_coords = astar(
                        width, height,
                        *self.pos, tx, ty,
                        occupancy_map, stop_map,
                        is_road_map, road_type_map, allowed_dirs_map,
                        respect_awareness=Defaults.VEHICLE_RESPECT_AWARENESS,
                        awareness_range=Defaults.VEHICLE_AWARENESS_RANGE,
                        density_map=density_map,
                        soft_obstacles=True,
                        ignore_flow=True,
                        maximum_steps=Defaults.VEHICLE_MAX_CONTRAFLOW_STUCK_DETOUR_STEPS
                    )
                    bypass_path = list(bypass_coords)
                    if bypass_path and bypass_path[-1] == (tx, ty) and len(bypass_path) > 1:
                        # splice and set patch flags
                        try:
                            merge_idx = next(i for i, c in enumerate(path) if c == (tx, ty))
                        except StopIteration:
                            merge_idx = None

                        if merge_idx is not None:
                            self.pre_stuck_detour_path = path.copy()
                            self.stuck_detour_path = bypass_path
                            self.stuck_detour_target = bypass_target
                            patched = bypass_path + path[merge_idx + 1:]
                            self.city_model.dynamic_traffic_generator.in_stuck_detour += 1
                            self.is_in_stuck_detour = True
                            self.stuck_detour_duration = 0
                            return patched

                    return path

        return path

    def _scan_ahead_for_obstacles(self, path: list[tuple[int, int]]) -> tuple[None | int, None | int]:
        """
        Return indices of the first Stop-cell or occupied cell on the path,
        using the cached stop_map and occupancy_map in CityModel.
        """
        if not path:
            return None, None

        idx_stop = None
        idx_vehicle = None

        max_lookahead = min(Defaults.VEHICLE_AWARENESS_RANGE, len(path))
        city = self.city_model
        stop_map = city.stop_map
        occupancy_map = city.occupancy_map

        for idx in range(max_lookahead):
            x, y = path[idx]

            # first stop-control obstacle
            if idx_stop is None and stop_map[y, x] == 1:
                idx_stop = idx

            # first vehicle obstacle
            if idx_vehicle is None and occupancy_map[y, x] == 1:
                idx_vehicle = idx

            if idx_stop == 0 or idx_vehicle == 0:
                break

        return idx_stop, idx_vehicle

    def _recompute_path_on_obstacle(self):
        """
        Normal re‑route ➜ if that yields *zero* moves *and* the first
        obstacle is a stranded/parked car, do an expensive contraflow search.
        """
        if (self.is_overtaking and
                (self.overtake_path is None or len(self.overtake_path) == 0 or self.pos not in self.overtake_path)):
            self.overtake_path = None
            self.is_overtaking = False
            self.city_model.dynamic_traffic_generator.overtaking += 0

        if (self.is_in_stuck_detour and
                (self.stuck_detour_path is None or len(self.stuck_detour_path) == 0 or self.pos not in self.stuck_detour_path)):
            self.stuck_detour_path = None
            self.is_in_stuck_detour = False
            self.city_model.dynamic_traffic_generator.in_stuck_detour += 0


        idx_stop, idx_vehicle = self._scan_ahead_for_obstacles(self.path)

        if self.is_overtaking:
            self.overtaking_duration += 1
            if self.overtaking_duration <= Defaults.VEHICLE_CONTRAFLOW_OVERTAKE_DURATION:
                return idx_stop, idx_vehicle

        if self.is_in_stuck_detour:
            self.stuck_detour_duration += 1
            if self.stuck_detour_duration <= Defaults.VEHICLE_CONTRAFLOW_STUCK_DETOUR_DURATION:
                return idx_stop, idx_vehicle

        if self.path_retry_cooldown > 0:
            # Exception: recalculate immediately if hard block at next cell
            if idx_vehicle == 0:
                blocker = self._vehicle_on_cell(self.path[0])
                if blocker and (blocker.is_stranded() or blocker.is_parked):
                    pass  # allow immediate pathfinding
                else:
                    self.path_retry_cooldown -= 1
                    return idx_stop, idx_vehicle
            else:
                self.path_retry_cooldown -= 1
                return idx_stop, idx_vehicle

        # —— First try the ordinary planner (same as before) ——
        if (idx_stop is not None) or (idx_vehicle is not None):
            path = self._compute_path(use_cache=False)
            if path:                     # success → adopt and leave
                self.path = path
                idx_stop, idx_vehicle = self._scan_ahead_for_obstacles(path)

        return idx_stop, idx_vehicle

    def _recompute_path_on_stuck(self):
        """
        If we’ve been in the same cell for too many ticks (and it’s not a Stop‐cell),
        force a fresh path compute (bypass cache).
        """
        if self._is_at_intersection():
            thresh = Defaults.VEHICLE_STUCK_RECOMPUTE_THRESHOLD_INTERSECTION
        else:
            thresh = Defaults.VEHICLE_STUCK_RECOMPUTE_THRESHOLD

        if self.stuck_ticks >= thresh:
            self.path = self._compute_path(use_cache=False)

    DIR_LOOKUP = ["N", "E", "S", "W"]

    def _move_to(self, new_pos: tuple[int, int], old_pos: tuple[int, int] = None):
        self.previous_pos = self.pos
        self.city_model.move_vehicle(self, new_pos, old_pos)
        dir_idx = compute_direction(old_pos, new_pos)
        if dir_idx != -1:
            self.direction = self.DIR_LOOKUP[dir_idx]

        if self.stuck_ticks > 0:
            if self.is_stuck:
                self.city_model.dynamic_traffic_generator.stuck -= 1
                self.is_stuck = False
            self.stuck_ticks = 0

    def _set_collision(self, ticks: int):
        """Mark the vehicle stranded due to a collision."""
        self.is_in_collision = True
        self.is_in_malfunction = False
        self._stranded_ticks_remaining = ticks
        self.base_speed = 0
        self.current_speed = 0
        self.city_model.dynamic_traffic_generator.collisions += 1

    def _set_malfunction(self, ticks: int):
        """Mark the vehicle stranded due to a malfunction."""
        self.is_in_malfunction = True
        self.is_in_collision = False
        self._stranded_ticks_remaining = ticks
        self.base_speed = 0
        self.current_speed = 0
        self.city_model.dynamic_traffic_generator.malfunctions += 1

    def _tick_stranded(self) -> bool:
        if not self.is_stranded():
            return False
        self._stranded_ticks_remaining -= 1
        if self._stranded_ticks_remaining <= 0:
            if self.is_in_collision:
                self.city_model.dynamic_traffic_generator.collisions -= 1
            if self.is_in_malfunction:
                self.city_model.dynamic_traffic_generator.malfunctions -= 1

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
            x, y   = self.pos
            nx, ny = x + dx, y + dy

            if not self.city_model.in_bounds(nx, ny):
                continue

            for ag in self.city_model.get_cell_contents(nx, ny):
                # only sideswipe if they’re moving alongside me
                if not isinstance(ag, VehicleAgent):
                    continue

                # 1) Skip if they’re not moving
                if ag.current_speed <= 0 or ag.is_stuck or ag.is_parked or ag.is_in_collision or ag.is_in_malfunction:
                    continue

                if ag.direction is not Defaults.DIRECTION_OPPOSITES[self.get_current_direction()]:
                    continue

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

    def step_decide(self):
        self._early_exit = False

        # 0) Stranded / malfunction checks
        if self._tick_stranded():
            self.base_speed = 0
            self.current_speed = 0
            self._early_exit = True
            return
        self._check_malfunction()
        if self.is_stranded():
            self.base_speed = 0
            self.current_speed = 0
            self._early_exit = True

            return
        self._check_sideswipe_collision()
        if self.is_stranded():
            self.base_speed = 0
            self.current_speed = 0
            self._early_exit = True

            return
        if self._is_at_stopped_cell():
            self.base_speed = 0
            self.current_speed = 0
            self._early_exit = True
            return

        # 1) Compute current speed
        self.current_speed = self._compute_speed()

        # 2) Maybe force a replan if stuck too long
        self._recompute_path_on_stuck()

        # 3) Existing obstacle-triggered replan (stop‐signs, blocked cells)
        idx_stop, idx_vehicle = self._recompute_path_on_obstacle()

        # 4) Determine how many steps we can take this tick
        self.max_steps, self.blocked_by_vehicle = self._determine_max_steps(idx_stop, idx_vehicle)
        if self.max_steps <= 0:
            self.base_speed = 0
            if self.pos == self.target.get_position():
                self.on_target_reached()
            self._early_exit = True
            return

        self._early_exit = False


    def step(self):
        """Advance the vehicle one tick, obeying traffic rules and lights."""

        if not Defaults.PATHFINDING_BATCHING:
            self.step_decide()

        if self._early_exit:
            self._early_exit = False
            self.tick_stuck()
            return


        self._execute_movement(self.max_steps)
        # moved_this_tick = True

        # if not moved_this_tick and (not self.path or len(self.path) == 0):
        #     self.path = self._compute_path()

        self.previous_pos = self.pos

        if self.pos == self.target.get_position():
            self.on_target_reached()

        self._despawn_check()

    def tick_stuck(self):
        if self.previous_pos == self.pos and not self._is_a_stop_cell_position(self.pos):
            self.stuck_ticks += 1
            if self.stuck_ticks > Defaults.VEHICLE_STUCK_RECOMPUTE_THRESHOLD and not self.is_stuck:
                self.city_model.dynamic_traffic_generator.stuck += 1
                self.is_stuck = True

    def _despawn_check(self):
        despawn_threshold = Defaults.VEHICLE_STUCK_DESPAWN_THRESHOLD_INTERSECTION if self._is_at_intersection() else Defaults.VEHICLE_STUCK_DESPAWN_THRESHOLD
        if Defaults.VEHICLE_STUCK_DESPAWN_ENABLED and self.stuck_ticks >= despawn_threshold:
            if self.population_type == "internal":
                self.city_model.dynamic_traffic_generator.errored_internal += 1
            else:
                self.city_model.dynamic_traffic_generator.errored_through += 1
            self._despawn()
    # ------------------------------------------------------------
    #  STEP helper methods (private)
    # ------------------------------------------------------------

    def _vehicle_on_cell(self, pos: tuple[int, int]) -> "VehicleAgent | None":
        """Return the *VehicleAgent* occupying *cell* or *None*."""
        x, y = pos
        for ag in self.city_model.get_cell_contents(x, y):
            if isinstance(ag, VehicleAgent):
                return ag
        return None

    def _determine_max_steps(self, idx_stop, idx_vehicle):
        """Return a tuple *(max_steps, blocked_by_vehicle)* for this tick."""
        max_steps = min(self.current_speed,len(self.path))
        blocked_by_vehicle = False

        if idx_stop is not None:
            max_steps = min(max_steps, idx_stop)  # allowed to *enter* Stop cell

        if idx_vehicle is not None:
            if idx_vehicle == 0:
                blocked_by_vehicle = True        # cannot even enter the next cell
            max_steps = min(max_steps, idx_vehicle)
        return max_steps, blocked_by_vehicle

    def _execute_movement(self, max_steps: int):
        """Move up to *max_steps* cells along `self.path`"""
        old_pos = self.pos

        for step_idx in range(max_steps):
            if not self.path:
                break  # already at destination
            x, y = self.path[0]

            # Safety re‑checks at runtime
            if self.city_model.occupancy_map[y, x] == 1 and (x, y) != self.pos:
                break
            if self._is_a_stop_cell_position((x, y)):
                if self.path[0] is not self.pos and step_idx != max_steps - 1:
                    break

            new_pos = (x, y)
            self._move_to(new_pos, old_pos)
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
        self.city_model.remove_vehicle(self, population_type=self.population_type, vehicle_type=self.vehicle_type)

    def _park(self) -> None:
        """Leave the vehicle in place but mark it as inactive."""
        if not self.is_parked:
            self.is_parked = True
            self.city_model.dynamic_traffic_generator.parked += 1
        pass

    def _unpark(self) -> None:
        if self.is_parked:
            self.is_parked = False
            self.city_model.dynamic_traffic_generator.parked -= 1

    def _get_base_color(self):
        if self.is_overtaking or self.is_in_stuck_detour:
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
        return self.path

    def get_target(self):
        return self.target.get_position()

    def get_description(self):
        return f"{self.population_type} Citizen"

    def get_portrayal(self):
        # Shallow-copy only the static bits cached at init
        p = self._base_portrayal.copy()

        if Defaults.AGENT_PORTRAYAL_LEVEL == 0:
            return p

        if Defaults.AGENT_PORTRAYAL_LEVEL >= 1:
            flash_on = (self.city_model.step_count % 2) == 0
            if self.is_in_collision:
                alt = Defaults.VEHICLE_COLLISION_COLOR
                p["Color"] = self._get_base_color() if flash_on else alt
            elif self.is_in_malfunction:
                alt = Defaults.VEHICLE_MALFUNCTION_COLOR
                p["Color"] = self._get_base_color() if flash_on else alt
            elif self.is_parked:
                alt = Defaults.VEHICLE_PARKED_COLOR
                p["Color"] = self._get_base_color() if flash_on else alt
            else:
                p["Color"] = self._get_base_color()

        if Defaults.AGENT_PORTRAYAL_LEVEL >= 2:
            p["Type"] = self.get_vehicle_type_name()
            p["Identifier"] = self.unique_id
            p["Description"] = self.get_description()

            # 2) patch dynamic vehicle fields
            p["Position"]            = self.pos
            p["Direction"]           = Defaults.DIRECTION_ICONS.get(self.get_current_direction(), "?")
            p["Speed"]               = self.current_speed
            p["Current Destination"] = self.target.id if self.target else "None"

            # 3) status flags & flashing color logic (reuse base logic)
            flags = []
            if self.is_in_stuck_detour:       flags.append("Detouring (Stuck)")
            if self.is_overtaking:     flags.append("Overtaking")
            if self.is_in_malfunction: flags.append("Malfunctioning")
            if self.is_in_collision:   flags.append("InCollision")
            if self.is_parked:         flags.append("Parked")
            if self.stuck_ticks > 0:
                flags.append(f"Stuck ({self.stuck_ticks})")

            p["Status"] = ", ".join(flags) if flags else "Ok"





        return p
