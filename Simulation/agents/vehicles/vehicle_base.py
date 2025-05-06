# vehicle_base.py – refactored step logic & vehicle‑block speed reset
from mesa import Agent
import random
import heapq
from typing import TYPE_CHECKING, cast
from collections import deque
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
        self.current_cell = None

        self.base_speed = 0      # «cruising» speed valid until a full stop
        self.current_speed = 0   # speed granted *this* tick
        self.direction: str | None = None

        self.is_parked = False
        self.is_in_collision = False
        self.is_in_malfunction = False

        self.remove_on_arrival = False

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
        return random.randint(*(Defaults.MIN_VEHICLE_SPEED, Defaults.MAX_VEHICLE_SPEED))


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

    def _compute_path(
            self,
            *,
            avoid_stops: bool = True,
            avoid_vehicles: bool = True,
            allow_probing: bool = False,
    ) -> list["CellAgent"]:
        """
        A* search from the agent’s current cell to its target.

        • First pass: only legal moves (forward + R1 side arrows)
        • If that fails, fall back to the *closest reachable* cell
          (minimum Manhattan distance to the goal).
        """

        # ── Early guard ───────────────────────────────────────────
        if self.current_cell is None:
            self.current_cell = self.start_cell  # type: ignore[attr-defined]

        city: "CityModel" = cast("CityModel", self.model)
        start = cast(tuple[int, int], self.current_cell.get_position())
        goal = cast(tuple[int, int], self.target.get_position())

        def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: list[
            tuple[int, int, tuple[int, int], list[tuple[int, int]]]
        ] = []
        heapq.heappush(open_set, (manhattan(start, goal), 0, start, []))
        closed: set[tuple[int, int]] = set()

        # NEW: remember the best‑so‑far node in case we never hit *goal*
        best_dist = float("inf")
        best_path: list[tuple[int, int]] | None = None

        while open_set:
            f, g, pos, path = heapq.heappop(open_set)
            if pos in closed:
                continue
            closed.add(pos)

            # Track closest‐yet node
            dist = manhattan(pos, goal)
            if dist < best_dist:
                best_dist, best_path = dist, path + [pos]

            # Goal reached
            if pos == goal:
                return self._path_from_positions(path + [pos])[1:]

            # Expand neighbours
            x, y = pos
            cell = city.get_cell_contents(x, y)[0]
            for d in cell.directions:
                nx, ny = city.next_cell_in_direction(x, y, d)
                if not city.in_bounds(nx, ny):
                    continue
                npos = (nx, ny)
                if npos in closed:
                    continue

                ncell = city.get_cell_contents(nx, ny)[0]
                if avoid_vehicles and ncell.occupied:
                    continue
                if avoid_stops and self._is_a_stop_cell(ncell):
                    continue

                heapq.heappush(
                    open_set,
                    (
                        g + 1 + manhattan(npos, goal),  # f
                        g + 1,  # g
                        npos,
                        path + [pos],
                    ),
                )

        # ── No path all the way to goal: use closest alternative ──
        if best_path:
            hop_path = self._path_from_positions(best_path)[1:]
            if hop_path:                         # already moves ≥ 1 step
                return hop_path

            if allow_probing:
            # ---------- single‑step probe that obeys lane flow ----------
                x0, y0 = start
                cell0 = city.get_cell_contents(x0, y0)[0]

                for d in cell0.directions:           # stay within allowed flow
                    nx, ny = city.next_cell_in_direction(x0, y0, d)
                    if not city.in_bounds(nx, ny):
                        continue
                    ncell = city.get_cell_contents(nx, ny)[0]

                    if avoid_vehicles and ncell.occupied:
                        continue
                    if avoid_stops and self._is_a_stop_cell(ncell):
                        continue

                    # legal first hop found → return two‑position path
                    return self._path_from_positions([(x0, y0), (nx, ny)])[1:]
                # -----------------------------------------------------------------

        return []  # truly boxed in – no legal move in permitted directions

    # ------------------------------------------------------------
    #  Local detour helper
    # ------------------------------------------------------------
    def _compute_adjacent_detour(self, original_path: list["CellAgent"]) -> list["CellAgent"]:
        """
        Breadth‑first search that stays within the 1‑block “band” around
        *original_path* (inclusive) and avoids Intersection cells.

        It ignores normal traffic flow, so neighbours are the four cardinal
        directions regardless of `cell.directions`.
        """
        city = cast("CityModel", self.model)

        # ── Build the allowed area (original path ± 1) ───────────
        band: set[tuple[int, int]] = set()
        for cell in original_path:
            x, y = cell.get_position()
            band.add((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not city.in_bounds(nx, ny):
                    continue
                nbr = city.get_cell_contents(nx, ny)[0]
                if nbr.cell_type != "Intersection":
                    band.add((nx, ny))

        start = self.current_cell.get_position()
        goal  = original_path[-1].get_position()  # the real destination

        # ── Plain BFS over the band ──────────────────────────────
        q = deque([(start, [])])
        visited = {start}

        while q:
            pos, path = q.popleft()
            if pos == goal:
                return self._path_from_positions(path + [pos])[1:]  # skip current cell

            x, y = pos
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                npos = (nx, ny)
                if npos not in band or npos in visited:
                    continue
                cell = city.get_cell_contents(nx, ny)[0]
                # avoid occupied cells; the *blocking* car itself is outside the band
                if cell.occupied:
                    continue
                visited.add(npos)
                q.append((npos, path + [pos]))

        return []  # no viable micro‑detour


# ────────────────────────────────────────────────────────────────

    def _scan_ahead_for_obstacles(self):
        """Return indices of the first Stop‑cell or occupied cell on the path."""
        idx_stop = idx_vehicle = None
        rng = min(Defaults.VEHICLE_AWARENESS_RANGE, len(self.path))
        for idx in range(rng):
            cell = self.path[idx]
            if idx_stop is None and self._is_a_stop_cell(cell):
                idx_stop = idx  # inclusive obstacle
            if idx_vehicle is None and cell.occupied:
                idx_vehicle = idx  # exclusive obstacle
            if (idx_stop == 0) or (idx_vehicle == 0):
                break
        return idx_stop, idx_vehicle

    def _recompute_path_on_obstacle(self):
        """
        Re‑route if we see a Stop ahead or a blocked lane.

        •If the obstacle is a **stranded / parked** vehicle, try the
          adjacent‑band detour described above.
        •Otherwise fall back to the normal A* re‑route.
        """
        idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()
        alt_path: list[CellAgent] | None = None

        # ── Special case: stranded / parked car ahead ────────────
        if idx_vehicle is not None and False:
            blocked_cell = self.path[idx_vehicle]
            veh = self._vehicle_on_cell(blocked_cell)
            if veh and (veh.is_stranded() or veh.is_parked):
                alt_path = self._compute_adjacent_detour(self.path)

        # ── Generic re‑route (e.g. Stop cell) ────────────────────
        if alt_path is None and (idx_stop is not None or idx_vehicle is not None):
            alt_path = self._compute_path(avoid_stops=True, avoid_vehicles=True)

        # ── Adopt the alternative if we found one ────────────────
        if alt_path and alt_path != self.path:
            self.path = alt_path
            idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()

        return idx_stop, idx_vehicle

    def _move_to(self, cell: CellAgent, new_pos: tuple[int, int], old_pos: tuple[int, int] = None):
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

    def _check_side_collision(self):

        if not self.direction:
            return
        right_dir = Defaults.DIRECTION_TO_THE_RIGHT[self.direction]
        left_dir = Defaults.DIRECTION_TO_THE_RIGHT[right_dir]
        for lat_dir in (left_dir, right_dir):
            dx, dy = Defaults.DIRECTION_VECTORS[lat_dir]
            x, y = self.current_cell.get_position()
            nx, ny = x + dx, y + dy
            if not self.city_model.in_bounds(nx, ny):
                continue
            for ag in self.city_model.get_cell_contents(nx, ny):
                if isinstance(ag, VehicleAgent) and ag.direction == Defaults.DIRECTION_OPPOSITES[self.direction]:
                    if random.random() >= Defaults.VEHICLE_SIDE_COLLISION_CHANCE:
                        return
                    self._set_collision(Defaults.VEHICLE_COLLISION_DURATION)
                    ag._set_collision(Defaults.VEHICLE_COLLISION_DURATION)
                    return

    def _check_front_collision(self):
        if not self.path or not self.direction:
            return
        next_cell = self.path[0]
        veh = self._vehicle_on_cell(next_cell)
        if veh and veh.direction == Defaults.DIRECTION_OPPOSITES[self.direction]:
            if random.random() >= Defaults.VEHICLE_FRONT_COLLISION_CHANCE:
                return
            self._set_collision(Defaults.VEHICLE_COLLISION_DURATION)
            veh._set_collision(Defaults.VEHICLE_COLLISION_DURATION)


    def _check_malfunction(self):
        if random.random() < Defaults.VEHICLE_MALFUNCTION_CHANCE:
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

        self._check_side_collision()
        if self.is_stranded():
            self.base_speed = 0
            return

        self._check_front_collision()
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
            self.city_model.move_vehicle(self, next_cell, new_pos, old_pos)
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

    # ══ Misc getters used by UI ══════════════════════════════════
    def is_stranded(self):
        return self.is_in_collision or self.is_in_malfunction

    def get_current_direction(self):
        return self.direction

    def get_current_path(self):
        return [agent.get_position() for agent in self.path]

    def get_target(self):
        return self.target.get_position()

    def get_portrayal(self):
        portrayal = {
            "Shape": "circle",
            "r": 0.66,
            "Filled": True,
            "Layer": 1,
            "Color": "black",
            "Type": "Vehicle",
            "Direction": self.direction or "?",
        }

        flash_on = (self.city_model.step_count % 2) == 0
        if self.is_in_collision:
            col_alt = Defaults.VEHICLE_COLLISION_COLOR
            portrayal["Color"] = self.base_color if flash_on else col_alt
        elif self.is_in_malfunction:
            mal_alt = Defaults.VEHICLE_MALFUNCTION_COLOR
            portrayal["Color"] = self.base_color if flash_on else mal_alt
        elif self.is_parked:
            park_alt = Defaults.VEHICLE_PARKED_COLOR
            portrayal["Color"] = self.base_color if flash_on else park_alt
        else:
            portrayal["Color"] = self.base_color

        return portrayal
