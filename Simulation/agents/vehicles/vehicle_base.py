# vehicle_base.py – refactored step logic & vehicle‑block speed reset
from mesa import Agent
import random
import heapq
from Simulation.config import Defaults
from Simulation.agents.cell import CellAgent
from Simulation.agents.dummy import DummyAgent
from Simulation.utilities import *


class VehicleAgent(Agent):
    """
    A vehicle agent that navigates from a start to a destination using A* path‑finding.

    Changes in this revision
    ------------------------
    • **Step refactor** – `step()` is now a thin orchestrator that delegates to
      dedicated private helpers for clarity and easier unit‑testing.
    • **Blocked‑by‑vehicle rule** – when the agent is forced to wait because the
      next cell is occupied by another vehicle, its `base_speed` is reset to
      **0 km/h** so the next acceleration is re‑sampled once traffic clears.
    """

    # ════════════════════════════════════════════════════════════
    #  INIT / HELPERS
    # ════════════════════════════════════════════════════════════
    def __init__(self, custom_id, model, start_cell: CellAgent, target_cell: CellAgent):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.start_cell = start_cell
        self.target = target_cell
        self.current_cell = start_cell
        self.base_speed = 0      # «cruising» speed valid until a full stop
        self.current_speed = 0   # speed granted *this* tick
        self.direction: str | None = None

        # occupy the start cell and remove its dummy
        self._occupy_initial_cell()
        self.model.grid.place_agent(self, self.current_cell.get_position())
        self.model.schedule.add(self)

        # initial path (ignoring Stop‑blocks so we can park *on* them if needed)
        self.path: list[CellAgent] = self.compute_path()

    # ------------------------------------------------------------
    #  Speed utilities
    # ------------------------------------------------------------
    def compute_speed(self) -> int:
        """Return the speed allowed this tick, weather‑adjusted if needed."""
        # Establish / keep the persistent cruising speed.
        if self.base_speed == 0:
            self.base_speed = self._sample_speed()

        speed = self.base_speed

        # --- Weather adjustment (rain) --------------------------------
        if self.current_cell.is_raining:
            speed = max(1, speed - Defaults.RAIN_SPEED_REDUCTION)

        return speed

    def _sample_speed(self) -> int:
        """Pick a persistent cruising speed until the next full stop."""
        return random.randint(*(Defaults.MIN_VEHICLE_SPEED, Defaults.MAX_VEHICLE_SPEED))

    # ------------------------------------------------------------
    #  Misc helpers
    # ------------------------------------------------------------
    def _is_stop_cell(self, cell: CellAgent) -> bool:
        """Return *True* if *cell* carries a traffic‑control status set to "Stop"."""
        return getattr(cell, "status", "Pass") == "Stop"

    # ════════════════════════════════════════════════════════════
    #  A*  PATH‑FINDING
    # ════════════════════════════════════════════════════════════
    def compute_path(self, *, avoid_stops: bool = False, avoid_vehicles: bool = False) -> list[CellAgent]:
        """Compute an A* path from *current* position to the *target*."""
        start = self.current_cell.get_position()
        goal = self.target.get_position()

        def heuristic(a: tuple[int, int], b: tuple[int, int]):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: list[tuple[int, int, tuple[int, int], list[tuple[int, int]]]] = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start, []))
        closed: set[tuple[int, int]] = set()

        while open_set:
            f, g, current_pos, path = heapq.heappop(open_set)
            if current_pos in closed:
                continue
            closed.add(current_pos)
            if current_pos == goal:
                # drop current cell, it is already occupied by the agent
                return self._path_from_positions(path + [current_pos])[1:]

            for neighbor_pos in self._neighbor_positions(current_pos):
                if neighbor_pos in closed:
                    continue
                neighbor_cell: CellAgent = self.model.get_cell_contents(neighbor_pos[0], neighbor_pos[1])[0]

                # -- dynamic obstacles ------------------------------------
                if avoid_vehicles and neighbor_cell.occupied:
                    continue
                if avoid_stops and self._is_stop_cell(neighbor_cell):
                    continue

                heapq.heappush(
                    open_set,
                    (
                        g + 1 + heuristic(neighbor_pos, goal),  # f = g + h
                        g + 1,                                  # new g
                        neighbor_pos,
                        path + [current_pos]
                    )
                )
        # no path found
        return []

    # ------------------------------------------------------------
    #  Neighbor / direction helpers
    # ------------------------------------------------------------
    def _path_from_positions(self, positions: list[tuple[int, int]]):
        return [self.model.get_cell_contents(pos[0], pos[1])[0] for pos in positions]

    def _neighbor_positions(self, pos: tuple[int, int]):
        x, y = pos
        cell = self.model.get_cell_contents(x, y)[0]
        for d in cell.directions:
            dx, dy = Defaults.DIRECTION_VECTORS[d]
            nx, ny = x + dx, y + dy
            if self.model.in_bounds(nx, ny):
                yield (nx, ny)

    # ════════════════════════════════════════════════════════════
    #  AWARENESS SCAN
    # ════════════════════════════════════════════════════════════
    def _scan_ahead_for_obstacles(self):
        """Return indices of the first Stop‑cell or occupied cell on the path."""
        idx_stop = idx_vehicle = None
        rng = min(Defaults.VEHICLE_AWARENESS_RANGE, len(self.path))
        for idx in range(rng):
            cell = self.path[idx]
            if idx_stop is None and self._is_stop_cell(cell):
                idx_stop = idx  # inclusive obstacle
            if idx_vehicle is None and cell.occupied:
                idx_vehicle = idx  # exclusive obstacle
            if (idx_stop == 0) or (idx_vehicle == 0):
                break
        return idx_stop, idx_vehicle

    # ════════════════════════════════════════════════════════════
    #  STEP  (public) – orchestrates one simulation tick
    # ════════════════════════════════════════════════════════════
    def step(self):
        """Advance the vehicle for one tick, obeying lights and traffic."""
        # 1) Hard‑stop if we are currently parked on a Stop cell.
        if self._is_in_stopped_cell():
            self.base_speed = 0
            return

        # 2) Update speed (initialise or retain previous *base_speed*).
        self.current_speed = self.compute_speed()

        # 3) Look‑ahead scan & optional re‑routing.
        idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()
        idx_stop, idx_vehicle = self._maybe_recompute_path(idx_stop, idx_vehicle)

        # 4) Work out how far we may advance *this* tick.
        max_steps, blocked_by_vehicle = self._determine_max_steps(idx_stop, idx_vehicle)

        # 5) If another car blocks us right in front, mark a full stop.
        if max_steps <= 0:
            self.base_speed = 0
            return  # no movement possible this tick

        # 6) Execute the actual movement along the path.
        self._execute_movement(max_steps)

        # 7) If we consumed the entire planned path, recompute (allows leaving a Stop on green).
        if not self.path:
            self.path = self.compute_path()

    # ------------------------------------------------------------
    #  STEP helper methods (private)
    # ------------------------------------------------------------
    def _is_in_stopped_cell(self) -> bool:
        """Return *True* when the agent must wait because it sits on a red‑light cell."""
        if self._is_stop_cell(self.current_cell):
            return True
        return False

    def _maybe_recompute_path(self, idx_stop, idx_vehicle):
        """Attempt detours to avoid visible Stop cells or blocked cars."""
        # --- detour around Stop ahead ----------------------------------
        if idx_stop is not None:
            alt = self.compute_path(avoid_stops=True)
            if alt and alt != self.path:
                self.path = alt
                idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()

        # --- detour around blocking vehicle ----------------------------
        if idx_vehicle is not None and idx_vehicle <= Defaults.VEHICLE_AWARENESS_RANGE:
            alt = self.compute_path(avoid_vehicles=True)
            if alt and alt != self.path:
                self.path = alt
                idx_stop, idx_vehicle = self._scan_ahead_for_obstacles()

        return idx_stop, idx_vehicle

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
        """Move up to *max_steps* cells along `self.path`, handling dummies & occupancy."""
        old_pos = self.current_cell.get_position()

        for step_idx in range(max_steps):
            if not self.path:
                break  # already at destination
            next_cell = self.path[0]

            # Safety re‑checks at runtime
            if next_cell.occupied and next_cell is not self.current_cell:
                break
            if self._is_stop_cell(next_cell):
                # May *enter* the Stop cell only as the very last permitted step
                if next_cell is not self.current_cell and step_idx != max_steps - 1:
                    break

            new_pos = next_cell.get_position()
            self._remove_dummy(new_pos)
            self._move_to(next_cell, old_pos, new_pos)
            old_pos = new_pos
            self.path.pop(0)

        # Leave a dummy where we vacated
        self._place_dummy(old_pos)

    # ------------------------------------------------------------
    #  Internal helpers (occupy / move / dummies)
    # ------------------------------------------------------------
    def _occupy_initial_cell(self):
        self.current_cell.occupied = True
        self._remove_dummy(self.current_cell.get_position())

    def _move_to(self, cell: CellAgent, old_pos: tuple[int, int], new_pos: tuple[int, int]):
        self.current_cell.occupied = False
        self.current_cell = cell
        self.current_cell.occupied = True
        self.model.grid.move_agent(self, new_pos)
        self.direction = self._compute_direction(old_pos, new_pos)

    def _compute_direction(self, old_pos: tuple[int, int], new_pos: tuple[int, int]):
        vector = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
        inv = {v: k for k, v in Defaults.DIRECTION_VECTORS.items()}
        return inv.get(vector, self.direction)

    # ------------------------------------------------------------
    #  Dummy‑agent management
    # ------------------------------------------------------------
    def _place_dummy(self, pos: tuple[int, int]):
        self.model.place_dummy(pos[0], pos[1])

    def _remove_dummy(self, pos: tuple[int, int]):
        self.model.remove_dummy(pos[0], pos[1])

    # ══ Misc getters used by UI ══════════════════════════════════
    def get_current_direction(self):
        return self.direction

    def get_current_path(self):
        return [agent.get_position() for agent in self.path]

    def get_target(self):
        return self.target.get_position()

    def get_portrayal(self):
        return {
            "Shape": "circle",
            "r": 0.5,
            "Filled": True,
            "Layer": 1,
            "Color": "black",
            "Type": "Vehicle",
            "Direction": self.direction or "?",
        }
