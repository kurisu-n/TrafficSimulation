import random
from typing import Dict, List, Set, cast, TYPE_CHECKING, Optional
from mesa import Agent
from Simulation.config import Defaults
from Simulation.utilities.general import *
from Simulation.utilities.numba_utilities import compute_max_pressure, compute_approach_queue, compute_total_queue

if TYPE_CHECKING:
    from Simulation.city_model import CityModel
    from Simulation.agents.city_structure_entities.cell import CellAgent

if Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM.startswith("NEIGHBOR_RL"):
    from Simulation.utilities.light_group_managment.reinforcement_learning import (
        make_policy_net,     # only used by NEIGHBOR_RL
        run_rl_control,      # single-agent
        run_batched_rl_control,   # batched (called from CityModel)
    )
    import tensorflow as tf

import numpy as np


class IntersectionLightGroup(Agent):
    """Bundle the TrafficLight agents that guard one road intersection."""

    def __init__(self, custom_id: str, model, traffic_lights: List[Agent]):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.traffic_lights = traffic_lights
        self.intersection_cells: List[CellAgent] = []
        self.neighbor_groups: Dict[str, "IntersectionLightGroup"] | None = None
        self.intermediate_groups: Set["IntersectionLightGroup"] | None = None
        self.opposite_pairs: dict[str, list] | None = None

        self.city_model = cast("CityModel", self.model)

        # --- Phase Change parameters (shared) ---
        self.current_phase: Optional[int] = None  # 0 = N–S, 1 = E–W
        self.pending_phase: Optional[int] = None
        self.transition_timer: int = 0
        self.clearance_timer: int = 0

        # --- Queue-actuated parameters ---
        self._ft_phase: int = 0  # 0 = NS green, 1 = EW green
        self.fixed_time_timer: int = 0
        self.green_duration: int = Defaults.TRAFFIC_LIGHT_GREEN_DURATION

        # --- Queue-actuated parameters ---
        self.queue_timer: int = 0
        self.gap_timer: int = 0
        self.last_arrival: int = 0

        # --- Pressure-based parameters ---
        self.ns_pressure = 0
        self.ew_pressure = 0

        # --- Neighbor-pressure trackers ---
        self._ne_timer: int = 0
        self._ne_last_request: Optional[int] = None

        # --- Neighbor green-wave trackers ---
        self._nc_timer: int = 0
        self._nc_gap_timer: int = 0
        self._nc_last_arrival: int = 0

        if Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM == "NEIGHBOR_RL":
            # each agent owns its own network
            input_dim = 9
            hidden = getattr(Defaults, "RL_POLICY_HIDDEN", 32)
            lr = getattr(Defaults, "RL_LEARNING_RATE", 1e-3)
            self.rl_policy = make_policy_net(input_dim, hidden)  # on GPU if available
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.memory: list = []
            self._rl_phase = 0
            self._rl_timer = 0

        elif Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM == "NEIGHBOR_RL_BATCHED":
            # placeholders – CityModel will attach shared policy/optimizer later
            self.rl_policy = None  # will be set to shared model
            self.optimizer = None
            self.memory = []
            self._rl_phase = 0
            self._rl_timer = 0

        self.ns_in_coords = []
        self.ns_out_coords = []
        self.ew_in_coords = []
        self.ew_out_coords = []

        self.initialize_cached_lane_coords()

        if Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM != "DISABLED":
            self.apply_phase(self._ft_phase)

    def initialize_cached_lane_coords(self):
        """
        Extract and store lane cell coordinates for each signal direction
        as NumPy arrays for fast JIT access.
        """

        for tl in self.traffic_lights:
            for rb in tl.assigned_road_blocks:  # ← controlled lane cells
                x, y = rb.position
                if "N" in rb.directions or "S" in rb.directions:
                    if y < tl.position[1]:
                        self.ns_in_coords.append((x, y))
                    else:
                        self.ns_out_coords.append((x, y))
                elif "E" in rb.directions or "W" in rb.directions:
                    if x < tl.position[0]:
                        self.ew_in_coords.append((x, y))
                    else:
                        self.ew_out_coords.append((x, y))

        # Convert to NumPy arrays with fixed dtype
        self.ns_in_coords = np.array(self.ns_in_coords, dtype=np.int32)
        self.ns_out_coords = np.array(self.ns_out_coords, dtype=np.int32)
        self.ew_in_coords = np.array(self.ew_in_coords, dtype=np.int32)
        self.ew_out_coords = np.array(self.ew_out_coords, dtype=np.int32)
    # ------------------------------------------------------------------
    # Link discovery
    # ------------------------------------------------------------------
    def populate_links(self, max_depth: int = 1000) -> None:
        """Fill *next*, *intermediate* and *opposite* group look-ups."""

        # ── helpers ──────────────────────────────────────────────────────
        def _band_or_single(idx, bands):
            band = self.city_model._find_band_covering(idx, bands)
            return band if band else (idx, idx, "R4", None)

        def blocks_all_lanes(ix, iy, d):
            def _band_clear(x0, x1, y0, y1):
                return all(self.city_model.is_type(xx, yy, "Intersection")
                           for yy in range(y0, y1 + 1)
                           for xx in range(x0, x1 + 1))

            if d in ("N", "S"):
                vx0, vx1, *_ = _band_or_single(ix, self.city_model.vertical_bands)
                if vx1 == vx0:
                    good_v = self.city_model.is_type(vx0, iy, "Intersection")
                    hy0, hy1, *_ = _band_or_single(iy, self.city_model.horizontal_bands)
                    return good_v and (hy1 != hy0 or self.city_model.is_type(ix, hy0, "Intersection"))
                return _band_clear(vx0, vx1, iy, iy)

            hy0, hy1, *_ = _band_or_single(iy, self.city_model.horizontal_bands)
            if hy1 == hy0:
                good_h = self.city_model.is_type(ix, hy0, "Intersection")
                vx0, vx1, *_ = _band_or_single(ix, self.city_model.vertical_bands)
                return good_h and (vx1 != vx0 or self.city_model.is_type(vx0, iy, "Intersection"))
            return _band_clear(ix, ix, hy0, hy1)

        # ── containers ───────────────────────────────────────────────────
        self.neighbor_groups = {}
        self.intermediate_groups = set()
        self.opposite_pairs = []

        # ── pick any diagonal-adjacent intersection as start ─────────────
        diag_intersections = []
        for tl in self.traffic_lights:
            tl = cast("CellAgent", tl)
            lx, ly = tl.position
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = lx + dx, ly + dy
                if self.city_model.in_bounds(nx, ny) and self.city_model.is_type(nx, ny, "Intersection"):
                    diag_intersections.append((nx, ny))

        for cx, cy in diag_intersections:
            for d in Defaults.AVAILABLE_DIRECTIONS:  # N,S,E,W
                x, y, steps = cx, cy, 0
                while steps < max_depth:
                    x, y = self.city_model.next_cell_in_direction(x, y, d)
                    if not self.city_model.in_bounds(x, y):
                        break
                    cell = self.city_model.get_cell_contents(x, y)[0]
                    if cell.cell_type != "Intersection":
                        steps += 1
                        continue
                    g = getattr(cell, "intersection_group", None)
                    if g is None or g is self:
                        steps += 1
                        continue
                    key = f"_blocks_{d}"
                    if not hasattr(g, key):
                        setattr(g, key, blocks_all_lanes(*cell.position, d))
                    if getattr(g, key):
                        self.neighbor_groups[d] = g
                        break
                    self.intermediate_groups.add(g)
                    steps += 1

        # ── detect opposite-axis traffic lights ─────────────────────────
        axis_dirs = {"vertical": {"N": [], "S": []},
                     "horizontal": {"E": [], "W": []}}

        for tl in self.traffic_lights:
            tl = cast("CellAgent", tl)
            for cb in tl.controlled_blocks:
                cbx, cby = cb.position
                for d in cb.directions:
                    nx, ny = self.city_model.next_cell_in_direction(cbx, cby, d)
                    if not self.city_model.in_bounds(nx, ny):
                        continue
                    if self.city_model.get_cell_contents(nx, ny)[0].cell_type == "Intersection" and \
                            getattr(self.city_model.get_cell_contents(nx, ny)[0], "intersection_group", None) is self:
                        axis = "vertical" if d in ("N", "S") else "horizontal"
                        axis_dirs[axis][d].append(tl)
                        break          # one direction per light is enough

        # ── build axis→lights dictionary  ───────────────────────────────
        self.opposite_pairs = {
            "N-S": [],   # vertical axis  (north <-> south)
            "W-E": []    # horizontal axis (west <-> east)
        }

        # unique order-preserving collection helper
        def _merge_unique(dst_list, src_iter):
            seen = set(dst_list)
            for item in src_iter:
                if item not in seen:
                    dst_list.append(item)
                    seen.add(item)

        # gather lights for each axis (0, 1 or 2 per axis)
        _merge_unique(self.opposite_pairs["N-S"],
                      axis_dirs["vertical"]["N"] + axis_dirs["vertical"]["S"])
        _merge_unique(self.opposite_pairs["W-E"],
                      axis_dirs["horizontal"]["E"] + axis_dirs["horizontal"]["W"])

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def is_intersection_occupied(self) -> bool:
        """Return True if any of this group’s intersection cells currently has a vehicle."""
        occ = self.city_model.occupancy_map
        return any(
            occ[cell.position[1], cell.position[0]]
            for cell in self.intersection_cells
        )

    def get_neighbor_groups(self):
        if self.neighbor_groups is None or self.neighbor_groups == []:
            self.populate_links()
        return self.neighbor_groups

    def get_intermediate_groups(self):
        if self.neighbor_groups is None or self.neighbor_groups == []:
            self.populate_links()
        return self.intermediate_groups

    def get_opposite_traffic_lights(self):
        if self.opposite_pairs is None or self.opposite_pairs == []:
            self.populate_links()
        return self.opposite_pairs

    # ------------------------------------------------------------------
    # Light control helpers
    # ------------------------------------------------------------------
    def _apply(self, fn):
        for tl in self.traffic_lights:
            fn(tl)

    def set_all_stop(self):
        self._apply(lambda tl: tl.set_light_stop())

    def set_all_go(self):
        self._apply(lambda tl: tl.set_light_go())

    def set_all_go_with_neighbors(self):
        self.set_all_go()
        for g in self.get_neighbor_groups().values():
            g.set_all_go()

    def set_all_stop_with_neighbors(self):
        self.set_all_stop()
        for g in self.get_neighbor_groups().values():
            g.set_all_stop()

    def set_all_go_with_neighbors_and_intermediate(self):
        self.set_all_go()
        for g in self.get_neighbor_groups().values():
            g.set_all_go()
        for g in self.get_intermediate_groups():
            g.set_all_go()

    def set_all_stop_with_neighbors_and_intermediate(self):
        self.set_all_stop()
        for g in self.get_neighbor_groups().values():
            g.set_all_stop()
        for g in self.get_intermediate_groups():
            g.set_all_stop()

    # ------------------------------------------------------------------

    def _execute_phase_change(self):
        """
        Execute any pending phase change with yellow/all-red transition
        and the "wait until clear" hard rule. Called at end of each step.
        """
        if self.pending_phase is None:
            return

        if Defaults.TRAFFIC_LIGHT_TRANSITION_DURATION_ENABLED and self.transition_timer > 0:
            self.transition_timer -= 1
            self.set_all_stop()
            return

        if Defaults.TRAFFIC_LIGHT_TRANSITION_CLEARANCE_ENABLED and self.is_intersection_occupied() and self.clearance_timer > 0:
            self.clearance_timer -= 1
            self.set_all_stop()
            return

        if Defaults.TRAFFIC_LIGHT_TRANSITION_DURATION_ENABLED:
            self.transition_timer = Defaults.TRAFFIC_LIGHT_YELLOW_DURATION + Defaults.TRAFFIC_LIGHT_ALL_RED_DURATION

        if Defaults.TRAFFIC_LIGHT_TRANSITION_CLEARANCE_ENABLED and self.is_intersection_occupied():
            self.clearance_timer = Defaults.TRAFFIC_LIGHT_CLEARANCE_MAX_DURATION

        # Actually set green for the committed phase
        ns_lights, ew_lights = self.get_opposite_traffic_lights().values()
        if self.pending_phase == 0:
            for tl in ns_lights:
                tl.set_light_go()
            for tl in ew_lights:
                tl.set_light_stop()
        else:
            for tl in ew_lights:
                tl.set_light_go()
            for tl in ns_lights:
                tl.set_light_stop()

        # Intersection clear and transition done: commit the phase
        new_phase = self.pending_phase
        self.pending_phase = None
        self.current_phase = new_phase

    def apply_phase(self, phase: int):
        """
        Called by control algorithms to *request* a phase change.
        Only registers the desired phase; execution deferred until end of step.
        """
        if phase == self.current_phase or phase == self.pending_phase:
            return
        self.pending_phase = phase

    # ------------------------------------------------------------------
    def step(self):

        if self.pending_phase is None:
            algo = Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM

            if algo == "FIXED_TIME":
                self.run_fixed_time()
            elif algo == "QUEUE_ACTUATED":
                self.run_queue_actuated()
            elif algo == "PRESSURE_CONTROL":
                self.run_pressure_control()
            elif algo == "NEIGHBOR_PRESSURE_CONTROL":
                self.run_neighbor_pressure_control()
            elif algo == "NEIGHBOR_GREEN_WAVE":
                self.run_neighbor_green_wave()
            elif algo == "NEIGHBOR_RL":
                run_rl_control(self)
            elif algo == "NEIGHBOR_RL_BATCHED":
                # decision handled centrally in CityModel → do nothing here
                pass


        self._execute_phase_change()

    # ------------------------------------------------------------------

    def run_fixed_time(self):
        """
        Two-phase fixed-time controller (N–S vs. E–W) using apply_phase for transitions.
        """
        # Advance timer
        self.fixed_time_timer += 1

        # On first tick of this phase, request green
        if self.fixed_time_timer == 1:
            self.apply_phase(self._ft_phase)

        # After full green duration, toggle phase and reset timer
        if self.fixed_time_timer >= self.green_duration:
            self._ft_phase = 1 - self._ft_phase  # Toggle between 0↔1
            self.fixed_time_timer = 0

    @staticmethod
    def to_int32(arr):
        arr = np.asarray(arr, dtype=np.int32)
        return np.ascontiguousarray(arr.reshape((-1, 2)), dtype=np.int32)

    def run_pressure_control(self):
        occ_map = self.city_model.occupancy_map.astype(np.int32)
        ns_pressure, ew_pressure = compute_max_pressure(
            self.to_int32(self.ns_in_coords), self.to_int32(self.ns_out_coords),
            self.to_int32(self.ew_in_coords), self.to_int32(self.ew_out_coords),
            self.to_int32(occ_map)
        )
        self.ns_pressure = ns_pressure
        self.ew_pressure = ew_pressure

        if ns_pressure > ew_pressure:
            self.apply_phase(0)
        else:
            self.apply_phase(1)

    def run_queue_actuated(self):
        self.queue_timer += 1
        occ = self.city_model.occupancy_map.astype(np.int32)

        ns_q = compute_approach_queue(occ, self.to_int32(self.ns_in_coords))
        ew_q = compute_approach_queue(occ, self.to_int32(self.ew_in_coords))

        if self.current_phase == 0:
            current_q, opp_q = ns_q, ew_q
        else:
            current_q, opp_q = ew_q, ns_q

        if self.queue_timer == 1:
            self.last_arrival = current_q
            self.gap_timer = 0

        if current_q > self.last_arrival:
            self.last_arrival = current_q
            self.gap_timer = 0
        else:
            self.gap_timer += 1

        if (
            self.queue_timer >= Defaults.TRAFFIC_LIGHT_QUEUE_ACTUATED_MIN_GREEN and (
                self.gap_timer >= Defaults.TRAFFIC_LIGHT_QUEUE_ACTUATED_GAP or
                self.queue_timer >= Defaults.TRAFFIC_LIGHT_QUEUE_ACTUATED_MAX_GREEN or
                (opp_q > current_q == 0)
            )
        ):
            next_phase = 1 - self.current_phase
            self.apply_phase(next_phase)
            self.queue_timer = 0

    def run_neighbor_pressure_control(self):
        occ_map = self.city_model.occupancy_map.astype(np.int32)

        ns_pressure, ew_pressure = compute_max_pressure(
            self.to_int32(self.ns_in_coords), self.to_int32(self.ns_out_coords),
            self.to_int32(self.ew_in_coords), self.to_int32(self.ew_out_coords),
            occ_map
        )

        neighbors = self.get_neighbor_groups()
        for d, g in neighbors.items():
            if hasattr(g, "ns_pressure") and hasattr(g, "ew_pressure"):
                if d in ("N", "S"):
                    ns_pressure -= getattr(g, "ns_pressure", 0)
                elif d in ("E", "W"):
                    ew_pressure -= getattr(g, "ew_pressure", 0)

        self.ns_pressure = ns_pressure
        self.ew_pressure = ew_pressure

        if ns_pressure > ew_pressure:
            self.apply_phase(0)
        else:
            self.apply_phase(1)

    def run_neighbor_green_wave(self):
        occ_map = self.city_model.occupancy_map.astype(np.int32)
        ns_q = compute_total_queue(self.to_int32(self.ns_in_coords), occ_map)
        ew_q = compute_total_queue(self.to_int32(self.ew_in_coords), occ_map)

        neighbors = self.get_neighbor_groups()
        favor_ns = False
        favor_ew = False

        for d, g in neighbors.items():
            if not hasattr(g, "current_phase"):
                continue
            if d in ("N", "S") and g.current_phase == 0:
                favor_ns = True
            if d in ("E", "W") and g.current_phase == 1:
                favor_ew = True

        if favor_ns and not favor_ew:
            self.apply_phase(0)
        elif favor_ew and not favor_ns:
            self.apply_phase(1)
        else:
            if ns_q > ew_q:
                self.apply_phase(0)
            else:
                self.apply_phase(1)



