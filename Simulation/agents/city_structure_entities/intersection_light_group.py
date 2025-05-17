from typing import Dict, List, Set, cast, TYPE_CHECKING
from mesa import Agent
from Simulation.config import Defaults
from Simulation.utilities.general import *

if TYPE_CHECKING:
    from Simulation.city_model import CityModel
    from Simulation.agents.city_structure_entities.cell import CellAgent


class IntersectionLightGroup(Agent):
    """Bundle the TrafficLight agents that guard one road intersection."""

    def __init__(self, custom_id: str, model, traffic_lights: List[Agent]):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.traffic_lights = traffic_lights
        self.neighbor_groups: Dict[str, "IntersectionLightGroup"] | None = None
        self.intermediate_groups: Set["IntersectionLightGroup"] | None = None
        self.opposite_pairs: dict[str, list] | None = None

        self.city_model = cast("CityModel", self.model)

        # Fixed-time state
        self.current_phase: int = 0  # 0 = N–S, 1 = E–W
        self.timer: int = 0
        self.green_duration: int = 20
        self.yellow_duration: int = 3
        self.all_red_duration: int = 2

        # Queue-actuated state
        self._qa_phase = 0  # 0=NS green,1=all-red→EW,2=EW green,3=all-red→NS
        self._qa_timer = 0
        self._gap_timer = 0
        self._last_arrival = 0
        self._ft_phase = 0
        self._ft_timer = 0

        # Pressure control state
        self._pc_timer = 0
        self._pc_phase = 0  # 0=NS green,1=all-red→EW,2=EW green,3=all-red→NS

        # Neighbor Pressure control state
        self._pressure_ns = 0
        self._pressure_ew = 0


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
    def step(self):
        algo = Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM
        if algo == "FIXED-TIME":
            self.run_fixed_time()
        elif algo == "QUEUE-ACTUATED":
            self.run_queue_actuated()
        elif algo == "PRESSURE-CONTROL":
            self.run_pressure_control()
        elif algo == "NEIGHBOR-ENHANCED-PRESSURE":
            self.run_neighbor_pressure_control()

    def run_fixed_time(self):
        """Two-phase fixed-time controller (N–S vs. E–W) with all-red interlocks."""
        # pull parameters
        green  = Defaults.TRAFFIC_LIGHT_GREEN_DURATION
        yellow = Defaults.TRAFFIC_LIGHT_YELLOW_DURATION
        redbuf = Defaults.TRAFFIC_LIGHT_ALL_RED_DURATION

        # initialize per-instance state on first call
        if not hasattr(self, "_ft_phase"):
            self._ft_phase = 0   # 0 = NS green, 1 = all-red, 2 = EW green, 3 = all-red
            self._ft_timer = 0

        self._ft_timer += 1
        ns_lights, ew_lights = self.get_opposite_traffic_lights().values()

        # Phase 0: NS green
        if self._ft_phase == 0:
            if self._ft_timer == 1:
                for tl in ns_lights: tl.set_light_go()
                for tl in ew_lights: tl.set_light_stop()
            if self._ft_timer >= green - yellow:
                self._ft_phase = 1
                self._ft_timer = 0

        # Phase 1: all-red before switching to EW
        elif self._ft_phase == 1:
            if self._ft_timer == 1:
                self.set_all_stop()
            if self._ft_timer >= redbuf:
                self._ft_phase = 2
                self._ft_timer = 0

        # Phase 2: EW green
        elif self._ft_phase == 2:
            if self._ft_timer == 1:
                for tl in ew_lights: tl.set_light_go()
                for tl in ns_lights: tl.set_light_stop()
            if self._ft_timer >= green - yellow:
                self._ft_phase = 3
                self._ft_timer = 0

        # Phase 3: all-red before switching back to NS
        elif self._ft_phase == 3:
            if self._ft_timer == 1:
                self.set_all_stop()
            if self._ft_timer >= redbuf:
                self._ft_phase = 0
                self._ft_timer = 0

    def run_queue_actuated(self):
        """
        Queue-actuated controller:
          - MIN_GREEN: minimum green interval before switching
          - MAX_GREEN: cap to prevent starvation
          - GAP: no-arrival gap-out interval after MIN_GREEN
        """
        # Parameters (fallbacks if not in Defaults)
        min_green = getattr(Defaults, 'TRAFFIC_LIGHT_MIN_GREEN', 5)
        max_green = getattr(Defaults, 'TRAFFIC_LIGHT_MAX_GREEN', 30)
        gap      = getattr(Defaults, 'TRAFFIC_LIGHT_GAP', 3)
        redbuf   = Defaults.TRAFFIC_LIGHT_ALL_RED_DURATION

        # Advance timers
        self._qa_timer += 1

        # Get queues: count occupied road blocks
        ns_cells, ew_cells = self.get_opposite_traffic_lights().values()
        ns_queue = sum(
            1
            for tl in ns_cells
            for rb in tl.assigned_road_blocks
            if rb.occupied
        )
        ew_queue = sum(
            1
            for tl in ew_cells
            for rb in tl.assigned_road_blocks
            if rb.occupied
        )

        # Determine current vs opposite queues
        if self._qa_phase == 0:
            current_q, opp_q = ns_queue, ew_queue
        elif self._qa_phase == 2:
            current_q, opp_q = ew_queue, ns_queue
        else:
            current_q = opp_q = None

        # Green phases
        if self._qa_phase in (0, 2):
            # On phase start, set lights
            if self._qa_timer == 1:
                lights_on  = ns_cells if self._qa_phase == 0 else ew_cells
                lights_off = ew_cells if self._qa_phase == 0 else ns_cells
                for tl in lights_on:
                    tl.set_light_go()
                for tl in lights_off:
                    tl.set_light_stop()
                self._last_arrival = current_q
                self._gap_timer = 0

            # Update gap timer on arrivals
            if current_q > self._last_arrival:
                self._last_arrival = current_q
                self._gap_timer = 0
            else:
                self._gap_timer += 1

            # Check for switch conditions
            if (
                self._qa_timer >= min_green and (
                    self._gap_timer >= gap or
                    self._qa_timer >= max_green or
                    (opp_q > current_q and current_q == 0)
                )
            ):
                # move to all-red before next green
                self._qa_phase = 1 if self._qa_phase == 0 else 3
                self._qa_timer = 0

        # All-red before switching
        elif self._qa_phase in (1, 3):
            if self._qa_timer == 1:
                self.set_all_stop()
            if self._qa_timer >= redbuf:
                # Next green phase
                self._qa_phase = 2 if self._qa_phase == 1 else 0
                self._qa_timer = 0

    def run_pressure_control(self):
        min_green = getattr(Defaults, 'TRAFFIC_LIGHT_MIN_GREEN', 5)
        max_green = getattr(Defaults, 'TRAFFIC_LIGHT_MAX_GREEN', 60)
        redbuf    = Defaults.TRAFFIC_LIGHT_ALL_RED_DURATION

        self._pc_timer += 1
        ns_cells, ew_cells = self.get_opposite_traffic_lights().values()

        # Local pressures
        local_ns = sum(1 for tl in ns_cells for rb in tl.assigned_road_blocks if rb.occupied)
        local_ew = sum(1 for tl in ew_cells for rb in tl.assigned_road_blocks if rb.occupied)
        p_ns = local_ns - local_ew
        p_ew = local_ew - local_ns

        # Sum neighbor pressure
        total_ns, total_ew = p_ns, p_ew
        for neighbor in self.get_neighbor_groups():
            n_ns, n_ew = 0, 0
            n_cells_ns, n_cells_ew = neighbor.get_opposite_traffic_lights().values()
            n_ns = sum(1 for tl in n_cells_ns for rb in tl.assigned_road_blocks if rb.occupied)
            n_ew = sum(1 for tl in n_cells_ew for rb in tl.assigned_road_blocks if rb.occupied)
            total_ns += (n_ns - n_ew)
            total_ew += (n_ew - n_ns)

        # Decide desired
        desired = 0 if total_ns >= total_ew else 2

        # Green execution
        if self._pc_phase in (0, 2):
            if self._pc_timer == 1:
                lights_on  = ns_cells if self._pc_phase == 0 else ew_cells
                lights_off = ew_cells if self._pc_phase == 0 else ns_cells
                for tl in lights_on: tl.set_light_go()
                for tl in lights_off: tl.set_light_stop()
            # Transition
            if self._pc_timer >= min_green and self._pc_phase != desired:
                self._pc_phase = 1 if self._pc_phase == 0 else 3
                self._pc_timer = 0

        # All-red buffer
        elif self._pc_phase in (1, 3):
            if self._pc_timer == 1: self.set_all_stop()
            if self._pc_timer >= redbuf:
                self._pc_phase = desired
                self._pc_timer = 0

    def run_neighbor_pressure_control(self):
        """
        Decentralized pressure‐based control using only immediate neighbors.
        Each group:
          1) computes and stores its own (p_ns, p_ew)
          2) sums its own p’s plus neighbor._pressure_* values
          3) runs the usual phase‐switch logic
        """

        # ── 0) init per‐instance state ───────────────────────────────────
        if not hasattr(self, '_ne_phase'):
            self._ne_phase = 0  # 0=NS-green,1=all-red→EW,2=EW-green,3=all-red→NS
            self._ne_timer = 0

        min_green = getattr(Defaults, 'TRAFFIC_LIGHT_MIN_GREEN', 5)
        redbuf = Defaults.TRAFFIC_LIGHT_ALL_RED_DURATION

        # ── 1) compute & store *local* pressure ─────────────────────────
        ns_cells, ew_cells = self.get_opposite_traffic_lights().values()
        local_ns = sum(1 for tl in ns_cells for rb in tl.assigned_road_blocks if rb.occupied)
        local_ew = sum(1 for tl in ew_cells for rb in tl.assigned_road_blocks if rb.occupied)
        # pressure in each direction
        self._pressure_ns = local_ns - local_ew
        self._pressure_ew = local_ew - local_ns

        # ── 2) aggregate with *neighbors’* pressures ────────────────────
        total_ns = self._pressure_ns
        total_ew = self._pressure_ew
        for neighbor in self.get_neighbor_groups().values():
            # direct attribute access—no re‐scanning
            total_ns += getattr(neighbor, '_pressure_ns', 0)
            total_ew += getattr(neighbor, '_pressure_ew', 0)

        # ── 3) decide desired axis ───────────────────────────────────────
        desired = 0 if total_ns >= total_ew else 2

        # ── 4) finite‐state timing & phase switching ─────────────────────
        self._ne_timer += 1

        if self._ne_phase in (0, 2):
            if self._ne_timer == 1:
                self._apply_phase(self._ne_phase)
            if self._ne_timer >= min_green and self._ne_phase != desired:
                # go into all‐red before the switch
                self._ne_phase = 1 if self._ne_phase == 0 else 3
                self._ne_timer = 0

        else:  # all‐red phases 1 or 3
            if self._ne_timer == 1:
                self.set_all_stop()
            if self._ne_timer >= redbuf:
                # switch to the chosen green
                self._ne_phase = desired
                self._ne_timer = 0

    def _apply_phase(self, phase):
        ns_lights, ew_lights = self.get_opposite_traffic_lights().values()
        if phase == 0:
            for tl in ns_lights: tl.set_light_go()
            for tl in ew_lights: tl.set_light_stop()
        else:
            for tl in ew_lights: tl.set_light_go()
            for tl in ns_lights: tl.set_light_stop()


