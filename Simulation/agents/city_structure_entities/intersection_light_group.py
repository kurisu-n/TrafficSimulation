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
        pass
