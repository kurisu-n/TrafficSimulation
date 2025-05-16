# city_model.py ─ refactored to share the central config
from __future__ import annotations
import numpy as np
import random
from typing import List, Any
from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from numba import njit

from Simulation.agents.rain import RainManager
from Simulation.agents.vehicles.vehicle_base import VehicleAgent
from Simulation.config import Defaults
from Simulation.agents.city_structure_entities.cell import CellAgent
from Simulation.agents.city_structure_entities.city_block import CityBlock
from Simulation.agents.city_structure_entities.intersection_light_group import IntersectionLightGroup
from Simulation.agents.dummy import DummyAgent
from Simulation.agents.dynamic_traffic_generator import DynamicTrafficAgent
from Simulation.utilities.general import overlay_dynamic
from Simulation.utilities.pathfinding import astar_jit
from Simulation.utilities.pathfinding.astar_cuda import PathPlanner

class CityModel(Model):
    def __init__(self,
                 width=Defaults.WIDTH,
                 height=Defaults.HEIGHT,
                 wall_thickness = Defaults.WALL_THICKNESS,
                 sidewalk_ring_width = Defaults.SIDEWALK_RING_WIDTH,
                 ring_road_type = Defaults.RING_ROAD_TYPE,
                 allow_extra_highways = Defaults.ALLOW_EXTRA_HIGHWAYS,
                 extra_highways_chance = Defaults.EXTRA_HIGHWAY_CHANCE,
                 r2_r3_chance_split = Defaults.R2_R3_CHANCE_SPLIT,
                 min_block_spacing = Defaults.MIN_BLOCK_SPACING,
                 max_block_spacing = Defaults.MAX_BLOCK_SPACING,
                 optimized_intersections = Defaults.OPTIMISED_INTERSECTIONS,
                 carve_subblock_roads = Defaults.CARVE_SUBBLOCK_ROADS,
                 subblock_roads_have_intersections = Defaults.SUBBLOCK_ROADS_HAVE_INTERSECTIONS,
                 subblock_chance = Defaults.SUBBLOCK_CHANGE,
                 subblock_road_type = Defaults.SUBBLOCK_ROAD_TYPE,
                 min_subblock_spacing = Defaults.MIN_SUBBLOCK_SPACING,
                 highway_offset_from_edges = Defaults.HIGHWAY_OFFSET,
                 traffic_light_range = Defaults.TRAFFIC_LIGHT_RANGE,
                 forward_traffic_light_range = Defaults.FORWARD_TRAFFIC_LIGHT_RANGE,
                 forward_traffic_light_range_intersections = Defaults.FORWARD_TRAFFIC_LIGHT_INTERSECTIONS,
                 gradual_city_block_resources = Defaults.GRADUAL_CITY_BLOCK_RESOURCES,
                 use_dummy_agents = Defaults.USE_DUMMY_AGENTS,
                 cache_cell_portrayal = Defaults.CACHE_CELL_PORTRAYAL,
                 seed=None):

        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.wall_thickness = wall_thickness
        self.sidewalk_ring_width = sidewalk_ring_width
        self.ring_road_type = ring_road_type
        self.highway_offset_from_edges = highway_offset_from_edges
        self.allow_extra_highways = allow_extra_highways
        self.extra_highways_chance = extra_highways_chance
        self.r2_r3_chance_split = r2_r3_chance_split
        self.min_block_spacing = min_block_spacing
        self.max_block_spacing = max_block_spacing
        self.optimized_intersections = optimized_intersections
        self.carve_subblock_roads = carve_subblock_roads
        self.subblock_chance = subblock_chance
        self.subblock_road_type = subblock_road_type
        self.min_subblock_spacing = min_subblock_spacing
        self.subblock_roads_have_intersections = subblock_roads_have_intersections
        self.traffic_light_range = traffic_light_range
        self.forward_traffic_light_range = forward_traffic_light_range
        self.forward_traffic_light_range_intersections = forward_traffic_light_range_intersections
        self.gradual_city_block_resources = gradual_city_block_resources

        self.use_dummy_agents = use_dummy_agents
        self.cache_cell_portrayal = cache_cell_portrayal

        self.user_selected_traffic_light = None
        self.user_selected_intersection = None
        self.user_selected_opposite = None

        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)

        self.stop_cells: set[tuple[int, int]] = set()
        self.vehicle_cells: set[tuple[int, int]] = set()

        # interior bounds
        self.interior_x_min = self.wall_thickness + self.sidewalk_ring_width
        self.interior_x_max = self.width  - (self.wall_thickness + self.sidewalk_ring_width) - 1
        self.interior_y_min = self.wall_thickness + self.sidewalk_ring_width
        self.interior_y_max = self.height - (self.wall_thickness + self.sidewalk_ring_width) - 1

        self._blocks_data = []
        self.step_count = 0

        # trackers for quick lookup
        self.block_entrances   = []
        self.highway_entrances = []
        self.highway_exits     = []
        self.controlled_roads  = []
        self.traffic_lights    = []
        self.intersection_light_groups = []

        self.cell_agent_cache = {}
        self.path_cache = {}
        self.tick_cache = {}
        self.tick_grid = None
        self.cell_lookup = {}
        self._flow_mask = None

        self.occ: np.ndarray = np.zeros((self.width, self.height), dtype=bool)

        # build sequence
        self._place_thick_wall()
        self._place_sidewalk_inner_ring()
        self._clear_interior()
        self._build_roads_and_sidewalks()

        if self.carve_subblock_roads:
            self._carve_subblock_roads()

        self._flood_fill_blocks_storing_data()
        self._eliminate_dead_ends()
        self._upgrade_r2_to_intersections()
        self._final_place_block_entrances()
        self._remove_invalid_intersection_directions()
        self._add_entrance_directions()
        self._add_traffic_lights()
        self._create_intersection_light_groups()
        self._instantiate_city_blocks()

        if self.use_dummy_agents:
            self._add_dummy_agents()

        self._populate_cell_cache()

        if Defaults.PATHFINDING_METHOD == "CUDA":
            self.path_planner = PathPlanner(self)

        if Defaults.RAIN_ENABLED:
            self.rains = []
            rain_manager = RainManager("RainManager", self)
            self.schedule.add(rain_manager)

        if Defaults.ENABLE_TRAFFIC:
            self.dynamic_traffic_generator = DynamicTrafficAgent("DTA",self)
            self.schedule.add(self.dynamic_traffic_generator)



    # -------------------------------------------------------------------
    #  intersection factory
    # -------------------------------------------------------------------
    def _make_intersection(self, x: int, y: int) -> None:
        """
        Upgrade (x,y) to a four‑way **Intersection** following these rules:

        ──  OPTIMISED=False  ───────────────────────────────────────────
        • Every overlap is turned into an intersection
          → the entire width of the multi‑lane road is upgraded.

        ──  OPTIMISED=True   ───────────────────────────────────────────
        • For single✕single → one normal intersection cell.
        • For single✕multi  → **only the outer‑most lane(s)** of the
          thicker road become intersections, inner lanes stay as
          normal road cells.
        """

        # ---------- helpers -------------------------------------------------
        def _dummy_band(coord: int, rtype: str):
            return coord, coord, rtype, None  # (start, end, type, dir)

        def _band_info(band, coord: int):
            st, en, rt, bd = band
            width = en - st + 1
            off   = coord - st
            return st, en, rt, bd, width, off

        def _ensure_intersection(cx: int, cy: int):
            cell_agents = self.get_cell_contents(cx, cy)
            if cell_agents and cell_agents[0].cell_type == "Intersection":
                return
            self.place_cell(cx, cy, "Intersection",
                               f"Intersection_{cx}_{cy}")
            intersection = self.get_cell_contents(cx, cy)[0]
            intersection.directions = Defaults.AVAILABLE_DIRECTIONS
            if hasattr(self, "_intersection_cells"):
                self._intersection_cells.add((cx, cy))
        # --------------------------------------------------------------------

        # ── 1. find (or fabricate) the covering bands ─────────────────────
        hband = self._find_band_covering(y, self.horizontal_bands)
        vband = self._find_band_covering(x, self.vertical_bands)

        if not hband and (
                self.is_type(x, y, self.subblock_road_type) or
                any(self.is_type(nx, y, self.subblock_road_type) for nx in (x - 1, x + 1))
        ):
            hband = _dummy_band(y, self.subblock_road_type)

        if not vband and (
                self.is_type(x, y, self.subblock_road_type) or
                any(self.is_type(x, ny, self.subblock_road_type) for ny in (y - 1, y + 1))
        ):
            vband = _dummy_band(x, self.subblock_road_type)

        if not (hband and vband):               # not a real crossing
            return

        # ── 2. band details ───────────────────────────────────────────────
        h_st, h_en, h_rt, h_bd, h_sz, h_off = _band_info(hband, y)
        v_st, v_en, v_rt, v_bd, v_sz, v_off = _band_info(vband, x)

        single_vs_multi = (
            (h_sz == 1 and v_sz > 1) or
            (v_sz == 1 and h_sz > 1)
        )

        # ── 3. OPTIMISED Mode – keep only outer‑most lanes ────────────────
        if self.optimized_intersections and single_vs_multi:
            if h_sz > 1:                         # horizontal is the multi
                multi_rt, multi_orient = h_rt, "horizontal"
                multi_off, multi_sz, bdir = h_off, h_sz, h_bd
            else:                               # vertical is the multi
                multi_rt, multi_orient = v_rt, "vertical"
                multi_off, multi_sz, bdir = v_off, v_sz, v_bd

            if multi_off not in (0, multi_sz - 1):
                # inner lane → revert to normal road cell
                dirs = self._compute_lane_dirs(x,y,
                    multi_rt, multi_orient, multi_off, multi_sz, bdir
                )
                self.place_cell(x, y, multi_rt, f"{multi_rt}_{x}_{y}")
                ag = self.get_cell_contents(x, y)[0]
                ag.directions = dirs
                if hasattr(self, "_intersection_cells"):
                    self._intersection_cells.discard((x, y))

                self._road_cells[(x, y)] = (
                    multi_rt, multi_orient, multi_off, multi_sz, bdir
                )
                return

            # outer‑most lane ➜ real intersection
            _ensure_intersection(x, y)
            return

        # ── 4. non‑optimised mode OR multi✕multi ───────────────────────
        _ensure_intersection(x, y)


    def _compute_highway_inset(self):
        return self.interior_x_min + self.highway_offset_from_edges

    # -----------------------------------------------------------------------
    # Boundary wall
    # -----------------------------------------------------------------------
    def _place_thick_wall(self):
        w, h = self.get_width(), self.get_height()
        for y in range(h):
            for x in range(w):
                self.place_cell(x, y, "Wall", f"Wall_{x}_{y}")

    def _replace_if_wall(self, x, y, new_type, new_id):
        ags = self.get_cell_contents(x, y)
        if ags and ags[0].cell_type == "Wall":
            self.place_cell(x, y, new_type, new_id)

    # -----------------------------------------------------------------------
    # Sidewalk ring (hug every wall cell’s inner face)
    # -----------------------------------------------------------------------
    def _place_sidewalk_inner_ring(self):
        w, h = self.get_width(), self.get_height()
        ws = self.wall_thickness
        sr = self.sidewalk_ring_width

        # carve each sidewalk “layer” depth sr
        for layer in range(sr):
            # compute which rows are this layer’s top & bottom
            y_top    = ws + layer
            y_bottom = h - ws - 1 - layer

            # === horizontal faces ===
            # only from x=ws … x=(w-ws-1) so we skip the corner columns
            for x in range(ws, w - ws):
                if self.is_type(x, y_top, "Wall"):
                    self.place_cell(x, y_top, "Sidewalk",
                                       f"Sidewalk_{x}_{y_top}")
                if self.is_type(x, y_bottom, "Wall"):
                    self.place_cell(x, y_bottom, "Sidewalk",
                                       f"Sidewalk_{x}_{y_bottom}")

            # === vertical faces ===
            x_left  = ws + layer
            x_right = w - ws - 1 - layer
            # only from y=ws … y=(h-ws-1) so we skip the corner rows
            for y in range(ws, h - ws):
                if self.is_type(x_left, y, "Wall"):
                    self.place_cell(x_left, y, "Sidewalk",
                                       f"Sidewalk_{x_left}_{y}")
                if self.is_type(x_right, y, "Wall"):
                    self.place_cell(x_right, y, "Sidewalk",
                                       f"Sidewalk_{x_right}_{y}")


    # -----------------------------------------------------------------------
    # Clear interior => "Nothing"
    # -----------------------------------------------------------------------
    def _clear_interior(self):
        for y in range(self.interior_y_min, self.interior_y_max + 1):
            for x in range(self.interior_x_min, self.interior_x_max + 1):
                self.place_cell(x, y, "Nothing", f"Nothing_{x}_{y}")

    # -----------------------------------------------------------------------
    # Build roads & sidewalks
    # -----------------------------------------------------------------------

    def _build_roads_and_sidewalks(self):
        w, h = self.get_width(), self.get_height()

        # (A) Make R2/R3 road bands in the interior.
        # We pass the forced initial_road value to both horizontal and vertical bands.
        self.horizontal_bands = self._make_road_bands_for_interior(
            self.interior_y_min, self.interior_y_max,
            orientation="horizontal", allow_highway=self.allow_extra_highways, initial_road=self.ring_road_type
        )
        self.vertical_bands = self._make_road_bands_for_interior(
            self.interior_x_min, self.interior_x_max,
            orientation="vertical", allow_highway=self.allow_extra_highways, initial_road=self.ring_road_type
        )

        # (B) Force 1 horizontal + 1 vertical R1 highway
        self._force_one_highway(self.horizontal_bands, total_size=h)
        self._force_one_highway(self.vertical_bands, total_size=w)

        # (C) Place roads in the grid
        self._intersection_cells = set()
        self._road_cells = {}

        # In _build_roads_and_sidewalks, for placing roads in the grid:
        for y in range(h):
            hband = self._find_band_covering(y, self.horizontal_bands)
            for x in range(w):
                vband = self._find_band_covering(x, self.vertical_bands)

                # If both horizontal and vertical bands apply.
                if hband and vband:
                    (hstart, hend, hrtype, hbdir) = hband
                    (vstart, vend, vrtype, vbdir) = vband
                    # Skip if non-highway roads are outside the interior.
                    if (hrtype != "R1" or vrtype != "R1") and not self._inside_interior(x, y):
                        continue

                    # If this cell lies in a forced boundary corner, mark it as a regular road.
                    if self.ring_road_type is not None:
                        forced_thick = Defaults.ROAD_THICKNESS[self.ring_road_type]
                        # Define forced boundary ranges.
                        bottom_range = range(self.interior_y_min, self.interior_y_min + forced_thick)
                        top_range = range(self.interior_y_max - forced_thick + 1, self.interior_y_max + 1)
                        left_range = range(self.interior_x_min, self.interior_x_min + forced_thick)
                        right_range = range(self.interior_x_max - forced_thick + 1, self.interior_x_max + 1)

                        if (y in bottom_range or y in top_range) and (x in left_range or x in right_range):
                            # Cell is in one of the forced corner regions.
                            # Mark it as a regular road cell. (Using horizontal band info here.)
                            band_size = (hend - hstart) + 1
                            offset = y - hstart
                            self._road_cells[(x, y)] = (hrtype, "horizontal", offset, band_size, hbdir)
                            continue

                    # Otherwise, treat this cell as an intersection.
                    self._intersection_cells.add((x, y))

                # Only horizontal band applies.
                elif hband:
                    (start, end, rtype, bdir) = hband
                    if rtype != "R1" and not self._inside_interior(x, y):
                        continue
                    band_size = (end - start) + 1
                    offset = y - start
                    self._road_cells[(x, y)] = (rtype, "horizontal", offset, band_size, bdir)

                # Only vertical band applies.
                elif vband:
                    (start, end, rtype, bdir) = vband
                    if rtype != "R1" and not self._inside_interior(x, y):
                        continue
                    band_size = (end - start) + 1
                    offset = x - start
                    self._road_cells[(x, y)] = (rtype, "vertical", offset, band_size, bdir)

        # Mark intersections
        for (ix, iy) in list(self._intersection_cells):  # ← copy ➜ no mutation
            self._make_intersection(ix, iy)

        # Mark roads
        for (rx, ry), (rtype, orientation, offset, band_size, bdir) in self._road_cells.items():
            if (rx, ry) in self._intersection_cells:
                continue
            self.place_cell(rx, ry, rtype, f"{rtype}_{rx}_{ry}")
            rag = self.get_cell_contents(rx, ry)[0]
            # Compute the default directions.
            directions = self._compute_lane_dirs(rx, ry, rtype, orientation, offset, band_size, bdir)
            # Override directions for forced corner cells if needed.
            # Determine the bands covering this cell.
            directions = self._override_corner_lane_dirs(rx, ry, directions)
            rag.directions = directions

        # Sidewalk around roads
        road_positions = set(self._road_cells.keys()).union(self._intersection_cells)
        for (rx, ry) in road_positions:
            for (nx, ny) in ((rx + 1, ry),
                             (rx - 1, ry),
                             (rx, ry + 1),
                             (rx, ry - 1)):
                if not self.in_bounds(nx, ny):
                    continue
                # skip other road/intersection cells
                if (nx, ny) in road_positions or (nx, ny) in self._intersection_cells:
                    continue

                neigh = self.get_cell_contents(nx, ny)[0]
                # 1) carve into empty space
                if neigh.cell_type == "Nothing":
                    self.place_cell(nx, ny, "Sidewalk", f"Sidewalk_{nx}_{ny}")

                # 2) *also* carve into the wall if this is a highway‐lane
                else:
                    curr = self.get_cell_contents(rx, ry)[0].cell_type
                    if neigh.cell_type == "Wall" and curr in {"R1", "HighwayEntrance", "HighwayExit"}:
                        self.place_cell(nx, ny, "Sidewalk", f"Sidewalk_{nx}_{ny}")

        # Convert boundary R1 => "HighwayEntrance"
        self._replace_boundary_highways_with_entrances()


    def _override_corner_lane_dirs(self, rx, ry, default_dirs):
        """
        If the model uses an initial R2 road, override the lane direction for cells
        in the forced corner blocks with a manually specified mapping.
        The forced bands (from self.horizontal_bands and self.vertical_bands) define
        the corner boundaries. The manual mapping is:

            Bottom-left corner:   (0,0)="E", (0,1)="S", (1,0)="W", (1,1)="N"
            Bottom-right corner:  (0,0)="S", (0,1)="S", (1,0)="E", (1,1)="W"
            Top-right corner:     (0,0)="S", (0,1)="W", (1,0)="N", (1,1)="W"
            Top-left corner:      (0,0)="E", (0,1)="W", (1,0)="N", (1,1)="N"

        Here, local indices are computed relative to the forced band's start (which is 0 for
        the first cell and 1 for the second since the forced R2 band thickness is 2).
        """

        # Only apply override if initial road is set to "R2".
        if self.ring_road_type != "R2":
            return default_dirs

        # Get forced bands (assumed to be forced by _make_road_bands_for_interior)
        h_bottom = self.horizontal_bands[0]   # forced bottom horizontal band
        h_top    = self.horizontal_bands[-1]  # forced top horizontal band
        v_left   = self.vertical_bands[0]      # forced left vertical band
        v_right  = self.vertical_bands[-1]     # forced right vertical band

        in_bottom = (h_bottom[0] <= ry <= h_bottom[1])
        in_top    = (h_top[0]    <= ry <= h_top[1])
        in_left   = (v_left[0]   <= rx <= v_left[1])
        in_right  = (v_right[0]  <= rx <= v_right[1])

        # If the cell is not in a forced corner region, return the default.
        if not ((in_bottom or in_top) and (in_left or in_right)):
            return default_dirs

        # Determine which forced corner the cell is in and set up the manual mapping.
        if in_bottom and in_left:
            mapping = {(0, 0): "E", (0, 1): "E", (1, 0): "S", (1, 1): "N"}
            local_row = ry - h_bottom[0]
            local_col = rx - v_left[0]
        elif in_bottom and in_right:
            mapping = {(0, 0): "E", (0, 1): "N", (1, 0): "W", (1, 1): "N"}
            local_row = ry - h_bottom[0]
            local_col = rx - v_right[0]
        elif in_top and in_right:
            mapping = {(0, 0): "S", (0, 1): "N", (1, 0): "W", (1, 1): "W"}
            local_row = ry - h_top[0]
            local_col = rx - v_right[0]
        elif in_top and in_left:
            mapping = {(0, 0): "S", (0, 1): "E", (1, 0): "S", (1, 1): "W"}
            local_row = ry - h_top[0]
            local_col = rx - v_left[0]
        else:
            return default_dirs

        # Since forced R2 bands are exactly 2 cells thick, local_row and local_col should be 0 or 1.
        if local_row in (0, 1) and local_col in (0, 1):
            new_dir = mapping.get((local_row, local_col))
            if new_dir is not None:
                return [new_dir]
        return default_dirs

    # -------------------------------------------------------------------
    #  Carve optional L‑shaped sub‑block roads
    # -------------------------------------------------------------------
    def _carve_subblock_roads(self):
        """
        1.  Randomly inserts a one‑cell‑wide L‑shaped road in large
            interior blocks.
        2.  Guarantees:
            • Smaller sub‑block ≥ MIN_SUB_BLOCK_SPACING in both axes.
            • One leg is inbound, the other outbound.
            • Entry cell (inbound leg) arrow → pivot.
            • Exit cell (outbound leg) arrow ← away from pivot.
            • Pivot cell shows **only** the outbound arrow.
            • Legs are extended until they touch an existing road (no
              sidewalk stubs).
            • Every non‑road neighbour (orthogonal & diagonal) of the pivot
              becomes Sidewalk, so blocks never touch the corner cell.
        """

        road_types = Defaults.ROAD_LIKE_TYPES

        # ---------------- helpers ----------------------------------------------
        def neighbours(x, y):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    yield nx, ny

        def lay_r4_cell(x, y, arrow):
            """Convert (x,y) → road (if not already road) and edge it with sidewalk."""
            ag = self.get_cell_contents(x, y)[0]
            if ag.cell_type not in road_types:
                self.place_cell(x, y, self.subblock_road_type, f"{self.subblock_road_type}_{x}_{y}")
                ag = self.get_cell_contents(x, y)[0]
                ag.directions = [arrow]
            elif ag.cell_type == "R4" and arrow not in ag.directions:
                ag.directions.append(arrow)

            # add sidewalk ring
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if self.in_bounds(nx, ny) and self.is_type(nx, ny, "Nothing"):
                    self.place_cell(nx, ny, "Sidewalk", f"Sidewalk_{nx}_{ny}")

        def extend_to_road(sx, sy, march_dir, arrow_dir):
            """
            March from (sx,sy) in *march_dir*.  Convert every Sidewalk or
            Nothing cell into road, giving it *arrow_dir*.  Stops at – and now
            also updates – the first pre‑existing road cell, ensuring the
            outside road can actually turn into the new sub‑block road.
            """
            dx, dy = Defaults.DIRECTION_VECTORS[march_dir]
            cx, cy = sx, sy
            while self.in_bounds(cx, cy):
                tgt = self.get_cell_contents(cx, cy)[0]
                if tgt.cell_type in road_types:
                    if self.subblock_roads_have_intersections:
                        self._make_intersection(cx, cy)
                        self._intersection_cells.add((cx, cy))
                    else:
                        # just give it the extra arrow into the road
                        if arrow_dir not in tgt.directions:
                            tgt.directions.append(arrow_dir)
                    break
                if tgt.cell_type in {"Sidewalk", "Nothing"}:
                    lay_r4_cell(cx, cy, arrow_dir)
                    cx, cy = cx + dx, cy + dy
                else:                                   # wall / zone
                    break

        # ---------------- iterate over all 'Nothing' blobs ---------------------
        visited = set()
        w, h = self.get_width(), self.get_height()
        for y in range(h):
            for x in range(w):
                if (x, y) in visited or not self.is_type(x, y, "Nothing"):
                    continue

                # ---- flood‑fill this blob ----
                stack, region = [(x, y)], []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in visited or not self.is_type(cx, cy, "Nothing"):
                        continue
                    visited.add((cx, cy))
                    region.append((cx, cy))
                    for nx, ny in neighbours(cx, cy):
                        if (nx, ny) not in visited and self.is_type(nx, ny, "Nothing"):
                            stack.append((nx, ny))

                if not region or random.random() > self.subblock_chance:
                    continue

                # ---- bounding box / quick reject ----
                min_x = min(pt[0] for pt in region)
                max_x = max(pt[0] for pt in region)
                min_y = min(pt[1] for pt in region)
                max_y = max(pt[1] for pt in region)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                if width < 2 * self.min_subblock_spacing + 1 or \
                        height < 2 * self.min_subblock_spacing + 1:
                    continue

                # ---------- pick pivot & orientation ---------------------------
                for _ in range(20):  # up to 20 attempts
                    px = random.randint(min_x + self.min_subblock_spacing,
                                        max_x - self.min_subblock_spacing)
                    py = random.randint(min_y + self.min_subblock_spacing,
                                        max_y - self.min_subblock_spacing)
                    hor_dir = random.choice(["W", "E"])
                    ver_dir = random.choice(["N", "S"])
                    small_w = (px - min_x) if hor_dir == "W" else (max_x - px)
                    small_h = (py - min_y) if ver_dir == "S" else (max_y - py)
                    if small_w >= self.min_subblock_spacing and \
                            small_h >= self.min_subblock_spacing:
                        break
                else:
                    continue

                # ---------- inbound / outbound assignment ----------------------
                inbound_leg, outbound_leg = random.choice(
                    [("horizontal", "vertical"), ("vertical", "horizontal")]
                )
                leg_dir = {
                    "horizontal": {"in": Defaults.DIRECTION_OPPOSITES[hor_dir], "out": hor_dir},
                    "vertical": {"in": Defaults.DIRECTION_OPPOSITES[ver_dir], "out": ver_dir},
                }

                # ---------- carve horizontal leg --------------------------------
                if hor_dir == "W":
                    xs = range(px - 1, min_x - 1, -1)
                    hx_end, hy_end = min_x, py
                else:
                    xs = range(px + 1, max_x + 1)
                    hx_end, hy_end = max_x, py
                h_dir_cells = leg_dir["horizontal"]["in" if inbound_leg == "horizontal"
                else "out"]
                for hx in xs:
                    lay_r4_cell(hx, py, h_dir_cells)

                # ---------- carve vertical leg ----------------------------------
                if ver_dir == "S":
                    ys = range(py, min_y - 1, -1)
                    vx_end, vy_end = px, min_y
                else:
                    ys = range(py, max_y + 1)
                    vx_end, vy_end = px, max_y
                v_dir_cells = leg_dir["vertical"]["in" if inbound_leg == "vertical"
                else "out"]
                for vy in ys:
                    lay_r4_cell(px, vy, v_dir_cells)

                # ---------- configure pivot (corner) ----------------------------
                pivot = self.get_cell_contents(px, py)[0]  # R4
                pivot.directions = [h_dir_cells if outbound_leg == "horizontal"
                                    else v_dir_cells]  # single outbound arrow

                # ---------- extend legs out to road -----------------------------
                # horizontal extension
                arrow_h = h_dir_cells
                extend_to_road(hx_end + Defaults.DIRECTION_VECTORS[hor_dir][0],
                               hy_end + Defaults.DIRECTION_VECTORS[hor_dir][1],
                               hor_dir, arrow_h)

                # vertical extension
                arrow_v = v_dir_cells
                extend_to_road(vx_end + Defaults.DIRECTION_VECTORS[ver_dir][0],
                               vy_end + Defaults.DIRECTION_VECTORS[ver_dir][1],
                               ver_dir, arrow_v)

                # ---------- surround pivot with sidewalk ------------------------
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (-1, 1), (1, -1), (-1, -1)):
                    nx, ny = px + dx, py + dy
                    if self.in_bounds(nx, ny):
                        nag = self.get_cell_contents(nx, ny)[0]
                        if nag.cell_type not in road_types and nag.cell_type != "Wall":
                            self.place_cell(nx, ny, "Sidewalk", f"Sidewalk_{nx}_{ny}")

    # -----------------------------------------------------------------------
    # Flood-fill leftover => blocks
    # -----------------------------------------------------------------------
    def _flood_fill_blocks_storing_data(self):
        visited = set()
        w, h = self.get_width(), self.get_height()

        for y in range(h):
            for x in range(w):
                # Skip anything already visited or not “Nothing”
                if (x, y) in visited or not self.is_type(x, y, "Nothing"):
                    continue

                # — Flood‑fill this empty region —
                stack, region = [(x, y)], []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in visited or not self.is_type(cx, cy, "Nothing"):
                        continue
                    visited.add((cx, cy))
                    region.append((cx, cy))
                    for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
                        if self.in_bounds(nx, ny) and (nx, ny) not in visited:
                            if self.is_type(nx, ny, "Nothing"):
                                stack.append((nx, ny))

                if not region:
                    continue  # safety

                # — Determine block_type via bounding‐box heuristics —
                min_x = min(px for px, _ in region)
                max_x = max(px for px, _ in region)
                min_y = min(py for _, py in region)
                max_y = max(py for _, py in region)
                w_bb = max_x - min_x + 1
                h_bb = max_y - min_y + 1

                if w_bb < 3 or h_bb < 3:
                    block_type = "Empty"
                else:
                    types = Defaults.AVAILABLE_CITY_BLOCKS
                    weights = [Defaults.CITY_BLOCK_CHANCE[bt] for bt in types]
                    block_type = random.choices(types, weights=weights, k=1)[0]


                # — Fill every cell in region with that block_type —
                for bx, by in region:
                    self.place_cell(bx, by, block_type, f"{block_type}_{bx}_{by}")

                # — Carve the “ring” around it for potential entrances —
                ring = set()
                for bx, by in region:
                    for nx, ny in [(bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)]:
                        if self.in_bounds(nx, ny) and (nx, ny) not in region:
                            ring.add((nx, ny))

                for sx, sy in ring:
                    if self.is_type(sx, sy, "Nothing"):
                        self.place_cell(sx, sy, "Sidewalk", f"Sidewalk_{sx}_{sy}")

                # — Assign a unique block_id and store all data —
                block_id = len(self._blocks_data) + 1
                self._blocks_data.append({
                    "block_id": block_id,
                    "block_type": block_type,
                    "region": region,
                    "ring": list(ring)
                })

    # -----------------------------------------------------------------------
    # (6) Eliminate dead ends
    # -----------------------------------------------------------------------
    def _eliminate_dead_ends(self):
        # Only consider these types as road cells.
        road_types = Defaults.ROAD_LIKE_TYPES
        # Marks these as removable types
        removable_types = Defaults.REMOVABLE_DEAD_END_TYPES

        changed = True
        while changed:
            changed = False
            for y in range(self.get_height()):
                for x in range(self.get_width()):
                    ags = self.get_cell_contents(x, y)
                    if not ags:
                        continue
                    ctype = ags[0].cell_type
                    if ctype in removable_types:
                        neighbors = self._road_neighbors(x, y, road_types)
                        if len(neighbors) < 2:
                            self.place_cell(x, y, "Sidewalk", f"Sidewalk_{x}_{y}")
                            changed = True


    def _road_neighbors(self, x, y, road_types):
        results = []
        for (nx, ny) in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if self.in_bounds(nx, ny):
                ags = self.get_cell_contents(nx, ny)
                if ags and ags[0].cell_type in road_types:
                    results.append((nx, ny))
        return results

    def _upgrade_r2_to_intersections(self):
        """
        Upgrades R2 road cells into intersections if they have sidewalks on at least 2 sides.
        This increases connectivity by turning such R2 blocks into intersections.
        However, if an R2 initial road is used, the forced corner cells are excluded.
        """
        for y in range(self.get_height()):
            for x in range(self.get_width()):
                ags = self.get_cell_contents(x, y)
                if not ags:
                    continue
                agent = ags[0]
                if agent.cell_type == "R2":
                    # If using an R2 initial road, skip forced corner cells.
                    if self.ring_road_type == "R2":
                        # Use the forced bands created in _make_road_bands_for_interior.
                        # Forced horizontal bands (bottom and top) are at positions:
                        forced_bottom = range(self.horizontal_bands[0][0], self.horizontal_bands[0][1] + 1)
                        forced_top = range(self.horizontal_bands[-1][0], self.horizontal_bands[-1][1] + 1)
                        # Forced vertical bands (left and right) are at positions:
                        forced_left = range(self.vertical_bands[0][0], self.vertical_bands[0][1] + 1)
                        forced_right = range(self.vertical_bands[-1][0], self.vertical_bands[-1][1] + 1)
                        # If the cell lies in any of the four forced corner blocks,
                        # skip the conversion to intersection.
                        if (y in forced_bottom or y in forced_top) and (x in forced_left or x in forced_right):
                            continue

                    sidewalk_count = 0
                    # Check each of the four cardinal neighbors.
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if self.in_bounds(nx, ny):
                            neighbor_ags = self.get_cell_contents(nx, ny)
                            if neighbor_ags and neighbor_ags[0].cell_type == "Sidewalk":
                                sidewalk_count += 1
                    # If two or more sides have sidewalk, convert to an intersection.
                    if sidewalk_count >= 2:
                        self._make_intersection(x, y)

    # -----------------------------------------------------------------------
    # Place block entrances
    # -----------------------------------------------------------------------
    def _final_place_block_entrances(self):
        valid_types = set(Defaults.AVAILABLE_CITY_BLOCKS)

        for info in self._blocks_data:
            if info["block_type"] not in valid_types:
                continue

            # find ring cells that touch a road
            road_candidates = [
                (rx, ry)
                for (rx, ry) in info["ring"]
                if self._touches_road((rx, ry))
            ]
            if not road_candidates:
                continue

            # pick one, make it a BlockEntrance
            cx, cy = random.choice(road_candidates)
            self.place_cell(cx, cy, "BlockEntrance", f"BlockEntrance_{cx}_{cy}")
            agent = self.get_cell_contents(cx, cy)[0]

            # annotate and track
            agent.block_id   = info["block_id"]
            agent.block_type = info["block_type"]
            self.block_entrances.append(agent)


    # -----------------------------------------------------------------------
    # (Remove invalid intersection directions
    # -----------------------------------------------------------------------
    def _remove_invalid_intersection_directions(self):
        road_types = Defaults.ROAD_LIKE_TYPES
        intersection_types = {"Intersection"}

        for y in range(self.get_height()):
            for x in range(self.get_width()):
                ags = self.get_cell_contents(x, y)
                if not ags:
                    continue
                agent = ags[0]
                if agent.cell_type in intersection_types:
                    old_dirs = agent.directions
                    valid_dirs = []

                    for d in old_dirs:
                        # If it's our special lane-switch direction, keep it:
                        if d == "LaneSwitch":
                            valid_dirs.append(d)
                            continue

                        # Only process the cardinal directions
                        if d not in Defaults.AVAILABLE_DIRECTIONS:
                            continue

                        nx, ny = self.next_cell_in_direction(x, y, d)
                        if not self.in_bounds(nx, ny):
                            continue

                        neighbor_ags = self.get_cell_contents(nx, ny)
                        if not neighbor_ags:
                            continue
                        neighbor_type = neighbor_ags[0].cell_type
                        neighbor_dirs = neighbor_ags[0].directions

                        # The neighbor must be a valid road cell.
                        if neighbor_type not in road_types:
                            continue

                        # If the neighboring cell is an intersection, always allow traffic toward it.
                        # Otherwise, require that the neighbor allows traffic coming from that direction.
                        if neighbor_type == "Intersection" or (d in neighbor_dirs):
                            valid_dirs.append(d)

                    agent.directions = valid_dirs


    def next_cell_in_direction(self, x, y, d):
        if d == "N":
            return x, y + 1
        elif d == "S":
            return x, y - 1
        elif d == "E":
            return x + 1, y
        elif d == "W":
            return x - 1, y
        return x, y

    def is_next_cell_in_direction_an_intersection(self, x, y, d):
        xd, yd = self.next_cell_in_direction(x, y, d)
        next_cell = self.get_cell_contents(xd, yd)[0]
        return next_cell.cell_type == "Intersection"

    # -----------------------------------------------------------------------
    # Ensure roads next to a block entrance have direction in
    #      **and** block‐entrances point away from the block
    # -----------------------------------------------------------------------
    def _add_entrance_directions(self):
        road_types = Defaults.ROAD_LIKE_TYPES

        for y in range(self.get_height()):
            for x in range(self.get_width()):
                ags = self.get_cell_contents(x, y)
                if not ags:
                    continue
                agent = ags[0]
                if agent.cell_type == "BlockEntrance":
                    # we'll collect the BlockEntrance's own arrows here
                    entrance_dirs = []
                    # for each neighboring road, point both ways:
                    for (nx, ny) in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                        if self.in_bounds(nx, ny):
                            nags = self.get_cell_contents(nx, ny)
                            if nags and nags[0].cell_type in road_types:
                                road_agent = nags[0]
                                dx = x - nx
                                dy = y - ny
                                if dx == 1 and dy == 0:
                                    needed_dir = "E"
                                elif dx == -1 and dy == 0:
                                    needed_dir = "W"
                                elif dx == 0 and dy == 1:
                                    needed_dir = "N"
                                elif dx == 0 and dy == -1:
                                    needed_dir = "S"
                                else:
                                    continue
                                # add the arrow INTO the entrance on the road
                                if needed_dir not in road_agent.directions:
                                    road_agent.directions.append(needed_dir)
                                # and _invert_ it for the BlockEntrance itself
                                entrance_dirs.append(Defaults.DIRECTION_OPPOSITES[needed_dir])
                    agent.directions = entrance_dirs


    # -----------------------------------------------------------------------
    # Road band helper
    # -----------------------------------------------------------------------
    def _make_road_bands_for_interior(self, start_coord, end_coord,
                                      orientation, allow_highway=False, initial_road=None):
        """
        Generates road bands between start_coord and end_coord.
        If `initial_road` is provided (either "R2" or "R3"), then the first band
        (starting at start_coord) and the last band (ending at end_coord) will be forced
        to be of that type.

        For horizontal bands, start_coord is the lower y (south edge) and end_coord
        the upper y (north edge) of the interior.
        For vertical bands, start_coord is the left (west edge) and end_coord is
        the right (east edge) of the interior.
        """



        bands = []
        current = start_coord
        last_r3_dir = None

        # Generate bands using the random scheme.
        while current <= end_coord:
            # Choose a road type randomly.
            rtype = self._choose_road_type(allow_highway=allow_highway)
            thick = Defaults.ROAD_THICKNESS[rtype]
            bstart = current
            bend = min(bstart + thick - 1, end_coord)

            if orientation == "horizontal":
                if rtype == "R3" and last_r3_dir is not None:
                    bdir = Defaults.DIRECTION_OPPOSITES[last_r3_dir]
                else:
                    bdir = random.choice(["E", "W"])
            else:
                if rtype == "R3" and last_r3_dir is not None:
                    bdir = Defaults.DIRECTION_OPPOSITES[last_r3_dir]
                else:
                    bdir = random.choice(["N", "S"])
            bands.append((bstart, bend, rtype, bdir))

            if rtype == "R3":
                last_r3_dir = bdir
            else:
                last_r3_dir = None

            next_pos = bend + 1
            if next_pos > end_coord:
                break
            block_size = random.randint(self.min_block_spacing, self.max_block_spacing)
            block_end = next_pos + block_size - 1
            if block_end > end_coord:
                break
            current = block_end + 1

        # --- Post-Processing: Force First and Last Bands if initial_road is set ---
        if initial_road is not None:
            forced_thick = Defaults.ROAD_THICKNESS[initial_road]
            # Determine forced directions:
            if initial_road == "R3":
                # For R3, use fixed directions.
                if orientation == "horizontal":
                    forced_first_dir = "E"  # bottom road faces right
                    forced_last_dir = "W"  # top road faces left
                else:
                    forced_first_dir = "S"  # left road faces down
                    forced_last_dir = "N"  # right road faces up
            else:
                # For R2 (or other types), use random assignment.
                if orientation == "horizontal":
                    forced_first_dir = random.choice(["E", "W"])
                    forced_last_dir = random.choice(["E", "W"])
                else:
                    forced_first_dir = random.choice(["N", "S"])
                    forced_last_dir = random.choice(["N", "S"])

            first_band = (start_coord,
                          start_coord + forced_thick - 1,
                          initial_road,
                          forced_first_dir)
            forced_last_start = end_coord - forced_thick + 1
            last_band = (forced_last_start,
                         end_coord,
                         initial_road,
                         forced_last_dir)

            # Now adjust the bands list based on its current length.
            if len(bands) == 0:
                # No bands produced by the random generator:
                bands.extend([first_band, last_band])
            elif len(bands) == 1:
                # Only one band exists:
                # Replace the only band with the forced first band...
                bands[0] = first_band
                # ...and then append the forced last band if it is different.
                if first_band != last_band:
                    bands.append(last_band)
            else:
                # If there are at least two bands, simply replace first and last.
                bands[0] = first_band
                bands[-1] = last_band

        return bands

    def _choose_road_type(self, allow_highway=True):
        if allow_highway:
            r1_chance = self.extra_highways_chance
            remaining = 1.0 - r1_chance
            r2_chance = remaining * self.r2_r3_chance_split
        else:
            r1_chance = 0.0
            r2_chance = self.r2_r3_chance_split

        r = random.random()

        if allow_highway:
            if r < r1_chance:
                return "R1"
            elif r < r1_chance + r2_chance:
                return "R2"
            else:
                return "R3"
        else:
            return "R2" if r < r2_chance else "R3"

    def _force_one_highway(self, bands, total_size):
        thick = Defaults.ROAD_THICKNESS["R1"]
        inset = self._compute_highway_inset()
        start_min = inset
        start_max = total_size - thick - inset
        if start_min > start_max:
            start_min = 0
            start_max = total_size - thick
            if start_max < 0:
                return
        hw_start = random.randint(start_min, start_max)
        hw_end   = hw_start + thick - 1
        bands.append((hw_start, hw_end, "R1", ""))
        bands.sort(key=lambda b: b[0])

        skip_low  = hw_start - self.min_block_spacing
        skip_high = hw_end   + self.min_block_spacing
        new_bands = []
        for (st, en, rt, bdir) in bands:
            if rt == "R1" and (st, en) == (hw_start, hw_end):
                new_bands.append((st, en, rt, bdir))
            else:
                if not (en < skip_low or st > skip_high):
                    continue
                new_bands.append((st, en, rt, bdir))
        bands[:] = new_bands

    def _find_band_covering(self, index, bands):
        for (st, en, rtype, bdir) in bands:
            if st <= index <= en:
                return st, en, rtype, bdir
        return None

    def _compute_lane_dirs(self, x, y, rtype, orientation, offset, band_size, band_dir):
        """
        Revised lane direction computation to enforce European right-hand traffic (RHT)
        for R1 and R2 roads.

        Assumptions:
          - For horizontal roads, the band cells are ordered from south to north.
          - For vertical roads, the band cells are ordered from west to east.
        """
        # R3 remains one-way using its given direction.
        if rtype == "R3":
            return [band_dir]

        # For two-lane roads (R2), assume band_size == 2.
        if rtype == "R2":
            if orientation == "horizontal":
                # For horizontal roads:
                # - For eastbound, the rightmost (southern) lane is at offset 0.
                # - For westbound, the rightmost (northern) lane is at offset 1.
                if offset == 0:
                    return ["E"]  # eastbound, right-hand lane (south)
                else:
                    return ["W"]  # westbound, right-hand lane (north)
            else:  # vertical road
                # For vertical roads:
                # - For southbound, the rightmost (western) lane is at offset 0.
                # - For northbound, the rightmost (eastern) lane is at offset 1.
                if offset == 0:
                    return ["S"]  # southbound, right-hand lane (west)
                else:
                    return ["N"]  # northbound, right-hand lane (east)

        # For highways (R1) assume band_size == 4 (2 lanes per direction).
        if rtype == "R1":
            half = band_size // 2  # for R1, half should be 2.

            if orientation == "horizontal":
                # Split the band into two groups along the vertical (y) axis.
                # Eastbound group: lanes in indices [0, half-1]. Their rightmost (correct) lane is the one with the lowest offset.
                # Westbound group: lanes in indices [half, band_size-1]. Their rightmost lane is the one with the highest offset.
                if offset < half:
                    # Eastbound lanes.
                    main_dir = "E"
                    side_dirs = []
                    # Allow a lane shift (if needed) only if not in the rightmost (offset==0) lane.
                    if (offset > 0
                            and not self.is_next_cell_in_direction_an_intersection (x, y, "S")):
                        side_dirs.append("S")  # shift right (toward south)
                    if (offset < half - 1
                            and not self.is_next_cell_in_direction_an_intersection(x, y, "N")):
                        side_dirs.append("N")  # optional left shift (toward north)
                    return [main_dir] + side_dirs
                else:
                    # Westbound lanes.
                    main_dir = "W"
                    side_dirs = []
                    # For westbound, the right-hand lane is the one with the highest offset.
                    if (offset < band_size - 1
                            and not self.is_next_cell_in_direction_an_intersection(x, y, "N")):
                        side_dirs.append("N")  # shift right (toward north)
                    if (offset > half
                            and not self.is_next_cell_in_direction_an_intersection(x, y, "S")):
                        side_dirs.append("S")  # optional left shift (toward south)
                    return [main_dir] + side_dirs

            else:  # vertical roads
                # Here, lanes vary along x: from west (offset 0) to east (offset band_size-1).
                # For northbound, the right-hand lane is the easternmost (highest offset).
                # For southbound, the right-hand lane is the westernmost (lowest offset).
                if offset < band_size // 2:
                    # Southbound group.
                    main_dir = "S"
                    side_dirs = []
                    if (offset > 0
                            and not self.is_next_cell_in_direction_an_intersection (x, y, "W")):
                        side_dirs.append("W")  # shift right (toward west)
                    if (offset < (band_size // 2) - 1
                            and not self.is_next_cell_in_direction_an_intersection (x, y, "E")):
                        side_dirs.append("E")  # optional left shift (toward east)
                    return [main_dir] + side_dirs
                else:
                    # Northbound group.
                    main_dir = "N"
                    side_dirs = []
                    if (offset < band_size - 1
                            and not self.is_next_cell_in_direction_an_intersection(x, y, "E")):
                        side_dirs.append("E")  # shift right (toward east)
                    if (offset > band_size // 2
                            and not self.is_next_cell_in_direction_an_intersection (x, y, "W")):
                        side_dirs.append("W")  # optional left shift (toward west)
                    return [main_dir] + side_dirs

        # Fallback: no direction assigned.
        return []

    def _replace_boundary_highways_with_entrances(self):
        w, h = self.get_width(), self.get_height()

        # (1) Collect all cells in the WALL_THICKNESS band
        edge_coords = set()
        for y in range(self.wall_thickness):
            for x in range(w):
                edge_coords.add((x, y))
        for y in range(h - self.wall_thickness, h):
            for x in range(w):
                edge_coords.add((x, y))
        for x in range(self.wall_thickness):
            for y in range(h):
                edge_coords.add((x, y))
        for x in range(w - self.wall_thickness, w):
            for y in range(h):
                edge_coords.add((x, y))

        # Maps for detecting “inward” direction
        inward_x = {0: "E", w - 1: "W"}
        inward_y = {0: "N", h - 1: "S"}

        # (2) Replace only the true boundary R1 → HighwayEntrance / Exit
        for (ex, ey) in edge_coords:
            ags = self.get_cell_contents(ex, ey)
            if not ags or ags[0].cell_type != "R1":
                continue
            # must actually be on the very outer edge:
            if not (ex in (0, w - 1) or ey in (0, h - 1)):
                continue

            old_dirs = list(ags[0].directions)
            # inward‐pointing arrow → Entrance, otherwise Exit
            if (ex in inward_x and inward_x[ex] in old_dirs) or \
               (ey in inward_y and inward_y[ey] in old_dirs):
                new_type = "HighwayEntrance"
            else:
                new_type = "HighwayExit"

            # swap it out
            self.place_cell(ex, ey, new_type, f"{new_type}_{ex}_{ey}")
            he = self.get_cell_contents(ex, ey)[0]
            he.directions = old_dirs
            he.highway_orientation = (
                "horizontal" if ey in (0, h - 1) else "vertical"
            )
            he.highway_id = f"highway_{he.highway_orientation}"
            if new_type == "HighwayEntrance":
                self.highway_entrances.append(he)
            else:
                self.highway_exits.append(he)

    def _add_traffic_lights(self):
        """
        Convert any road‐type cell that points into an Intersection
        into a ControlledRoad, and carve TrafficLight agents on
        all adjacent Sidewalks.
        This scans *all* road cells (including sub‑block roads).
        """
        road_types = Defaults.ROAD_LIKE_TYPES_WITHOUT_INTERSECTIONS

        w, h = self.get_width(), self.get_height()
        for road_block_x in range(w):
            for road_block_y in range(h):
                # fetch the only agent here
                ags = self.get_cell_contents(road_block_x, road_block_y)
                if not ags:
                    continue
                road_block = ags[0]
                if road_block.cell_type not in road_types:
                    continue

                road_directions = road_block.directions.copy()
                original_road_type = road_block.cell_type

                # check each arrow for an Intersection neighbor
                for r_d in road_directions:
                    nx, ny = self.next_cell_in_direction(road_block_x, road_block_y, r_d)
                    if not self.in_bounds(nx, ny):
                        continue
                    road_lead_to = self.get_cell_contents(nx, ny)[0]
                    if road_lead_to.cell_type != "Intersection":
                        continue

                    # → convert to ControlledRoad
                    self.place_cell(road_block_x, road_block_y, "ControlledRoad", f"ControlledRoad_{road_block_x}_{road_block_y}")
                    controlled_road = self.get_cell_contents(road_block_x, road_block_y)[0]
                    controlled_road.directions = road_directions
                    controlled_road.status = "Pass"
                    controlled_road.base_color = Defaults.ZONE_COLORS.get(original_road_type)
                    self.controlled_roads.append(controlled_road)

                    # …inside your loop or function where you have:
                    #    road_block_x, road_block_y
                    #    controlled_road.directions  (e.g. ["N","W"] etc.)
                    valid_blocks = []
                    for cr_d in controlled_road.directions:
                        rd = Defaults.DIRECTION_TO_THE_RIGHT[cr_d]  # e.g. "N" → "E"
                        dx, dy = Defaults.DIRECTION_VECTORS.get(rd, (0, 0))  # e.g. (1,0)
                        valid_blocks.append((road_block_x + dx,
                                             road_block_y + dy))

                    # right_blocks is now a list of coords to the right of the road.
                    # If you need them unique:
                    valid_blocks = list(set(valid_blocks))

                    # → carve lights on every adjoining sidewalk (or controlled‑road neighbor)
                    for valid_neighbor_x, valid_neighbor_y in valid_blocks:
                        if self.in_bounds(valid_neighbor_x, valid_neighbor_y):
                            scanned_lead_to = self.get_cell_contents(valid_neighbor_x, valid_neighbor_y)[0]  # grab the single agent

                            # ── if it's a ControlledRoad, look one more cell out in that same direction ──
                            if scanned_lead_to.cell_type == "ControlledRoad" or scanned_lead_to.cell_type == original_road_type:
                                if not any(dir_ in controlled_road.directions for dir_ in scanned_lead_to.directions):
                                    continue
                                # compute direction vector from (x,y) → (sx,sy)
                                dx, dy = valid_neighbor_x - road_block_x, valid_neighbor_y - road_block_y
                                fx, fy = valid_neighbor_x + dx, valid_neighbor_y + dy
                                if self.in_bounds(fx, fy):
                                    neigh3 = self.get_cell_contents(fx, fy)[0]
                                    # assign the same traffic light to that further block
                                    self._assign_traffic_light(controlled_road, neigh3,
                                                               original_road_type, road_directions,
                                                               fx, fy)

                            self._assign_traffic_light(controlled_road, scanned_lead_to,
                                                       original_road_type, road_directions,
                                                       valid_neighbor_x, valid_neighbor_y)
                    # only one set of lights per road cell
                    break

    def _assign_traffic_light(self, controlled_road, aspiring_traffic_light, original_road_type,
                              scanning_directions, x, y):

        if aspiring_traffic_light.cell_type == "TrafficLight":
            tl = aspiring_traffic_light
        elif aspiring_traffic_light.cell_type == "Sidewalk":
            # replace with a TrafficLight
            self.place_cell(x, y, "TrafficLight", f"TrafficLight_{x}_{y}")
            tl = self.get_cell_contents(x, y)[0]
            self.traffic_lights.append(tl)
            tl.status = "Pass"
        else:
            return

        if tl.status == "Stop":
            self.stop_cells.add(tl.get_position())
        else:
            self.stop_cells.discard(tl.get_position())

        controlled_road.light = tl
        tl.controlled_blocks.append(controlled_road)

        self._scan_for_traffic_flow(controlled_road, scanning_directions, original_road_type, tl, 0)

    def _scan_for_traffic_flow(self, road, scanning_directions, original_road_type, traffic_light, scan_depth):

        self._scan_for_traffic_flow_reverse(road, scanning_directions, original_road_type, traffic_light, scan_depth)
        if self.forward_traffic_light_range:
            self._scan_for_traffic_flow_forward(road, scanning_directions, original_road_type, traffic_light, scan_depth)

    def _scan_for_traffic_flow_reverse(self, road, scanning_directions, original_road_type, traffic_light, scan_depth):

        reversed_directions = []
        for fd in scanning_directions:
            reversed_directions.append(Defaults.DIRECTION_OPPOSITES[fd])

        for rd in reversed_directions:
            ctrl_x, ctrl_y = road.get_position()
            bx, by = self.next_cell_in_direction(ctrl_x, ctrl_y, rd)
            while scan_depth <= self.traffic_light_range:
                if self.in_bounds(bx, by):
                    nb = self.get_cell_contents(bx, by)[0]
                    if nb.cell_type == original_road_type and nb.leads_to(road):
                        traffic_light.assigned_road_blocks.append(nb)
                        nb.light = traffic_light
                        bx, by = self.next_cell_in_direction(bx, by, rd)
                        scan_depth += 1
                    else:
                        break
                else:
                    break

    def _scan_for_traffic_flow_forward(self, road, scanning_directions, original_road_type, traffic_light, scan_depth):

        for rd in scanning_directions:
            ctrl_x, ctrl_y = road.get_position()
            bx, by = self.next_cell_in_direction(ctrl_x, ctrl_y, rd)

            while scan_depth <= self.traffic_light_range:
                if self.in_bounds(bx, by):
                    currently_scanned_road = self.get_cell_contents(bx, by)[0]
                    if currently_scanned_road.cell_type == "Intersection":

                        if self.forward_traffic_light_range_intersections == Defaults.FORWARD_TRAFFIC_LIGHT_INTERSECTION_OPTIONS[1]:
                            traffic_light.assigned_road_blocks.append(currently_scanned_road)
                            currently_scanned_road.light = traffic_light
                            scan_depth += 1
                        elif self.forward_traffic_light_range_intersections == Defaults.FORWARD_TRAFFIC_LIGHT_INTERSECTION_OPTIONS[2]:
                            traffic_light.assigned_road_blocks.append(currently_scanned_road)
                            currently_scanned_road.light = traffic_light

                        bx, by = self.next_cell_in_direction(bx, by, rd)
                    elif currently_scanned_road.cell_type == original_road_type:
                        if currently_scanned_road.directly_leads_to(road):
                            self._scan_for_traffic_flow_forward(currently_scanned_road, original_road_type, original_road_type, traffic_light, scan_depth)
                        elif rd in currently_scanned_road.directions:
                            traffic_light.assigned_road_blocks.append(currently_scanned_road)
                            currently_scanned_road.light = traffic_light
                            scan_depth += 1

                        bx, by = self.next_cell_in_direction(bx, by, rd)
                    else:
                        break
                else:
                    break

    def _create_intersection_light_groups(self) -> None:
        """Scan every *contiguous* block of Intersection cells, work out its
        outer bounding box and collect the lights that sit on the four diagonal
        sidewalk corners.  Each quartet becomes one IntersectionLightGroup."""

        visited = set()
        comp_idx = 0

        for seed in getattr(self, "_intersection_cells", []):
            if seed in visited:
                continue

            # ── flood-fill the whole block of Intersection cells ──────────
            stack = [seed]
            cluster = []
            while stack:
                x, y = stack.pop()
                if (x, y) in visited or (x, y) not in self._intersection_cells:
                    continue
                visited.add((x, y))
                cluster.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in self._intersection_cells and (nx, ny) not in visited:
                        stack.append((nx, ny))

            # nothing to group
            if not cluster:
                continue

            # ── bounding box & corner-coords (one cell outside) ───────────
            min_x = min(p[0] for p in cluster)
            max_x = max(p[0] for p in cluster)
            min_y = min(p[1] for p in cluster)
            max_y = max(p[1] for p in cluster)

            corner_cells = [(min_x - 1, min_y - 1), (max_x + 1, min_y - 1),
                            (min_x - 1, max_y + 1), (max_x + 1, max_y + 1)]

            lights = []
            for cx, cy in corner_cells:
                if not self.in_bounds(cx, cy):
                    continue
                ags = self.get_cell_contents(cx, cy)
                if ags and ags[0].cell_type == "TrafficLight":
                    lights.append(ags[0])

            if not lights:  # no lights ⇒ nothing to create
                continue

            comp_idx += 1
            group = IntersectionLightGroup(f"Intersection_{comp_idx}", self, lights)
            self.intersection_light_groups.append(group)
            self.schedule.add(group)  # optional, keeps API consistent

            for tl in lights:
                tl.intersection_group = group

            # ② every Intersection cell inside the cluster
            for ix, iy in cluster:
                ia = self.get_cell_contents(ix, iy)[0]  # the single Intersection CellAgent
                ia.intersection_group = group

    # ------------------------------------------------------------------
    #  City-block construction helper
    # ------------------------------------------------------------------

      # ---------------------------------------------------------------
      # Spawn all blocks recorded during flood-fill
      # ---------------------------------------------------------------


    def _instantiate_city_blocks(self) -> None:
        """Convert every entry in ``self._blocks_data`` into a `CityBlock`."""


        for info in self._blocks_data:
        # ignore “Empty” blocks

            if info["block_type"] not in Defaults.AVAILABLE_CITY_BLOCKS:
                continue

            self._spawn_city_block(
            block_id = info["block_id"],
            block_type = info["block_type"],
            inner_coords = set(info["region"]),)

    def _spawn_city_block(
            self,
            block_id: int,
            block_type: str,
            inner_coords: set[tuple[int, int]],
    ) -> CityBlock:
        """
        Create a ``CityBlock`` agent, given the coordinates of every
        *inner* buildable cell.

        Parameters
        ----------
        block_id
            Unique identifier for the new block (usually an int).
        block_type
            \"Residential\", \"Office\", … – must be one of
            ``Defaults.AVAILABLE_CITY_BLOCKS``.
        inner_coords
            Coordinates **inside** the block’s perimeter *only* (no sidewalks).

        Returns
        -------
        CityBlock
            The freshly spawned agent.
        """
        grid = self.grid  # Mesa MultiGrid
        cell_c = self.get_cell_contents  # tiny shorthand

        # ── collect all cells that belong to the block ─────────────────
        inner_blocks: list[CellAgent] = []
        sidewalks: list[CellAgent] = []
        entrances: list[CellAgent] = []

        for x, y in inner_coords:
            cell: CellAgent = cell_c(x, y)[0]
            inner_blocks.append(cell)
            cell.block_id = block_id
            cell.block_type = block_type

            # find immediate neighbours (4-neighbourhood is enough here)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not self.in_bounds(nx, ny):
                    continue
                nbr: CellAgent = cell_c(nx, ny)[0]

                if nbr.cell_type == "Sidewalk" and nbr not in sidewalks:
                    sidewalks.append(nbr)
                elif nbr.cell_type == "BlockEntrance" and nbr not in entrances:
                    entrances.append(nbr)

        # ── create the agent & register it ──────────────────────────────
        cb = CityBlock(
            custom_id=f"CB_{block_id}",
            model=self,
            block_type=block_type,
            inner_blocks=inner_blocks,
            sidewalks=sidewalks,
            entrances=entrances,
            gradual_resources=self.gradual_city_block_resources
        )

        self.schedule.add(cb)

        # Keep a reference if that’s handy elsewhere
        if not hasattr(self, "city_blocks"):
            self.city_blocks: dict[int, CityBlock] = {}
        self.city_blocks[block_id] = cb

        return cb

    def _add_dummy_agents(self):
        from Simulation.agents.dummy import DummyAgent

        uid = 0
        for x in range(self.width):
            for y in range(self.height):
                dummy = DummyAgent(f"Dummy_{x}_{y}", self, (x, y))
                # place_agent will now set dummy.pos internally
                self.grid.place_agent(dummy, (x, y))
                self.schedule.add(dummy)
                uid += 1

    def _populate_cell_cache(self):
        """Caches all agents for quick access by (x,y) position."""
        for content, (x, y) in self.grid.coord_iter():  # Correct unpacking
            self.cell_agent_cache[(x, y)] = content
    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _touches_road(self, cell):
        """
        Return True if the given (x,y) is adjacent to any road‐type cell
        (so we know where a BlockEntrance can go).
        """
        x, y = cell
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if self.in_bounds(nx, ny):
                ags = self.get_cell_contents(nx, ny)
                if ags and ags[0].cell_type in [
                    "R1", "R2", "R3", "Intersection", "HighwayEntrance", "ControlledRoad"
                ]:
                    return True
        return False

    def _inside_interior(self, x, y):
        return (self.interior_x_min <= x <= self.interior_x_max) and \
               (self.interior_y_min <= y <= self.interior_y_max)


    def is_type(self, x, y, ctype):
        ags = self.get_cell_contents(x, y)
        return bool(ags and ags[0].cell_type == ctype)

    def in_bounds(self, x, y):
        return (0 <= x < self.get_width()) and (0 <= y < self.get_height())

    def step(self):
        self.tick_cache.clear()
        self.tick_grid = None
        self.schedule.step()
        if Defaults.PATHFINDING_METHOD == "CUDA":
            self.path_planner.solve_all()     # ← GPU crunch
        self.step_count += 1

    # — Convenience getters —

    def place_cell(self, x, y, new_type, new_id):
        self.remove_cell(x,y)

        ag = CellAgent(new_id, self, (x, y), new_type)
        self.grid.place_agent(ag, (x, y))
        self.cell_agent_cache[(x, y)] = [ag]  # Update cache here
        #self.schedule.add(ag)

    def remove_cell(self, x, y):
        old_list = self.get_cell_contents(x, y)
        for oa in old_list:
            self.grid.remove_agent(oa)
            if oa in self.schedule.agents:
                self.schedule.remove(oa)

    def place_dummy(self, x , y):
        """Place a new DummyAgent at the vacated position."""
        dummy = DummyAgent(f"Dummy_{x}_{y}", self, (x,y))
        dummy.pos = None
        self.grid.place_agent(dummy, (x,y))
        #self.schedule.add(dummy)

    def remove_dummy(self, x , y):
        """Remove a DummyAgent at the given position if present."""
        for a in self.get_cell_contents(x, y):
            if isinstance(a, DummyAgent):
                self.grid.remove_agent(a)
                if a in self.schedule.agents:
                    self.schedule.remove(a)
                # clear pos so no warning on next place
                a.pos = None
                break

    def place_vehicle(self, vehicle: VehicleAgent, pos: tuple[int, int]):
        """Place a new VehicleAgent at the vacated position."""
        vehicle.current_cell = self.get_cell_contents(pos[0], pos[1])[0]
        vehicle.current_cell.occupied = True
        self.remove_dummy(pos[0], pos[1])
        self.grid.place_agent(vehicle, vehicle.current_cell.get_position())
        self.vehicle_cells.add(pos)
        self.schedule.add(vehicle)

    def remove_vehicle(self, vehicle: VehicleAgent):
        v_pos = None

        if vehicle.current_cell is not None:
            vehicle.current_cell.occupied = False
            v_pos = vehicle.current_cell.get_position()
            self.vehicle_cells.discard(v_pos)

        self.grid.remove_agent(vehicle)
        self.schedule.remove(vehicle)

        if self.use_dummy_agents and  v_pos is not None:
            self.place_dummy(v_pos[0], v_pos[1])

    def move_vehicle(self, vehicle: VehicleAgent, new_cell: CellAgent, new_pos: tuple[int, int], old_pos: tuple[int, int]):
        if self.use_dummy_agents:
            self.remove_dummy(new_pos[0], new_pos[1])

        if vehicle.current_cell is not None:
            self.vehicle_cells.discard(vehicle.current_cell.get_position())
            vehicle.current_cell.occupied = False

        vehicle.current_cell = new_cell
        new_cell.occupied = True

        self.grid.move_agent(vehicle, new_pos)
        self.vehicle_cells.add(new_pos)

        if self.use_dummy_agents and old_pos is not None:
            self.place_dummy(old_pos[0], old_pos[1])

    def get_cell_contents(self, x: int, y: int) -> Agent | None | list[Agent | None] | list[Any]:
        """
        Return the live list of agents at (x, y). Always a list,
        so callers can safely do `[0]`.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x, y]  # tuple indexing → works
        return []

    def get_width(self) -> int:
        return self.grid.width

    def get_height(self) -> int:
        return self.grid.height

    def get_block_entrances(self):
        return self.block_entrances

    def get_highway_entrances(self):
        return self.highway_entrances

    def get_highway_exits(self):
        return self.highway_exits

    def get_controlled_roads(self):
        return self.controlled_roads

    def get_traffic_lights(self):
        return self.traffic_lights

    def set_traffic_lights_go(self):
        for tl in self.traffic_lights:
            tl.set_light_go()

    def set_traffic_lights_stop(self):
        for tl in self.traffic_lights:
            tl.set_light_stop()

    def get_intersection_light_groups(self):
        return self.intersection_light_groups

    def set_intersections_go(self):
        for Intersection in self.intersection_light_groups:
            Intersection.set_go()

    def set_intersections_stop(self):
        for Intersection in self.intersection_light_groups:
            Intersection.set_stop()

    # ═══════════════════════════════════════════════════════════════
    #  City-block querying helpers
    # ═══════════════════════════════════════════════════════════════
    def _sort_blocks(self, blocks, by: str):
        if by == "food":
            blocks = sorted(blocks, key=lambda b: b.get_food_units())
        elif by == "waste":
            blocks = sorted(blocks, key=lambda b: -b.get_waste_units())
        return blocks

    def get_all_city_blocks(self, sort_by: str = "unsorted"):
        """Return **every** CityBlock (optionally sorted)."""
        return self._sort_blocks(list(self.city_blocks.values()), sort_by)

    def get_blocks_needing_food(self, sort_by: str = "unsorted"):
        """Blocks whose ``needs_food()`` is True."""
        return self._sort_blocks(
            [b for b in self.city_blocks.values() if b.needs_food()],
            sort_by,
        )

    def get_blocks_producing_waste(self, sort_by: str = "unsorted"):
        """Blocks whose ``produces_waste()`` is True."""
        return self._sort_blocks(
            [b for b in self.city_blocks.values() if b.produces_waste()],
            sort_by,
        )

    def get_city_blocks_by_type(self, block_type, sort_by: str = "unsorted"):
        return self.get_city_blocks_by_types([block_type], sort_by)

    def get_city_blocks_by_types(self, block_types, sort_by: str = "unsorted"):
        """
        Generic picker.

        * ``types`` may be a single str or any iterable of types.
        * Pass ``types=None`` to get *all* blocks (identical to ``get_all_city_blocks``).
        """
        if block_types is None:
            subset = list(self.city_blocks.values())
        else:
            wanted = {block_types} if isinstance(block_types, str) else set(block_types)
            subset = [b for b in self.city_blocks.values() if b.block_type in wanted]
        return self._sort_blocks(subset, sort_by)

    # ──────────────────────────────────────────────────────────────
    #  Typed-block helpers (static, readable)
    # ──────────────────────────────────────────────────────────────
    def get_residential_city_blocks(self, sort_by: str = "unsorted"):
        return self.get_city_blocks_by_types("Residential", sort_by)

    def get_office_city_blocks(self, sort_by: str = "unsorted"):
        return self.get_city_blocks_by_types("Office", sort_by)

    def get_market_city_blocks(self, sort_by: str = "unsorted"):
        return self.get_city_blocks_by_types("Market", sort_by)

    def get_leisure_city_blocks(self, sort_by: str = "unsorted"):
        return self.get_city_blocks_by_types("Leisure", sort_by)

    def get_other_city_blocks(self, sort_by: str = "unsorted"):
        return self.get_city_blocks_by_types("Other", sort_by)

    # “Highest-need” selectors ------------------------------------------------
    def get_block_most_in_need_of_food(self):
        """The block with *least* food remaining (needs topping up most)."""
        needy = self.get_blocks_needing_food(sort_by="food")
        return needy[0] if needy else None

    def get_block_most_in_need_of_waste_pickup(self):
        """The block with *most* waste accumulated."""
        dirty = self.get_blocks_producing_waste(sort_by="waste")
        return dirty[0] if dirty else None

    # ═══════════════════════════════════════════════════════════════
    #  ENTRY / EXIT HELPERS
    # ═══════════════════════════════════════════════════════════════
    # ── private ────────────────────────────────────────────────────
    @staticmethod
    def _are_adjacent(a, b) -> bool:
        """
        Return True when the two CellAgent instances occupy
        4-neighbouring grid positions (N, S, E, W).
        """
        (x1, y1), (x2, y2) = a.position, b.position
        return abs(x1 - x2) + abs(y1 - y2) == 1

    # ── public API ─────────────────────────────────────────────────
    def get_start_blocks(self):
        """
        All cells a vehicle may **spawn** on:

          • every *BlockEntrance*
          • every *HighwayEntrance*
        """
        return list(self.block_entrances) + list(self.highway_entrances)

    def get_exit_blocks(self):
        """
        All cells that qualify as an **end-point**:

          • every *BlockEntrance*
          • every *HighwayExit*
        """
        return list(self.block_entrances) + list(self.highway_exits)

    def get_valid_exits(self, entry_cell):
        """
        Given a starting entrance cell, return every permissible exit:

        • If *entry_cell* is a **BlockEntrance** →
            ⟶ all other *BlockEntrances* **and** every *HighwayExit*.

        • If *entry_cell* is a **HighwayEntrance** →
            ⟶ every *BlockEntrance* **plus** every *HighwayExit*
               that is **not adjacent** to the starting entrance.

        For any other cell type the function returns an empty list.
        """
        if entry_cell.cell_type == "BlockEntrance":
            exits = [be for be in self.block_entrances if be is not entry_cell]
            exits += self.highway_exits
            return exits

        if entry_cell.cell_type == "HighwayEntrance":
            exits = [
                        be for be in self.block_entrances
                        if not self._are_adjacent(be, entry_cell)
                    ] + [
                        hx for hx in self.highway_exits
                        if not self._are_adjacent(hx, entry_cell)
                    ]
            return exits

        # not a recognised start block
        return []

    # ═══════════════════════════════════════════════════════════════
    #  NUMBA OPTIMIZATION HELPERS
    # ═══════════════════════════════════════════════════════════════

    def export_to_grid(self, *, dtype=np.int8) -> np.ndarray:
        """
        Fast grid export for A*:
            0 = drivable cell
            3 = impassable cell (building, sidewalk, wall, …)
        The static 0/3 mask is cached once; the caller will paint
        1 = vehicle   and   2 = stop/red-light   on top.
        """
        w, h = self.get_width(), self.get_height()

        # ── 1. build & cache static mask on first call ──────────────────
        if getattr(self, "_road_mask", None) is None:
            road_mask = np.full((w, h), 3, dtype=dtype)  # default = blocked
            road_like = Defaults.ROAD_LIKE_TYPES
            for x in range(w):
                for y in range(h):
                    ags = self.get_cell_contents(x, y)
                    if ags and ags[0].cell_type in road_like:
                        road_mask[x, y] = 0  # drivable
            self._road_mask = road_mask

        # ── 2. cheap copy; caller mutates it ----------------------------
        return self._road_mask.copy()

    def export_to_grid_cuda(self, *, dtype=np.int8) -> np.ndarray:
        """
        Fast grid export for A* with CUDA support:
            0 = drivable cell
            255 = impassable cell (building, …)
            1 = live vehicle
            2 = stop/red-light
        The static mask is cached once; dynamic overlays are JIT-compiled.
        """
        w, h = self.get_width(), self.get_height()

        # ── 1) Build & cache static mask on first call ───────────────────
        if getattr(self, "_road_mask", None) is None:
            road_mask = np.full((w, h), 255, dtype=dtype)  # default = wall
            road_like = Defaults.ROAD_LIKE_TYPES
            for x in range(w):
                for y in range(h):
                    ags = self.get_cell_contents(x, y)
                    if ags and ags[0].cell_type in road_like:
                        road_mask[x, y] = 0  # drivable
                    elif ags and ags[0].cell_type == "Stop":
                        road_mask[x, y] = 2  # static stop
            self._road_mask = road_mask

        # ── 2) Create a mutable copy for this step ───────────────────────
        grid = self._road_mask.copy()

        # ── 3) Gather dynamic positions (always 2-D) ─────────────────────
        vlist = list(self.occupied_vehicle_positions())
        vpos = np.array(vlist, dtype=np.int64).reshape(-1, 2)
        slist = list(self.stop_cells)
        spos = np.array(slist, dtype=np.int64).reshape(-1, 2)

        # ── 4) Overlay via JIT-compiled function ─────────────────────────
        grid = overlay_dynamic(grid, vpos, spos)

        return grid

    def occupied_vehicle_positions(self):
        """
        Yield ``(x, y)`` for every grid cell that is *currently occupied*
        by a **VehicleAgent** which is *not* parked.
        """
        return self.vehicle_cells

    def stop_cell_positions(self):
        """
        Yield ``(x, y)`` for every cell whose traffic-control *status*
        is ``"Stop"`` …
        """
        for pos in self.stop_cells:
            yield pos

    _DIR_TO_K: dict[str, int] = {}
    for k, (dx, dy) in enumerate(astar_jit.NEIGHBOR_VECTORS):
        for d, vec in astar_jit.DIRECTION_VECTORS.items():
            if vec == (dx, dy):
                _DIR_TO_K[d] = k

    def get_flow_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (rows, cols, 4) where mask[x,y,k] is True
        iff from cell (x,y) there is allowed flow/movement in the k-th neighbor
        direction (using the same NEIGHBOR_VECTORS as the JIT A* expects).
        """
        grid = self.export_to_grid(dtype=np.int8)
        rows, cols = grid.shape
        mask = np.zeros((rows, cols, len(astar_jit.NEIGHBOR_VECTORS)), dtype=bool)

        for x in range(rows):
            for y in range(cols):
                # get_cell_contents always returns a list, road cells first
                cell = self.get_cell_contents(x, y)[0]
                for d in cell.directions:
                    # look up the exact k that matches this direction
                    k = self._DIR_TO_K.get(d)
                    if k is None:
                        raise KeyError(f"Unknown direction '{d}' at cell ({x},{y})")
                    mask[x, y, k] = True

        return mask