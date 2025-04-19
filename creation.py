#creation.py

import random
from mesa import Model
from mesa.space import MultiGrid

from cell import CellAgent
from cell import ZONE_COLORS

from typing import List


GRID_WIDTH = 100
GRID_HEIGHT = 100

WALL_THICKNESS = 5
SIDEWALK_RING_WIDTH = 2

ROAD_THICKNESS = {
    "R1": 4,
    "R2": 2,
    "R3": 1,
    "R4": 1,                # NEW
}

OPTIMIZED_INTERSECTIONS = True

EMPTY_BLOCK_CHANGE = 0.1

MIN_BLOCK_SPACING = 8
MAX_BLOCK_SPACING = 16

CARVE_SUBBLOCK_ROADS = True
SUBBLOCK_ROADS_HAVE_INTERSECTIONS = False
MIN_SUBBLOCK_SPACING = 4
SUBBLOCK_CHANCE    = 0.75
SUBBLOCK_ROAD_TYPE = "R3"

HIGHWAY_OFFSET_FROM_EDGES = 5

RING_ROAD_TYPE = "R2"

class StructuredCityModel(Model):
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, seed=None):
        """
        ring_road: None (default) means no ring road; otherwise it should be one of the
                   road types "R1", "R2", or "R3". When provided, the ring road is drawn
                   inside the sidewalk around the city block. This also shifts the interior
                   boundaries inward so that other roads (and highways) are placed further from
                   the grid edge.
        """
        super().__init__(seed=seed)  # Required in Mesa 3.x
        self.grid = MultiGrid(width, height, torus=False)
        self.space = self.grid

        # interior bounds
        self.interior_x_min = WALL_THICKNESS + SIDEWALK_RING_WIDTH
        self.interior_x_max = width  - (WALL_THICKNESS + SIDEWALK_RING_WIDTH) - 1
        self.interior_y_min = WALL_THICKNESS + SIDEWALK_RING_WIDTH
        self.interior_y_max = height - (WALL_THICKNESS + SIDEWALK_RING_WIDTH) - 1

        self.initial_road = RING_ROAD_TYPE
        self._blocks_data = []

        # trackers for quick lookup
        self.block_entrances   = []
        self.highway_entrances = []
        self.highway_exits     = []
        self.controlled_roads  = []
        self.traffic_lights    = []

        # build sequence
        self._place_thick_wall()
        self._place_sidewalk_inner_ring()
        self._clear_interior()
        self._build_roads_and_sidewalks()

        if CARVE_SUBBLOCK_ROADS:
            self._carve_subblock_roads()

        self._flood_fill_blocks_storing_data()
        self._eliminate_dead_ends()
        self._upgrade_r2_to_intersections()
        self._final_place_block_entrances()
        self._remove_invalid_intersection_directions()
        self._add_entrance_directions()
        self._add_traffic_lights()

    def cell_contents(self, x: int, y: int) -> List[CellAgent]:
        """
        Return the list of CellAgent in the given [x, y] cell.
        """
        # We know this grid only ever holds CellAgent instances
        return self.grid.get_cell_list_contents([(x, y)])  # type: ignore[return-value]
    
    def get_width(self) -> int:
        return self.grid.width
    def get_height(self) -> int:
        return self.grid.height
    
    # -------------------------------------------------------------------
    #  Generic intersection factory (optimised + sub‑block aware)
    # -------------------------------------------------------------------
    def _make_intersection(self, x: int, y: int) -> None:
        """
        Upgrade (x, y) to a four‑way **Intersection** following these rules:

        ──  OPTIMISED = False  ───────────────────────────────────────────
        • Every overlap of single ✕ multi is turned into an intersection
          → the entire width of the multi‑lane road is upgraded.

        ──  OPTIMISED = True   ───────────────────────────────────────────
        • For single ✕ single  → one normal intersection cell.
        • For single ✕ multi   → **only the outer‑most lane(s)** of the
          thicker road become intersections
          (offset 0 and offset band_size‑1).  Inner lanes stay as normal
          road cells.

        Sub‑block R4 legs are handled by synthesising one‑cell “dummy
        bands” because they are not present in the main band tables.
        """

        # ---------- helpers -------------------------------------------------
        def _dummy_band(coord: int, rtype: str):
            return coord, coord, rtype, None        # (start, end, type, dir)

        def _band_info(band, coord: int):
            st, en, rt, bd = band
            width = en - st + 1
            off   = coord - st
            return st, en, rt, bd, width, off

        def _ensure_intersection(cx: int, cy: int):
            """Create / register a 4‑way intersection at (cx, cy)."""
            ags = self.cell_contents(cx,cy)
            if ags and ags[0].cell_type == "Intersection":
                return
            self._replace_cell(cx, cy, "Intersection")
            ag_s = self.cell_contents(cx,cy)[0]
            ag_s.directions = ["N", "S", "E", "W"]
            if hasattr(self, "_intersection_cells"):
                self._intersection_cells.add((cx, cy))
        # --------------------------------------------------------------------

        # ── 1. find (or fabricate) the covering bands ─────────────────────
        hband = self._find_band_covering(y, self.horizontal_bands)
        vband = self._find_band_covering(x, self.vertical_bands)

        # inject dummy bands when an R4 sub‑block road is involved
        if not hband and (
            self._is_type(x, y, SUBBLOCK_ROAD_TYPE) or
            any(self._is_type(nx, y, SUBBLOCK_ROAD_TYPE) for nx in (x - 1, x + 1))
        ):
            hband = _dummy_band(y, SUBBLOCK_ROAD_TYPE)

        if not vband and (
            self._is_type(x, y, SUBBLOCK_ROAD_TYPE) or
            any(self._is_type(x, ny, SUBBLOCK_ROAD_TYPE) for ny in (y - 1, y + 1))
        ):
            vband = _dummy_band(x, SUBBLOCK_ROAD_TYPE)

        if not (hband and vband):               # not a real crossing
            return

        # ── 2. band details ───────────────────────────────────────────────
        h_st, h_en, h_rt, h_bd, h_sz, h_off = _band_info(hband, y)
        v_st, v_en, v_rt, v_bd, v_sz, v_off = _band_info(vband, x)

        single_vs_multi = (
            (h_sz == 1 and v_sz > 1) or
            (v_sz == 1 and h_sz > 1)
        )

        # ── 3. OPTIMISED mode – keep only outer‑most lanes ────────────────
        if OPTIMIZED_INTERSECTIONS and single_vs_multi:
            if h_sz > 1:                         # horizontal is the multi
                multi_rt, multi_orient = h_rt, "horizontal"
                multi_off, multi_sz, bdir = h_off, h_sz, h_bd
            else:                               # vertical is the multi
                multi_rt, multi_orient = v_rt, "vertical"
                multi_off, multi_sz, bdir = v_off, v_sz, v_bd

            # keep only offset 0 or offset band_size‑1
            if multi_off not in (0, multi_sz - 1):
                # inner lane → revert to normal road cell
                dirs = self._compute_lane_dirs(
                    multi_rt, multi_orient, multi_off, multi_sz, bdir
                )
                self._replace_cell(x, y, multi_rt)
                ag = self.cell_contents(x,y)[0]
                ag.directions = dirs
                if hasattr(self, "_intersection_cells"):
                    self._intersection_cells.discard((x, y))
                # keep bookkeeping consistent
                self._road_cells[(x, y)] = (
                    multi_rt, multi_orient, multi_off, multi_sz, bdir
                )
                return

            # outer‑most lane ➜ real intersection
            _ensure_intersection(x, y)
            return

        # ── 4. non‑optimised mode OR multi ✕ multi ───────────────────────
        _ensure_intersection(x, y)


    def _compute_highway_inset(self):
        return self.interior_x_min + HIGHWAY_OFFSET_FROM_EDGES

    # -----------------------------------------------------------------------
    # (1) Boundary wall
    # -----------------------------------------------------------------------
    def _place_thick_wall(self):
        w, h = self.get_width(), self.get_height()
        for y in range(h):
            for x in range(w):
                ag = CellAgent(self, "Wall", (x, y))
                self._place_agent(ag, (x, y))

    def _replace_if_wall(self, x, y, new_type):
        ags = self.cell_contents(x,y)
        if ags and ags[0].cell_type == "Wall":
            self._replace_cell(x, y, new_type)

    # -----------------------------------------------------------------------
    # (2) Sidewalk ring (hug every wall cell’s inner face)
    # -----------------------------------------------------------------------
    def _place_sidewalk_inner_ring(self):
        w, h = self.get_width(), self.get_height()
        ws = WALL_THICKNESS
        sr = SIDEWALK_RING_WIDTH

        # carve each sidewalk “layer” depth sr
        for layer in range(sr):
            # compute which rows are this layer’s top & bottom
            y_top    = ws + layer
            y_bottom = h - ws - 1 - layer

            # === horizontal faces ===
            # only from x=ws … x=(w-ws-1) so we skip the corner columns
            for x in range(ws, w - ws):
                if self._is_type(x, y_top, "Wall"):
                    self._replace_cell(x, y_top, "Sidewalk")
                if self._is_type(x, y_bottom, "Wall"):
                    self._replace_cell(x, y_bottom, "Sidewalk")

            # === vertical faces ===
            x_left  = y_top
            x_right = y_bottom
            # only from y=ws … y=(h-ws-1) so we skip the corner rows
            for y in range(ws, h - ws):
                if self._is_type(x_left, y, "Wall"):
                    self._replace_cell(x_left, y, "Sidewalk")
                if self._is_type(x_right, y, "Wall"):
                    self._replace_cell(x_right, y, "Sidewalk")


    # -----------------------------------------------------------------------
    # (3) Clear interior => "Nothing"
    # -----------------------------------------------------------------------
    def _clear_interior(self):
        for y in range(self.interior_y_min, self.interior_y_max + 1):
            for x in range(self.interior_x_min, self.interior_x_max + 1):
                self._replace_cell(x, y, "Nothing")

        # -----------------------------------------------------------------------
        # (4) Build roads & sidewalks
        # -----------------------------------------------------------------------

    def _build_roads_and_sidewalks(self):
        w, h = self.get_width(), self.get_height()

        # (A) Make R2/R3 road bands in the interior.
        # We pass the forced initial_road value to both horizontal and vertical bands.
        self.horizontal_bands = self._make_road_bands_for_interior(
            self.interior_y_min, self.interior_y_max,
            orientation="horizontal", allow_highway=False, initial_road=self.initial_road
        )
        self.vertical_bands = self._make_road_bands_for_interior(
            self.interior_x_min, self.interior_x_max,
            orientation="vertical", allow_highway=False, initial_road=self.initial_road
        )

        # (B) Force 1 horizontal + 1 vertical R1 highway
        self._force_one_highway(self.horizontal_bands, total_size=h, orientation="horizontal")
        self._force_one_highway(self.vertical_bands, total_size=w, orientation="vertical")

        # (C) Place roads in the grid
        self._intersection_cells = set()
        self._road_cells = {}

        # In _build_roads_and_sidewalks, for placing roads in the grid:
        for y in range(h):
            hband = self._find_band_covering(y, self.horizontal_bands)
            for x in range(w):
                vband = self._find_band_covering(x, self.vertical_bands)

                # If both horizontal and vertical bands apply...
                if hband and vband:
                    (hstart, hend, hrtype, hbdir) = hband
                    (vstart, vend, vrtype, vbdir) = vband
                    # Skip if non-highway roads are outside the interior.
                    if (hrtype != "R1" or vrtype != "R1") and not self._inside_interior(x, y):
                        continue

                    # --- New Check: If this cell lies in a forced boundary corner, mark it as a regular road.
                    if self.initial_road is not None:
                        forced_thick = ROAD_THICKNESS[self.initial_road]
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
        # Mark roads
        for (rx, ry), (rtype, orientation, offset, band_size, bdir) in self._road_cells.items():
            if (rx, ry) in self._intersection_cells:
                continue
            self._replace_cell(rx, ry, rtype)
            rag = self.cell_contents(rx,ry)[0]
            # Compute the default directions.
            directions = self._compute_lane_dirs(rtype, orientation, offset, band_size, bdir)
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
                if not self._in_bounds(nx, ny):
                    continue
                # skip other road/intersection cells
                if (nx, ny) in road_positions or (nx, ny) in self._intersection_cells:
                    continue

                neigh = self.cell_contents(nx,ny)[0]
                # 1) carve into empty space
                if neigh.cell_type == "Nothing":
                    self._replace_cell(nx, ny, "Sidewalk")

                # 2) *also* carve into the wall if this is a highway‐lane
                else:
                    curr = self.cell_contents(rx,ry)[0].cell_type
                    if neigh.cell_type == "Wall" and curr in {"R1", "HighwayEntrance", "HighwayExit"}:
                        self._replace_cell(nx, ny, "Sidewalk")

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
        if self.initial_road != "R2":
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
    #  (4b)  Carve optional L‑shaped sub‑block roads (type R4)
    # -------------------------------------------------------------------
    def _carve_subblock_roads(self):
        """
        1.  Randomly inserts a one‑cell‑wide L‑shaped R4 road in large
            interior blocks (probability SUB_BLOCK_L_CHANCE).
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
        dir_vec = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
        opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}.__getitem__
        road = {"R1", "R2", "R3", "R4", "Intersection", "HighwayEntrance"}

        # ---------------- helpers ----------------------------------------------
        def neighbours(cx, cy):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if self._in_bounds(nx, ny):
                    yield nx, ny

        def lay_r4_cell(x, y, arrow):
            """Convert (x,y) → R4 (if not already road) and edge it with sidewalk."""
            ag = self.cell_contents(x,y)[0]
            if ag.cell_type not in road:
                self._replace_cell(x, y, SUBBLOCK_ROAD_TYPE)
                ag = self.cell_contents(x,y)[0]
                ag.directions = [arrow]
            elif ag.cell_type == "R4" and arrow not in ag.directions:
                ag.directions.append(arrow)

            # add sidewalk ring
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if self._in_bounds(nx, ny) and self._is_type(nx, ny, "Nothing"):
                    self._replace_cell(nx, ny, "Sidewalk")

        def extend_to_road(sx, sy, march_dir, arrow_dir):
            """
            March from (sx,sy) in *march_dir*.  Convert every Sidewalk or
            Nothing cell into R4, giving it *arrow_dir*.  Stops at – and now
            also updates – the first pre‑existing road cell, ensuring the
            outside road can actually turn into the new sub‑block road.
            """
            dx, dy = dir_vec[march_dir]
            cx, cy = sx, sy
            while self._in_bounds(cx, cy):
                tgt = self.cell_contents(cx,cy)[0]
                if tgt.cell_type in road:  # reached network
                    if SUBBLOCK_ROADS_HAVE_INTERSECTIONS:
                        self._make_intersection(cx, cy)
                        self._intersection_cells.add((cx, cy))
                    else:
                        # just give it the extra arrow into the R4
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
        W, H = self.get_width(), self.get_height()
        for y in range(H):
            for x in range(W):
                if (x, y) in visited or not self._is_type(x, y, "Nothing"):
                    continue

                # ---- flood‑fill this blob ----
                stack, region = [(x, y)], []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in visited or not self._is_type(cx, cy, "Nothing"):
                        continue
                    visited.add((cx, cy))
                    region.append((cx, cy))
                    for nx, ny in neighbours(cx, cy):
                        if (nx, ny) not in visited and self._is_type(nx, ny, "Nothing"):
                            stack.append((nx, ny))

                if not region or random.random() > SUBBLOCK_CHANCE:
                    continue

                # ---- bounding box / quick reject ----
                min_x = min(pt[0] for pt in region)
                max_x = max(pt[0] for pt in region)
                min_y = min(pt[1] for pt in region)
                max_y = max(pt[1] for pt in region)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                if width < 2 * MIN_SUBBLOCK_SPACING + 1 or \
                        height < 2 * MIN_SUBBLOCK_SPACING + 1:
                    continue

                # ---------- pick pivot & orientation ---------------------------
                for _ in range(20):  # up to 20 attempts
                    px = random.randint(min_x + MIN_SUBBLOCK_SPACING,
                                        max_x - MIN_SUBBLOCK_SPACING)
                    py = random.randint(min_y + MIN_SUBBLOCK_SPACING,
                                        max_y - MIN_SUBBLOCK_SPACING)
                    hor_dir = random.choice(["W", "E"])
                    ver_dir = random.choice(["N", "S"])
                    small_w = (px - min_x) if hor_dir == "W" else (max_x - px)
                    small_h = (py - min_y) if ver_dir == "S" else (max_y - py)
                    if small_w >= MIN_SUBBLOCK_SPACING and \
                            small_h >= MIN_SUBBLOCK_SPACING:
                        break
                else:
                    continue

                # ---------- inbound / outbound assignment ----------------------
                inbound_leg, outbound_leg = random.choice(
                    [("horizontal", "vertical"), ("vertical", "horizontal")]
                )
                leg_dir = {
                    "horizontal": {"in": opposite(hor_dir), "out": hor_dir},
                    "vertical": {"in": opposite(ver_dir), "out": ver_dir},
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
                pivot = self.cell_contents(px,py)[0]  # R4
                pivot.directions = [h_dir_cells if outbound_leg == "horizontal"
                                    else v_dir_cells]  # single outbound arrow

                # ---------- extend legs out to road -----------------------------
                # horizontal extension
                arrow_h = h_dir_cells  # arrow on extension
                extend_to_road(hx_end + dir_vec[hor_dir][0],
                               hy_end + dir_vec[hor_dir][1],
                               hor_dir, arrow_h)

                # vertical extension
                arrow_v = v_dir_cells
                extend_to_road(vx_end + dir_vec[ver_dir][0],
                               vy_end + dir_vec[ver_dir][1],
                               ver_dir, arrow_v)

                # ---------- surround pivot with sidewalk ------------------------
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (-1, 1), (1, -1), (-1, -1)):
                    nx, ny = px + dx, py + dy
                    if self._in_bounds(nx, ny):
                        nag = self.cell_contents(nx,ny)[0]
                        if nag.cell_type not in road and nag.cell_type != "Wall":
                            self._replace_cell(nx, ny, "Sidewalk")

    # -----------------------------------------------------------------------
    # (5) Flood-fill leftover => blocks (store for entrances)
    # -----------------------------------------------------------------------
    def _flood_fill_blocks_storing_data(self):
        visited = set()
        w, h = self.get_width(), self.get_height()

        for y in range(h):
            for x in range(w):
                # Skip anything already visited or not “Nothing”
                if (x, y) in visited or not self._is_type(x, y, "Nothing"):
                    continue

                # — Flood‑fill this empty region —
                stack, region = [(x, y)], []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in visited or not self._is_type(cx, cy, "Nothing"):
                        continue
                    visited.add((cx, cy))
                    region.append((cx, cy))
                    for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
                        if self._in_bounds(nx, ny) and (nx, ny) not in visited:
                            if self._is_type(nx, ny, "Nothing"):
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
                    if random.random() < (1 - EMPTY_BLOCK_CHANGE):
                        block_type = random.choice(["Residential", "Office", "Market", "Leisure"])
                    else:
                        block_type = "Empty"

                # — Fill every cell in region with that block_type —
                for bx, by in region:
                    self._replace_cell(bx, by, block_type)

                # — Carve the “ring” around it for potential entrances —
                ring = set()
                for bx, by in region:
                    for nx, ny in [(bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)]:
                        if self._in_bounds(nx, ny) and (nx, ny) not in region:
                            ring.add((nx, ny))

                for sx, sy in ring:
                    if self._is_type(sx, sy, "Nothing"):
                        self._replace_cell(sx, sy, "Sidewalk")

                # — Assign a unique block_id and store all data —
                block_id = len(self._blocks_data) + 1
                self._blocks_data.append({
                    "block_id": block_id,
                    "block_type": block_type,
                    "region": region,
                    "ring": list(ring)
                })

    # -----------------------------------------------------------------------
    # (6) Eliminate dead ends for R3 & normal Intersections
    # -----------------------------------------------------------------------
    def _eliminate_dead_ends(self):
        # Only consider these types as road cells.
        road_types = {"R1", "R2", "R3", "R4", "Intersection", "HighwayEntrance"}
        # Mark R2, R3, and Intersection as removable if they become dead-ends.
        removable_types = {"R2", "R3", "Intersection"}

        changed = True
        while changed:
            changed = False
            for y in range(self.get_height()):
                for x in range(self.get_width()):
                    ags = self.cell_contents(x,y)
                    if not ags:
                        continue
                    ctype = ags[0].cell_type
                    if ctype in removable_types:
                        neighbors = self._road_neighbors(x, y, road_types)
                        if len(neighbors) < 2:
                            self._replace_cell(x, y, "Sidewalk")
                            changed = True


    def _road_neighbors(self, x, y, road_types):
        results = []
        for (nx, ny) in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if self._in_bounds(nx, ny):
                ags = self.cell_contents(nx,ny)
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
                ags = self.cell_contents(x,y)
                if not ags:
                    continue
                agent = ags[0]
                if agent.cell_type == "R2":
                    # If using an R2 initial road, skip forced corner cells.
                    if self.initial_road == "R2":
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
                        if self._in_bounds(nx, ny):
                            neighbor_ags = self.cell_contents(nx,ny)
                            if neighbor_ags and neighbor_ags[0].cell_type == "Sidewalk":
                                sidewalk_count += 1
                    # If two or more sides have sidewalk, convert to an intersection.
                    if sidewalk_count >= 2:
                        self._make_intersection(x, y)

    # -----------------------------------------------------------------------
    # (7) Place block entrances
    # -----------------------------------------------------------------------
    def _final_place_block_entrances(self):
        valid_types = {"Residential", "Office", "Market", "Leisure"}

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
            self._replace_cell(cx, cy, "BlockEntrance")
            agent = self.cell_contents(cx,cy)[0]

            # annotate and track
            agent.block_id   = info["block_id"]
            agent.block_type = info["block_type"]
            self.block_entrances.append(agent)


    # -----------------------------------------------------------------------
    # (8A) Remove invalid intersection directions
    # -----------------------------------------------------------------------
    def _remove_invalid_intersection_directions(self):
        road_types = {"R1", "R2", "R3", "R4", "Intersection", "HighwayEntrance"}
        intersection_types = {"Intersection"}

        for y in range(self.get_height()):
            for x in range(self.get_width()):
                ags = self.cell_contents(x,y)
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
                        if d not in ["N", "S", "E", "W"]:
                            continue

                        nx, ny = self._next_cell_in_direction(x, y, d)
                        if not self._in_bounds(nx, ny):
                            continue

                        neighbor_ags = self.cell_contents(nx,ny)
                        if not neighbor_ags:
                            continue
                        neighbor_type = neighbor_ags[0].cell_type
                        neighbor_dirs = neighbor_ags[0].directions

                        # The neighbor must be a valid road cell.
                        if neighbor_type not in road_types:
                            continue

                        # NEW BEHAVIOR:
                        # If the neighboring cell is an intersection, always allow traffic toward it.
                        # Otherwise, require that the neighbor allows traffic coming from that direction.
                        if neighbor_type == "Intersection" or (d in neighbor_dirs):
                            valid_dirs.append(d)

                    agent.directions = valid_dirs


    def _next_cell_in_direction(self, x, y, d):
        if d == "N":
            return (x, y+1)
        elif d == "S":
            return (x, y-1)
        elif d == "E":
            return (x+1, y)
        elif d == "W":
            return (x-1, y)
        return (x, y)

    # -----------------------------------------------------------------------
    # (8B) Ensure roads next to a block entrance have direction in
    #      **and** block‐entrances point away from the block
    # -----------------------------------------------------------------------
    def _add_entrance_directions(self):
        road_types = {"R1", "R2", "R3", "R4", "Intersection", "HighwayEntrance"}

        opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}

        for y in range(self.get_height()):
            for x in range(self.get_width()):
                ags = self.cell_contents(x,y)
                if not ags:
                    continue
                agent = ags[0]
                if agent.cell_type == "BlockEntrance":
                    # we'll collect the BlockEntrance's own arrows here
                    entrance_dirs = []
                    # for each neighboring road, point both ways:
                    for (nx, ny) in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                        if self._in_bounds(nx, ny):
                            nags = self.cell_contents(nx,ny)
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
                                entrance_dirs.append(opposite[needed_dir])
                    agent.directions = entrance_dirs


    # -----------------------------------------------------------------------
    # Road band helper (modified)
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

        def opposite_dir(d):
            if d == "N": return "S"
            if d == "S": return "N"
            if d == "E": return "W"
            if d == "W": return "E"
            return d

        bands = []
        current = start_coord
        last_r3_dir = None

        # Generate bands using the random scheme.
        while current <= end_coord:
            # Choose a road type randomly.
            rtype = self._choose_road_type(allow_highway=allow_highway)
            thick = ROAD_THICKNESS[rtype]
            bstart = current
            bend = min(bstart + thick - 1, end_coord)

            if orientation == "horizontal":
                if rtype == "R3" and last_r3_dir is not None:
                    bdir = opposite_dir(last_r3_dir)
                else:
                    bdir = random.choice(["E", "W"])
            else:
                if rtype == "R3" and last_r3_dir is not None:
                    bdir = opposite_dir(last_r3_dir)
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
            block_size = random.randint(MIN_BLOCK_SPACING, MAX_BLOCK_SPACING)
            block_end = next_pos + block_size - 1
            if block_end > end_coord:
                break
            current = block_end + 1

        # --- Post-Processing: Force First and Last Bands if initial_road is set ---
        # --- Post-Processing: Force First and Last Bands if initial_road is set ---
        if initial_road is not None:
            forced_thick = ROAD_THICKNESS[initial_road]
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
        # Weighted random => 20% R1, 50% R2, 30% R3 if allow_highway
        r = random.random()
        if allow_highway:
            if r < 0.2:
                return "R1"
            elif r < 0.7:
                return "R2"
            else:
                return "R3"
        else:
            return "R2" if (random.random() < 0.5) else "R3"

    def _force_one_highway(self, bands, total_size, orientation):
        thick = ROAD_THICKNESS["R1"]
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

        skip_low  = hw_start - MIN_BLOCK_SPACING
        skip_high = hw_end   + MIN_BLOCK_SPACING
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
                return (st, en, rtype, bdir)
        return None

    def _compute_lane_dirs(self, rtype, orientation, offset, band_size, band_dir):
        """
        Revised lane direction computation to enforce European right-hand traffic (RHT)
        for R1 and R2 roads.

        Assumptions:
          - For horizontal roads, the band cells are ordered from south (offset 0) to north.
          - For vertical roads, the band cells are ordered from west (offset 0) to east.
        """
        # R3 remains one-way using its given direction.
        if rtype in ("R3", "R4"):  # <── was just R3
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
                    if offset > 0:
                        side_dirs.append("S")  # shift right (toward south)
                    if offset < half - 1:
                        side_dirs.append("N")  # optional left shift (toward north)
                    return [main_dir] + side_dirs
                else:
                    # Westbound lanes.
                    main_dir = "W"
                    side_dirs = []
                    # For westbound, the right-hand lane is the one with the highest offset.
                    if offset < band_size - 1:
                        side_dirs.append("N")  # shift right (toward north)
                    if offset > half:
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
                    if offset > 0:
                        side_dirs.append("W")  # shift right (toward west)
                    if offset < (band_size // 2) - 1:
                        side_dirs.append("E")  # optional left shift (toward east)
                    return [main_dir] + side_dirs
                else:
                    # Northbound group.
                    main_dir = "N"
                    side_dirs = []
                    if offset < band_size - 1:
                        side_dirs.append("E")  # shift right (toward east)
                    if offset > band_size // 2:
                        side_dirs.append("W")  # optional left shift (toward west)
                    return [main_dir] + side_dirs

        # Fallback: no direction assigned.
        return []

    def _replace_boundary_highways_with_entrances(self):
        w, h = self.get_width(), self.get_height()

        # (1) Collect all cells in the WALL_THICKNESS band
        edge_coords = set()
        for y in range(WALL_THICKNESS):
            for x in range(w):
                edge_coords.add((x, y))
        for y in range(h - WALL_THICKNESS, h):
            for x in range(w):
                edge_coords.add((x, y))
        for x in range(WALL_THICKNESS):
            for y in range(h):
                edge_coords.add((x, y))
        for x in range(w - WALL_THICKNESS, w):
            for y in range(h):
                edge_coords.add((x, y))

        # Maps for detecting “inward” direction
        inward_x = {0: "E", w - 1: "W"}
        inward_y = {0: "N", h - 1: "S"}

        # (2) Replace only the true boundary R1 → HighwayEntrance / Exit
        for (ex, ey) in edge_coords:
            ags = self.cell_contents(ex,ey)
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
            self._replace_cell(ex, ey, new_type)
            he = self.cell_contents(ex,ey)[0]
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
        road_types = {"R1", "R2", "R3", "R4", "HighwayEntrance"}

        w, h = self.get_width(), self.get_height()
        for x in range(w):
            for y in range(h):
                # fetch the only agent here
                ags = self.cell_contents(x,y)
                if not ags:
                    continue
                ag = ags[0]
                if ag.cell_type not in road_types:
                    continue

                dirs = ag.directions.copy()
                old_type = ag.cell_type

                # check each arrow for an Intersection neighbor
                for d in dirs:
                    nx, ny = self._next_cell_in_direction(x, y, d)
                    if not self._in_bounds(nx, ny):
                        continue
                    neigh = self.cell_contents(nx,ny)[0]
                    if neigh.cell_type != "Intersection":
                        continue

                    # → convert to ControlledRoad
                    self._replace_cell(x, y, "ControlledRoad")
                    ctrl = self.cell_contents(x,y)[0]
                    ctrl.directions = dirs
                    ctrl.status = "Pass"
                    ctrl.base_color = ZONE_COLORS.get(old_type)
                    self.controlled_roads.append(ctrl)

                    # → carve lights on every adjoining sidewalk
                    for sx, sy in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                        if not self._in_bounds(sx, sy):
                            continue
                        neigh2 = self.cell_contents(sx,sy)[0]
                        if neigh2.cell_type != "Sidewalk":
                            continue

                        # replace with a TrafficLight
                        self._replace_cell(sx, sy, "TrafficLight")
                        tl = self.cell_contents(sx,sy)[0]
                        tl.status = "Pass"
                        tl.controlled_road = ctrl
                        ctrl.lights.append(tl)
                        self.traffic_lights.append(tl)

                        # ray‑cast out along the light’s own arrows to find block entrances
                        for dd in tl.directions:
                            bx, by = sx, sy
                            while True:
                                bx, by = self._next_cell_in_direction(bx, by, dd)
                                if not self._in_bounds(bx, by):
                                    break
                                nb = self.cell_contents(bx,by)[0]
                                if nb.cell_type == "BlockEntrance":
                                    tl.controlled_blocks.append(nb.block_id)
                                    break

                    # only one set of lights per road cell
                    break

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
            if self._in_bounds(nx, ny):
                ags = self.cell_contents(nx,ny)
                if ags and ags[0].cell_type in [
                    "R1", "R2", "R3", "Intersection", "HighwayEntrance", "ControlledRoad"
                ]:
                    return True
        return False

    def _place_agent(self, agent: CellAgent, pos: tuple[int,int]):
        """
        Helper to place a freshly‑constructed CellAgent on the grid,
        making sure it has no stale .pos and that any existing agent
        at that cell is removed first.
        """
        # 1) remove any old agent(s) at that spot
        for old in list(self.grid.get_cell_list_contents([pos])):
            old.remove()
        # 2) clear any pre‑set position
        agent.pos = None
        # 3) put it on the grid (auto‑registers into self.agents)
        self.grid.place_agent(agent, pos)
        return agent

    def _replace_cell(self, x, y, new_type):
        """
        Thin wrapper around _place_agent that also lets you set
        a label/unique_id if you like.
        """
        ag = CellAgent(self, new_type, (x, y))
        return self._place_agent(ag, (x, y))

    def _is_type(self, x, y, ctype):
        ags = self.cell_contents(x,y)
        return bool(ags and ags[0].cell_type == ctype)

    def _in_bounds(self, x, y):
        return (0 <= x < self.get_width()) and (0 <= y < self.get_height())

    def _inside_interior(self, x, y):
        return (self.interior_x_min <= x <= self.interior_x_max) and \
               (self.interior_y_min <= y <= self.interior_y_max)

    # — Convenience getters —

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

    def step(self):
        # Calls every agent.step() in the current AgentSet
        self.agents.do("step")
