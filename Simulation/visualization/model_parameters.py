# model_parameters.py
# ---------------------------------------------------------------------------
#  Parameter panel with (1) tidy headings / spacers   (2) label-left layout
# ---------------------------------------------------------------------------
import mesa
from itertools import count
from Simulation.config import Defaults

# ───────────────────────── 1 · GLOBAL LAYOUT CSS ────────────────────────────
PARAM_PANEL_CSS = mesa.visualization.StaticText("""
<style id="inline-param-layout">
  #elements_container > div[style^='margin: 10px']{
      display:flex;               
      align-items:center;
      gap:12px;
  }
  #elements_container > div[style^='margin: 10px'] br{display:none;}

  #elements_container label{
      flex:1 0 45%;               
      margin:0;
      font-weight:500;
  }
  #elements_container input[type='range'],
  #elements_container select,
  #elements_container input[type='checkbox']{
      flex:1 0 55%;
  }
  #elements_container span[id$='_value']{
      margin-left:8px;
      min-width:32px;
      text-align:right;
      display:inline-block;
  }
</style>
""")

# ───────────────────────── 2 · SMALL HTML HELPERS ───────────────────────────
def h(title: str) -> mesa.visualization.StaticText:
    return mesa.visualization.StaticText(
        f'''
        <div style="
            width:250px;              /* ← static width — tweak as needed */
            background:#6d28d9;       /* purple fill   */
            color:#fff;               /* white letters */
            padding:6px 0;            /* vertical breathing room */
            margin:18px 0 10px 0;     /* space above / below */
            font-size:16px;
            line-height:1.25;
            font-weight:600;
            border-radius:6px;        /* rounded corners */
            text-align:center;        /* centred title */
        ">{title}</div>
        '''
    )

def spacer(px: int = 8) -> mesa.visualization.StaticText:
    """Vertical breathing room."""
    return mesa.visualization.StaticText(f'<div style="height:{px}px;"></div>')

# unique keys for every spacer so JSON serialisation is happy
_sp = count()
def sp_key() -> str:
    return f"__sp{next(_sp)}"

# ───────────────────────── 3 · PARAMETER DICTIONARY ─────────────────────────
model_params = {
    "_css": PARAM_PANEL_CSS,
    # ── Outer frame ───────────────────────────────────────────────────────
    "_hdr_frame": h("Outer frame"),
    sp_key(): spacer(),
    "wall_thickness": mesa.visualization.Slider(
        name="Wall thickness", value=Defaults.WALL_THICKNESS, min_value=4, max_value=20, step=1,
        description="Outer wall thickness",
    ),
    sp_key(): spacer(),
    "sidewalk_ring_width": mesa.visualization.Slider(
        name="Sidewalk ring width", value=Defaults.SIDEWALK_RING_WIDTH, min_value=1, max_value=10, step=1,
        description="Sidewalk ring width",
    ),
    sp_key(): spacer(),

    # ── Road network ──────────────────────────────────────────────────────
    "_hdr_road": h("Road network"),
    sp_key(): spacer(),
    "ring_road_type": mesa.visualization.Choice(
        name="Ring road type", value=Defaults.RING_ROAD_TYPE,
        choices=Defaults.ROADS,
        description="Road type for the perimeter ring",
    ),
    sp_key(): spacer(),
    "highway_offset_from_edges": mesa.visualization.Slider(
        name="Highway offset", value=Defaults.HIGHWAY_OFFSET, min_value=0, max_value=20, step=1,
        description="Distance from grid edge to highway centre-line",
    ),
    sp_key(): spacer(),
    "r1_chance_mean": mesa.visualization.Slider(
        name="R1 chance μ", value=Defaults.R1_CHANCE_MEAN,
        min_value=0.0, max_value=1.0, step=0.01,
        description="Average probability of drawing an R1 (highway) band."
    ),
    "r1_chance_std": mesa.visualization.Slider(
        name="R1 chance σ", value=Defaults.R1_CHANCE_STD,
        min_value=0.0, max_value=0.25, step=0.01,
        description="Spread (std-dev) of the R1 probability each draw."
    ),
    "r2_chance_mean": mesa.visualization.Slider(
        name="R2 chance μ", value=Defaults.R2_CHANCE_MEAN,
        min_value=0.0, max_value=1.0, step=0.01,
        description="Average probability of drawing an R2 band."
    ),
    "r2_chance_std": mesa.visualization.Slider(
        name="R2 chance σ", value=Defaults.R2_CHANCE_STD,
        min_value=0.0, max_value=0.25, step=0.01,
        description="Spread (std-dev) of the R2 probability."
    ),
    "min_r1_bands": mesa.visualization.Slider(
        name="Min R1 per axis", value=Defaults.MIN_R1_BANDS,
        min_value=0, max_value=6, step=1,
        description="Guarantee at least this many R1 bands horizontally "
                    "AND vertically (ring-road excluded)."
    ),
    sp_key(): spacer(),

    # ── Block layout ──────────────────────────────────────────────────────
    "_hdr_blocks": h("Block layout"),
    sp_key(): spacer(),
    "min_block_spacing": mesa.visualization.Slider(
        name="Min block size", value=Defaults.MIN_BLOCK_SPACING,  min_value=3,  max_value=24, step=1,
    ),
    sp_key(): spacer(),
    "max_block_spacing": mesa.visualization.Slider(
        name="Max block size", value=Defaults.MAX_BLOCK_SPACING, min_value=8,  max_value=48, step=1,
    ),
    sp_key(): spacer(),

    # ── Sub-block roads ───────────────────────────────────────────────────
    "_hdr_subblocks": h("Sub-block roads"),
    sp_key(): spacer(),
    "carve_subblock_roads": mesa.visualization.Checkbox(
        name="Enable sub-block roads", value=Defaults.CARVE_SUBBLOCK_ROADS,
    ),
    sp_key(): spacer(),
    "min_subblock_spacing": mesa.visualization.Slider(
        name="Min sub-block spacing", value=Defaults.MIN_SUBBLOCK_SPACING,
        min_value=2, max_value=24, step=1,
    ),
    sp_key(): spacer(),
    "subblock_chance": mesa.visualization.Slider(
        name="Subblock Chance", value=Defaults.SUBBLOCK_CHANGE,
        min_value=0, max_value=1, step=0.05,
    ),
    sp_key(): spacer(),

    # ── Traffic control ───────────────────────────────────────────────────
    "_hdr_control": h("Traffic control"),
    sp_key(): spacer(),
    "subblock_roads_have_intersections": mesa.visualization.Checkbox(
        name="Subblock roads can intersect", value=Defaults.SUBBLOCK_ROADS_HAVE_INTERSECTIONS,
    ),
    sp_key(): spacer(),
    "optimized_intersections": mesa.visualization.Checkbox(
        name="Optimised intersections", value=Defaults.OPTIMISED_INTERSECTIONS,
    ),
    sp_key(): spacer(),
    "traffic_light_range": mesa.visualization.Slider(
        name="Traffic-light sensor range", value=Defaults.TRAFFIC_LIGHT_RANGE,
        min_value=0, max_value=20, step=1,
    ),
    sp_key(): spacer(),
    "forward_traffic_light_range": mesa.visualization.Checkbox(
        name="Traffic-light Forward range", value=Defaults.FORWARD_TRAFFIC_LIGHT_RANGE,
        description="Include the road blocks after the traffic light in the traffic light sensor range."
    ),
    sp_key(): spacer(),
    "forward_traffic_light_range_intersections": mesa.visualization.Choice(
        name="Traffic-light Intersection range", value=Defaults.FORWARD_TRAFFIC_LIGHT_INTERSECTIONS,
        choices=Defaults.FORWARD_TRAFFIC_LIGHT_INTERSECTION_OPTIONS,
        description="How to handle traffic light sensor range for intersections."
    ),
    "_hdr_city_blocks": h("City Blocks"),
    sp_key(): spacer(),
    "gradual_city_block_resources": mesa.visualization.Checkbox(
        name="Gradual Resources", value=Defaults.GRADUAL_CITY_BLOCK_RESOURCES,
        description="Apply gradual production or consumption of city block resources, instead of doing a bunk per X ticks"
    ),
    sp_key(): spacer(),
    sp_key(): spacer(),
}
