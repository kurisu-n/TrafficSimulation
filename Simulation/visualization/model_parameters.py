# model_parameters.py
# ---------------------------------------------------------------------------
#  Parameter panel with (1) tidy headings / spacers   (2) label-left layout
# ---------------------------------------------------------------------------
import mesa
from itertools import count

# ───────────────────────── 1 · GLOBAL LAYOUT CSS ────────────────────────────
PARAM_PANEL_CSS = mesa.visualization.StaticText("""
<style id="inline-param-layout">
  /* every auto-generated parameter block */
  #elements_container > div[style^='margin: 10px']{
      display:flex;               /* label ‖ widget  */
      align-items:center;
      gap:12px;
  }
  /* kill the hard-coded <br> after each label */
  #elements_container > div[style^='margin: 10px'] br{display:none;}

  /* label (left) */
  #elements_container label{
      flex:1 0 45%;               /* ≤ 45 % but can grow */
      margin:0;
      font-weight:500;
  }
  /* slider / checkbox / select (right) */
  #elements_container input[type='range'],
  #elements_container select,
  #elements_container input[type='checkbox']{
      flex:1 0 55%;
  }
  /* numeric value read-out */
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
    """Purple header pill – fixed width, centred text."""
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
model_params_mesa = {
    # (a) inject global CSS first
    "_css": PARAM_PANEL_CSS,

    # ── Grid geometry ─────────────────────────────────────────────────────
    "_hdr_grid": h("Grid geometry"),
    sp_key(): spacer(),
    "width":  mesa.visualization.Slider(
        name="Grid Width", value=100, min_value=40, max_value=200, step=10,
        description="Square grid size",
    ),
    sp_key(): spacer(),
    "height": mesa.visualization.Slider(
        name="Grid Height", value=100, min_value=40, max_value=200, step=10,
        description="Square grid size",
    ),
    sp_key(): spacer(),

    # ── Outer frame ───────────────────────────────────────────────────────
    "_hdr_frame": h("Outer frame"),
    sp_key(): spacer(),
    "wall_thickness": mesa.visualization.Slider(
        name="Wall thickness", value=5, min_value=4, max_value=20, step=1,
        description="Outer wall thickness",
    ),
    sp_key(): spacer(),
    "sidewalk_ring_width": mesa.visualization.Slider(
        name="Sidewalk ring width", value=2, min_value=1, max_value=10, step=1,
        description="Sidewalk ring width",
    ),
    sp_key(): spacer(),

    # ── Road network ──────────────────────────────────────────────────────
    "_hdr_road": h("Road network"),
    sp_key(): spacer(),
    "ring_road_type": mesa.visualization.Choice(
        name="Ring road type", value="R2",
        choices=["R1", "R2", "R3"],
        description="Road type for the perimeter ring",
    ),
    sp_key(): spacer(),
    "highway_offset_from_edges": mesa.visualization.Slider(
        name="Highway offset", value=5, min_value=0, max_value=20, step=1,
        description="Distance from grid edge to highway centre-line",
    ),
    sp_key(): spacer(),

    # ── Block layout ──────────────────────────────────────────────────────
    "_hdr_blocks": h("Block layout"),
    sp_key(): spacer(),
    "min_block_spacing": mesa.visualization.Slider(
        name="Min block size", value=8,  min_value=3,  max_value=12, step=1,
    ),
    sp_key(): spacer(),
    "max_block_spacing": mesa.visualization.Slider(
        name="Max block size", value=16, min_value=8,  max_value=24, step=1,
    ),
    sp_key(): spacer(),
    "empty_block_chance": mesa.visualization.Slider(
        name="Empty block chance", value=0.1,
        min_value=0.0, max_value=1.0, step=0.05,
    ),
    sp_key(): spacer(),

    # ── Sub-block roads ───────────────────────────────────────────────────
    "_hdr_subblocks": h("Sub-block roads"),
    sp_key(): spacer(),
    "carve_subblock_roads": mesa.visualization.Checkbox(
        name="Enable sub-block roads", value=True,
    ),
    sp_key(): spacer(),
    "min_subblock_spacing": mesa.visualization.Slider(
        name="Min sub-block spacing", value=4,
        min_value=2, max_value=8, step=1,
    ),
    sp_key(): spacer(),

    # ── Traffic control ───────────────────────────────────────────────────
    "_hdr_control": h("Traffic control"),
    sp_key(): spacer(),
    "subblock_roads_have_intersections": mesa.visualization.Checkbox(
        name="Subblock roads can intersect", value=False,
    ),
    sp_key(): spacer(),
    "optimized_intersections": mesa.visualization.Checkbox(
        name="Optimised intersections", value=True,
    ),
    sp_key(): spacer(),
    "traffic_light_range": mesa.visualization.Slider(
        name="Traffic-light sensor range", value=4,
        min_value=0, max_value=20, step=1,
    )
}
