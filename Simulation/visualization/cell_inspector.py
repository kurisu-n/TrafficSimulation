# Simulation/visualization/cell_inspector.py
from mesa.visualization.modules import TextElement
from tornado.web import RequestHandler
import traceback, tornado.escape

# ───────────────────────────────────────────────────────────────
# 1.  Front-end snippet – installs one click handler on the grid
# ───────────────────────────────────────────────────────────────
class CellInspectorJS(TextElement):
    """
    Injects JS that sends the clicked cell’s x/y to /inspect_cell.
    Needs the grid’s logical width/height so we can map pixels → cells.
    """
    package_includes = ()       # none – the JS is inlined
    local_includes   = ()

    def __init__(self, grid_width: int, grid_height: int):
        super().__init__()
        self.grid_width  = grid_width
        self.grid_height = grid_height

    def render(self, model) -> str:
        # The little helper below runs once (idempotent) and never touches
        # Mesa’s own websocket – we just use a plain fetch() POST.
        return f"""
<script>
(function() {{
  if (window.__cellInspectorHooked) return;
  const canvas = document.querySelector("canvas");
  if (!canvas) return;  // grid not rendered yet
  const gw = {self.grid_width};
  const gh = {self.grid_height};

  canvas.addEventListener("click", ev => {{
    const rect = canvas.getBoundingClientRect();
    // canvas (0,0) is top-left; Mesa cell (0,0) is *bottom-left*.
    const cellX = Math.floor((ev.clientX - rect.left) / (rect.width  / gw));
    const cellY = gh - 1 - Math.floor((ev.clientY - rect.top)  / (rect.height / gh));

    fetch("/inspect_cell", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ x: cellX, y: cellY }})
    }}).catch(console.error);
  }});
  window.__cellInspectorHooked = true;
}})();
</script>
"""


# ───────────────────────────────────────────────────────────────
# 2.  Back-end handler – prints info to the server console
# ───────────────────────────────────────────────────────────────
class InspectCellHandler(RequestHandler):
    """
    POST body: {"x": <int>, "y": <int>}
    Prints a short description of every agent in that grid cell.
    """
    def initialize(self, server):
        self.server = server

    def post(self):
        try:
            data = tornado.escape.json_decode(self.request.body)
            x, y = int(data["x"]), int(data["y"])

            agents = self.server.model.get_cell_contents(x, y)
            if not agents:
                info = {"x": x, "y": y, "msg": "empty"}
            else:
                info = {
                    "x": x,
                    "y": y,
                    "agents": [
                        {
                            "id":     ag.unique_id,
                            "type":   ag.cell_type,
                            "dirs":   ag.directions,
                            "status": getattr(ag, "status", None)
                        }
                        for ag in agents
                    ]
                }

            # ---- report to console -----------------------------------------
            print("\n[Cell Inspector]", info)
            # -----------------------------------------------------------------
            self.write(info)               # handy for debugging in DevTools

        except Exception:
            traceback.print_exc()
            self.set_status(500)
            self.write("ERROR: "+ traceback.format_exc())


def add_cell_inspector(server):
    # Tornado route
    server.add_handlers(
        r".*",
        [(r"/inspect_cell", InspectCellHandler, dict(server=server))]
    )