from __future__ import annotations
import traceback, tornado.escape
from mesa.visualization.modules import TextElement
from tornado.web import RequestHandler
from typing import List

class TrafficLightControl(TextElement):
    def render(self, model):
        # ── FIRST dropdown selection ────────────────────────────────
        tls: List = list(model.get_traffic_lights())
        if not hasattr(model, "user_selected_traffic_light"):
            model.user_selected_traffic_light = None
        if model.user_selected_traffic_light is None and tls:
            model.user_selected_traffic_light = tls[0].unique_id

        # ── SECOND dropdown selection ─────────
        ilgs: List = []
        if hasattr(model, "get_intersection_light_groups"):
            ilgs = list(model.get_intersection_light_groups())
        if not hasattr(model, "user_selected_intersection"):
            model.user_selected_intersection = None
        if model.user_selected_intersection is None and ilgs:
            model.user_selected_intersection = ilgs[0].unique_id

        # Build <option> lists ------------------------------------------------
        tl_opts = "\n".join(
            f'<option value="{tl.unique_id}"'
            f'{" selected" if str(tl.unique_id)==str(model.user_selected_traffic_light) else ""}>'
            f'{tl.unique_id}</option>' for tl in tls
        )
        ilg_opts = "\n".join(
            f'<option value="{g.unique_id}"'
            f'{" selected" if str(g.unique_id)==str(model.user_selected_intersection) else ""}>'
            f'{g.unique_id}</option>' for g in ilgs
        )
        return f"""
<style>
:root {{
  --purple:#6d28d9; --go:#34c759; --stop:#ff453a; --ring:rgba(0,0,0,.08);
}}
#tl-card {{
  font:14px/1.35 system-ui,sans-serif; background:var(--purple); color:#fff;
  padding:8px 12px 10px; margin:6px; border-radius:12px;
  box-shadow:0 2px 5px var(--ring); display:grid; gap:12px;
}}
#tl-card h3 {{ margin:0; font-size:17px; font-weight:600; }}

#tl-toolbar {{ display:flex; flex-wrap:wrap; gap:30px; align-items:center; justify-content:space-evenly;}}
.tl-group    {{ display:flex; gap:5px; align-items:center; }}

select.tlc-select {{
  padding:6px 10px; border:1px solid var(--ring);
  border-radius:8px; min-width:140px; color:#222;
}}

.tl-btn {{
  appearance:none; border:none; padding:8px 16px; border-radius:24px;
  font-weight:600; cursor:pointer; transition:transform .05s ease, box-shadow .15s ease;
  box-shadow:0 1px 2px var(--ring);
}}
.tl-btn:hover  {{ transform:translateY(-1px); box-shadow:0 4px 6px var(--ring); }}
.tl-btn:active {{ transform:translateY(0);    box-shadow:0 1px 2px var(--ring) inset; }}

.go-one, .go-ilg, .go-all        {{ background:var(--go);   color:#fff; }}
.stop-one, .stop-ilg, .stop-all  {{ background:var(--stop); color:#fff; }}

</style>

<div id=\"tl-card\">
  <h3>Manual Traffic Light Control</h3>
  <div id=\"tl-toolbar\">

    <!-- ◇ Individual light --------------------------------------------- -->
    <div class=\"tl-group\">
      <select id=\"tl_select\" class=\"tlc-select\"
        onchange=\"fetch('/set_user_selected_traffic_light',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id:this.value}})}}).catch(console.error);\">{tl_opts}</select>
      <button class=\"tl-btn go-one\"  onclick=\"fetch('/set_traffic_light_go',{{method:'POST'}}).then(()=>{{(document.getElementById('step_button')||document.getElementById('step'))?.click();}}).catch(console.error)\">Go</button>
      <button class=\"tl-btn stop-one\" onclick=\"fetch('/set_traffic_light_stop',{{method:'POST'}}).then(()=>{{(document.getElementById('step_button')||document.getElementById('step'))?.click();}}).catch(console.error)\">Stop</button>
    </div>

    <!-- ◇ Intersection group ------------------------------------------- -->
    <div class=\"tl-group\">
      <select id=\"ilg_select\" class=\"tlc-select\"
        onchange=\"fetch('/set_user_selected_intersection',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id:this.value}})}}).catch(console.error);\">{ilg_opts}</select>
      <button class=\"tl-btn go-ilg\"  onclick=\"fetch('/set_ilg_go',{{method:'POST'}}).then(()=>{{(document.getElementById('step_button')||document.getElementById('step'))?.click();}}).catch(console.error)\">Go</button>
      <button class=\"tl-btn stop-ilg\" onclick=\"fetch('/set_ilg_stop',{{method:'POST'}}).then(()=>{{(document.getElementById('step_button')||document.getElementById('step'))?.click();}}).catch(console.error)\">Stop</button>
    </div>

    <!-- ◇ Global -------------------------------------------------------- -->
    <div class=\"tl-group\">
      <button class=\"tl-btn go-all\"  onclick=\"fetch('/set_traffic_lights_go',{{method:'POST'}}).then(()=>{{(document.getElementById('step_button')||document.getElementById('step'))?.click();}}).catch(console.error)\">All GO</button>
      <button class=\"tl-btn stop-all\" onclick=\"fetch('/set_traffic_lights_stop',{{method:'POST'}}).then(()=>{{(document.getElementById('step_button')||document.getElementById('step'))?.click();}}).catch(console.error)\">All STOP</button>
      <button class=\"tl-btn\" style=\"background:#fff;color:var(--purple);\" onclick=\"(document.getElementById('step_button')||document.getElementById('step'))?.click();\">Step</button>
    </div>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
#  Tornado – helpers & handlers
# ---------------------------------------------------------------------------
class _Base(RequestHandler):
    def initialize(self, server):
        self.server = server
    def _json(self):
        try:
            return tornado.escape.json_decode(self.request.body) or {}
        except Exception:
            return {}
    def _ok(self, msg="OK"):
        self.write(msg)
    def _err(self, status, msg):
        self.set_status(status); self.write(msg)

class SetUserSelectionHandler(_Base):
    def post(self):
        try:
            uid = self._json().get("id")
            self.server.model.user_selected_traffic_light = uid
            self._ok(f"SELECT OK {uid}")
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())

class SetSingleGoHandler(_Base):
    def post(self):
        try:
            uid = self.server.model.user_selected_traffic_light
            next(tl for tl in self.server.model.get_traffic_lights() if str(tl.unique_id)==str(uid)).set_light_go()
            self._ok(f"GO OK {uid}")
        except StopIteration:
            self._err(404, f"ERROR No traffic-light {uid}")
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())

class SetSingleStopHandler(_Base):
    def post(self):
        try:
            uid = self.server.model.user_selected_traffic_light
            next(tl for tl in self.server.model.get_traffic_lights() if str(tl.unique_id)==str(uid)).set_light_stop()
            self._ok(f"STOP OK {uid}")
        except StopIteration:
            self._err(404, f"ERROR No traffic-light {uid}")
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())

class SetGoHandler(_Base):
    def post(self):
        try:
            self.server.model.set_traffic_lights_go()
            self._ok()
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())

class SetStopHandler(_Base):
    def post(self):
        try:
            self.server.model.set_traffic_lights_stop()
            self._ok()
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())


class SetUserIntersectionSelectionHandler(_Base):
    def post(self):
        try:
            uid = self._json().get("id")
            self.server.model.user_selected_intersection = uid
            self._ok(f"ILG SELECT OK {uid}")
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())

class SetIlgGoHandler(_Base):
    def post(self):
        try:
            uid = self.server.model.user_selected_intersection
            group = next(g for g in self.server.model.get_intersection_light_groups() if str(g.unique_id)==str(uid))
            group.set_all_go()
            self._ok(f"ILG GO OK {uid}")
        except StopIteration:
            self._err(404, f"ERROR No intersection‑group {uid}")
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())

class SetIlgStopHandler(_Base):
    def post(self):
        try:
            uid = self.server.model.user_selected_intersection
            group = next(g for g in self.server.model.get_intersection_light_groups() if str(g.unique_id)==str(uid))
            group.set_all_stop()
            self._ok(f"ILG STOP OK {uid}")
        except StopIteration:
            self._err(404, f"ERROR No intersection‑group {uid}")
        except Exception:
            self._err(500, "ERROR " + traceback.format_exc())


def add_traffic_light_routes(server):
    """Attach *all* traffic‑light + group control routes to the Tornado app."""
    print("️🚦 Adding Traffic & ILG Control to server…")
    server.add_handlers(
        r".*",
        [
            # individual lights
            (r"/set_user_selected_traffic_light", SetUserSelectionHandler, dict(server=server)),
            (r"/set_traffic_light_go",    SetSingleGoHandler,        dict(server=server)),
            (r"/set_traffic_light_stop",  SetSingleStopHandler,      dict(server=server)),
            (r"/set_traffic_lights_go",   SetGoHandler,              dict(server=server)),
            (r"/set_traffic_lights_stop", SetStopHandler,            dict(server=server)),

            # intersection‑level control
            (r"/set_user_selected_intersection", SetUserIntersectionSelectionHandler, dict(server=server)),
            (r"/set_ilg_go",              SetIlgGoHandler,           dict(server=server)),
            (r"/set_ilg_stop",            SetIlgStopHandler,         dict(server=server)),
        ],
    )
