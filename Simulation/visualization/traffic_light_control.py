from __future__ import annotations
import traceback, tornado.escape
from mesa.visualization.modules import TextElement
from tornado.web import RequestHandler

class TrafficLightControl(TextElement):
    def render(self, model):
        # ensure a default selection if none set
        lights = list(model.get_traffic_lights())
        if model.user_selected_traffic_light is None and lights:
            model.user_selected_traffic_light = lights[0].unique_id

        # Build options, marking the user-selected one as 'selected'
        options = "\n".join(
            f'<option value="{tl.unique_id}"'
            f'{" selected" if str(tl.unique_id)==str(model.user_selected_traffic_light) else ""}>'
            f'{tl.unique_id}</option>'
            for tl in lights
        )

        return f"""
<style>
:root {{
  --purple:#6d28d9; --go:#34c759; --stop:#ff453a; --ring:rgba(0,0,0,.08);
}}
#tl-card {{
  font:14px/1.35 system-ui,sans-serif;
  background:var(--purple); color:#fff;
  padding:16px 18px 10px; margin:12px; border-radius:12px;
  box-shadow:0 2px 5px var(--ring); display:grid; gap:14px;
}}
#tl-card h3 {{margin:0; font-size:17px; font-weight:600;}}

#tl-toolbar {{
  display:flex; gap:10px; align-items:center;
}}

#tl-select {{
  padding:6px 10px; border:1px solid var(--ring);
  border-radius:8px; min-width:140px; color:#222;
}}

.tl-btn {{
  appearance:none; border:none; padding:8px 16px;
  border-radius:24px; font-weight:600; cursor:pointer;
  transition:transform .05s ease, box-shadow .15s ease;
  box-shadow:0 1px 2px var(--ring);
}}
.tl-btn:hover  {{transform:translateY(-1px); box-shadow:0 4px 6px var(--ring);}}
.tl-btn:active {{transform:translateY(0); box-shadow:0 1px 2px var(--ring) inset;}}

.go-one, .go-all   {{background:var(--go);   color:#fff;}}
.stop-one, .stop-all{{background:var(--stop); color:#fff;}}

.go-all {{ margin-left:auto; }}     /* ← pushes both “all” buttons right */

#tl-note {{font-size:12px; opacity:.8; margin-top:4px;}}
</style>

<div id="tl-card">
  <h3>Traffic-Light Control</h3>

  <div id="tl-toolbar">
    <!-- update model selection on change -->
    <select id="tl_select"
      onchange="fetch('/set_user_selected_traffic_light',{{
        method:'POST',
        headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{id:this.value}})
      }}).catch(console.error);">{options}</select>

    <!-- single light: Go -->
    <button class="tl-btn go-one"
      onclick="fetch('/set_traffic_light_go',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id: document.getElementById('tl_select').value}})}})
        .then(()=>{{ const ds=document.getElementById('step_button')||document.getElementById('step'); if(ds) ds.click(); }})
        .catch(console.error)">
      Go
    </button>

    <!-- single light: Stop -->
    <button class="tl-btn stop-one"
      onclick="fetch('/set_traffic_light_stop',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id: document.getElementById('tl_select').value}})}})
        .then(()=>{{ const ds=document.getElementById('step_button')||document.getElementById('step'); if(ds) ds.click(); }})
        .catch(console.error)">
      Stop
    </button>

    <!-- all lights: All GO -->
    <button class="tl-btn go-all"
      onclick="fetch('/set_traffic_lights_go',{{method:'POST'}})
        .then(()=>{{ const ds=document.getElementById('step_button')||document.getElementById('step'); if(ds) ds.click(); }})
        .catch(console.error)">
      All GO
    </button>

    <!-- all lights: All STOP -->
    <button class="tl-btn stop-all"
      onclick="fetch('/set_traffic_lights_stop',{{method:'POST'}})
        .then(()=>{{ const ds=document.getElementById('step_button')||document.getElementById('step'); if(ds) ds.click(); }})
        .catch(console.error)">
      All STOP
    </button>

    <!-- standalone Step -->
    <button class="tl-btn" style="background:#fff;color:var(--purple);" onclick="{{ const ds=document.getElementById('step_button')||document.getElementById('step'); if(ds) ds.click(); }}">
      Step
    </button>
  </div>

  <p id="tl-note">(take one simulation step to see the effect)</p>
</div>
"""

# ─── Handlers & Routes ──────────────────────────────────────────────────────
class _Base(RequestHandler):
    def initialize(self, server): self.server = server
    def _json(self):
        try: return tornado.escape.json_decode(self.request.body) or {}
        except Exception: return {}
    def _ok(self, msg="OK"): self.write(msg)
    def _err(self, status, msg): self.set_status(status); self.write(msg)

# handler to persist user selection in the model
class SetUserSelectionHandler(_Base):
    def post(self):
        try:
            uid = self._json().get("id")
            # save selection on the model
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

# Route attachment

def add_traffic_light_routes(server):
    print("️🚦 Adding Traffic Control to server…")
    server.add_handlers(
        r".*",
        [
            (r"/set_user_selected_traffic_light", SetUserSelectionHandler, dict(server=server)),
            (r"/set_traffic_light_go",    SetSingleGoHandler,        dict(server=server)),
            (r"/set_traffic_light_stop",  SetSingleStopHandler,      dict(server=server)),
            (r"/set_traffic_lights_go",   SetGoHandler,              dict(server=server)),
            (r"/set_traffic_lights_stop", SetStopHandler,            dict(server=server)),
        ],
    )