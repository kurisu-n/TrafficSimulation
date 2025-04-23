# traffic_light_control.py
from mesa.visualization.modules import TextElement
from tornado.web import RequestHandler
import traceback, tornado.escape

class TrafficLightControl(TextElement):
    """Purple card + title + colour-coded buttons + step reminder."""
    def render(self, model):
        options = "\n".join(
            f'<option value="{tl.unique_id}">{tl.unique_id}</option>'
            for tl in model.get_traffic_lights()
        )

        return f'''
<style>
:root {{
  --purple:#6d28d9;
  --go:#34c759; --stop:#ff453a; --ring:rgba(0,0,0,.08);
}}
#tl-card {{
  font:14px/1.35 system-ui,sans-serif;
  background:var(--purple); color:#fff;
  padding:16px 18px 8px; margin:12px; border-radius:12px;
  box-shadow:0 2px 5px var(--ring); display:grid; gap:14px;
}}
#tl-card h3 {{ margin:0; font-size:17px; font-weight:600; }}
#tl-toolbar {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(110px,max-content));
  gap:10px; align-items:center;
}}
#tl-select {{
  padding:6px 10px; border:1px solid var(--ring);
  border-radius:8px; min-width:140px; color:#222;
}}
.tl-btn {{
  appearance:none; border:none; padding:8px 16px; border-radius:24px;
  font-weight:600; cursor:pointer;
  transition:transform .05s ease, box-shadow .15s ease;
  box-shadow:0 1px 2px var(--ring);
}}
.tl-btn:hover  {{ transform:translateY(-1px); box-shadow:0 4px 6px var(--ring); }}
.tl-btn:active {{ transform:translateY(0);    box-shadow:0 1px 2px var(--ring) inset; }}
.go-one,.go-all   {{ background:var(--go);   color:#fff; }}
.stop-one,.stop-all{{ background:var(--stop); color:#fff; }}

/* small footnote style */
#tl-note {{ font-size:12px; opacity:.8; margin-top:4px 0 0 0; }}
</style>

<div id="tl-card">
  <h3>Traffic-Light&nbsp;Control</h3>

  <div id="tl-toolbar">
    <select id="tl-select">{options}</select>

    <button class="tl-btn go-one"   onclick="tlSetOne('go')">Go</button>
    <button class="tl-btn stop-one" onclick="tlSetOne('stop')">Stop</button>

    <button class="tl-btn go-all"
            onclick="fetch('/set_traffic_lights_go', {{method:'POST'}})">All&nbsp;GO</button>
    <button class="tl-btn stop-all"
            onclick="fetch('/set_traffic_lights_stop', {{method:'POST'}})">All&nbsp;STOP</button>
  </div>

  <!-- new footnote ------------------------------------------------------ -->
  <p id="tl-note">(simulation must take one step for these changes to apply)</p>
</div>

<script>
function tlSetOne(mode){{
  const id=document.getElementById('tl-select').value;
  fetch(mode==='go'?'/set_traffic_light_go':'/set_traffic_light_stop',{{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{id}})
  }}).catch(console.error);
}}
</script>
'''

# ---------------------------------------------------------------------------
#               (handler classes below)
# ---------------------------------------------------------------------------
class SetSingleGoHandler(RequestHandler):
    def initialize(self, server): self.server = server
    def post(self):
        try:
            uid = tornado.escape.json_decode(self.request.body)["id"]
            next(t for t in self.server.model.get_traffic_lights()
                 if t.unique_id == uid).set_light_go()
            self.write(f"GO OK: {uid}")
        except StopIteration:
            self.set_status(404); self.write(f"ERROR: No traffic light {uid}")
        except Exception:
            traceback.print_exc(); self.set_status(500)
            self.write("ERROR: "+ traceback.format_exc())

class SetSingleStopHandler(RequestHandler):
    def initialize(self, server): self.server = server
    def post(self):
        try:
            uid = tornado.escape.json_decode(self.request.body)["id"]
            next(t for t in self.server.model.get_traffic_lights()
                 if t.unique_id == uid).set_light_stop()
            self.write(f"STOP OK: {uid}")
        except StopIteration:
            self.set_status(404); self.write(f"ERROR: No traffic light {uid}")
        except Exception:
            traceback.print_exc(); self.set_status(500)
            self.write("ERROR: "+ traceback.format_exc())

class SetGoHandler(RequestHandler):
    def initialize(self, server): self.server = server
    def post(self):
        try: self.server.model.set_traffic_lights_go();  self.write("GO OK")
        except Exception:
            traceback.print_exc(); self.set_status(500)
            self.write("ERROR: "+ traceback.format_exc())

class SetStopHandler(RequestHandler):
    def initialize(self, server): self.server = server
    def post(self):
        try:
            self.server.model.set_traffic_lights_stop(); self.write("STOP OK")
        except Exception:
            traceback.print_exc(); self.set_status(500)
            self.write("ERROR: "+ traceback.format_exc())
