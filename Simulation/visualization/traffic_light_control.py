from mesa.visualization.modules import TextElement
from tornado.web import RequestHandler
import traceback
import tornado.escape

class TrafficLightControl(TextElement):
    """
    Renders:
      - A dropdown of all traffic lights
      - “Go” / “Stop” buttons for the selected light
      - (Keeps your existing “all lights” buttons)
    """
    def render(self, model):
        # Build <option> list from current traffic‐light IDs
        options = "\n".join(
            f'<option value="{tl.unique_id}">{tl.unique_id}</option>'
            for tl in model.get_traffic_lights()
        )

        return f'''
        <div style="margin:10px; display:flex; gap:10px; align-items:center;">
            <!-- Dropdown to pick one light -->
            <select id="tl_select" style="padding:8px; font-size:14px; border-radius:4px;">
                {options}
            </select>

            <!-- Set only selected light to GO -->
            <button
                style="padding:8px 16px; font-size:14px; border-radius:4px; background:lightgreen;"
                onclick="
                  const id = document.getElementById('tl_select').value;
                  fetch('/set_traffic_light_go', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ id }})
                  }})
                  .then(resp => resp.text()).then(console.log).catch(console.error);
                ">
                Go
            </button>

            <!-- Set only selected light to STOP -->
            <button
                style="padding:8px 16px; font-size:14px; border-radius:4px; background:salmon;"
                onclick="
                  const id = document.getElementById('tl_select').value;
                  fetch('/set_traffic_light_stop', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ id }})
                  }})
                  .then(resp => resp.text()).then(console.log).catch(console.error);
                ">
                Stop
            </button>

            <!-- existing: all‐lights controls -->
            <button
                style="padding:8px 16px; font-size:14px; border-radius:4px;"
                onclick="fetch('/set_traffic_lights_go', {{ method: 'POST' }})
                         .then(resp => resp.text()).then(console.log).catch(console.error);">
                Set All GO
            </button>

            <button
                style="padding:8px 16px; font-size:14px; border-radius:4px;"
                onclick="fetch('/set_traffic_lights_stop', {{ method: 'POST' }})
                         .then(resp => resp.text()).then(console.log).catch(console.error);">
                Set All STOP
            </button>
        </div>
        '''



# ——— New handlers for single‐light control —————————————————————

class SetSingleGoHandler(RequestHandler):
    def initialize(self, server):
        self.server = server

    def post(self):
        try:
            data = tornado.escape.json_decode(self.request.body)
            uid = data.get("id")
            # find the traffic light by its unique_id
            tl = next(t for t in self.server.model.get_traffic_lights()
                      if t.unique_id == uid)
            tl.set_light_go()
            self.write(f"GO OK: {uid}")
        except StopIteration:
            self.set_status(404)
            self.write(f"ERROR: No traffic light {uid}")
        except Exception:
            traceback.print_exc()
            self.set_status(500)
            self.write("ERROR: " + traceback.format_exc())


class SetSingleStopHandler(RequestHandler):
    def initialize(self, server):
        self.server = server

    def post(self):
        try:
            data = tornado.escape.json_decode(self.request.body)
            uid = data.get("id")
            tl = next(t for t in self.server.model.get_traffic_lights()
                      if t.unique_id == uid)
            tl.set_light_stop()
            self.write(f"STOP OK: {uid}")
        except StopIteration:
            self.set_status(404)
            self.write(f"ERROR: No traffic light {uid}")
        except Exception:
            traceback.print_exc()
            self.set_status(500)
            self.write("ERROR: " + traceback.format_exc())


# Handler for green light
class SetGoHandler(RequestHandler):
    def initialize(self, server):
        self.server = server

    def post(self):
        try:
            self.server.model.set_traffic_lights_go()
            self.write("GO OK")
        except Exception:
            # Print the full traceback to your terminal
            traceback.print_exc()
            self.set_status(500)
            self.write("ERROR: " + traceback.format_exc())

class SetStopHandler(RequestHandler):
    def initialize(self, server):
        self.server = server

    def post(self):
        try:
            self.server.model.set_traffic_lights_stop()
            self.write("STOP OK")
        except Exception:
            traceback.print_exc()
            self.set_status(500)
            self.write("ERROR: " + traceback.format_exc())