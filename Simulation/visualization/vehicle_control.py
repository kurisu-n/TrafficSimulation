# vehicle_control.py
# =============================================================
from __future__ import annotations
import tornado.escape
from mesa.visualization.modules import TextElement
from Simulation.visualization.traffic_light_control import _Base

# =============================================================
#  ⚑  UI TEXT ELEMENT
# =============================================================
class ManualVehicleControl(TextElement):
    """Control card for manually spawning vehicles."""
    def render(self, model):
        # --- gather possible starts ---
        starts = list(model.get_start_blocks())
        if not hasattr(model, 'user_selected_start'):
            model.user_selected_start = starts[0].id if starts else None
        # validate current selection
        start_ids = [s.id for s in starts]
        if model.user_selected_start not in start_ids and start_ids:
            model.user_selected_start = start_ids[0]

        # --- gather valid targets based on selected start ---
        sel_start = next((s for s in starts if s.id == model.user_selected_start), None)
        targets = model.get_valid_exits(sel_start) if sel_start else []
        if not hasattr(model, 'user_selected_target'):
            model.user_selected_target = targets[0].id if targets else None
        target_ids = [t.id for t in targets]
        if model.user_selected_target not in target_ids and target_ids:
            model.user_selected_target = target_ids[0]

        # --- options HTML ---
        start_opts = "\n".join(
            f'<option value=\"{s.id}\"' +
            (' selected' if s.id==model.user_selected_start else '') +
            f'>{s.id}</option>' for s in starts
        )
        target_opts = "\n".join(
            f'<option value=\"{t.id}\"' +
            (' selected' if t.id==model.user_selected_target else '') +
            f'>{t.id}</option>' for t in targets
        )

        # --- render HTML with client-side persistence ---
        return f"""
        <style>
        #vc-card {{
          font:14px/1.4 system-ui,sans-serif; background:#444; color:#fff;
          padding:10px; margin:2px; border-radius:8px;
          display:flex; align-items:center; gap:10px;
        }}
        .vc-label {{ font-weight:600; white-space:nowrap; }}
        .vc-select {{ padding:4px; border-radius:6px; border:1px solid #666; color:#222; }}
        .vc-btn {{ padding:6px 12px; border:none; border-radius:6px;
                   background:#4CAF50; color:#fff; cursor:pointer; }}
        .vc-btn:hover {{ background:#45A049; }}
        </style>
        <div id="vc-card">
          <span class="vc-label">Create New Base Vehicle:</span>
          <select id="start_select" class="vc-select"
            onchange="localStorage.setItem('manual_selected_start', this.value);
                       fetch('/set_user_selected_start',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id:this.value}})}})
                       .then(()=>{{(document.getElementById('step_button')||document.getElementById('step')).click();}});">
            {start_opts}
          </select>
          <select id="target_select" class="vc-select"
            onchange="localStorage.setItem('manual_selected_target', this.value);
                       fetch('/set_user_selected_target',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id:this.value}})}})
                       .then(()=>{{(document.getElementById('step_button')||document.getElementById('step')).click();}});">
            {target_opts}
          </select>
          <button class="vc-btn start-vehicle"
            onclick="fetch('/create_vehicle',{{method:'POST'}})
                     .then(()=>{{(document.getElementById('step_button')||document.getElementById('step')).click();}});">
            Start
          </button>
        </div>
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
          const startSelect = document.getElementById('start_select');
          const targetSelect = document.getElementById('target_select');
          const savedStart = localStorage.getItem('manual_selected_start');
          if(savedStart && Array.from(startSelect.options).some(o=>o.value===savedStart)) {{
            startSelect.value = savedStart;
            startSelect.dispatchEvent(new Event('change'));
          }}
          const savedTarget = localStorage.getItem('manual_selected_target');
          if(savedTarget && Array.from(targetSelect.options).some(o=>o.value===savedTarget)) {{
            targetSelect.value = savedTarget;
          }}
        }});
        </script>
        """

# =============================================================
#  ⚑  TORNADO HELPERS & HANDLERS
# =============================================================
class SetUserStartSelectionHandler(_Base):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        self.server.model.user_selected_start = data.get('id')
        self._ok()

class SetUserTargetSelectionHandler(_Base):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        self.server.model.user_selected_target = data.get('id')
        self._ok()

class CreateVehicleHandler(_Base):
    def post(self):
        model = self.server.model
        try:
            start = next(s for s in model.get_start_blocks()
                         if str(s.id) == str(model.user_selected_start))
            target = next(t for t in model.get_valid_exits(start)
                          if str(t.id) == str(model.user_selected_target))
        except StopIteration:
            self._err(404, 'Invalid start or target')
            return
        if not hasattr(model, 'vehicle_count'):
            model.vehicle_count = 0
        model.vehicle_count += 1
        vid = f"V{model.vehicle_count}"
        from Simulation.agents.vehicles.vehicle_base import VehicleAgent
        VehicleAgent(vid, model, start, target)
        self._ok()

# =============================================================
#  ⚑  ROUTE REGISTRATION
# =============================================================

def add_manual_vehicle_routes(server):
    """Attach manual vehicle control endpoints to the server."""
    print("🚗 Adding Manual Vehicle Control endpoints …")
    server.add_handlers(
        r".*",
        [
            (r"/set_user_selected_start",  SetUserStartSelectionHandler,  dict(server=server)),
            (r"/set_user_selected_target", SetUserTargetSelectionHandler, dict(server=server)),
            (r"/create_vehicle",           CreateVehicleHandler,         dict(server=server)),
        ],
    )
