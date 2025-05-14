# vehicle_control.py
# =============================================================
from __future__ import annotations
import tornado.escape
from mesa.visualization.modules import TextElement
from Simulation.visualization.ui_modules.traffic_light_control import _Base

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

        # --- gather possible service types & entrances ---
        sv_types = ["Food", "Waste"]
        entrances = list(model.get_highway_entrances())
        if not hasattr(model, 'user_selected_sv_type'):
            model.user_selected_sv_type = sv_types[0]
        if not hasattr(model, 'user_selected_sv_entrance'):
            model.user_selected_sv_entrance = entrances[0].id if entrances else None

        # --- options HTML ---
        start_opts = "\n".join(
            f'<option value=\"{s.id}\"' +
            (' selected' if s.id==model.user_selected_start else '') +
            f'>{s.get_display_name()}</option>' for s in starts
        )
        target_opts = "\n".join(
            f'<option value=\"{t.id}\"' +
            (' selected' if t.id==model.user_selected_target else '') +
            f'>{t.get_display_name()}</option>' for t in targets
        )

        sv_type_opts = "\n".join(
            f'<option value="{t}"' +
            (' selected' if t==model.user_selected_sv_type else '') +
            f'>{t}</option>' for t in sv_types
        )
        sv_entrance_opts = "\n".join(
            f'<option value="{e.id}"' +
            (' selected' if str(e.id)==str(model.user_selected_sv_entrance) else '') +
            f'>{e.get_display_name()}</option>' for e in entrances
        )

        # --- render HTML with client-side persistence ---
        return f"""
        <style>
        #vc-card {{
          font:14px/1.4 system-ui,sans-serif; background:#444; color:#fff;
          padding:10px; margin:2px; border-radius:8px;
          display:grid; row-gap:10px;
          align-items:center; gap:10px;
        }}
        .vc-label {{ font-weight:600; white-space:nowrap; }}
        .vc-select {{ padding:4px; border-radius:6px; border:1px solid #666; color:#222; }}
        .vc-btn {{ padding:6px 12px; border:none; border-radius:6px;
                   background:#4CAF50; color:#fff; cursor:pointer; }}
        .vc-btn:hover {{ background:#45A049; }}
        
        #row-1, #row-2 {{
          display:flex; flex-wrap:wrap; gap:10px; align-items:center;
          justify-content:space-between;
        }}
        </style>
        <div id="vc-card">
            <div id="row-1">
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
            <div id="row-2">
              <span class="vc-label">New Service Vehicle:</span>
                <select id="sv_type_select" class="vc-select"
                  onchange="localStorage.setItem('manual_selected_sv_type', this.value);
                             fetch('/set_user_selected_sv_type',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{type:this.value}})}})
                           .then(()=>{{(document.getElementById('step_button')||document.getElementById('step')).click();}});">
                  {sv_type_opts}
                </select>
                <select id="sv_entrance_select" class="vc-select"
                  onchange="localStorage.setItem('manual_selected_sv_entrance', this.value);
                             fetch('/set_user_selected_sv_entrance',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id:this.value}})}})
                           .then(()=>{{(document.getElementById('step_button')||document.getElementById('step')).click();}});">
                  {sv_entrance_opts}
                </select>
              <button class="vc-btn vc-btn-sv start-service"
                onclick="fetch('/create_service_vehicle',{{method:'POST'}})
                         .then(()=>{{(document.getElementById('step_button')||document.getElementById('step')).click();}});">
                Start</button>
            </div>
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
            targetSelect.dispatchEvent(new Event('change'));
          }}
          
          // restore service vehicle selections
          const svType = localStorage.getItem('manual_selected_sv_type');
          const svTypeSelect = document.getElementById('sv_type_select');
          if(svType && Array.from(svTypeSelect.options).some(o=>o.value===svType)){{
            svTypeSelect.value = svType;
            svTypeSelect.dispatchEvent(new Event('change'));
          }}
          const svEnt = localStorage.getItem('manual_selected_sv_entrance');
          const svEntSelect = document.getElementById('sv_entrance_select');
          if(svEnt && Array.from(svEntSelect.options).some(o=>o.value===svEnt)){{
            svEntSelect.value = svEnt;
            svEntSelect.dispatchEvent(new Event('change'));
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

class SetUserServiceTypeHandler(_Base):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        self.server.model.user_selected_sv_type = data.get('type')
        self._ok()

class SetUserServiceEntranceHandler(_Base):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        self.server.model.user_selected_sv_entrance = data.get('id')
        self._ok()

class CreateServiceVehicleHandler(_Base):
    """Handler that spawns a new ServiceVehicleAgent"""
    def post(self):
        model = self.server.model
        sv_type = getattr(model, 'user_selected_sv_type', 'Food')
        ent_id  = getattr(model, 'user_selected_sv_entrance', None)
        try:
            entrance = next(e for e in model.get_highway_entrances()
                            if str(e.id)==str(ent_id))
        except StopIteration:
            self._err(404, "Invalid service entrance")
            return

        if getattr(entrance, "occupied", False):
            self._err(409, "Selected service entrance is currently occupied")
            return

        if not hasattr(model, 'service_vehicle_count'):
            model.service_vehicle_count = 0
        model.service_vehicle_count += 1
        vid = f"SV{model.service_vehicle_count}"
        from Simulation.agents.vehicles.vehicle_service import ServiceVehicleAgent
        ServiceVehicleAgent(vid, model, entrance, sv_type)
        self._ok()

class CreateVehicleHandler(_Base):
    """HTTP handler that spawns a new VehicleAgent
    ------------------------------------------------
    Adds a safety guard: if the user‑selected *start* Cell is already
    occupied by another vehicle, the request is rejected with **409 Conflict**
    instead of quietly stacking two cars in the same spot.
    """

    def post(self):
        model = self.server.model

        # 1️⃣ Resolve the chosen start & target cells ------------------------
        try:
            start = next(
                s for s in model.get_start_blocks()
                if str(s.id) == str(model.user_selected_start)
            )
            target = next(
                t for t in model.get_valid_exits(start)
                if str(t.id) == str(model.user_selected_target)
            )
        except StopIteration:
            self._err(404, "Invalid start or target")
            return

        # 2️⃣ NEW: Abort if start cell is already occupied -------------------
        # The CellAgent interface exposes an ``occupied`` boolean that is
        # set to *True* whenever a vehicle is currently present in that cell.
        if getattr(start, "occupied", False):
            # 409 ‑ Conflict: the request cannot be completed because it
            # would result in a resource conflict on the server.
            self._err(409, "Selected start block is currently occupied")
            return

        # 3️⃣ Create a new vehicle ------------------------------------------
        if not hasattr(model, "vehicle_count"):
            model.vehicle_count = 0
        model.vehicle_count += 1
        vid = f"V{model.vehicle_count}"

        from Simulation.agents.vehicles.vehicle_base import VehicleAgent
        VehicleAgent(vid, model, start, target)

        # 4️⃣ Respond OK ------------------------------------------------------
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
            (r"/set_user_selected_start",       SetUserStartSelectionHandler,    dict(server=server)),
            (r"/set_user_selected_target",      SetUserTargetSelectionHandler,   dict(server=server)),
            (r"/create_vehicle",                CreateVehicleHandler,            dict(server=server)),
            (r"/set_user_selected_sv_type",     SetUserServiceTypeHandler,       dict(server=server)),
            (r"/set_user_selected_sv_entrance", SetUserServiceEntranceHandler,   dict(server=server)),
            (r"/create_service_vehicle",        CreateServiceVehicleHandler,     dict(server=server)),
        ],
    )
