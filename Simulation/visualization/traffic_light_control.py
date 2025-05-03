# traffic_light_control.py
# =============================================================
from __future__ import annotations
import tornado.escape
from mesa.visualization.modules import TextElement
from tornado.web import RequestHandler
from typing import List

# -----------------------------------------------------------------
#  Browser-side helper (little wrapper for the Mesa “Step” button)
# -----------------------------------------------------------------
_STEP_JS = """
function step(){ (document.getElementById('step_button')||
                  document.getElementById('step')).click(); }
"""

# =============================================================
#  ⚑  UI TEXT ELEMENT
# =============================================================
class TrafficLightControl(TextElement):
    """Nice little controller card shown above the grid."""
    def render(self, model):
        # =============== pull data from the model =================
        tls:  List = list(model.get_traffic_lights())
        ilgs: List = list(model.get_intersection_light_groups())

        # ---------- ensure we always have *a* selected TL ----------
        if not hasattr(model, "user_selected_traffic_light"):
            model.user_selected_traffic_light = None
        if model.user_selected_traffic_light is None and tls:
            model.user_selected_traffic_light = tls[0].id

        # ---------- ensure we always have *a* selected ILG ----------
        if not hasattr(model, "user_selected_intersection"):
            model.user_selected_intersection = None
        if model.user_selected_intersection is None and ilgs:
            model.user_selected_intersection = ilgs[0].id

        # ---------------- dropdown 〈option〉 lists -----------------
        tl_opts = "\n".join(
            f'<option value="{tl.id}"'
            f'{" selected" if str(tl.id)==str(model.user_selected_traffic_light) else ""}>'
            f'{tl.id}</option>'
            for tl in tls
        )

        ilg_opts = "\n".join(
            f'<option value="{g.id}"'
            f'{" selected" if str(g.id)==str(model.user_selected_intersection) else ""}>'
            f'{g.id}</option>'
            for g in ilgs
        )

        # ----------------------------------------------------------- #
        #  Build the opposite-*lights* list (depends on TL pick)      #
        # ----------------------------------------------------------- #
        opp_opts = ""
        try:
            sel_tl = next(tl for tl in tls
                          if str(tl.id) == str(model.user_selected_traffic_light))
            parent_ilg = next(g for g in ilgs if sel_tl in g.traffic_lights)

            opp_dict = parent_ilg.get_opposite_traffic_lights()     # {"N-S":[…], "W-E":[…]}

            # keep only non-empty axes, preserve order N-S → W-E
            axes_available = [axis for axis in ("N-S", "W-E") if opp_dict.get(axis)]

            # pick a default axis if nothing chosen yet
            if getattr(model, "user_selected_opposite", None) not in axes_available:
                model.user_selected_opposite = axes_available[0] if axes_available else None

            opp_opts = "\n".join(
                f'<option value="{axis}"'
                f'{" selected" if axis == model.user_selected_opposite else ""}>{axis}</option>'
                for axis in axes_available
            )
        except StopIteration:
            pass

        # =========================================================
        #  HTML / CSS
        # =========================================================
        return f"""
        <style>
        :root {{
          --purple:#6d28d9; --go:#34c759; --stop:#ff453a; --ring:rgba(0,0,0,.08);
        }}
        #tl-card {{
          font:14px/1.35 system-ui,sans-serif; background:var(--purple); color:#fff;
          padding:10px 10px 10px 10px; margin: 2px; border-radius:10px;
          box-shadow:0 2px 5px var(--ring);
          display:grid; row-gap:10px;
          width: 100%;
          height: 100%;
          box-sizing: border-box;
        }}
        #row-1, #row-2 {{
          display:flex; flex-wrap:wrap; gap:10px; align-items:center;
          justify-content:space-between;
        }}
        
        #col-1 {{ display:flex; flex-direction:column; align-items:center; height:100%; gap:10px; }}

        
        .tl-group {{ display:flex; gap:5px; align-items:center; justify-content:space-between; }}

        .tl-label {{ font-weight:600; margin-right:4px; white-space:nowrap; }}

        select.tlc-select {{
          padding:4px 4px; border:1px solid var(--ring);
          border-radius:8px; min-width:80px; color:#222;
        }}

        .tl-btn {{
          appearance:none; border:none; padding:6px 10px; border-radius:10px;
          font-weight:600; cursor:pointer;
          transition:transform .05s ease, box-shadow .15s ease;
          box-shadow:0 1px 2px var(--ring);
        }}
        .tl-btn:hover  {{ transform:translateY(-1px); box-shadow:0 4px 6px var(--ring); }}
        .tl-btn:active {{ transform:translateY(0);    box-shadow:0 1px 2px var(--ring) inset; }}

        .go-one, .go-ilg, .go-opp, .go-all, .go-neighbors {{ background:var(--go);   color:#fff; }}
        .stop-one,.stop-ilg,.stop-opp,.stop-all,.stop-neighbors{{ background:var(--stop); color:#fff; }}
        </style>

        <div id="tl-card">

          <!-- ────────── ROW 1 ────────── -->
          <div id="row-1">
        
            <!-- ◇ single traffic-light -->
            <div id="col-1">
                <span class="tl-label">Individual Light</span>
                <div class="tl-group">
                  <select id="tl_select" class="tlc-select"
                    onchange="fetch('/set_user_selected_traffic_light',{{method:'POST',
                       headers:{{'Content-Type':'application/json'}},
                       body:JSON.stringify({{id:this.value}})}})
                       .then(()=>{{(document.getElementById('step_button')||
                                    document.getElementById('step')).click();}});">
                    {tl_opts}
                  </select>
                  <button class="tl-btn go-one"
                    onclick="fetch('/set_traffic_light_go',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">Go</button>
                  <button class="tl-btn stop-one"
                    onclick="fetch('/set_traffic_light_stop',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">STOP</button>
                </div>
            </div>    

            <!-- ◇ global -->
            <div id="col-1">
                <span class="tl-label">All Lights</span>
                <div class="tl-group">
                  <button class="tl-btn go-all"
                    onclick="fetch('/set_traffic_lights_go',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">GO</button>
                  <button class="tl-btn stop-all"
                    onclick="fetch('/set_traffic_lights_stop',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">STOP</button>
                </div>
            </div>
          </div>

          <!-- ────────── ROW 2 ────────── -->
          <div id="row-2">

            <!-- ◇ intersection group -->
            <div id="col-1">
                <span class="tl-label">Intersection Group</span>
                
                <div class="tl-group">
                  <select id="ilg_select" class="tlc-select"
                    onchange="fetch('/set_user_selected_intersection',{{method:'POST',
                       headers:{{'Content-Type':'application/json'}},
                       body:JSON.stringify({{id:this.value}})}});">
                    {ilg_opts}
                  </select>
                  <button class="tl-btn go-ilg"
                    onclick="fetch('/set_ilg_go',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">GO</button>
                  <button class="tl-btn stop-ilg"
                    onclick="fetch('/set_ilg_stop',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">STOP</button>
                </div>                         
            </div>

            <!-- ◇ opposite group -->
            <div id="col-1">
                <span class="tl-label">Opposite Pairs</span>
                <div class="tl-group">
                  <select id="opp_select" class="tlc-select"
                    onchange="fetch('/set_user_selected_opposite',{{method:'POST',
                       headers:{{'Content-Type':'application/json'}},
                       body:JSON.stringify({{id:this.value}})}});">
                    {opp_opts}
                  </select>
                  <button class="tl-btn go-opp"
                    onclick="fetch('/set_opp_go',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">Go</button>
                  <button class="tl-btn stop-opp"
                    onclick="fetch('/set_opp_stop',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">Stop</button>
                </div>
            </div>
                                      
            <!-- ◇ ILG + next_groups -->
            <div id="col-1">
                <span class="tl-label">Intersection and Neighbors</span>
                <div class="tl-group">
                  <button class="tl-btn go-neighbors"
                    onclick="fetch('/set_ilg_neighbors_go',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">GO</button>
                  <button class="tl-btn stop-neighbors"
                    onclick="fetch('/set_ilg_neighbors_stop',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">STOP</button>
                </div>
            </div>
            
            <!-- ◇ ILG + next_groups + intermediate -->
            <div id="col-1">
                <span class="tl-label">Inter., Neighbors and Betweens</span>
                <div class="tl-group">
                  <button class="tl-btn go-neighbors"
                    onclick="fetch('/set_group_neighbors_intermediate_go',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">GO</button>
                  <button class="tl-btn stop-neighbors"
                    onclick="fetch('/set_group_neighbors_intermediate_stop',{{method:'POST'}})
                             .then(()=>{{(document.getElementById('step_button')||
                                          document.getElementById('step')).click();}});">STOP</button>
                </div>
            </div>
          </div>
        </div>
        """


# =============================================================
#  ⚑  TORNADO HELPERS & HANDLERS
# =============================================================
class _Base(RequestHandler):
    """Base Tornado handler offering tiny JSON helpers."""
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

# --------------- single light ---------------------------------
class SetUserSelectionHandler(_Base):
    def post(self):
        self.server.model.user_selected_traffic_light = self._json().get("id")
        self._ok()

class SetSingleGoHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_traffic_light
        try:
            next(tl for tl in self.server.model.get_traffic_lights()
                 if str(tl.id) == str(uid)).set_light_go()
            self._ok()
        except StopIteration:
            self._err(404, "No TL")

class SetSingleStopHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_traffic_light
        try:
            next(tl for tl in self.server.model.get_traffic_lights()
                 if str(tl.id) == str(uid)).set_light_stop()
            self._ok()
        except StopIteration:
            self._err(404, "No TL")

class SetGoHandler(_Base):
    def post(self):
        self.server.model.set_traffic_lights_go();   self._ok()

class SetStopHandler(_Base):
    def post(self):
        self.server.model.set_traffic_lights_stop(); self._ok()

# --------------- ILG (intersection) ---------------------------
class SetUserIntersectionSelectionHandler(_Base):
    def post(self):
        self.server.model.user_selected_intersection = self._json().get("id")
        self._ok()

class SetIlgGoHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_intersection
        try:
            next(g for g in self.server.model.get_intersection_light_groups()
                 if str(g.id)==str(uid)).set_all_go()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")

class SetIlgStopHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_intersection
        try:
            next(g for g in self.server.model.get_intersection_light_groups()
                 if str(g.id)==str(uid)).set_all_stop()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")

# --------------- opposite ILG ---------------------------------
# --------------- opposite traffic-LIGHT ---------------------
class SetUserOppositeSelectionHandler(_Base):
    def post(self):
        self.server.model.user_selected_opposite = self._json().get("id")
        self._ok()

class SetOppGoHandler(_Base):
    def post(self):
        axis = self.server.model.user_selected_opposite          # "N-S" or "W-E"
        ilg_id = self.server.model.user_selected_intersection    # current ILG
        try:
            ilg = next(g for g in self.server.model.get_intersection_light_groups()
                       if str(g.id) == str(ilg_id))
            for tl in ilg.get_opposite_traffic_lights().get(axis, []):
                tl.set_light_go()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")

class SetOppStopHandler(_Base):
    def post(self):
        axis = self.server.model.user_selected_opposite
        ilg_id = self.server.model.user_selected_intersection
        try:
            ilg = next(g for g in self.server.model.get_intersection_light_groups()
                       if str(g.id) == str(ilg_id))
            for tl in ilg.get_opposite_traffic_lights().get(axis, []):
                tl.set_light_stop()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")


# --------------- ILG + next_groups ----------------------------
class SetIlgNeighborsGoHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_intersection
        try:
            next(g for g in self.server.model.get_intersection_light_groups()
                 if str(g.id)==str(uid)).set_all_go_with_neighbors()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")

class SetIlgNeighborsStopHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_intersection
        try:
            next(g for g in self.server.model.get_intersection_light_groups()
                 if str(g.id)==str(uid)).set_all_stop_with_neighbors()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")


class SetGroupNeighborsIntermediateGoHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_intersection
        try:
            next(g for g in self.server.model.get_intersection_light_groups()
                 if str(g.id)==str(uid)).set_all_go_with_neighbors_and_intermediate()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")

class SetGroupNeighborsIntermediateStopHandler(_Base):
    def post(self):
        uid = self.server.model.user_selected_intersection
        try:
            next(g for g in self.server.model.get_intersection_light_groups()
                 if str(g.id)==str(uid)).set_all_stop_with_neighbors_and_intermediate()
            self._ok()
        except StopIteration:
            self._err(404, "No ILG")

# =============================================================
#  ⚑  ROUTE REGISTRATION
# =============================================================
def add_traffic_light_routes(server):
    """Call once from your Mesa server to attach every route."""
    print("🚦  Adding Traffic-Light/ILG control endpoints …")
    server.add_handlers(
        r".*",
        [
            # single TL
            (r"/set_user_selected_traffic_light", SetUserSelectionHandler, dict(server=server)),
            (r"/set_traffic_light_go",            SetSingleGoHandler,      dict(server=server)),
            (r"/set_traffic_light_stop",          SetSingleStopHandler,    dict(server=server)),
            (r"/set_traffic_lights_go",           SetGoHandler,            dict(server=server)),
            (r"/set_traffic_lights_stop",         SetStopHandler,          dict(server=server)),

            # ILG
            (r"/set_user_selected_intersection",  SetUserIntersectionSelectionHandler, dict(server=server)),
            (r"/set_ilg_go",                      SetIlgGoHandler,         dict(server=server)),
            (r"/set_ilg_stop",                    SetIlgStopHandler,       dict(server=server)),

            # opposite ILG
            (r"/set_user_selected_opposite",      SetUserOppositeSelectionHandler, dict(server=server)),
            (r"/set_opp_go",                      SetOppGoHandler,         dict(server=server)),
            (r"/set_opp_stop",                    SetOppStopHandler,       dict(server=server)),

            # ILG + next
            (r"/set_ilg_neighbors_go",                 SetIlgNeighborsGoHandler, dict(server=server)),
            (r"/set_ilg_neighbors_stop",               SetIlgNeighborsStopHandler, dict(server=server)),

            # IGL + next + intermediate
            (r"/set_group_neighbors_intermediate_go",  SetGroupNeighborsIntermediateGoHandler, dict(server=server)),
            (r"/set_group_neighbors_intermediate_stop",SetGroupNeighborsIntermediateStopHandler, dict(server=server)),
        ],
    )
