# Simulation/visualization/rain_control.py
"""
A simple control UI for manually spawning RainAgent instances via RainManager,
respecting cooldown and max occurrences. Similar to TrafficLightControl.
"""
from mesa.visualization.modules import TextElement
from tornado.web import RequestHandler
from Simulation.config import Defaults
from Simulation.agents.rain import RainManager

# inject a small script to trigger the Mesa step button
_STEP_JS = """
<script type=\"text/javascript\">  
function rainStep(){  
  (document.getElementById('step_button')||
   document.getElementById('step')).click();
}
</script>
"""

class RainControl(TextElement):
    """UI card: one button to spawn rain when allowed."""
    def render(self, model):
        # locate and cache RainManager
        if not hasattr(model, 'rain_manager'):
            for a in model.schedule.agents:
                if isinstance(a, RainManager):
                    model.rain_manager = a
                    break

        rm = getattr(model, 'rain_manager', None)
        cooling = rm.cooldown if rm else 0
        active = len(model.rains)
        # disable button if on cooldown or at max
        disabled = 'disabled' if (cooling > 0 or active >= Defaults.RAIN_OCCURRENCES_MAX) else ''
        label = ('Spawn Rain' if not disabled
                 else f'Cooldown: {cooling} step' + ('s' if cooling != 1 else ''))

        # return script + styled button
        return f"""
        {_STEP_JS}
        <style>
        #rain-card {{ font:14px system-ui; padding:8px; margin:4px; background:#0ea5e9; color:#fff; border-radius:8px; }}
        #rain-card button {{ padding:8px 16px; border:none; border-radius:6px; font-weight:600; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.2); }}
        #rain-card button:disabled {{ opacity:0.5; cursor:default; }}
        </style>
        <div id="rain-card">
          <button id="rain_btn" {disabled}
            onclick="fetch('/spawn_rain', {{method:'POST'}})
                     .then(()=>{{rainStep();}});">{label}</button>
        </div>
        """

class SpawnRainHandler(RequestHandler):
    """Tornado handler: delegates spawning to RainManager."""
    def initialize(self, server):
        self.server = server

    def post(self):
        model = self.server.model
        # ensure manager reference
        rm = getattr(model, 'rain_manager', None)
        if rm is None:
            rm = next(a for a in model.schedule.agents if isinstance(a, RainManager))
            model.rain_manager = rm
        # enforce rules
        if rm.cooldown > 0 or len(model.rains) >= Defaults.RAIN_OCCURRENCES_MAX:
            self.set_status(400)
            self.write('Cannot spawn rain: cooldown or max reached')
            return
        # spawn via manager
        rm.add_random_rain()
        self.write('OK')

# Route registration (call once in your server setup)
def add_rain_routes(server):
    print("🌧️ Adding Rain Control endpoints …")
    server.add_handlers(r".*", [
        (r"/spawn_rain", SpawnRainHandler, dict(server=server)),
    ])