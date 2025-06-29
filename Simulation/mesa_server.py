#mesa_server.py
import tornado.autoreload
tornado.autoreload.start = lambda *args, **kwargs: None

import socket
from mesa.visualization.modules import CanvasGrid
from mesa_viz_tornado.ModularVisualization import VisualizationElement

from Simulation.config import Defaults
from Simulation.visualization.dynamic_grid_server import DynamicGridServer
from Simulation.visualization.ui_modules.traffic_light_control import TrafficLightControl, add_traffic_light_routes
from Simulation.visualization.ui_modules.traffic_statistics import TrafficStatistics
from Simulation.visualization.ui_modules.vehicle_control import ManualVehicleControl, add_manual_vehicle_routes
from Simulation.visualization.ui_modules.rain_control import RainControl, add_rain_routes
from Simulation.visualization.model_parameters import model_params
from Simulation.city_model import CityModel
from Simulation.visualization.agent_portrayal import agent_portrayal

def get_free_port(default=9250, max_tries=100):
    for offset in range(max_tries):
        port = default + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available ports found!")

# Build the grid with our click handler
canvas = CanvasGrid(
    portrayal_method = agent_portrayal,
    grid_width       = Defaults.WIDTH,
    grid_height      = Defaults.HEIGHT,
    canvas_height    = 1000,
    canvas_width     = 1000)

visualization_elements: list[VisualizationElement] = []


if Defaults.ENABLE_TRAFFIC:
    visualization_elements.append(TrafficStatistics())

if Defaults.ENABLE_AGENT_PORTRAYAL:
    visualization_elements.extend([ canvas,
                                    TrafficLightControl(),
                                    ManualVehicleControl(),
                                  ])

    if Defaults.RAIN_ENABLED:
        visualization_elements.append(RainControl())

server = DynamicGridServer(
    model_cls = CityModel,
    visualization_elements = visualization_elements,
    name = "Structured Urban Grid World",
    model_params = model_params,
)

add_traffic_light_routes(server)
add_manual_vehicle_routes(server)
add_rain_routes(server)

print("  → Elements:", [type(el).__name__ for el in server.visualization_elements])
print("  → Handlers:", [h[0] for h in server.handlers])

server.render_schedule = 20
server.port = get_free_port()