#server.py

import socket
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from Simulation.visualization.model_parameters import model_params_mesa
from Simulation.visualization.traffic_light_control import SetGoHandler, SetStopHandler, SetSingleGoHandler, SetSingleStopHandler, TrafficLightControl
from Simulation.city_model import CityModel, GRID_WIDTH, GRID_HEIGHT
from Simulation.agents.cell import agent_portrayal

def get_free_port(default=8000, max_tries=100):
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
canvas = CanvasGrid(agent_portrayal,
                    model_params_mesa["width"].value, model_params_mesa["height"].value,
                    900, 900)

server = ModularServer(
    model_cls = CityModel,
    visualization_elements = [canvas, TrafficLightControl()],
    name = "Structured Urban Grid World",
    model_params = model_params_mesa
)

server.add_handlers(r".*", [
    (r"/set_traffic_lights_go",  SetGoHandler,  dict(server=server)),
    (r"/set_traffic_lights_stop", SetStopHandler, dict(server=server)),
    (r"/set_traffic_light_go", SetSingleGoHandler, dict(server=server)),
    (r"/set_traffic_light_stop", SetSingleStopHandler, dict(server=server)),
])

server.port = get_free_port()