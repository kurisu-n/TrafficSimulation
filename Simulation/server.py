#server.py

import socket
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from Simulation.visualization.cell_inspector import CellInspectorJS, add_cell_inspector
from Simulation.visualization.model_parameters import model_params
from Simulation.visualization.traffic_light_control import add_traffic_light_routes, TrafficLightControl
from Simulation.city_model import CityModel
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
canvas = CanvasGrid(
    portrayal_method = agent_portrayal,
    grid_width       = model_params["width"].value,
    grid_height      = model_params["height"].value,
    canvas_height    = 1000,
    canvas_width     = 1000)

server = ModularServer(
    model_cls = CityModel,
    visualization_elements = [canvas, TrafficLightControl(), CellInspectorJS(model_params["width"].value, model_params["height"].value)],
    name = "Structured Urban Grid World",
    model_params = model_params,
)

add_traffic_light_routes(server)
add_cell_inspector(server)

print("  → Elements:", [type(el).__name__ for el in server.visualization_elements])
print("  → Handlers:", [h[0] for h in server.handlers])

server.port = get_free_port()