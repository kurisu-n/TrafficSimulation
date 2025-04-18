#server.py
import sys, os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import socket
from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from creation import StructuredCityModel, GRID_WIDTH, GRID_HEIGHT
from cell import agent_portrayal

def get_free_port(default=9000, max_tries=100):
    for offset in range(max_tries):
        port = default + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available ports found!")


canvas = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, 900, 900)

elements = [
    canvas
]

server = ModularServer(
    StructuredCityModel,
    elements,
    "Structured Urban Grid World",
    {"width": GRID_WIDTH, "height": GRID_HEIGHT},
)


server.port = get_free_port()
server.launch()
