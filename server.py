#server.py

import socket
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from creation import StructuredCityModel, GRID_WIDTH, GRID_HEIGHT
from cell import agent_portrayal
from cellinspector import CellInspector

def get_free_port(default=8521, max_tries=100):
    for offset in range(max_tries):
        port = default + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available ports found!")


# This will be called whenever you click on the grid.
def cell_click_handler(model, x, y):
    # Store the last‐clicked position on the model
    model.last_click = (x, y)

# Build the grid with our click handler
canvas = CanvasGrid(agent_portrayal,
                  GRID_WIDTH, GRID_HEIGHT,
                  900, 900)

server = ModularServer(
    StructuredCityModel,
    [canvas],        # ← INFO PANE **first**, canvas second
    "Structured Urban Grid World",
    {"width": GRID_WIDTH, "height": GRID_HEIGHT}
)

server.port = get_free_port()
print(f"Server on http://localhost:{server.port}")
server.launch()
