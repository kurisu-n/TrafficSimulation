# server.py
import socket
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from creation import StructuredCityModel, GRID_WIDTH, GRID_HEIGHT
from cell import agent_portrayal

# Helper function to find a free port
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

canvas_element = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, 900, 900)

model_params = {
    "width": GRID_WIDTH,
    "height": GRID_HEIGHT,
}

server = ModularServer(
    StructuredCityModel,
    [canvas_element],
    "Structured Urban Grid World",
    model_params
)

# Automatically assign a free port
server.port = get_free_port()

# Note: Mesa's server does not currently support a built-in shutdown button.
# To end the process gracefully, instruct users to stop the script manually (Ctrl+C)
# or close the browser tab if run interactively.

print(f"\nServer running on http://localhost:{server.port}\nPress Ctrl+C in the terminal to stop it.\n")
server.launch()
