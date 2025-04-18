# app.py

import solara
from mesa.visualization import make_space_component, SolaraViz
from cell import agent_portrayal

from creation import StructuredCityModel, GRID_WIDTH, GRID_HEIGHT
from cell import ZONE_COLORS, desaturate

# Optional: improve aesthetics of Matplotlib grid
def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")  # optional background color

# Create the grid visualization component
SpaceGrid = make_space_component(agent_portrayal, post_process=post_process, draw_grid=False)

# Define interactive widgets for model configuration
model_params = {
    "width": {
        "type": "SliderInt",
        "value": GRID_WIDTH,
        "min": 50,
        "max": 150,
        "step": 10,
        "label": "Grid Width"
    },
    "height": {
        "type": "SliderInt",
        "value": GRID_HEIGHT,
        "min": 50,
        "max": 150,
        "step": 10,
        "label": "Grid Height"
    },
    "seed": {
        "type": "InputText",
        "value": "",
        "label": "Random Seed (leave blank for random)"
    }
}

# Create the Solara page
def model_factory(**kwargs):
    return StructuredCityModel(**kwargs)

page = SolaraViz(
    model_factory,
    components=[SpaceGrid],
    model_params=model_params,
    name="Structured Urban Grid World"
)
