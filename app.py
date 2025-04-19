# app.py

import solara
from mesa.visualization import make_space_component, SolaraViz
from cell import agent_portrayal

from creation import StructuredCityModel, GRID_WIDTH, GRID_HEIGHT
from cell import ZONE_COLORS, desaturate

# Optional: improve aesthetics of Matplotlib grid
def post_process(ax):
    ax.set_aspect("equal")                   # 1: make each cell square
    ax.set_xticks(range(GRID_WIDTH + 1))     # 2: tick at every column line
    ax.set_yticks(range(GRID_HEIGHT + 1))    # 3: tick at every row line
    ax.grid(True, color="lightgray", linewidth=0.5)  # 4: draw grid on top
    ax.set_facecolor("white")

# Create the grid visualization component
SpaceGrid = make_space_component(
    agent_portrayal,
    post_process=post_process,
    draw_grid=True,
)

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
def model_factory(
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
    seed: str | None = None,
):
    # Convert seed to int if you like, or let the model handle None
    return StructuredCityModel(width=width, height=height, seed=seed)

static_model = StructuredCityModel(width=GRID_WIDTH, height=GRID_HEIGHT)
reactive_model = solara.reactive(lambda **p: StructuredCityModel(**p))
factory_model = model_factory()
reactive_factory_model = solara.reactive(model_factory)


#page = SolaraViz(static_model, components=[SpaceGrid])
Page = SolaraViz(factory_model, components=[SpaceGrid], model_params=model_params, name="Structured Urban Grid World")
#page = SolaraViz(model, components=[SpaceGrid])

Page