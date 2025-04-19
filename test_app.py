import mesa
print(f"Mesa version: {mesa.__version__}")

from mesa.visualization import SolaraViz, make_plot_component, make_space_component

# Import the local MoneyModel.py
from moneymodel import MoneyModel

def agent_portrayal(agent):
    return {
        "color": "tab:blue",
        "size": 50,
    }

model_params = {
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": 10,
    "height": 10,
}