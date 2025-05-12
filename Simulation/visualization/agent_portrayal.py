from Simulation.agents.cell import CellAgent
from Simulation.agents.vehicles.vehicle_base import VehicleAgent
from Simulation.agents.dummy import DummyAgent

from typing import cast, TYPE_CHECKING

from Simulation.config import Defaults

if TYPE_CHECKING:
    from Simulation.city_model import CityModel
    from Simulation.agents.cell import CellAgent

# Properties from the CellAgent portrayal to exclude when showing DummyAgent
EXCLUDE_PROPERTIES = {}
RENDER_EVERY_X_STEPS = 5


def agent_portrayal(agent):
    """
    Dispatch portrayal based on agent type:
      - CellAgent: draw self on City Layer
      - VehicleAgent: draw self on Vehicle Layer
      - DummyAgent: mirror underlying CellAgent properties on Vehicle Layer,
        excluding a small set of keys
    """
    if not Defaults.ENABLE_AGENT_PORTRAYAL:
        return {}

    if isinstance(agent, DummyAgent):
        # fetch the cell underneath
        city_model = cast("CityModel", agent.model)
        x, y = agent.pos
        cell = next(a for a in city_model.get_cell_contents(x, y)
                    if isinstance(a, CellAgent))

        cell_por = cell.get_portrayal()
        # filter out excluded properties
        filtered = {k: v for k, v in cell_por.items() if k not in EXCLUDE_PROPERTIES}
        # build dummy portrayal
        filtered["Layer"] = 1
        filtered["Shape"] = "rect"
        filtered["Color"] = "rgba(0,0,0,0)"
        return filtered

    if isinstance(agent, CellAgent):
        por = agent.get_portrayal()
        return por

    if isinstance(agent, VehicleAgent):
        por = agent.get_portrayal()
        return por

    # fallback: nothing
    return {}
