from mesa import Agent
from Simulation.utilities.general import *
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.city_model import CityModel
    from Simulation.agents.city_structure_entities.cell import CellAgent

class DummyAgent(Agent):
    """
    Always sits atop a CellAgent when no vehicle is present.
    Portrayal simply duplicates the cell's portrayal on layer 1.
    """
    def __init__(self, custom_id, model, pos):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self._initial_pos = pos

    def get_portrayal(self):
        city_model = cast("CityModel", self.model)
        cell = city_model.get_cell_contents(*self.pos)
        if cell is None:
            return {"Layer": 1}

        cell_agent = cast("CellAgent", cell[0])
        p = cell_agent.get_portrayal()
        p["Layer"] = 1
        return p