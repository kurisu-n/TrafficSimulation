from mesa import Agent
from Simulation.agents.cell import CellAgent
from Simulation.config import Defaults

class DummyAgent(Agent):
    """
    Always sits atop a CellAgent when no vehicle is present.
    Portrayal simply duplicates the cell's portrayal on layer 1.
    """
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self._initial_pos = pos

    def get_portrayal(self):
        # find the CellAgent below us
        cell = next(a for a in self.model.grid.get_cell_list_contents([self.pos])
                    if isinstance(a, CellAgent))
        p = cell.get_portrayal()
        p["Layer"] = 1
        return p