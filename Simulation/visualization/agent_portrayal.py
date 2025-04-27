from Simulation.agents.cell import CellAgent
from Simulation.agents.vehicles.vehicle_base import VehicleAgent
from Simulation.agents.dummy import DummyAgent

# Properties from the CellAgent portrayal to exclude when showing DummyAgent
EXCLUDE_PROPERTIES = {
    "Shape", "Color"
}

def agent_portrayal(agent):
    """
    Dispatch portrayal based on agent type:
      - CellAgent: draw self on City Layer
      - VehicleAgent: draw self on Vehicle Layer
      - DummyAgent: mirror underlying CellAgent properties on Vehicle Layer,
        excluding a small set of keys
    """

    # Case 1: Dummy – show underlying cell info on vehicle layer
    if isinstance(agent, DummyAgent):
        # fetch the cell underneath
        x, y = agent.pos
        cell = next(a for a in agent.model.grid.get_cell_list_contents([(x, y)])
                    if isinstance(a, CellAgent))

        cell_por = cell.get_portrayal()
        # filter out excluded properties
        filtered = {k: v for k, v in cell_por.items() if k not in EXCLUDE_PROPERTIES}
        # build dummy portrayal
        filtered["Layer"] = 1
        return filtered

    # Case 2: Real cell – city layer
    if isinstance(agent, CellAgent):
        por = agent.get_portrayal()
        return por

    # Case 3: Real vehicle – vehicle layer
    if isinstance(agent, VehicleAgent):
        por = agent.get_portrayal()
        return por

    # fallback: nothing
    return {}
