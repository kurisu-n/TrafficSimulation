# Simulation/portrayal.py
# ============================================================
from Simulation.agents.cell import CellAgent
from Simulation.agents.vehicles.vehicle_base import VehicleAgent

def agent_portrayal(agent):

    # ── Case 1: the item handed to us already *is* a CellAgent ───────────
    if isinstance(agent, CellAgent):
        # CellAgents implement get_portrayal() themselves
        return agent.get_portrayal()

    if isinstance(agent, VehicleAgent):
        return agent.get_portrayal()

    # Safety fallback: nothing to draw if somehow no CellAgent is present
    return {}