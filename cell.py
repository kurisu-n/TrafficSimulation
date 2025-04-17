#cell.py

from mesa import Agent

# Colors for each zone/road type
ZONE_COLORS = {
    "Residential": "cyan",
    "Office": "orange",
    "Market": "green",
    "Leisure": "pink",
    "Other": "gray",
    "R1": "dodgerblue",        # 4‑lane highway (unchanged)
    "R2": "saddlebrown",       # 2‑lane major
    "R3": "teal",         # darker than before
    "R4": "darkgreen",    # one‑way L‑shaped cuts Local Road (1 lane)
    "Sidewalk": "gray",
    "Intersection": "yellow",
    "BlockEntrance": "red",
    "HighwayEntrance": "navy", # Dark blue for boundary highway entrance
    "HighwayExit": "lightsteelblue",
    "TrafficLight":"magenta",
    "ControlledRoadOpen":"thistle",
    "ControlledRoadClosed":"crimson",
    "Wall": "black",
    "Nothing":"white"
}

# Direction icons for visualization
DIRECTION_ICONS = {
    "N": "↑",
    "S": "↓",
    "E": "→",
    "W": "←",
}

class CellAgent(Agent):
    """
    A cell that can be:
      - Road (R1, R2, R3)
      - Intersection
      - HighwayEntrance
      - Wall
      - Sidewalk
      - Zone block (Residential, Office, etc.)
    'directions' can store lane directions, e.g. ['E','W'] or ['N','S','N','S'].
    """
    def __init__(self, unique_id, model, cell_type):
        super().__init__(unique_id, model)
        self.cell_type = cell_type
        self.directions = []

    def step(self):
        pass

def agent_portrayal(agent):
    """
    Portrayal function for Mesa visualization.
    Renders each CellAgent with color + optional direction arrows.
    """
    # Convert lane directions to arrow icons
    arrows = [DIRECTION_ICONS.get(d, '') for d in agent.directions]
    direction_text = ' '.join(arrows)

    # Descriptions for GUI tooltips
    desc_map = {
        "R1": "Highway (4 cells wide, 2 lanes/direction)",
        "R2": "Major Road (2-lane, each direction)",
        "R3": "Local Road (1‑lane, one‑way)",
        "R4": "Sub‑block Road (L‑shaped, 1‑lane, one‑way)",
        "Sidewalk": "Pedestrian walkway",
        "Intersection": "Road intersection",
        "BlockEntrance": "Block entrance",
        "HighwayEntrance": "Highway entrance (punches through wall)",
        "Nothing": "Empty/unused space",
        "Wall": "Outer wall (4-cells thick)",

        # Zones
        "Residential": "Residential block",
        "Office": "Office block",
        "Market": "Market block",
        "Leisure": "Leisure block",
        "Empty": "Empty block"
    }
    description = desc_map.get(agent.cell_type, "Zone")

    portrayal = {
        "Shape": "rect",
        "w": 1.0,
        "h": 1.0,
        "Filled": True,
        "Color": ZONE_COLORS.get(agent.cell_type, "white"),
        "Layer": 0,
        "Type": agent.cell_type,
        "Description": description,
    }

    if direction_text:
        portrayal["Directions"] = direction_text

    return portrayal
