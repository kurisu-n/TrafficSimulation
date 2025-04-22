# cell.py

from mesa import Agent
import colorsys
import matplotlib.colors as mcolors

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )

def desaturate(color, sat_factor=0.5, light_factor=0.0):
    if isinstance(color, str) and color.startswith('#'):
        r, g, b = _hex_to_rgb(color)
    else:
        r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s *= sat_factor
    l = min(1.0, l + light_factor)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((r2, g2, b2))

# Colors for each zone/road type
ZONE_COLORS = {
    "Residential": "cadetblue",
    "Office": "orange",
    "Market": "green",
    "Leisure": "palevioletred",
    "Empty":"papayawhip",
    "R1": "dodgerblue",
    "R2": "saddlebrown",
    "R3": "darkgreen",
    "Sidewalk": "grey",
    "Intersection": "yellow",
    "BlockEntrance": "magenta",
    "HighwayEntrance": "royalblue",
    "HighwayExit": "blue",
    "TrafficLight":"lime",
    "TrafficLightStop":"red",
    "ControlledRoad":"thistle",
    "ControlledRoadStop":"salmon",
    "Wall": "black",
    "Other": "white",
    "Nothing":"white"
}

DIRECTION_ICONS = {"N":"↑","S":"↓","E":"→","W":"←"}

ROADS = ["R1","R2","R3"]

class CellAgent(Agent):
    """
    A cell that can be:
      - Road (R1, R2, R3, R4)
      - Intersection
      - BlockEntrance
      - HighwayEntrance / Exit
      - TrafficLight
      - ControlledRoad
      - Wall, Sidewalk, Zone blocks...
    """
    def __init__(self, unique_id, model, cell_type):
        super().__init__(unique_id, model)
        self.cell_type = cell_type
        self.directions = []
        self.status = None
        self.base_color = ZONE_COLORS.get(cell_type)

        # new fields:
        self.occupied = False
        self.block_id = None
        self.block_type = None
        self.highway_id = None
        self.highway_orientation = None

        # for intersections controlled by traffic lights:
        self.light = None
        self.assigned_road_blocks = []
        self.controlled_blocks = []

    def step(self):
        pass

    # ———— utility/query methods ————

    def get_position(self):
        # Assumes unique_id like "Type_x_y"
        parts = self.unique_id.split('_')
        return int(parts[-2]), int(parts[-1])

    def successors(self):
        """Return dict direction → list of neighbor CellAgents."""
        x, y = self.get_position()
        nbrs = {}
        for d in self.directions:
            nx, ny = self.model._next_cell_in_direction(x, y, d)
            if self.model._in_bounds(nx, ny):
                nbr = self.model.grid.get_cell_list_contents((nx, ny))[0]
                nbrs.setdefault(d, []).append(nbr)
        return nbrs

    def is_block_entrance(self):
        return self.cell_type == "BlockEntrance"

    def is_highway_entrance(self):
        return self.cell_type == "HighwayEntrance"

    def is_highway_exit(self):
        return self.cell_type == "HighwayExit"

    def is_controlled_road(self):
        return self.cell_type == "ControlledRoad"

    def is_traffic_light(self):
        return self.cell_type == "TrafficLight"

    def leads_to(self, other: "CellAgent", visited=None) -> bool:
        """
        Return True if there is a directed path of road‐arrows from self to other.
        Searches successors following each arrow in self.directions.
        """
        if visited is None:
            visited = set()
        # If we’ve arrived…
        if self is other:
            return True
        visited.add(self)
        # Explore every outgoing arrow
        x, y = self.get_position()
        for d in self.directions:
            nx, ny = self.model._next_cell_in_direction(x, y, d)
            if not self.model._in_bounds(nx, ny):
                continue
            nbr = self.model.grid.get_cell_list_contents((nx, ny))[0]
            if nbr in visited:
                continue
            if nbr.leads_to(other, visited):
                return True
        return False

    def set_light_go(self):
        if self.is_traffic_light():
            self.status = "Pass"
            for controlled_block in self.controlled_blocks:
                controlled_block.status = "Pass"

    def set_light_stop(self):
        if self.is_traffic_light():
            self.status = "Stop"
            for controlled_block in self.controlled_blocks:
                controlled_block.status = "Stop"



def agent_portrayal(agent):
    arrows = [DIRECTION_ICONS.get(d, '') for d in agent.directions]
    direction_text = ' '.join(arrows)
    desc_map = {
        "Residential": "Residential City Block",
        "Office": "Office City Block",
        "Market": "Market City Block",
        "Leisure": "Leisure City Block",
        "Empty": "Empty City Block",
        "R1": "Highway (4 Lanes, 2/Dir)",
        "R2": "Major Road (2 Lanes, 1/Dir)",
        "R3": "Local Road (1 Lane, One Dir)",
        "R4": "Sub‑block Road (L‑shaped)",
        "Sidewalk": "Pedestrian Walkway",
        "Intersection": "Road intersection",
        "BlockEntrance": "City Block Entrance & Exit",
        "HighwayEntrance": "Highway Entrance",
        "HighwayExit": "Highway Exit",
        "TrafficLight":"Intersection Traffic Light",
        "ControlledRoad":"Road Controlled by Traffic Light",
        "Wall": "Outer Wall",
        "Other": "Unknown",
        "Nothing":"Empty/unused space",
    }
    portrayal = {
        "Shape": "rect", "w":1.0, "h":1.0, "Filled":True,
        "Color": agent.base_color,
        "Layer": 0,
        "Type": agent.cell_type,
        "Description": desc_map.get(agent.cell_type, ""),
    }

    if agent.cell_type in ROADS:
        portrayal["Assigned"] = agent.light is not None
        portrayal["Color"] = (
            desaturate(agent.base_color, sat_factor=0.75, light_factor=0.25)
            if agent.light is not None and agent.light.status == "Stop"
            else agent.base_color
        )

    if agent.cell_type=="ControlledRoad":
        portrayal["Color"] = (
            ZONE_COLORS["ControlledRoadStop"]
            if agent.status=="Stop"
            else desaturate(agent.base_color, sat_factor=0.75, light_factor=0.25)
        )
        portrayal["Control State"] = agent.status

    if agent.cell_type =="TrafficLight":
        portrayal["Color"] = (
            ZONE_COLORS["TrafficLightStop"]
            if agent.status == "Stop"
            else ZONE_COLORS["TrafficLight"]
        )



    if direction_text:
        portrayal["Directions"] = direction_text
    return portrayal
