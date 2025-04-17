#cell.py

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
    """
    Accepts a CSS name (e.g. "thistle") or "#RRGGBB".
    Returns a hex string with its saturation scaled by `sat_factor`
    and its lightness increased by `light_factor`.
    - sat_factor: 0 = gray, 1 = original saturation
    - light_factor: added to L component (0 = no change, up to 1 = full white)
    """
    # get RGB floats
    if isinstance(color, str) and color.startswith('#'):
        r, g, b = _hex_to_rgb(color)
    else:
        r, g, b = mcolors.to_rgb(color)
    # convert to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # adjust saturation and lightness
    s *= sat_factor
    l = min(1.0, l + light_factor)
    # back to RGB
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((r2, g2, b2))


# Colors for each zone/road type
ZONE_COLORS = {
    "Residential": "cadetblue",
    "Office": "orange",
    "Market": "green",
    "Leisure": "palevioletred",
    "Empty":"papayawhip",
    "R1": "dodgerblue",        # 4‑lane highway (unchanged)
    "R2": "saddlebrown",       # 2‑lane major
    "R3": "darkgreen",         # darker than before
    "Sidewalk": "grey",
    "Intersection": "yellow",
    "BlockEntrance": "red",
    "HighwayEntrance": "royalblue", # Dark blue for boundary highway entrance
    "HighwayExit": "blue",
    "TrafficLight":"magenta",
    "ControlledRoad":"thistle",
    "ControlledRoadClosed":"crimson",
    "Wall": "black",
    "Other": "white",
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
        self.status = None
        self.base_color = ZONE_COLORS.get(cell_type)

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
        # Zones
        "Residential": "Residential City Block",
        "Office": "Office City Block",
        "Market": "Market City Block",
        "Leisure": "Leisure City Block",
        "Empty": "Empty City Block",
        "R1": "Highway (4 Lanes, 2/Dir)",
        "R2": "Major Road (2 Lanes, 1/Dir)",
        "R3": "Local Road (1 Lane, One Dir)",
        "R4": "Sub‑block Road (L‑shaped, 1 Lane, One Dir)",
        "Sidewalk": "Pedestrian Walkway",
        "Intersection": "Road intersection",
        "BlockEntrance": "City Block Entrance & Exit",
        "HighwayEntrance": "Highway Entrance",
        "HighwayExit": "Highway Exit",
        "TrafficLight":"Intersection Traffic Light",
        "ControlledRoad":"Road Controlled by Traffic Light",
        "Wall": "Outer Wall",
        "Other": "Unknown",
        "Nothing": "Empty/unused space",


    }
    description = desc_map.get(agent.cell_type, "Zone")

    portrayal = {
        "Shape": "rect",
        "w": 1.0,
        "h": 1.0,
        "Filled": True,
        "Color": agent.base_color,
        "Layer": 0,
        "Type": agent.cell_type,
        "Description": description,
    }

    if agent.cell_type == "ControlledRoad":
        if agent.status == "Stop":
            # red light
            portrayal["Color"] = ZONE_COLORS["ControlledRoadClosed"]
        else:
            # pass → desaturate the **original** road color
            base = agent.base_color
            portrayal["Color"] = desaturate(base, sat_factor=0.75, light_factor=0.25)

    if direction_text:
        portrayal["Directions"] = direction_text

    return portrayal
