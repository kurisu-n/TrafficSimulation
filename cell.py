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
    "BlockEntrance": "red",
    "HighwayEntrance": "royalblue",
    "HighwayExit": "blue",
    "TrafficLight":"magenta",
    "ControlledRoad":"thistle",
    "ControlledRoadClosed":"crimson",
    "Wall": "black",
    "Other": "white",
    "Nothing":"white"
}

DIRECTION_ICONS = {"N":"↑","S":"↓","E":"→","W":"←"}

class CellAgent(Agent):
    def __init__(self, model, cell_type, pos):
        super().__init__(model)
        self.cell_type = cell_type
        self.pos = pos  # REQUIRED for Mesa 3.x visualization
        self.directions = []
        self.status = None
        self.base_color = ZONE_COLORS.get(cell_type)

        # additional fields
        self.occupied = False
        self.block_id = None
        self.block_type = None
        self.highway_id = None
        self.highway_orientation = None

        self.lights = []
        self.controlled_road = None
        self.controlled_blocks = []

    def step(self):
        pass

    def get_position(self):
        return self.pos

    def successors(self):
        x, y = self.pos
        nbrs = {}
        for d in self.directions:
            nx, ny = self.model._next_cell_in_direction(x, y, d)
            if self.model._in_bounds(nx, ny):
                neighbor = self.model.grid.get_cell_list_contents((nx, ny))[0]
                nbrs.setdefault(d, []).append(neighbor)
        return nbrs

    def set_pass(self):
        if self.cell_type == "TrafficLight":
            self.status = "Pass"
            if self.controlled_road is not None:
                self.controlled_road.status = "Pass"

    def set_stop(self):
        if self.cell_type == "TrafficLight":
            self.status = "Stop"
            if self.controlled_road is not None:
                self.controlled_road.status = "Stop"

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


def agent_portrayal(agent):
    return {"color": "black", "marker": "s", "size": 10}