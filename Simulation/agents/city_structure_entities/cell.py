# cell.py
from typing import cast, TYPE_CHECKING
from mesa import Agent
from Simulation.config import Defaults
from collections import deque
from Simulation.utilities.general import *

if TYPE_CHECKING:
    from Simulation.agents.city_structure_entities.city_block import CityBlock
    from Simulation.city_model import CityModel

class CellAgent(Agent):
    """
    A cell that can be:
      - Road (R1, R2, R3, R4)
      - Intersection
      - BlockEntrance
      - HighwayEntrance / Exit
      - TrafficLight
      - ControlledRoad
      - Wall, Sidewalk, Zone blocks
    """
    def __init__(self, custom_id, model, position, cell_type):
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.position = position
        self.cell_type = cell_type
        self.directions = []
        self.status = None
        self.base_color = Defaults.ZONE_COLORS.get(cell_type)

        self.is_raining = False

        self.occupied = False
        self.block_id = None
        self.block_type = None
        self.highway_id = None
        self.highway_orientation = None

        self.intersection_group = None
        self.light = None
        self.assigned_road_blocks = []
        self.controlled_blocks = []

        self.get_city_model().cell_lookup[self.position] = self

        self.has_been_displayed = False

        self._cached_portrayal: dict | None = None

    def _is_cacheable(self) -> bool:
        """
        Return True when cell_type is contained (even inside a nested
        list / set / tuple) in Defaults.CACHED_TYPES.
        """
        for entry in Defaults.CACHED_TYPES:
            if isinstance(entry, (list, set, tuple)):
                if self.cell_type in entry:
                    return True
            elif self.cell_type == entry:
                return True
        return False

    def _format_highway_label(self) -> str:
        """
        Build a name like "Horizontal_1_South_Entrance_2" or
        "Vertical_3_East_Exit_1" for any HighwayEntrance/Exit cell.
        """
        model = self.get_city_model()
        x, y = self.position
        max_x, max_y = model.width - 1, model.height - 1

        # 1) Cardinal edge
        if y == 0:
            cardinal = "South"
        elif y == max_y:
            cardinal = "North"
        elif x == 0:
            cardinal = "West"
        elif x == max_x:
            cardinal = "East"
        else:
            cardinal = "Center"  # should never happen for highway cells

        # 2) Orientation
        orientation = "Horizontal" if cardinal in ("South", "North") else "Vertical"

        # 3) GroupIdx: gather all highway IDs of this orientation
        ents = model.get_highway_entrances()
        exts = model.get_highway_exits()
        # pick all IDs where the cell sits on the matching edge
        hw_ids = set(
            c.highway_id
            for c in (ents + exts)
            if ((orientation == "Horizontal") and c.position[1] in (0, max_y))
            or ((orientation == "Vertical")   and c.position[0] in (0, max_x))
        )
        # For each ID, pick one representative cell to sort by
        reps = []
        for hid in hw_ids:
            # prefer the entrance cell if it exists, else an exit
            candidates = [c for c in ents if c.highway_id == hid] or [c for c in exts if c.highway_id == hid]
            rep_pos = candidates[0].position
            reps.append((hid, rep_pos))

        # sort them into reading order
        if orientation == "Horizontal":
            reps.sort(key=lambda t: (t[1][1], t[1][0]))  # by y (south→north), then x
        else:
            reps.sort(key=lambda t: (t[1][0], t[1][1]))  # by x (west→east), then y

        ordered_hids = [t[0] for t in reps]
        group_idx = ordered_hids.index(self.highway_id) + 1

        # 4) PairIdx: among entrances *or* exits on this same edge
        if self.cell_type == "HighwayEntrance":
            same_edge = [c for c in ents if c.position == c.position and c.position[0] == x and c.position[1] == y]
            # actually you want *all* entrances on that same edge:
            if orientation == "Horizontal":
                coll = [c for c in ents if c.position[1] == y]
                coll.sort(key=lambda c: c.position[0])
            else:
                coll = [c for c in ents if c.position[0] == x]
                coll.sort(key=lambda c: c.position[1])
        else:
            # exits
            if orientation == "Horizontal":
                coll = [c for c in exts if c.position[1] == y]
                coll.sort(key=lambda c: c.position[0])
            else:
                coll = [c for c in exts if c.position[0] == x]
                coll.sort(key=lambda c: c.position[1])

        pair_idx = coll.index(self) + 1

        typ = "Entrance" if self.cell_type == "HighwayEntrance" else "Exit"
        return f"{orientation}_{group_idx}_{cardinal}_{typ}_{pair_idx}"


    def step(self):
        self.is_raining = False

    # ———— utility/query methods ————

    def get_display_name(self):
        if self.cell_type == "Intersection":
            return f"{self.id}"
        elif self.cell_type == "BlockEntrance":
            return f"BlockEntrance_{self.block_id}"
        elif self.cell_type in ("HighwayEntrance", "HighwayExit"):
            return self._format_highway_label()
        elif self.cell_type == "TrafficLight":
            return f"TrafficLight_I{self.intersection_group.id}_#{self.intersection_group.traffic_lights.index(self)}"
        else:
            return self.get_position()

    def get_city_model(self) -> "CityModel":
        from Simulation.city_model import CityModel
        return cast(CityModel, self.model)

    def get_position(self):
        return self.position

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

    def outgoing_cells(self):
        """Return dict direction → list of neighbor CellAgents."""
        x, y = self.get_position()
        nbrs = {}
        for d in self.directions:
            nx, ny = self.get_city_model().next_cell_in_direction(x, y, d)
            if self.get_city_model().in_bounds(nx, ny):
                nbr = self.get_city_model().get_cell_contents((nx, ny))[0]
                nbrs.setdefault(d, []).append(nbr)
        return nbrs

    def leads_to(self, other: "CellAgent") -> bool:
        city_model = self.get_city_model()
        queue = deque([self])
        visited = {self}

        while queue:
            current = queue.popleft()
            if current is other:
                return True

            for d in current.directions:
                nx, ny = city_model.next_cell_in_direction(*current.position, d)
                if not city_model.in_bounds(nx, ny):
                    continue

                # Cache neighbor to avoid frequent grid access
                neighbors = city_model.get_cell_contents(nx, ny)
                if not neighbors:
                    continue
                neighbor = neighbors[0]  # Extract single CellAgent
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append(neighbor)

        return False

    def directly_leads_to(self, other: "CellAgent") -> bool:
        x,y = self.get_position()
        for d in self.directions:
            nx, ny = self.get_city_model().next_cell_in_direction(x,y,d)
            if not self.get_city_model().in_bounds(nx, ny):
                continue
            nbr = self.get_city_model().get_cell_contents((nx, ny))[0]
            if nbr is other:
                return True

        return False

    def set_light_go(self):
        if self.is_traffic_light():
            self.status = "Pass"
            self.get_city_model().stop_cells.discard(self.position)
            for controlled_block in self.controlled_blocks:
                controlled_block.status = "Pass"

    def set_light_stop(self):
        if self.is_traffic_light():
            self.status = "Stop"
            self.get_city_model().stop_cells.add(self.position)
            for controlled_block in self.controlled_blocks:
                controlled_block.status = "Stop"

    def get_description(self):
        return Defaults.DESCRIPTION_MAP.get(self.cell_type, "")

    def should_portray(self):
        return not Defaults.CACHE_CELL_PORTRAYAL or not self._is_cacheable() or not self.has_been_displayed

    def get_portrayal(self):
        # ① return cached copy when allowed
        if (
            self.get_city_model().cache_cell_portrayal         # global flag
            and self._cached_portrayal is not None             # already built
            and self._is_cacheable()                           # type whitelisted
        ):
            return self._cached_portrayal

        arrows = [Defaults.DIRECTION_ICONS.get(d, '') for d in self.directions]
        direction_text = ' '.join(arrows)

        portrayal = {
            "Shape": "rect", "w":1.0, "h":1.0, "Filled":True,
            "Color": self.base_color,
            "Layer": 0,
            "Type": self.cell_type,
            "Identifier": self.get_display_name(),
            "Position": self.get_position(),
            "Description": self.get_description(),
            "Rain": self.is_raining,
        }

        if self.cell_type in Defaults.ROADS:
            portrayal["Light"] = self.light is not None
            if Defaults.CHANGE_ASSIGNED_CELL_COLOR_ON_STOP:
                portrayal["Color"] = (
                    desaturate(self.base_color, sat_factor=0.75, light_factor=0.25)
                    if self.light is not None and self.light.status == "Stop"
                    else self.base_color
                )

        if self.cell_type=="ControlledRoad":
            portrayal["Color"] = (
                Defaults.ZONE_COLORS["ControlledRoadStop"]
                if self.status=="Stop"
                else desaturate(self.base_color, sat_factor=0.75, light_factor=0.25)
            )
            portrayal["Control State"] = self.status

        if self.cell_type =="TrafficLight":
            portrayal["Color"] = (
                Defaults.ZONE_COLORS["TrafficLightStop"]
                if self.status == "Stop"
                else Defaults.ZONE_COLORS["TrafficLight"]
            )

        if self.cell_type=="Intersection":
            portrayal["Intersection Group"] = None if self.intersection_group is None else self.intersection_group.id
            portrayal["Color"] = (
                desaturate(self.base_color, sat_factor=0.75, light_factor=0.25)
                if self.light is not None and self.light.status == "Stop"
                else self.base_color
            )

        if self.cell_type == "BlockEntrance":
            portrayal["Block ID"] = self.block_id

            city = cast("CityModel", self.get_city_model())
            city_block = cast("CityBlock", city.city_blocks.get(self.block_id))

            if city_block is not None:
                if city_block.needs_food():
                    portrayal["Food"] = (
                        f"{int(city_block.get_food_units())}/{int(city_block.max_food_units)}")
                if city_block.produces_waste():
                    portrayal["Waste"] = (
                        f"{int(city_block.get_waste_units())}/{int(city_block.max_waste_units)}")

        if self.cell_type in Defaults.AVAILABLE_CITY_BLOCKS:
            portrayal["Block ID"] = self.block_id

        if self.cell_type == "Sidewalk" and self.block_id is not None:
            portrayal["Block ID"] = self.block_id

        if direction_text:
            portrayal["Directions"] = direction_text

        if self.get_city_model().cache_cell_portrayal and self._is_cacheable():
            self._cached_portrayal = dict(portrayal)
            self.has_been_displayed = True

        if self.is_raining:
            portrayal["Color"] = desaturate(portrayal["Color"], sat_factor=0.95, light_factor=-0.05)

        return portrayal
