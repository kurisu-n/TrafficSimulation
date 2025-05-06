# cell.py
from typing import cast
from mesa import Agent
from Simulation.config import Defaults
from collections import deque
from Simulation.utilities import *



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

    def step(self):
        pass

    # ———— utility/query methods ————

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
            for controlled_block in self.controlled_blocks:
                controlled_block.status = "Pass"

    def set_light_stop(self):
        if self.is_traffic_light():
            self.status = "Stop"
            for controlled_block in self.controlled_blocks:
                controlled_block.status = "Stop"



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
            "Description": Defaults.DESCRIPTION_MAP.get(self.cell_type, ""),
        }

        if self.cell_type in Defaults.ROADS:
            portrayal["Assigned"] = self.light is not None
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
            portrayal["Assigned"] = self.light is not None
            portrayal["Intersection Group"] = None if self.intersection_group is None else self.intersection_group.id
            portrayal["Color"] = (
                desaturate(self.base_color, sat_factor=0.75, light_factor=0.25)
                if self.light is not None and self.light.status == "Stop"
                else self.base_color
            )


        if self.cell_type in Defaults.AVAILABLE_CITY_BLOCKS:
            portrayal["Block ID"] = self.block_id
            city = self.get_city_model()
            cb = getattr(city, "city_blocks", {}).get(self.block_id)

            if cb is not None:
                if cb.needs_food():
                   portrayal["Food"] = (
                                f"{int(cb.get_food_units())}/{int(cb.max_food_units)}")
                if cb.produces_waste():
                    portrayal["Waste"] = (
                                f"{int(cb.get_waste_units())}/{int(cb.max_waste_units)}")


        if self.cell_type == "BlockEntrance":
            portrayal["Block ID"] = self.block_id
            city = self.get_city_model()
            cb = getattr(city, "city_blocks", {}).get(self.block_id)

        if self.cell_type == "Sidewalk":
            if self.block_id is not None:
                portrayal["Block ID"] = self.block_id

        if direction_text:
            portrayal["Directions"] = direction_text

        if self.get_city_model().cache_cell_portrayal and self._is_cacheable():
            self._cached_portrayal = dict(portrayal)

        return portrayal
