# cell.py
from typing import cast, TYPE_CHECKING
from mesa import Agent
from Simulation.config import Defaults
from collections import deque
from Simulation.utilities.general import *

if TYPE_CHECKING:
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
        self.city_model = cast("CityModel", model)
        self.cell_type = cell_type

        if self.cell_type in Defaults.ROADS:
            self.road_type = self.cell_type
        else :
            self.road_type = None

        self.directions = []
        self.base_color = Defaults.ZONE_COLORS.get(cell_type)

        self.block_id = None
        self.block_type = None
        self.highway_id = None
        self.highway_orientation = None

        self.intersection_group = None
        self.light = None
        self.assigned_road_blocks = []
        self.controlled_blocks = []

        self.city_model.cell_lookup[self.position] = self

        self.has_been_displayed = False

        self._base_portrayal = {
            "Shape":      "rect",
            "w":          1.0,
            "h":          1.0,
            "Filled":     True,
            "Layer":      0,
            "Color":      self.base_color
        }
        self._cached_portrayal = None

    def expand_static_portrayal(self):
        self._base_portrayal["Identifier"] = self.get_display_name()
        self._base_portrayal["Description"] = self.get_description()

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
        model = self.city_model
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
            nx, ny = self.city_model.next_cell_in_direction(x, y, d)
            if self.city_model.in_bounds(nx, ny):
                nbr = self.city_model.get_cell_contents(nx, ny)[0]
                nbrs.setdefault(d, []).append(nbr)
        return nbrs

    def leads_to(self, other: "CellAgent") -> bool:
        city_model = self.city_model
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
        x, y = self.get_position()
        for d in self.directions:
            nx, ny = self.city_model.next_cell_in_direction(x, y, d)
            if not self.city_model.in_bounds(nx, ny):
                continue
            # unpack x, y rather than passing a tuple
            nbr = self.city_model.get_cell_contents(nx, ny)[0]
            if nbr is other:
                return True
        return False

    def set_light_stop(self):
        x, y = self.get_position()
        self.city_model.stop_map[y, x] = 1
        for controlled_block in self.controlled_blocks:
            self.city_model.stop_map[controlled_block.position[1], controlled_block.position[0]] = 1

    def set_light_go(self):
        x, y = self.get_position()
        self.city_model.stop_map[y, x] = 0
        for controlled_block in self.controlled_blocks:
            self.city_model.stop_map[controlled_block.position[1], controlled_block.position[0]] = 0

    def get_description(self):
        return Defaults.DESCRIPTION_MAP.get(self.cell_type, "")

    def should_portray(self):
        return not Defaults.CACHE_CELL_PORTRAYAL or not self._is_cacheable() or not self.has_been_displayed

    def get_portrayal(self):
        # 1) if you already fully cached it, just return that
        if (
            self.city_model.cache_cell_portrayal
            and self._cached_portrayal is not None
            and self._is_cacheable()
        ):
            return self._cached_portrayal

        # 2) shallow-copy the static bits
        p = self._base_portrayal.copy()

        if Defaults.AGENT_PORTRAYAL_LEVEL == 0:
            return p

        if Defaults.AGENT_PORTRAYAL_LEVEL >= 1:
            color = self.base_color

            x, y = self.pos
            is_stop = self.city_model.stop_map[y, x] == 1
            is_raining = Defaults.RAIN_ENABLED and self.city_model.rain_map[y, x] > 0

            if self.cell_type in Defaults.ROADS and Defaults.CHANGE_ASSIGNED_CELL_COLOR_ON_STOP and self.light and is_stop:
                color = desaturate(color, sat_factor=0.75, light_factor=0.25)

            if self.cell_type == "ControlledRoad":
                color = Defaults.ZONE_COLORS["ControlledRoadStop"] if is_stop else desaturate(color, sat_factor=0.75, light_factor=0.25)

            if self.cell_type == "TrafficLight":
                color = Defaults.ZONE_COLORS["TrafficLightStop"] if is_stop else Defaults.ZONE_COLORS["TrafficLight"]

            if self.cell_type == "Intersection" and self.intersection_group.pending_phase is not None:
                color = self.base_color if self.intersection_group.pending_phase is None else Defaults.ZONE_COLORS["IntersectionPending"]

            # if self.cell_type == "Intersection" and self.light and self.light.status == "Stop":
            #     color = desaturate(color, sat_factor=0.75, light_factor=0.25)

            if is_raining:
                color = desaturate(color, sat_factor=0.95, light_factor=-0.05)

            p["Color"] = color


            if Defaults.AGENT_PORTRAYAL_LEVEL >= 2:
                p["Position"] = self.position
                p["Rain"]     = "Yes" if is_raining else "No"

                if self.cell_type in Defaults.ROADS:
                    p["Light"] = self.light is not None

                if self.cell_type=="ControlledRoad":
                    p["Control State"] = "Stop" if is_stop else "Go"

                if self.cell_type=="Intersection":
                    if self.light is not None and self.intersection_group is not None:
                        p["Intersection Group"] = self.intersection_group.id
                        p["Algorithm"] = Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM
                        p["Current Phase"] = self.intersection_group.current_phase
                        p["Pending Phase"] = self.intersection_group.pending_phase
                        if Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM == "FIXED_TIME":
                            p["Fixed-Time Timer"] = self.intersection_group.fixed_time_timer
                        elif Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM == "QUEUE_ACTUATED":
                            p["Queue Actuated Timer"] = self.intersection_group.queue_timer
                        elif Defaults.TRAFFIC_LIGHT_AGENT_ALGORITHM == "PRESSURE_CONTROL":
                            p["N-S Pressure"] = self.intersection_group.ns_pressure
                            p["E-W Pressure"] = self.intersection_group.ew_pressure

                if self.cell_type == "BlockEntrance" or self.cell_type in Defaults.AVAILABLE_CITY_BLOCKS:
                    p["Block ID"] = self.block_id

                if self.cell_type=="BlockEntrance" and self.city_model.city_blocks.get(self.block_id):
                    blk = self.city_model.city_blocks[self.block_id]
                    if blk.needs_food():
                        p["Food"]  = f"{int(blk.get_food_units())}/{int(blk.max_food_units)}"
                    if blk.produces_waste():
                        p["Waste"] = f"{int(blk.get_waste_units())}/{int(blk.max_waste_units)}"

                # directions arrow text
                arrows = [Defaults.DIRECTION_ICONS.get(d, "") for d in self.directions]
                if arrows:
                    p["Directions"] = " ".join(arrows)

                if self.city_model.cache_cell_portrayal and self._is_cacheable():
                    self._cached_portrayal = p

        return p

