# config.py
from dataclasses import dataclass
from enum import Enum, auto

@dataclass(frozen=True)
class Defaults:
    # grid
    WIDTH:  int = 100
    HEIGHT: int = 100
    # frame
    WALL_THICKNESS:       int   = 4
    SIDEWALK_RING_WIDTH:  int   = 2

    ROADS = ["R1", "R2", "R3"]

    # road network
    RING_ROAD_TYPE:       str = "R2"
    HIGHWAY_OFFSET:       int   = 4
    ALLOW_EXTRA_HIGHWAYS: bool = False
    EXTRA_HIGHWAY_CHANCE: float = 0.05
    R2_R3_CHANCE_SPLIT:   float = 0.5

    # blocks
    MIN_BLOCK_SPACING:    int   = 12
    MAX_BLOCK_SPACING:    int   = 24
    EMPTY_BLOCK_CHANCE:   float = 0.10
    # sub-blocks
    SUBBLOCK_CHANGE:      float = 0.7
    CARVE_SUBBLOCK_ROADS: bool  = True
    MIN_SUBBLOCK_SPACING: int   = 4
    SUBBLOCK_ROADS_HAVE_INTERSECTIONS: bool = False
    SUBBLOCK_ROAD_TYPE:   str = "R3"
    # control
    OPTIMISED_INTERSECTIONS:           bool = True
    TRAFFIC_LIGHT_RANGE:               int  = 10
    FORWARD_TRAFFIC_LIGHT_RANGE:         bool = False

    FORWARD_TRAFFIC_LIGHT_INTERSECTION_OPTIONS = ["Skip", "Include in Range", "Include as Extra"]
    FORWARD_TRAFFIC_LIGHT_INTERSECTIONS: str = FORWARD_TRAFFIC_LIGHT_INTERSECTION_OPTIONS[0]

    ROAD_THICKNESS = {
        "R1": 4,
        "R2": 2,
        "R3": 1
    }

    AVAILABLE_CITY_BLOCKS = ["Residential", "Office", "Market", "Leisure", "Other"]

    AVAILABLE_DIRECTIONS = ["N", "S", "E", "W"]

    DIRECTION_VECTORS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
    DIRECTION_OPPOSITES = {"N": "S", "S": "N", "E": "W", "W": "E"}
    DIRECTION_TO_THE_RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}

    ROAD_LIKE_TYPES = {"R1", "R2", "R3", "Intersection", "HighwayEntrance"}
    ROAD_LIKE_TYPES_WITHOUT_INTERSECTIONS = {"R1", "R2", "R3", "HighwayEntrance"}
    REMOVABLE_DEAD_END_TYPES = {"R2", "R3", "Intersection"}

    DIRECTION_ICONS = {"N": "↑", "S": "↓", "E": "→", "W": "←"}

    ZONES = [
        "Residential",
        "Office",
        "Market",
        "Leisure",
        "Other",
        "Empty",
        "Nothing",
        "Sidewalk",
        "Wall",
        "R1",
        "R2",
        "R3",
        "Intersection",
        "HighwayEntrance",
        "HighwayExit",
        "TrafficLight",
        "TrafficLightStop",
        "ControlledRoad",
        "ControlledRoadStop",
        "BlockEntrance",
    ]

    # Colors for each zone/road type
    ZONE_COLORS = {
        "Residential": "cadetblue",
        "Office": "orange",
        "Market": "green",
        "Leisure": "palevioletred",
        "Other": "darkkhaki",
        "Empty": "papayawhip",
        "Nothing": "white",
        "Sidewalk": "grey",
        "Wall": "black",
        "R1": "dodgerblue",
        "R2": "saddlebrown",
        "R3": "darkgreen",
        "Intersection": "yellow",
        "HighwayEntrance": "royalblue",
        "HighwayExit": "blue",
        "TrafficLight": "lime",
        "TrafficLightStop": "red",
        "ControlledRoad": "thistle",
        "ControlledRoadStop": "salmon",
        "BlockEntrance": "magenta",
    }

    DESCRIPTION_MAP = {
        "Residential": "Residential City Block",
        "Office": "Office City Block",
        "Market": "Market City Block",
        "Leisure": "Leisure City Block",
        "Other": "Miscellaneous City Block",
        "Empty": "Empty City Block",
        "Nothing": "Empty/unused space",
        "Sidewalk": "Pedestrian Walkway",
        "Wall": "Outer Wall",
        "R1": "Highway (4 Lanes, 2/Dir)",
        "R2": "Major Road (2 Lanes, 1/Dir)",
        "R3": "Local Road (1 Lane, One Dir)",
        "Intersection": "Road intersection",
        "HighwayEntrance": "Highway Entrance",
        "HighwayExit": "Highway Exit",
        "TrafficLight": "Intersection Traffic Light",
        "ControlledRoad": "Road Controlled by Traffic Light",
        "BlockEntrance": "City Block Entrance & Exit",
    }

    # CITY RESOURCES

    FOOD_CAPACITY_PER_CELL = 2  # units per inner cell
    FOOD_CONSUMPTION_TICKS = 50  # ticks between consumption
    WASTE_CAPACITY_PER_CELL = 1.5  # units per inner cell
    WASTE_PRODUCTION_TICKS = 100  # ticks between production

    CITY_BLOCK_THAT_NEED_FOOD = ["Market", "Leisure"]
    CITY_BLOCK_THAT_PRODUCE_WASTE = AVAILABLE_CITY_BLOCKS

    GRADUAL_CITY_BLOCK_RESOURCES = True

    # WEATHER

    RAIN_SPEED_REDUCTION = 2

    # VEHICLE SETTINGS

    VEHICLE_MIN_SPEED: int = 1
    VEHICLE_MAX_SPEED: int = 5

    VEHICLE_RESPECT_AWARENESS: bool = False
    VEHICLE_AWARENESS_RANGE: int = 10
    VEHICLE_AWARENESS_WIDTH: int = 3

    VEHICLE_OBSTACLE_PENALTY_VEHICLE = 1_000
    VEHICLE_OBSTACLE_PENALTY_STOP = 500

    VEHICLE_BASE_COLOR = "black"
    VEHICLE_PARKED_COLOR = "seagreen"

    VEHICLE_CONTRAFLOW_OVERTAKE_ACTIVE = True
    VEHICLE_CONTRAFLOW_PENALTY = 500
    VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS: int = 6
    VEHICLE_CONTRAFLOW_OVERTAKE_COLOR = "orange"

    VEHICLE_MALFUNCTION_ACTIVE: bool = True
    VEHICLE_MALFUNCTION_CHANCE: float = 1E-7
    VEHICLE_MALFUNCTION_DURATION: int = 400
    VEHICLE_MALFUNCTION_COLOR = "yellow"

    VEHICLE_SIDESWIPE_COLLISION_ACTIVE: bool = False
    VEHICLE_SIDESWIPE_COLLISION_CHANCE: float = 1E-3
    VEHICLE_SIDESWIPE_COLLISION_DURATION: int = 600

    VEHICLE_COLLISION_COLOR = "red"

    # OPTIMIZATION AND DEBUGGING

    USE_DUMMY_AGENTS: bool = True
    CACHE_CELL_PORTRAYAL: bool = True

    CACHED_TYPES = [z for z in ZONES if z not in [
        "HighwayEntrance",
        "HighwayExit",
        "TrafficLight",
        "TrafficLightStop",
        "ControlledRoad",
        "ControlledRoadStop",
        "BlockEntrance"]]

    CHANGE_ASSIGNED_CELL_COLOR_ON_STOP: bool = False











