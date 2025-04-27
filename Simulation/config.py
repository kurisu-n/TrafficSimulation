# config.py
from dataclasses import dataclass
from enum import Enum, auto

@dataclass(frozen=True)
class Defaults:
    # grid
    WIDTH:  int = 200
    HEIGHT: int = 200
    # frame
    WALL_THICKNESS:       int   = 10
    SIDEWALK_RING_WIDTH:  int   = 2

    ROADS = ["R1", "R2", "R3"]

    # road network
    RING_ROAD_TYPE:       str = "R2"
    HIGHWAY_OFFSET:       int   = 10
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

    # Colors for each zone/road type
    ZONE_COLORS = {
        "Residential": "cadetblue",
        "Office": "orange",
        "Market": "green",
        "Leisure": "palevioletred",
        "Other": "darkkhaki",
        "Empty": "papayawhip",
        "R1": "dodgerblue",
        "R2": "saddlebrown",
        "R3": "darkgreen",
        "Sidewalk": "grey",
        "Intersection": "yellow",
        "BlockEntrance": "magenta",
        "HighwayEntrance": "royalblue",
        "HighwayExit": "blue",
        "TrafficLight": "lime",
        "TrafficLightStop": "red",
        "ControlledRoad": "thistle",
        "ControlledRoadStop": "salmon",
        "Wall": "black",
        "Nothing": "white"
    }

    DESCRIPTION_MAP = {
        "Residential": "Residential City Block",
        "Office": "Office City Block",
        "Market": "Market City Block",
        "Leisure": "Leisure City Block",
        "Other": "Miscellaneous City Block",
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
        "Nothing":"Empty/unused space",
    }

    FOOD_CAPACITY_PER_CELL = 2  # units per inner cell
    FOOD_CONSUMPTION_TICKS = 50  # ticks between consumption
    WASTE_CAPACITY_PER_CELL = 1.5  # units per inner cell
    WASTE_PRODUCTION_TICKS = 100  # ticks between production

    CITY_BLOCK_THAT_NEED_FOOD = ["Market", "Leisure"]
    CITY_BLOCK_THAT_PRODUCE_WASTE = AVAILABLE_CITY_BLOCKS

    GRADUAL_CITY_BLOCK_RESOURCES = True
