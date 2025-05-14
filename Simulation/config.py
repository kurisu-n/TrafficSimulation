# config.py
from dataclasses import dataclass
from enum import Enum, auto

@dataclass(frozen=True)
class Defaults:
    # grid
    WIDTH:  int = 200
    HEIGHT: int = 200
    # frame
    WALL_THICKNESS:       int   = 4
    SIDEWALK_RING_WIDTH:  int   = 2

    ROADS = ["R1", "R2", "R3"]

    # road network
    RING_ROAD_TYPE:       str = "R2"
    HIGHWAY_OFFSET:       int   = 4
    ALLOW_EXTRA_HIGHWAYS: bool = False
    EXTRA_HIGHWAY_CHANCE: float = 0.05
    R2_R3_CHANCE_SPLIT:   float = 0.7

    # blocks
    MIN_BLOCK_SPACING:    int   = 12
    MAX_BLOCK_SPACING:    int   = 24
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

    CITY_BLOCK_CHANCE = {
        "Residential": 0.25,
        "Office": 0.25,
        "Market": 0.2,
        "Leisure": 0.2,
        "Other": 0.1,
        "Empty": 0.0
    }

    AVAILABLE_DIRECTIONS = ["N", "S", "E", "W"]

    DIRECTION_VECTORS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
    DIRECTION_OPPOSITES = {"N": "S", "S": "N", "E": "W", "W": "E"}
    DIRECTION_TO_THE_RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}

    ROAD_LIKE_TYPES = {"R1", "R2", "R3", "Intersection", "HighwayEntrance", "HighwayExit", "BlockEntrance"}
    ROAD_LIKE_TYPES_WITHOUT_INTERSECTIONS = {"R1", "R2", "R3", "HighwayEntrance", "HighwayExit", "BlockEntrance"}
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
        "HighwayEntrance": "blue",
        "HighwayExit": "royalblue",
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

    RAIN_ENABLED: bool = False
    RAIN_SPEED_REDUCTION = 2
    RAIN_RADIUS_MIN = 50
    RAIN_RADIUS_MAX = 100
    RAIN_SPEED_MIN = 1
    RAIN_SPEED_MAX = 10
    RAIN_OCCURRENCES_MAX = 3
    RAIN_COOLDOWN = 86400
    RAIN_SPAWN_CHANCE = 0.0001
    RAIN_SPAWN_OFFSET: int = 10

    # VEHICLE SETTINGS

    VEHICLE_MIN_SPEED: int = 1
    VEHICLE_MAX_SPEED: int = 5

    VEHICLE_RESPECT_AWARENESS: bool = False
    VEHICLE_AWARENESS_RANGE: int = 10
    VEHICLE_AWARENESS_WIDTH: int = 3

    VEHICLE_OBSTACLE_PENALTY_VEHICLE = 1_000
    VEHICLE_OBSTACLE_PENALTY_STOP = 500

    VEHICLE_BASE_COLOR = "black"
    VEHICLE_PARKED_COLOR = "aliceblue"

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

    # SERVICE VEHICLE SETTINGS

    SERVICE_VEHICLE_BASE_COLOR = "darkolivegreen"
    SERVICE_VEHICLE_MAX_LOAD_FOOD: float = 50.0
    SERVICE_VEHICLE_MAX_LOAD_WASTE: float = 250.0
    SERVICE_VEHICLE_LOAD_TIME: int = 20

    # CITY FLOW SETTINGS

    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Zone & transition definitions
    # ──────────────────────────────────────────────────────────────────────────────

    # Helper to map the 2-letter codes in your spec to block_type strings
    ABBR = {
        "Res": "Residential",
        "Off": "Office",
        "Mar": "Market",
        "Lei": "Leisure",
        "Oth": "Other"
    }

    # Define each 3-hour zone with its through-prob and a dict of internal transitions
    TIME_ZONES = [
        {  # Zone 1: 06:00–09:00
            "start_hour": 6, "end_hour": 9,
            "through_distribution": 0.15,
            "internal_distribution": {
                ("Res", "Off"): 0.05,
                ("Res", "Mar"): 0.05,
                ("Res", "Lei"): 0.02,
                ("Res", "Oth"): 0.03,
            },
        },
        {  # Zone 2: 09:00–12:00
            "start_hour": 9, "end_hour": 12,
            "through_distribution": 0.20,
            "internal_distribution": {
                ("Res", "Mar"): 0.10,
                ("Res", "Oth"): 0.04,
                ("Off", "Oth"): 0.06,
            },
        },
        {  # Zone 3: 12:00–15:00
            "start_hour": 12, "end_hour": 15,
            "through_distribution": 0.15,
            "internal_distribution": {
                ("Res", "Mar"): 0.07,
                ("Res", "Oth"): 0.03,
                ("Off", "Oth"): 0.05,
            },
        },
        {  # Zone 4: 15:00–18:00
            "start_hour": 15, "end_hour": 18,
            "through_distribution": 0.15,
            "internal_distribution": {
                ("Res", "Mar"): 0.03,
                ("Off", "Oth"): 0.05,
                ("Mar", "Oth"): 0.05,
                ("Lei", "Oth"): 0.02,
            },
        },
        {  # Zone 5: 18:00–21:00
            "start_hour": 18, "end_hour": 21,
            "through_distribution": 0.12,
            "internal_distribution": {
                ("Res", "Oth"): 0.02,
                ("Res", "Lei"): 0.02,
                ("Off", "Lei"): 0.02,
                ("Mar", "Lei"): 0.02,
                ("Oth", "Lei"): 0.02,
                ("Mar", "Oth"): 0.01,
                ("Lei", "Oth"): 0.01,
            },
        },
        {  # Zone 6: 21:00–24:00
            "start_hour": 21, "end_hour": 24,
            "through_distribution": 0.10,
            "internal_distribution": {
                ("Off", "Res"): 0.03,
                ("Mar", "Res"): 0.03,
                ("Lei", "Res"): 0.02,
                ("Oth", "Res"): 0.02,
            },
        },
        {  # Zone 7: 00:00–03:00
            "start_hour": 0, "end_hour": 3,
            "through_distribution": 0.08,
            "internal_distribution": {
                ("Off", "Res"): 0.02,
                ("Lei", "Res"): 0.04,
                ("Oth", "Res"): 0.01,
                ("Res", "Lei"): 0.01,
            },
        },
        {  # Zone 8: 03:00–06:00
            "start_hour": 3, "end_hour": 6,
            "through_distribution": 0.05,
            "internal_distribution": {
                ("Res", "Mar"): 0.02,
                ("Res", "Lei"): 0.02,
                ("Res", "Oth"): 0.01,
            },
        },
    ]

    TIME_PER_STEP_IN_SECONDS = 6
    SIMULATION_STARTING_TIME_OF_DAY_HOURS = 6
    SIMULATION_STARTING_TIME_OF_DAY_MINUTES = 0

    INTERNAL_POPULATION_TRAFFIC_PER_DAY = 100000
    PASSING_POPULATION_TRAFFIC_PER_DAY = 24000
    TOTAL_SERVICE_VEHICLES_FOOD = 50
    TOTAL_SERVICE_VEHICLES_WASTE = 50
    INDIVIDUAL_SERVICE_VEHICLE_COOLDOWN = 3600

    # RECORDING

    SAVE_TOTAL_RESULTS: bool = True
    RESULTS_TOTAL_INTERVAL_UNIT: str = "minutes"
    RESULTS_TOTAL_INTERVAL_VALUE: int = 5
    SAVE_INDIVIDUAL_RESULTS: bool = True
    RESULTS_INDIVIDUAL_INTERVAL_UNIT: str = "minutes"
    RESULTS_INDIVIDUAL_INTERVAL_VALUE: int = 5

    # OPTIMIZATION AND DEBUGGING

    USE_DUMMY_AGENTS: bool = False
    CACHE_CELL_PORTRAYAL: bool = False
    ENABLE_AGENT_PORTRAYAL: bool = True
    ENABLE_TRAFFIC:bool = False
    USE_CUDA_PATHFINDING: bool = False

    CACHED_TYPES = [z for z in ZONES if z not in [
        "HighwayEntrance",
        "HighwayExit",
        "TrafficLight",
        "TrafficLightStop",
        "ControlledRoad",
        "ControlledRoadStop",
        "BlockEntrance"]]

    CHANGE_ASSIGNED_CELL_COLOR_ON_STOP: bool = False











