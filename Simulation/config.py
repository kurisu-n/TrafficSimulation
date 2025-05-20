# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Defaults:
    # grid
    WIDTH:  int = 200
    HEIGHT: int = 200
    # frame
    WALL_THICKNESS:       int   = 15
    SIDEWALK_RING_WIDTH:  int   = 2

    ROADS = ["R1", "R2", "R3"]

    # road network
    RING_ROAD_TYPE:       str = "R2"
    HIGHWAY_OFFSET:       int   = 7

    R1_CHANCE_MEAN: float = 0.15
    R1_CHANCE_STD: float = 0.03
    R2_CHANCE_MEAN: float = 0.70
    R2_CHANCE_STD: float = 0.05
    MIN_R1_BANDS: int = 2

    # blocks
    BLOCK_ENTRANCE_ROAD_LEVEL: int = 1
    BLOCK_ENTRANCE_AVOID_TRAFFIC_LIGHTS = True
    MIN_BLOCK_SPACING:    int   = 12
    MAX_BLOCK_SPACING:    int   = 24

    # sub-blocks
    SUBBLOCK_CHANGE:      float = 0.2
    CARVE_SUBBLOCK_ROADS: bool  = True
    MIN_SUBBLOCK_SPACING: int   = 6
    SUBBLOCK_ROADS_HAVE_INTERSECTIONS: bool = False
    SUBBLOCK_ROAD_TYPE:   str = "R3"
    # control
    OPTIMISED_INTERSECTIONS:           bool = True
    TRAFFIC_LIGHT_RANGE:               int  = 10
    FORWARD_TRAFFIC_LIGHT_RANGE:         bool = True

    FORWARD_TRAFFIC_LIGHT_INTERSECTION_OPTIONS = ["Skip", "Include in Range", "Include as Extra"]
    FORWARD_TRAFFIC_LIGHT_INTERSECTIONS: str = FORWARD_TRAFFIC_LIGHT_INTERSECTION_OPTIONS[2]

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

    DIRECTION_VECTORS = {"N": (0, 1), "S": (0, -1), "W": (-1, 0), "E": (1, 0),}
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
        "IntersectionPending": "darkkhaki",
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

    # CITY FLOW SETTINGS

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

    INTERNAL_POPULATION_TRAFFIC_PER_DAY = 10000 * 1
    PASSING_POPULATION_TRAFFIC_PER_DAY = 2400  * 1
    TOTAL_SERVICE_VEHICLES_FOOD = 50
    TOTAL_SERVICE_VEHICLES_WASTE = 50
    INDIVIDUAL_SERVICE_VEHICLE_COOLDOWN = 3600

    # CITY RESOURCES

    FOOD_CAPACITY_PER_CELL = 2  # units per inner cell
    FOOD_CONSUMPTION_TICKS = 50  # ticks between consumption
    WASTE_CAPACITY_PER_CELL = 1.5  # units per inner cell
    WASTE_PRODUCTION_TICKS = 100  # ticks between production

    CITY_BLOCK_THAT_NEED_FOOD = ["Market", "Leisure"]
    CITY_BLOCK_THAT_PRODUCE_WASTE = AVAILABLE_CITY_BLOCKS

    GRADUAL_CITY_BLOCK_RESOURCES = True

    # WEATHER

    RAIN_ENABLED: bool = True
    RAIN_SPEED_REDUCTION = 2
    RAIN_RADIUS_MIN = 50
    RAIN_RADIUS_MAX = 100
    RAIN_SPEED_MIN = 1
    RAIN_SPEED_MAX = 10
    RAIN_OCCURRENCES_MAX = 3
    RAIN_COOLDOWN = 86400
    RAIN_SPAWN_CHANCE = 0.1
    RAIN_SPAWN_OFFSET: int = 10

    # VEHICLE SETTINGS

    VEHICLE_MIN_SPEED: int = 1
    VEHICLE_MAX_SPEED: int = 5

    VEHICLE_RESPECT_AWARENESS: bool = False
    VEHICLE_AWARENESS_RANGE: int = 10
    VEHICLE_AWARENESS_WIDTH: int = 3


    VEHICLE_ROAD_TYPES_PENALTIES_ENABLED = True
    VEHICLE_ROAD_TYPES_PENALTY_R1 = 1.0
    VEHICLE_ROAD_TYPES_PENALTY_R2 = 5.0
    VEHICLE_ROAD_TYPES_PENALTY_R3 = 10.0

    VEHICLE_TURN_PENALTY_ENABLED = True
    VEHICLE_TURN_PENALTY = 10

    VEHICLE_DYNAMIC_PENALTIES_ENABLED: bool = True
    VEHICLE_DYNAMIC_PENALTY_SCALE: float = 4.0

    VEHICLE_OBSTACLE_PENALTY_VEHICLE = 1_000
    VEHICLE_OBSTACLE_PENALTY_STOP = 500

    VEHICLE_BASE_COLOR = "black"
    VEHICLE_PARKED_COLOR = "aliceblue"

    VEHICLE_CONTRAFLOW_OVERTAKE_ACTIVE = True
    VEHICLE_CONTRAFLOW_PENALTY = 5_000
    VEHICLE_MAX_CONTRAFLOW_OVERTAKE_STEPS: int = 6
    VEHICLE_CONTRAFLOW_OVERTAKE_COLOR = "orange"
    VEHICLE_CONTRAFLOW_OVERTAKE_DURATION = 30

    VEHICLE_STUCK_RECOMPUTE_THRESHOLD = 30
    VEHICLE_STUCK_RECOMPUTE_THRESHOLD_INTERSECTION = 1

    VEHICLE_STUCK_CONTRAFLOW_ENABLED = True
    VEHICLE_STUCK_CONTRAFLOW_THRESHOLD = 60
    VEHICLE_STUCK_CONTRAFLOW_THRESHOLD_INTERSECTION = 10
    VEHICLE_MAX_CONTRAFLOW_STUCK_DETOUR_STEPS: int = 20
    VEHICLE_CONTRAFLOW_STUCK_DETOUR_DURATION = 10

    VEHICLE_STUCK_DESPAWN_ENABLED = True
    VEHICLE_STUCK_DESPAWN_THRESHOLD = 80
    VEHICLE_STUCK_DESPAWN_THRESHOLD_INTERSECTION = 20

    VEHICLE_MALFUNCTION_ACTIVE: bool = True
    VEHICLE_MALFUNCTION_CHANCE: float = 1E-7
    VEHICLE_MALFUNCTION_DURATION: int = 400
    VEHICLE_MALFUNCTION_COLOR = "yellow"

    VEHICLE_SIDESWIPE_COLLISION_ACTIVE: bool = True
    VEHICLE_SIDESWIPE_COLLISION_CHANCE: float = 1E-9
    VEHICLE_SIDESWIPE_COLLISION_DURATION: int = 600

    VEHICLE_COLLISION_COLOR = "red"

    # SERVICE VEHICLE SETTINGS

    SERVICE_VEHICLE_BASE_COLOR = "darkolivegreen"
    SERVICE_VEHICLE_MAX_LOAD_FOOD: float = 50.0
    SERVICE_VEHICLE_MAX_LOAD_WASTE: float = 250.0
    SERVICE_VEHICLE_LOAD_TIME: int = 20

    # LIGHT AGENT SETTINGS
    TRAFFIC_LIGHT_TRANSITION_DURATION_ENABLED: bool = False
    TRAFFIC_LIGHT_TRANSITION_CLEARANCE_ENABLED: bool = True

    TRAFFIC_LIGHT_AGENT_ALGORITHM = "NEIGHBOR_RL_BATCHED"
    # "DISABLED", "FIXED_TIME", "QUEUE_ACTUATED
    # "PRESSURE_CONTROL", "NEIGHBOR_PRESSURE_CONTROL"
    # "NEIGHBOR_GREEN_WAVE"
    # "NEIGHBOR_RL", "NEIGHBOR_RL_BATCHED"
    # "RL_A2C_BATCHED"
    # "GAT_DQN", "GAT_DQN_BATCHED"

    # Parameters for the all controllers
    TRAFFIC_LIGHT_YELLOW_DURATION = 3  # ticks before red
    TRAFFIC_LIGHT_ALL_RED_DURATION = 2
    TRAFFIC_LIGHT_CLEARANCE_MAX_DURATION = 15

    # Parameters for the FIXED_TIME controller
    TRAFFIC_LIGHT_GREEN_DURATION = 20  # ticks

    # Parameters for the QUEUE-ACTUATED controller
    TRAFFIC_LIGHT_QUEUE_ACTUATED_MIN_GREEN = 5
    TRAFFIC_LIGHT_QUEUE_ACTUATED_MAX_GREEN = 30
    TRAFFIC_LIGHT_QUEUE_ACTUATED_GAP = 3

    # Parameters for the PRESSURE_CONTROL controller
    TRAFFIC_LIGHT_PRESSURE_CONTROL_MIN_GREEN = 5

    # Parameters for Simple Reinforcement Learning
    SRL_HIDDEN_LAYERS = 30
    SRL_HIDDEN_LAYER_SIZE = 512
    SRL_LEARNING_RATE = 0.005
    SRL_UPDATE_EVERY = 32
    SRL_BATCH_SIZE = 512
    SRL_DROPOUT = 0.1

    # Parameters for the A2C controller
    A2C_HIDDEN_LAYERS = 20
    A2C_HIDDEN_LAYER_SIZE = 256

    A2C_TRAFFIC_RL_MAX_GREEN = 60

    A2C_GAMMA = 0.995  # discount
    A2C_LAMBDA = 0.95  # GAE(λ)
    A2C_UPDATE_EVERY = 32  # env steps before an update
    A2C_BATCH_SIZE = 256  # SGD minibatch
    A2C_ENTROPY_MAX = 0.01
    A2C_ENTROPY_MIN = 0.001
    A2C_ENTROPY_DECAY_STEPS = 500

    # Parameters for the GAT-DQN controller
    GAT_GAMMA = 0.99  # Discount factor for Q-learning
    GAT_BATCH_SIZE = 256  # Mini-batch size for training
    GAT_MEMORY_CAPACITY = 10000  # Replay memory capacity per agent
    GAT_TARGET_UPDATE_EVERY = 32  # Frequency (in training steps) to update target network
    EPS_INITIAL = 1.0  # Starting epsilon for ε-greedy
    EPS_MIN = 0.1  # Minimum epsilon (end of decay)
    EPS_DECAY_RATE = 1e-5  # Decay rate per decision step for epsilon

    GAT_TRAFFIC_RL_MIN_GREEN = 5



    # PATHFINDING

    PATHFINDING_METHOD = "NUMBA"
    # "NUMBA" "CYTHON"
    # "TENSORFLOW" "TENSORFLOW_VEC"
    PATHFINDING_COOLDOWN: int = 5
    PATHFINDING_CACHE: bool = True
    PATHFINDING_BATCHING: bool = True

    # TRAFFIC
    ENABLE_TRAFFIC:bool = True

    # RECORDING
    SAVE_TOTAL_RESULTS: bool = True
    RESULTS_TOTAL_INTERVAL_UNIT: str = "minutes"
    RESULTS_TOTAL_INTERVAL_VALUE: int = 30
    SAVE_INDIVIDUAL_RESULTS: bool = True
    RESULTS_INDIVIDUAL_INTERVAL_UNIT: str = "minutes"
    RESULTS_INDIVIDUAL_INTERVAL_VALUE: int = 60

    # STATISTICS
    SHOW_TIME_STATISTICS: bool = True
    SHOW_TRAFFIC_STATISTICS: bool = True
    SHOW_METRICS_STATISTICS: bool = True
    STATISTICS_UPDATE_INTERVAL: int = 100

    # OPTIMIZATION AND DEBUGGING

    USE_DUMMY_AGENTS: bool = False
    CACHE_CELL_PORTRAYAL: bool = True
    ENABLE_AGENT_PORTRAYAL: bool = False
    AGENT_PORTRAYAL_LEVEL: int = 2

    CUDA_GPU_ENABLED: bool = False

    CACHED_TYPES = [z for z in ZONES if z not in [
        "HighwayEntrance",
        "HighwayExit",
        "TrafficLight",
        "TrafficLightStop",
        "ControlledRoad",
        "ControlledRoadStop",
        "Intersection",
        "BlockEntrance"]]

    CHANGE_ASSIGNED_CELL_COLOR_ON_STOP: bool = False











