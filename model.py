from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation  # or StagedActivation if needed
from mesa.datacollection import DataCollector

# Import agent classes from agents.py
from agents import VehicleAgent, TrafficLightAgent, ResidentAgent, LogisticsAgent

# Global model parameters (easy to tweak)
GRID_WIDTH = 20
GRID_HEIGHT = 20
INIT_NUM_VEHICLES = 10
INIT_NUM_LIGHTS = 4
TORUS = False  # non-wrapping grid (city has boundaries)
TICK_DURATION = 1.0  # (for conceptual use – how many "seconds" a tick represents)


class UrbanTrafficModel(Model):
    """The main model class for the urban traffic simulation."""

    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, num_vehicles=INIT_NUM_VEHICLES,
                 num_lights=INIT_NUM_LIGHTS):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=TORUS)
        self.schedule = RandomActivation(self)  # using random order activation
        self.step_count = 0

        # Set up road network (for simplicity, let's define a grid cross pattern of roads)
        # For example, vertical road at middle column, horizontal road at middle row.
        self.roads = set()
        mid_x = width // 2
        mid_y = height // 2
        for x in range(width):
            self.roads.add((x, mid_y))
        for y in range(height):
            self.roads.add((mid_x, y))
        # (All cells in self.roads set are drivable road cells; others are buildings or obstacles)

        # Create Traffic Light agents at the intersection of the cross (and maybe at other intersections)
        # In this simple pattern, the main intersection is at (mid_x, mid_y).
        # We can also place lights at other points, e.g., quarter intersections.
        light_positions = [(mid_x, mid_y)]  # center intersection
        # (Add more intersections if desired, e.g., at quarter points)
        # light_positions.append((mid_x, height//4))
        # light_positions.append((mid_x, 3*height//4))
        # ... (depending on how many traffic lights we want)

        for i, pos in enumerate(light_positions):
            light_agent = TrafficLightAgent(unique_id=f"TL{i}", model=self, state="GREEN", cycle_length=10)
            self.schedule.add(light_agent)
            self.grid.place_agent(light_agent, pos)

        # Create Vehicle agents and place them on random road cells
        for i in range(num_vehicles):
            vehicle = VehicleAgent(unique_id=i, model=self)
            self.schedule.add(vehicle)
            # Place vehicle on a random road position
            if self.roads:
                start_pos = self.random.choice(list(self.roads))
            else:
                # If no road network defined, place randomly anywhere
                start_pos = (self.random.randrange(width), self.random.randrange(height))
            self.grid.place_agent(vehicle, start_pos)

        # (Optional) create residents or logistics agents similarly:
        # for j in range(num_logistics):
        #     truck = LogisticsAgent(unique_id=f"Truck{j}", model=self)
        #     self.schedule.add(truck)
        #     self.grid.place_agent(truck, some_road_position)
        # for k in range(num_residents):
        #     res = ResidentAgent(unique_id=f"Res{k}", model=self)
        #     self.schedule.add(res)
        #     self.grid.place_agent(res, (some_x, some_y))  # maybe on non-road cells if representing houses

        # Initialize data collector for any metrics (if needed)
        self.datacollector = DataCollector({
            # Example metrics (to be computed in step or agent):
            # "AverageTravelTime": lambda m: m.compute_average_travel_time(),
            # "NumVehicles": lambda m: m.schedule.get_agent_count(),
        })

    def is_road(self, pos):
        """Check if a given position is a road (driveable cell)."""
        return pos in self.roads

    def step(self):
        """Advance the model by one tick."""
        self.schedule.step()
        self.datacollector.collect(self)
        self.step_count += 1
