from mesa import Agent

# Global constants (parameters) for agent behavior can be defined here or in model.py
# For example:
# VEHICLE_SPEED = 1  # cells per tick (for now, 1 cell move per step)
# LIGHT_CYCLE = 10   # ticks a traffic light stays green before switching

class VehicleAgent(Agent):
    """An agent representing a vehicle moving through the city."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # For simplicity, assign a random initial direction to the vehicle
        self.direction = self.random.choice(["N", "S", "E", "W"])
        # (In future, could add destination, speed, etc.)

    def step(self):
        # Basic movement behavior: attempt to move in current direction or random if blocked
        # (Placeholder logic for scaffold)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        # Filter to road cells if we have a road layout (to be implemented in model)
        valid_moves = [cell for cell in possible_steps if self.model.is_road(cell)]
        if valid_moves:
            new_position = self.random.choice(valid_moves)
            self.model.grid.move_agent(self, new_position)
        # In future, incorporate traffic light state (stop if red) and more logic.


class TrafficLightAgent(Agent):
    """An agent representing a traffic light at an intersection."""

    def __init__(self, unique_id, model, state="GREEN", cycle_length=10):
        super().__init__(unique_id, model)
        self.state = state  # Current state: "GREEN" or "RED"
        self.cycle_length = cycle_length  # how many ticks to stay in one state
        self.ticks = 0  # counter for ticks in current state

    def step(self):
        # Increment tick counter and switch state when cycle_length is reached
        self.ticks += 1
        if self.ticks >= self.cycle_length:
            # Toggle the light
            if self.state == "GREEN":
                self.state = "RED"
            else:
                self.state = "GREEN"
            self.ticks = 0
        # (Later, we can add logic to coordinate multiple lights or yellow phase, etc.)


# (Optional) Placeholder classes for future agents:
class ResidentAgent(Agent):
    """Agent representing a resident (could generate trips or move as pedestrian)."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Additional attributes like home location, work location, etc., could be added.

    def step(self):
        # For now, no behavior (could be used later to spawn VehicleAgents or similar).
        pass


class LogisticsAgent(Agent):
    """Agent for logistics (delivery trucks, etc.), subclassing VehicleAgent for now."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # This could share behavior with VehicleAgent or have a different pattern.
        self.direction = self.random.choice(["N", "S", "E", "W"])

    def step(self):
        # For now, behave like a VehicleAgent (move randomly).
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        valid_moves = [cell for cell in possible_steps if self.model.is_road(cell)]
        if valid_moves:
            self.model.grid.move_agent(self, self.random.choice(valid_moves))
