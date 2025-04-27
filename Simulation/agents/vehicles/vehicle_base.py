from mesa import Agent
import random
import heapq
from Simulation.config import Defaults
from Simulation.agents.cell import CellAgent
from Simulation.agents.dummy import DummyAgent

class VehicleAgent(Agent):
    """
    A vehicle agent that navigates from a start to a destination using A* pathfinding.
    Refactored to isolate dummy-agent logic into helper methods for clarity.
    """
    def __init__(self, unique_id, model, start_cell: CellAgent, target_cell: CellAgent):
        super().__init__(unique_id, model)
        self.start_cell = start_cell
        self.target = target_cell
        self.current_cell = start_cell
        self.base_speed = random.randint(Defaults.MIN_VEHICLE_SPEED, Defaults.MAX_VEHICLE_SPEED)
        self.current_speed = self.compute_speed()
        self.direction = None

        # Occupy the start cell and remove dummy
        self._occupy_initial_cell()
        self.model.grid.place_agent(self, self.current_cell.get_position())
        self.model.schedule.add(self)
        self.path = self.compute_path()

    def compute_speed(self):
        """Compute current speed. Can be extended for dynamic speed changes."""
        return self.base_speed

    def compute_path(self):
        """
        Compute an A* path from current to target, avoiding occupied cells.
        """
        start = self.current_cell.get_position()
        goal = self.target.get_position()

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start, []))
        closed = set()

        while open_set:
            f, g, current_pos, path = heapq.heappop(open_set)
            if current_pos in closed:
                continue
            closed.add(current_pos)
            if current_pos == goal:
                return self._path_from_positions(path + [current_pos])[1:]

            for neighbor_pos in self._neighbor_positions(current_pos):
                if neighbor_pos in closed:
                    continue
                neighbor_cell = self.model.grid.get_cell_list_contents([neighbor_pos])[0]
                if neighbor_cell.occupied or self._detect_vehicle_in_direction(cell=self.model.grid.get_cell_list_contents([current_pos])[0], direction=None):
                    continue
                heapq.heappush(
                    open_set,
                    (g + 1 + heuristic(neighbor_pos, goal), g + 1, neighbor_pos, path + [current_pos])
                )
        return []

    def _path_from_positions(self, positions):
        """Convert list of grid positions to CellAgent list."""
        return [self.model.grid.get_cell_list_contents([pos])[0] for pos in positions]

    def _neighbor_positions(self, pos):
        """Yield valid neighbor positions based on cell directions."""
        x, y = pos
        cell = self.model.grid.get_cell_list_contents([pos])[0]
        for d in cell.directions:
            dx, dy = Defaults.DIRECTION_VECTORS[d]
            nx, ny = x + dx, y + dy
            if self.model.in_bounds(nx, ny):
                yield (nx, ny)

    def _detect_vehicle_in_direction(self, cell, direction):
        """
        Scan ahead up to awareness range for other vehicles.
        If direction is None, skip detection.
        """
        if direction is None:
            return False
        dx, dy = Defaults.DIRECTION_VECTORS[direction]
        x, y = cell.get_position()
        for step in range(1, Defaults.VEHICLE_AWARENESS_RANGE + 1):
            pos = (x + dx * step, y + dy * step)
            if not self.model.in_bounds(*pos):
                break
            for a in self.model.grid.get_cell_list_contents([pos]):
                if isinstance(a, VehicleAgent):
                    return True
        return False

    def step(self):
        """
        Perform movement: remove dummy at destination, move, then spawn dummy at vacated cell.
        """
        self.current_speed = self.compute_speed()
        old_pos = self.current_cell.get_position()

        for _ in range(self.current_speed):
            if not self.path:
                break
            next_cell = self.path[0]
            if next_cell.occupied:
                break
            new_pos = next_cell.get_position()

            self._remove_dummy(new_pos)
            self._move_to(next_cell, old_pos, new_pos)
            old_pos = new_pos
            self.path.pop(0)

        self._spawn_dummy(old_pos)
        self.path = self.compute_path()

    def _occupy_initial_cell(self):
        """Removed dummy and mark start cell occupied at initialization."""
        self.current_cell.occupied = True
        self._remove_dummy(self.current_cell.get_position())

    def _remove_dummy(self, pos):
        """Remove a DummyAgent at the given position if present."""
        for a in self.model.grid.get_cell_list_contents([pos]):
            if isinstance(a, DummyAgent):
                self.model.grid.remove_agent(a)
                self.model.schedule.remove(a)
                # clear pos so no warning on next place
                a.pos = None
                break

    def _spawn_dummy(self, pos):
        """Place a new DummyAgent at the vacated position."""
        dummy = DummyAgent(f"Dummy_{pos[0]}_{pos[1]}", self.model, pos)
        dummy.pos = None
        self.model.grid.place_agent(dummy, pos)
        self.model.schedule.add(dummy)

    def _move_to(self, cell, old_pos, new_pos):
        """Handle occupancy, grid move, and direction update."""
        self.current_cell.occupied = False
        self.current_cell = cell
        self.current_cell.occupied = True
        self.model.grid.move_agent(self, new_pos)
        self.direction = self._compute_direction(old_pos, new_pos)

    def _compute_direction(self, old_pos, new_pos):
        """Compute direction from old_pos to new_pos."""
        vector = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
        inv = {v: k for k, v in Defaults.DIRECTION_VECTORS.items()}
        return inv.get(vector, self.direction)

    def get_current_direction(self):
        return self.direction

    def get_current_path(self):
        return [agent.get_position() for agent in self.path]

    def get_target(self):
        return self.target.get_position()

    def get_portrayal(self):
        """Visual representation of the vehicle on layer 1."""
        return {
            "Shape": "circle",
            "r": 0.5,
            "Filled": True,
            "Layer": 1,
            "Color": "black",
            "Type": "Vehicle",
            "Direction": self.direction or "?",
        }
