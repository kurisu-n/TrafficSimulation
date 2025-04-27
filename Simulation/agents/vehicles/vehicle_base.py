from mesa import Agent
import random
import heapq
from Simulation.config import Defaults
from Simulation.agents.cell import CellAgent
from Simulation.agents.dummy import DummyAgent

class VehicleAgent(Agent):
    """
    A vehicle agent that navigates from a start to a destination using A* pathfinding.
    """
    def __init__(self, unique_id, model, start_cell: CellAgent, target_cell: CellAgent):
        super().__init__(unique_id, model)
        self.start_cell = start_cell
        self.target = target_cell
        self.current_cell = start_cell
        # Base speed is randomly chosen within configured range
        self.base_speed = random.randint(Defaults.MIN_VEHICLE_SPEED, Defaults.MAX_VEHICLE_SPEED)
        self.current_speed = self.compute_speed()
        self.direction = None

        # Mark occupancy on the starting cell and place on the grid
        self.current_cell.occupied = True
        for a in self.model.grid.get_cell_list_contents([self.start_cell.get_position()]):
            if isinstance(a, DummyAgent):
                self.model.grid.remove_agent(a)
                self.model.schedule.remove(a)
                break

        self.model.grid.place_agent(self, self.current_cell.get_position())
        # Register with the model scheduler
        self.model.schedule.add(self)

        # Compute initial path
        self.path = self.compute_path()

    def compute_speed(self):
        """
        Compute current speed. Can be extended for dynamic speed changes.
        """
        return self.base_speed

    def compute_path(self):
        """
        Compute an A* path (list of CellAgent) from current position to target,
        avoiding occupied cells and vehicles in awareness range.
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
                # Build the CellAgent path
                full_path = []
                for pos in path + [current_pos]:
                    agent = self.model.grid.get_cell_list_contents([pos])[0]
                    full_path.append(agent)
                return full_path[1:]  # Exclude starting cell

            x, y = current_pos
            cell = self.model.grid.get_cell_list_contents([current_pos])[0]
            for d in cell.directions:
                dx, dy = Defaults.DIRECTION_VECTORS[d]
                neighbor_pos = (x + dx, y + dy)
                if not self.model.in_bounds(*neighbor_pos):
                    continue
                neighbor_cell = self.model.grid.get_cell_list_contents([neighbor_pos])[0]
                # Skip occupied cells
                if neighbor_cell.occupied:
                    continue
                # Skip if a vehicle is detected ahead
                if self._detect_vehicle_in_direction(d):
                    continue
                if neighbor_pos in closed:
                    continue
                heapq.heappush(open_set, (g + 1 + heuristic(neighbor_pos, goal), g + 1, neighbor_pos, path + [current_pos]))

        return []

    def _detect_vehicle_in_direction(self, direction):
        """
        Scan straight in the given direction up to awareness range
        for other vehicles.
        """
        dx, dy = Defaults.DIRECTION_VECTORS[direction]
        x, y = self.current_cell.get_position()

        for step in range(1, Defaults.VEHICLE_AWARENESS_RANGE + 1):
            nx, ny = x + dx * step, y + dy * step
            if not self.model.in_bounds(nx, ny):
                break
            for agent in self.model.grid.get_cell_list_contents([(nx, ny)]):
                if isinstance(agent, VehicleAgent):
                    return True
        return False

    def step(self):
        """
        Move along the path according to current speed, updating occupancy,
        swapping out DummyAgents when entering or leaving a cell.
        """
        # 1) Update speed
        self.current_speed = self.compute_speed()

        # 2) Remember the old position
        old_pos = self.current_cell.get_position()

        # 3) Walk up to current_speed steps
        for _ in range(self.current_speed):
            if not self.path:
                break
            next_cell = self.path[0]
            # stop if occupied
            if next_cell.occupied:
                break

            new_pos = next_cell.get_position()

            # 3a) Remove the DummyAgent at the destination cell (if any)
            for a in self.model.grid.get_cell_list_contents([new_pos]):
                if isinstance(a, DummyAgent):
                    self.model.grid.remove_agent(a)
                    self.model.schedule.remove(a)
                    break

            # 3b) Free the old cell and occupy the new one
            self.current_cell.occupied = False
            self.current_cell = next_cell
            self.current_cell.occupied = True

            # 3c) Physically move on the grid
            self.model.grid.move_agent(self, new_pos)

            # 3d) Update direction arrow
            dx = new_pos[0] - old_pos[0]
            dy = new_pos[1] - old_pos[1]
            # Invert your DIRECTION_VECTORS mapping for reverse lookup
            inv = {v: k for k, v in Defaults.DIRECTION_VECTORS.items()}
            self.direction = inv.get((dx, dy), self.direction)

            # 3e) Pop the step from the path
            self.path.pop(0)

            # 3f) Prepare for next iteration
            old_pos = new_pos

        # 4) After moving, respawn a DummyAgent at the last old_pos
        dummy = DummyAgent(f"Dummy_{old_pos[0]}_{old_pos[1]}", self.model, old_pos)
        self.model.grid.place_agent(dummy, old_pos)
        self.model.schedule.add(dummy)

        # 5) Optionally, recompute path for next tick
        self.path = self.compute_path()

    def get_current_direction(self):
        return self.direction

    def get_current_path(self):
        return [agent.get_position() for agent in self.path]

    def get_target(self):
        return self.target.get_position()

    def get_portrayal(self):
        # show a circle on layer 1, colored however you like
        return {
            "Shape": "circle",
            "r": 0.5,
            "Filled": True,
            "Layer": 1,
            "Color": "black",
            "Type": "Vehicle",
            "Direction": self.direction or "?",
        }
