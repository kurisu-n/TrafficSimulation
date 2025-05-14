# Simulation/agents/rain_agent.py
"""
Defines RainAgent (a moving rain cloud that tracks covered cells)
and RainManager (spawns RainAgents, enforces max/cooldown, and centrally sets cell rain flags by querying each cloud).
"""
import random
import math
from typing import TYPE_CHECKING, Set

from mesa import Agent
from Simulation.config import Defaults

if TYPE_CHECKING:
    from Simulation.city_model import CityModel
    from Simulation.agents.city_structure_entities.cell import CellAgent

class RainAgent(Agent):
    """
    A rain cloud that:
      - moves each step by a unit vector
      - tracks the set of CellAgents within its radius (covered_cells)
      - notifies manager on exit, then removes itself
    """
    def __init__(
        self,
        unique_id: str,
        city_model: "CityModel",
        pos: tuple[float, float],
        direction: tuple[float, float]
    ):
        super().__init__(unique_id, city_model)
        self.city_model = city_model

        # floating position and movement
        self.x, self.y = pos
        dx, dy = direction
        length = math.hypot(dx, dy) or 1.0
        self.dx, self.dy = dx / length, dy / length

        # rain radius and covered cell set
        self.radius = random.randint(Defaults.RAIN_RADIUS_MIN, Defaults.RAIN_RADIUS_MAX)
        self.covered_cells: Set[CellAgent] = set()
        # Precompute integer offsets within radius for efficient coverage
        self._offsets = [
            (dx, dy)
            for dx in range(-self.radius, self.radius + 1)
            for dy in range(-self.radius, self.radius + 1)
            if dx*dx + dy*dy <= self.radius*self.radius
        ]

        # off-map removal thresholds
        self._xmin = -self.radius
        self._xmax = city_model.get_width() + self.radius
        self._ymin = -self.radius
        self._ymax = city_model.get_height() + self.radius

    def step(self):
        # move cloud
        self.x += self.dx
        self.y += self.dy

        # update covered_cells using precomputed offsets
        self.covered_cells.clear()
        cx, cy = int(self.x), int(self.y)
        for dx, dy in self._offsets:
            xi, yi = cx + dx, cy + dy
            cell = self.city_model.cell_lookup.get((xi, yi))
            if cell:
                self.covered_cells.add(cell)

        # handle exit
        if (
            self.x < self._xmin or self.x > self._xmax or
            self.y < self._ymin or self.y > self._ymax
        ):
            manager = getattr(self.city_model, 'rain_manager', None)
            if manager:
                manager.on_rain_exit(self)
            # unregister
            self.city_model.rains.remove(self)
            self.city_model.schedule.remove(self)

class RainManager(Agent):
    """
    Controls rain lifecycle and sets cell rain flags by querying each RainAgent.
    """
    def __init__(self, unique_id: str, city_model: "CityModel"):
        super().__init__(unique_id, city_model)
        self.city_model = city_model
        self.counter = 0
        self.cooldown = 0  # in model steps
        # Track previously raining cells to avoid full-grid clears
        self._prev_raining: Set[CellAgent] = set()

    def add_random_rain(self):
        """Spawn one RainAgent just inside a random edge, heading toward a corner."""
        w = self.city_model.get_width()
        h = self.city_model.get_height()
        off = Defaults.RAIN_SPAWN_OFFSET

        # 1) pick which edge to spawn from
        edge = random.choice(['N', 'S', 'E', 'W'])

        # 2) compute spawn position just inside that edge
        if edge == 'N':
            x0 = random.uniform(0, w)
            y0 = h - off
            # opposite edge is South; lateral edges are E/W → corners SW or SE
            corner = random.choice(['SW', 'SE'])
        elif edge == 'S':
            x0 = random.uniform(0, w)
            y0 = off
            corner = random.choice(['NW', 'NE'])
        elif edge == 'E':
            x0 = w - off
            y0 = random.uniform(0, h)
            corner = random.choice(['NW', 'SW'])
        else:  # 'W'
            x0 = off
            y0 = random.uniform(0, h)
            corner = random.choice(['NE', 'SE'])

        # 3) figure out which map‐corner to aim at
        if corner == 'NW':
            xt, yt = 0, h
        elif corner == 'NE':
            xt, yt = w, h
        elif corner == 'SW':
            xt, yt = 0, 0
        else:  # SE
            xt, yt = w, 0

        # 4) normalize that vector
        dx = xt - x0
        dy = yt - y0
        length = math.hypot(dx, dy) or 1.0
        direction = (dx/length, dy/length)

        rain = RainAgent(f"Rain_{self.counter}", self.city_model, (x0, y0), direction)
        self.city_model.schedule.add(rain)
        self.city_model.rains.append(rain)
        self.counter += 1

    def on_rain_exit(self, rain_agent: RainAgent):
        """Called when a RainAgent fully exits the map"""
        # start cooldown immediately when last cloud leaves
        if not self.city_model.rains:
            self.cooldown = Defaults.RAIN_COOLDOWN // Defaults.TIME_PER_STEP_IN_SECONDS

    def step(self):
        # 1) clear rain flag on only previously raining cells
        for cell in self._prev_raining:
            cell.is_raining = False

        # 2) decrement cooldown
        if self.cooldown > 0:
            self.cooldown -= 1

        # 3) automatic spawn of new rains
        if (
            len(self.city_model.rains) < Defaults.RAIN_OCCURRENCES_MAX and
            self.cooldown == 0 and
            random.random() < Defaults.RAIN_SPAWN_CHANCE
        ):
            self.add_random_rain()

        # 4) set new raining cells
        new_raining: Set[CellAgent] = set()
        for rain in self.city_model.rains:
            new_raining |= rain.covered_cells
        for cell in new_raining:
            cell.is_raining = True

        # 5) remember for next tick
        self._prev_raining = new_raining

