from __future__ import annotations

import random
from typing import List, Sequence, TYPE_CHECKING

from mesa import Agent

from Simulation.agents.vehicles.vehicle_base import VehicleAgent
from Simulation.config import Defaults
from Simulation.agents.city_structure_entities.cell import CellAgent
from Simulation.utilities.general import *


class CityBlock(Agent):
    """A whole city block (Residential, Office, …) that tracks Food & Waste.

    ── **Two resource modes** ────────────────────────────────────────
    ``self.gradual_resources`` (bool) decides **how** the block’s resources
    change:

    * **True → Gradual mode** – every tick a fractional amount is applied
      (size/TICKS).  Remainders accumulate so no units are lost.
    * **False → Burst mode** – the legacy behaviour: one big update every
      *N* ticks.
    """

    # ------------------------------------------------------------------
    # Capacity helpers – scale with number of inner cells
    # ------------------------------------------------------------------
    @staticmethod
    def max_food_capacity(cells: int) -> float:
        return cells * Defaults.FOOD_CAPACITY_PER_CELL

    @staticmethod
    def max_waste_capacity(cells: int) -> float:
        return cells * Defaults.WASTE_CAPACITY_PER_CELL

    # ------------------------------------------------------------------
    def __init__(
        self,
        custom_id: str,
        model,
        block_type: str,
        inner_blocks: Sequence[CellAgent],
        sidewalks: Sequence[CellAgent],
        entrances: Sequence[CellAgent],
        gradual_resources: bool = False,
    ) -> None:
        super().__init__(str_to_unique_int(custom_id), model)
        self.id = custom_id
        self.block_type = block_type
        self._inner_blocks: List[CellAgent] = list(inner_blocks)
        self._sidewalks: List[CellAgent] = list(sidewalks)
        self._entrances: List[CellAgent] = list(entrances)

        # external flag (bool) requested by the user
        self.gradual_resources: bool = gradual_resources

        # ── capacities & current stock ───────────────────────────────
        self.max_food_units  = self.max_food_capacity(len(self._inner_blocks))
        self.max_waste_units = self.max_waste_capacity(len(self._inner_blocks))
        self._food_units:  float = self.max_food_units   # start full
        self._waste_units: float = 0.0                  # start empty

        # ── internal state for both modes ───────────────────────────
        # Burst mode counters
        self._ticks_since_food  = 0
        self._ticks_since_waste = 0

        # Gradual mode rates & fractional remainders
        self._food_rate_per_tick  = len(self._inner_blocks) / Defaults.FOOD_CONSUMPTION_TICKS
        self._waste_rate_per_tick = len(self._inner_blocks) / Defaults.WASTE_PRODUCTION_TICKS
        self._food_remainder  = 0.0  # fractional units carried over
        self._waste_remainder = 0.0

    # ------------------------------------------------------------------
    # Convenience predicates & metrics
    # ------------------------------------------------------------------
    def needs_food(self) -> bool:
        """Return *True* if this block type is listed as food‑dependent."""
        return self.block_type in Defaults.CITY_BLOCK_THAT_NEED_FOOD

    def produces_waste(self) -> bool:
        """Return *True* if this block type produces waste by default."""
        return self.block_type in Defaults.CITY_BLOCK_THAT_PRODUCE_WASTE

    # Numeric helpers frequently used for sorting/ranking
    def food_shortage(self) -> float:      # higher ⇒ more urgent
        return self.max_food_units - self._food_units

    def waste_surplus(self) -> float:      # higher ⇒ more urgent
        return self._waste_units

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------
    def get_inner_blocks(self): return self._inner_blocks
    def get_sidewalks(self):    return self._sidewalks
    def get_entrances(self):    return self._entrances
    def get_food_units(self):   return self._food_units
    def get_waste_units(self):  return self._waste_units

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_food(self, n: float):     self._food_units  = min(self._food_units + n, self.max_food_units)
    def _consume_food(self, n: float): self._food_units  = max(self._food_units - n, 0.0)
    def _add_waste(self, n: float):    self._waste_units = min(self._waste_units + n, self.max_waste_units)
    def _remove_waste(self, n: float): self._waste_units = max(self._waste_units - n, 0.0)

    # ------------------------------------------------------------------
    # Update logic (called every tick)
    # ------------------------------------------------------------------
    def _update_food(self):
        if not self.needs_food():
            return

        if self.gradual_resources:
            # accumulate fractional amount
            self._food_remainder += self._food_rate_per_tick
            if self._food_remainder >= 1.0:
                whole = int(self._food_remainder)
                self._consume_food(whole)
                self._food_remainder -= whole
        else:
            self._ticks_since_food += 1
            if self._ticks_since_food >= Defaults.FOOD_CONSUMPTION_TICKS:
                self._consume_food(len(self._inner_blocks))
                self._ticks_since_food = 0

    def _update_waste(self):
        if not self.produces_waste():
            return

        if self.gradual_resources:
            self._waste_remainder += self._waste_rate_per_tick
            if self._waste_remainder >= 1.0:
                whole = int(self._waste_remainder)
                self._add_waste(whole)
                self._waste_remainder -= whole
        else:
            self._ticks_since_waste += 1
            if self._ticks_since_waste >= Defaults.WASTE_PRODUCTION_TICKS:
                self._add_waste(len(self._inner_blocks))
                self._ticks_since_waste = 0

    # ------------------------------------------------------------------
    def step(self):
        self._update_food()
        self._update_waste()

    def get_service_road_cell(self, model) -> "CellAgent | None":
        """
        Return the nearest *free* road-cell that

        • touches this block’s sidewalk ring,
        • is *not* the road directly in front of any entrance, and
        • has a cell_type that belongs to ``Defaults.ROADS``.

        When every candidate is busy the method returns ``None``.
        """
        # 1) collect every road cell adjacent to *any* sidewalk tile
        sidewalk_coords = [sw.get_position() for sw in self._sidewalks]
        candidates: set[tuple[int, int]] = set()
        for sx, sy in sidewalk_coords:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rx, ry = sx + dx, sy + dy
                if not model.in_bounds(rx, ry):
                    continue
                agents = model.get_cell_contents(rx, ry)
                if agents and agents[0].cell_type in Defaults.ROADS:
                    candidates.add((rx, ry))

        if not candidates:
            return None

        # 2) remove the road directly in front of each entrance
        entrance_coords = [e.get_position() for e in self._entrances]
        for ex, ey in entrance_coords:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidates.discard((ex + dx, ey + dy))

        if not candidates:
            return None

        # 3) rank by shortest distance to the nearest entrance
        ranked = sorted(
            candidates,
            key=lambda rc: min(abs(rc[0] - ex) + abs(rc[1] - ey)
                               for ex, ey in entrance_coords)
        )

        # 4) return the first spot not occupied by a parked vehicle
        from Simulation.agents.vehicles.vehicle_base import VehicleAgent  # avoid circular import
        for rx, ry in ranked:
            agents = model.get_cell_contents(rx, ry)
            if any(isinstance(a, VehicleAgent) and getattr(a, "is_parked", False)
                   for a in agents[1:]):
                continue
            return agents[0]          # the road CellAgent itself

        return None


    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"<CityBlock {self.id} | {self.block_type} | "
            f"Food {self._food_units:.1f}/{self.max_food_units}, "
            f"Waste {self._waste_units:.1f}/{self.max_waste_units}>"
        )
