from __future__ import annotations

from typing import List, Sequence, TYPE_CHECKING

from mesa import Agent
from Simulation.config import Defaults
from Simulation.agents.cell import CellAgent
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

    def get_service_road_cell(self, model) -> CellAgent | None:
        """
        Starting from each BlockEntrance, alternate scanning its adjacent
        sidewalk cells (that belong to this block) until you find a
        neighboring road cell of type R1/R2/R3 that isn’t occupied.
        Returns that road CellAgent, or None if none is available.
        """
        from Simulation.config import Defaults
        from Simulation.agents.cell import CellAgent

        for ent in self._entrances:
            x0, y0 = ent.get_position()
            # collect only the sidewalks that were registered for this block
            sidewalks: list[CellAgent] = []
            for d, (dx, dy) in Defaults.DIRECTION_VECTORS.items():
                nx, ny = x0 + dx, y0 + dy
                if not model.in_bounds(nx, ny):
                    continue
                nbrs = model.get_cell_contents(nx, ny)
                if not nbrs:
                    continue
                cell = nbrs[0]
                if cell.cell_type == "Sidewalk" and cell in self._sidewalks:
                    sidewalks.append(cell)

            # alternate from start/end of the list
            i, j, toggle = 0, len(sidewalks) - 1, True
            while i <= j:
                sw = sidewalks[i] if toggle else sidewalks[j]

                # look for *one* adjacent road cell off this sidewalk
                for d2, (dx2, dy2) in Defaults.DIRECTION_VECTORS.items():
                    rx, ry = sw.get_position()[0] + dx2, sw.get_position()[1] + dy2
                    if not model.in_bounds(rx, ry):
                        continue
                    rcands = model.get_cell_contents(rx, ry)
                    if not rcands:
                        continue
                    road = rcands[0]
                    # only basic roads (R1/R2/R3) and must be free
                    if road.cell_type in Defaults.ROADS and not getattr(road, "occupied", False):
                        return road

                # advance pointers & flip
                if toggle:
                    i += 1
                else:
                    j -= 1
                toggle = not toggle

        # no valid service road found around any entrance
        return None


    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"<CityBlock {self.id} | {self.block_type} | "
            f"Food {self._food_units:.1f}/{self.max_food_units}, "
            f"Waste {self._waste_units:.1f}/{self.max_waste_units}>"
        )
