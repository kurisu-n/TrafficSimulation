from __future__ import annotations

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

    def get_service_road_cell(self, model) -> CellAgent | None:
        """
        Return one adjacent road cell off any block-entrance,
        rejecting cells occupied by parked vehicles.
        """
        for ent in self._entrances:  # list of BlockEntrance agents :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
            x_e, y_e = ent.get_position()
            # check each cardinal neighbor of the entrance
            for _, (dx, dy) in Defaults.DIRECTION_VECTORS.items():
                rx, ry = x_e + dx, y_e + dy
                if not model.in_bounds(rx, ry):
                    continue
                rcands = model.get_cell_contents(rx, ry)
                if not rcands:
                    continue
                road = rcands[0]
                if road.cell_type in Defaults.ROADS:
                    # reject if any parked VehicleAgent is here
                    has_parked = any(
                        isinstance(ag, VehicleAgent) and getattr(ag, "is_parked", False)
                        for ag in rcands[1:]
                    )  # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
                    if not has_parked:
                        return road
        return None


    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"<CityBlock {self.id} | {self.block_type} | "
            f"Food {self._food_units:.1f}/{self.max_food_units}, "
            f"Waste {self._waste_units:.1f}/{self.max_waste_units}>"
        )
