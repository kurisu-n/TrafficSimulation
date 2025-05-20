# vehicle_service.py

from mesa import Agent
from typing import Optional, TYPE_CHECKING, cast
from Simulation.config import Defaults
from Simulation.agents.vehicles.vehicle_base import VehicleAgent
from Simulation.agents.city_structure_entities.city_block import CityBlock
from Simulation.utilities.general import *

if TYPE_CHECKING:
    from Simulation.city_model import CityModel

class ServiceVehicleAgent(VehicleAgent):
    """
    A vehicle that carries either Food (starts full) or Waste (starts empty),
    services the highest‐need blocks in turn, and then exits via the closest highway.
    """

    def __init__(
        self,
        custom_id: str,
        model,
        start_cell,
        service_type: str,        # "Food" or "Waste"
        max_load: float = None,
    ):
        # pick initial load
        self.service_type = service_type
        self.max_load = max_load if max_load is not None else (Defaults.SERVICE_VEHICLE_MAX_LOAD_FOOD if self.service_type == "Food" else Defaults.SERVICE_VEHICLE_MAX_LOAD_WASTE)
        self.current_load = self.max_load if service_type == "Food" else 0.0
        self.city_model = cast("CityModel", model)

        target_cell = self._find_initial_target()

        # prevent base class from despawning when hitting our intermediate targets
        super().__init__(custom_id, model, start_cell, target_cell, population_type="through", vehicle_type=self.service_type.lower())
        self.remove_on_arrival = False
        # state machine
        self.service_ticks = 0

    def step(self):
        # if we're in the middle of a load/unload, count down
        if self.phase == "servicing":
            self.service_ticks -= 1
            if self.service_ticks <= 0:
                self._finish_service()
            return

        # otherwise do normal movement / A*
        super().step()

    def on_target_reached(self):
        # when we pull up to a block‐service point...
        if self.phase == "to_block":
            self._start_service()
        else:
            # must be heading for exit → let base class despawn us
            super().on_target_reached()

    def _find_initial_target(self):
        target_cell = None
        attempt = 0
        valid_blocks = self.city_model.get_blocks_needing_food() if self.service_type == "Food" else self.city_model.get_blocks_producing_waste()
        valid_blocks_count = len(valid_blocks)
        while target_cell is None and attempt < valid_blocks_count:
            self.current_block: Optional[CityBlock] = valid_blocks[attempt]
            # find the road‐cell to go to
            target_cell = (
                self.current_block.get_service_road_cell(self.city_model)
                if self.current_block
                else None
            )

        if target_cell is None:
            # no valid blocks found, so just go to the nearest highway exit
            self.current_block = None
            target_cell = self.city_model.get_highway_exits()[0]
            self.phase = "to_exit"
        else:
            self.phase = "to_block"

        return target_cell

    def _start_service(self):
        """Perform the actual load/unload and begin the wait timer."""
        # ─── mark as parked while we load/unload ───
        self._park()
        blk = self.current_block
        if self.service_type == "Food":
            need = blk.food_shortage()
            amt  = min(self.current_load, need)
            blk._add_food(amt)
            self.current_load -= amt
        else:  # Waste
            surplus = blk.waste_surplus()
            cap     = self.max_load - self.current_load
            amt     = min(cap, surplus)
            blk._remove_waste(amt)
            self.current_load += amt

        # now wait to simulate loading/unloading
        self.service_ticks = Defaults.SERVICE_VEHICLE_LOAD_TIME
        self.phase = "servicing"

    def _finish_service(self):
        """After unload/load delay, clear parked status then pick next block or head for exit."""
        # ─── un‐park so we can move again ───
        self._unpark()
        more = (
            (self.service_type == "Food"  and self.current_load > 0) or
            (self.service_type == "Waste" and self.current_load < self.max_load)
        )
        if more:
            # pick next highest‐need block
            next_blk = (
                self.city_model.get_block_most_in_need_of_food()
                if self.service_type == "Food"
                else self.city_model.get_block_most_in_need_of_waste_pickup()
            )
            if next_blk:
                self.current_block = next_blk
                next_cell = next_blk.get_service_road_cell(self.city_model)
                self.target = next_cell
                self.path = self._compute_path()
                self.phase = "to_block"
                return

        # otherwise, head for the nearest highway exit
        exits = self.city_model.get_highway_exits()
        # pick by Manhattan distance
        curr_x, curr_y = self.pos
        best = min(
            exits,
            key=lambda e: abs(e.get_position()[0] - curr_x)
                        + abs(e.get_position()[1] - curr_y)
        )
        self.target = best
        self.path = self._compute_path()
        self.remove_on_arrival = True
        self.phase = "to_exit"

    def _get_base_color(self):
        return Defaults.SERVICE_VEHICLE_BASE_COLOR

    def get_vehicle_type_name(self):
        return f"{self.service_type}ServiceVehicle"

    def get_description(self):
        return f"{self.service_type}-Service Vehicle ({self.current_load:.1f}/{self.max_load})"

    def get_portrayal(self):
        p = super().get_portrayal()
        p["Destination Block"] = (
            f"{self.current_block.id}, {self.current_block.block_type}"
            if self.current_block else "None"
        )
        return p

