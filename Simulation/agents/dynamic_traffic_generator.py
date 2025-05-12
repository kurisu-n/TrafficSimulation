from __future__ import annotations

"""Dynamic traffic & service‑vehicle scheduler **with convenient time helpers**.

Changes in this revision
========================
*  **Helper properties** – quick access to:
   • ``day``  – simulation day index (int)
   • ``hour`` / ``minute`` / ``second``
   • ``time_of_day_seconds`` – seconds since midnight
   • ``elapsed_(seconds|minutes|hours|days)`` – wall‑clock since sim start
   • ``now_dhms()`` – tuple ``(day, hour, min, sec)`` for compact use.
*  **Population‑type tagging** – every *VehicleAgent* now gets a
   ``population_type`` string ("through" or "internal") so statistics can
   classify live agents.
*  Slight tidy‑ups of docstrings and imports.
"""

import random
from typing import TYPE_CHECKING, cast, Literal

from mesa import Agent

from Simulation.config import Defaults
from Simulation.agents.vehicles.vehicle_base import VehicleAgent
from Simulation.agents.vehicles.vehicle_service import ServiceVehicleAgent
from Simulation.agents.cell import CellAgent

if TYPE_CHECKING:  # avoid circular at runtime
    from Simulation.city_model import CityModel

# ──────────────────────────────────────────────────────────────────────
#  Helper dataclass for scheduled spawns
# ──────────────────────────────────────────────────────────────────────
class Trip:
    __slots__ = ("origin", "destination", "depart_secs", "kind")

    def __init__(
        self,
        origin: CellAgent,
        destination: CellAgent | None,
        depart_secs: float,
        kind: Literal[
            "internal",
            "through",
            "service_food",
            "service_waste",
        ],
    ) -> None:
        self.origin = origin
        self.destination = destination  # None for service vehicles – they pick a target themselves
        self.depart_secs = depart_secs
        self.kind = kind

    # nice repr for debugging / logging
    def __repr__(self):
        h = int(self.depart_secs // 3600) % 24
        m = int((self.depart_secs % 3600) // 60)
        s = int(self.depart_secs % 60)
        return f"<Trip {self.kind} @{h:02d}:{m:02d}:{s:02d}>"


# ──────────────────────────────────────────────────────────────────────
#  Main agent
# ──────────────────────────────────────────────────────────────────────
class DynamicTrafficAgent(Agent):
    """Generates **internal**, **through** *and* **service** traffic on‑the‑fly."""

    # ------------------------------------------------------------------
    def __init__(self, unique_id: str, model):
        super().__init__(unique_id, model)

        # — cached config —
        self.P_int = Defaults.INTERNAL_POPULATION_TRAFFIC_PER_DAY
        self.P_thr = Defaults.PASSING_POPULATION_TRAFFIC_PER_DAY

        # service fleets
        self.sv_food_ids = [f"SF_{i:03d}" for i in range(Defaults.TOTAL_SERVICE_VEHICLES_FOOD)]
        self.sv_waste_ids = [f"SW_{i:03d}" for i in range(Defaults.TOTAL_SERVICE_VEHICLES_WASTE)]
        self._sv_next_time: dict[str, float] = {vid: 0.0 for vid in (*self.sv_food_ids, *self.sv_waste_ids)}

        # constants
        self.dt = Defaults.TIME_PER_STEP_IN_SECONDS
        self.cooldown = Defaults.INDIVIDUAL_SERVICE_VEHICLE_COOLDOWN
        self.count_sv_as_through = getattr(Defaults, "SERVICE_VEHICLES_COUNT_AS_THROUGH", True)

        # simulation clock (seconds)
        self.start_offset = (
            Defaults.SIMULATION_STARTING_TIME_OF_DAY_HOURS * 3600
            + Defaults.SIMULATION_STARTING_TIME_OF_DAY_MINUTES * 60
        )
        self.elapsed = 0.0          # seconds since *simulation* start
        self.current_day = 0        # 0‑based day index
        self.pending: list[Trip] = []

        # first‑day generation
        self._generate_day(0)

    # ════════════════════════════════════════════════════════════════
    #  Public step
    # ════════════════════════════════════════════════════════════════
    def step(self):
        prev = self.elapsed
        self.elapsed += self.dt

        # —— day rollover ——
        total_secs = self.start_offset + self.elapsed
        new_day = int(total_secs // 86_400)
        if new_day > self.current_day:
            for d in range(self.current_day + 1, new_day + 1):
                self._generate_day(d)
            self.current_day = new_day

        # —— spawn trips in (prev, elapsed] ——
        to_spawn = [t for t in self.pending if prev < t.depart_secs <= self.elapsed]
        for trip in to_spawn:
            self._spawn(trip)
        self.pending = [t for t in self.pending if t not in to_spawn]

    # ════════════════════════════════════════════════════════════════
    #  Time helper properties
    # ════════════════════════════════════════════════════════════════
    @property
    def time_of_day_seconds(self) -> int:
        """Seconds since *midnight* of the current simulation day."""
        return int((self.start_offset + self.elapsed) % 86_400)

    @property
    def hour(self) -> int:          # 0‑23
        return self.time_of_day_seconds // 3_600

    @property
    def minute(self) -> int:        # 0‑59
        return (self.time_of_day_seconds % 3_600) // 60

    @property
    def second(self) -> int:        # 0‑59
        return self.time_of_day_seconds % 60

    @property
    def day(self) -> int:
        return self.current_day

    # —— elapsed wall‑clock helpers ——
    @property
    def elapsed_seconds(self) -> int:
        return int(self.elapsed)

    @property
    def elapsed_minutes(self) -> int:
        return int(self.elapsed // 60)

    @property
    def elapsed_hours(self) -> int:
        return int(self.elapsed // 3_600)

    @property
    def elapsed_days(self) -> int:
        return int(self.elapsed // 86_400)

    # convenience tuple
    def now_dhms(self) -> tuple[int, int, int, int]:
        """Return ``(day, hour, minute, second)``."""
        return self.day, self.hour, self.minute, self.second

    # ════════════════════════════════════════════════════════════════
    #  Trip generation helpers
    # ════════════════════════════════════════════════════════════════
    def _generate_day(self, day_idx: int):
        """Populate *pending* with all trips for *day_idx*."""
        city: CityModel = cast("CityModel", self.model)
        entrances = city.get_highway_entrances()
        exits = city.get_highway_exits()

        for zone in Defaults.TIME_ZONES:
            z0 = day_idx * 86_400 + zone["start_hour"] * 3_600 - self.start_offset
            z1 = day_idx * 86_400 + zone["end_hour"] * 3_600 - self.start_offset
            span = z1 - z0

            # — 1) INTERNAL —
            for (abbr_o, abbr_d), frac in zone["internal_distribution"].items():
                count = round(self.P_int * frac)
                if count == 0:
                    continue
                o_type = Defaults.ABBR[abbr_o]
                d_type = Defaults.ABBR[abbr_d]
                origins = city.get_city_blocks_by_type(o_type)
                dests = city.get_city_blocks_by_type(d_type)
                if not origins or not dests:
                    continue
                for _ in range(count):
                    t = z0 + random.random() * span
                    oblk = random.choice(origins)
                    dblk = random.choice(dests)
                    o_cell = random.choice(oblk.get_entrances())  # type: ignore[attr-defined]
                    d_cell = random.choice(dblk.get_entrances())  # type: ignore[attr-defined]
                    self.pending.append(Trip(o_cell, d_cell, t, "internal"))

            # — 2) SERVICE VEHICLES (Food & Waste) —
            sv_zone_share = zone["through_distribution"]
            sv_food_quota = round(len(self.sv_food_ids) * sv_zone_share)
            sv_waste_quota = round(len(self.sv_waste_ids) * sv_zone_share)

            def _schedule_sv(kind: Literal["service_food", "service_waste"]):
                pool = self.sv_food_ids if kind == "service_food" else self.sv_waste_ids
                for _ in range(10):  # up to 10 attempts to respect cooldown
                    vid = random.choice(pool)
                    t = z0 + random.random() * span
                    if t >= self._sv_next_time[vid]:
                        start = random.choice(entrances)
                        self.pending.append(Trip(start, None, t, kind))
                        self._sv_next_time[vid] = t + self.cooldown
                        return True
                return False

            for _ in range(sv_food_quota):
                _schedule_sv("service_food")
            for _ in range(sv_waste_quota):
                _schedule_sv("service_waste")

            # — 3) THROUGH —
            base_thr = round(self.P_thr * zone["through_distribution"])
            if self.count_sv_as_through:
                base_thr = max(0, base_thr - (sv_food_quota + sv_waste_quota))
            for _ in range(base_thr):
                t = z0 + random.random() * span
                ent = random.choice(entrances)
                ex = random.choice(exits)
                self.pending.append(Trip(ent, ex, t, "through"))

    # ════════════════════════════════════════════════════════════════
    #  Spawn helper
    # ════════════════════════════════════════════════════════════════
    def _spawn(self, trip: Trip):
        city: CityModel = cast("CityModel", self.model)

        if trip.kind in ("internal", "through"):
            vid = f"V_{int(trip.depart_secs):06d}_{random.randint(0, 9999):04d}"
            VehicleAgent(vid, city, trip.origin, cast(CellAgent, trip.destination), population_type=trip.kind)
            return

        # —— service vehicles ——
        pool = self.sv_food_ids if trip.kind == "service_food" else self.sv_waste_ids
        vid = random.choice(pool)
        sv_type = "Food" if trip.kind == "service_food" else "Waste"
        ServiceVehicleAgent(vid, city, trip.origin, sv_type)
