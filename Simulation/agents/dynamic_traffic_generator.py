from __future__ import annotations

import datetime
import math
import os
import random
from typing import TYPE_CHECKING, cast, Literal

from mesa import Agent

from Simulation.config import Defaults
from Simulation.agents.vehicles.vehicle_base import VehicleAgent
from Simulation.agents.vehicles.vehicle_service import ServiceVehicleAgent
from Simulation.agents.city_structure_entities.cell import CellAgent

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
        self.pending_by_day: dict[int, list[Trip]] = {}

        self.created_internal: int = 0
        self.created_through: int = 0
        self.created_service_food: int = 0
        self.created_service_waste: int = 0

        self.total_duration_internal = 0.0
        self.count_completed_internal = 0
        self.total_distance_internal = 0.0

        self.total_duration_through = 0.0
        self.count_completed_through = 0
        self.total_distance_through = 0.0

        self.daily_finished_internal = 0
        self.daily_finished_through = 0
        self.daily_difference_history: list[int] = []

        self._setup_results_dir()

        if Defaults.SAVE_TOTAL_RESULTS:
            self._initialize_total_saving()

        if Defaults.SAVE_INDIVIDUAL_RESULTS:
            self._initialize_individual_saving()

        self._generate_day(0)



    # ════════════════════════════════════════════════════════════════
    #  Public step
    # ════════════════════════════════════════════════════════════════
    def step(self):
        prev = self.elapsed
        self.elapsed += self.dt

        self._maybe_save_totals()
        self._maybe_save_individual()

        # —— day rollover ——
        total_secs = self.start_offset + self.elapsed
        new_day = int(total_secs // 86_400)
        if new_day > self.current_day:

            spawned = self.created_internal + self.created_through
            finished = self.daily_finished_internal + self.daily_finished_through
            self.daily_difference_history.append(finished - spawned)

            for d in range(self.current_day + 1, new_day + 1):
                self._generate_day(d)
            self.current_day = new_day

            # reset all counters at the start of the new day
            self.created_internal = 0
            self.created_through = 0
            self.created_service_food = 0
            self.created_service_waste = 0
            self.daily_finished_internal = 0
            self.daily_finished_through = 0

        # —— spawn trips in (prev, elapsed] ——
        to_spawn = [t for t in self.pending if prev < t.depart_secs <= self.elapsed]
        for trip in to_spawn:
            self._spawn(trip)
            self.pending.remove(trip)
            self.pending_by_day[self.current_day].remove(trip)

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

    @property
    def avg_duration_internal(self) -> float:
        return (self.total_duration_internal / self.count_completed_internal
                if self.count_completed_internal else 0.0)

    @property
    def avg_duration_through(self) -> float:
        return (self.total_duration_through / self.count_completed_through
                if self.count_completed_through else 0.0)

    @property
    def avg_time_per_unit_internal(self) -> float:
        return ((self.total_duration_internal / self.total_distance_internal)
                if self.total_distance_internal else 0.0)

    @property
    def avg_time_per_unit_through(self) -> float:
        return ((self.total_duration_through / self.total_distance_through)
                if self.total_distance_through else 0.0)

    @property
    def avg_daily_difference(self) -> float:
        if not self.daily_difference_history:
            return 0.0
        return sum(self.daily_difference_history) / len(self.daily_difference_history)

    # convenience tuple
    def now_dhms(self) -> tuple[int, int, int, int]:
        """Return ``(day, hour, minute, second)``."""
        return self.day, self.hour, self.minute, self.second

    # ────────── Traffic‐stats helper methods ──────────
    def pending_trips_today(self, kind: str | None = None) -> list[Trip]:
        trips = self.pending_by_day.get(self.current_day, [])
        if kind is None:
            return list(trips)  # return a copy if you’re going to mutate
        return [t for t in trips if t.kind == kind]

    def daily_total(self, kind: str) -> int:
        """Configured total for 'internal', 'through', or scheduled-today count for service types."""
        if kind == "internal":
            return self.P_int
        if kind == "through":
            return self.P_thr
        # for service, total = already created + still pending
        if kind in ("service_food", "service_waste"):
            created = getattr(self, f"created_{kind}")
            return created + len(self.pending_trips_today(kind))
        raise ValueError(f"Unknown kind: {kind}")

    def created_count(self, kind: str) -> int:
        """Number of trips of this kind spawned so far today."""
        if kind in ("internal", "through"):
            return getattr(self, f"created_{kind}")
        # service_food → created_service_food, service_waste → created_service_waste
        return getattr(self, f"created_{kind}")

    def remaining(self, kind: str) -> int:
        """How many more trips of this kind can spawn today."""
        return self.daily_total(kind) - self.created_count(kind)

    def percentage_created(self, kind: str) -> float:
        """What percent of the daily_total has been created so far."""
        total = self.daily_total(kind)
        return (self.created_count(kind) / total * 100) if total else 0.0

    def next_service_eta(self, kind: str) -> float | None:
        """
        Seconds until the next pending service trip of this kind
        (or None if none remain today).
        """
        future = [
            t.depart_secs - self.elapsed
            for t in self.pending_trips_today(kind)
            if t.depart_secs > self.elapsed
        ]
        return min(future) if future else None

    def live_count(self, kind: str) -> int:
        """
        Current active vehicles:
         - for "internal"/"through", counts VehicleAgent.population_type
         - for service types, counts ServiceVehicleAgent.service_type
        """
        agents = self.model.schedule.agents
        if kind in ("internal", "through"):
            from Simulation.agents.vehicles.vehicle_base import VehicleAgent
            return sum(
                1
                for ag in agents
                if isinstance(ag, VehicleAgent) and ag.population_type == kind
            )
        from Simulation.agents.vehicles.vehicle_service import ServiceVehicleAgent
        svc = "Food" if kind == "service_food" else "Waste"
        return sum(
            1
            for ag in agents
            if isinstance(ag, ServiceVehicleAgent) and ag.service_type == svc
        )


    def record_internal_trip(self, duration: float, distance: float):
        self.total_duration_internal += duration
        self.total_distance_internal += distance
        self.count_completed_internal += 1
        self.daily_finished_internal += 1


    def record_through_trip(self, duration: float, distance: float):
        self.total_duration_through += duration
        self.total_distance_through += distance
        self.count_completed_through += 1
        self.daily_finished_through += 1

    def _generate_day(self, day_idx: int):
        """Populate *pending* with all trips for *day_idx*, and index them by day."""
        city: CityModel = cast("CityModel", self.model)
        entrances = city.get_highway_entrances()  # list of CellAgent
        exits = city.get_highway_exits()  # list of CellAgent

        zones = Defaults.TIME_ZONES

        # collect only today's trips for quick lookup later
        day_trips: list[Trip] = []

        # 0) Pre-compute per-zone quotas so they sum exactly to total SV counts
        def compute_quotas(total: int, shares: list[float]) -> list[int]:
            float_counts = [total * s for s in shares]
            floors = [math.floor(x) for x in float_counts]
            rem = total - sum(floors)
            fracs = sorted(
                enumerate(float_counts),
                key=lambda iv: (iv[1] - math.floor(iv[1])),
                reverse=True
            )
            for i in range(rem):
                idx, _ = fracs[i]
                floors[idx] += 1
            return floors

        shares = [z["through_distribution"] for z in zones]
        food_total = len(self.sv_food_ids)
        waste_total = len(self.sv_waste_ids)
        food_quotas = compute_quotas(food_total, shares)
        waste_quotas = compute_quotas(waste_total, shares)

        # 1) Loop zones and spawn traffic
        for idx, zone in enumerate(zones):
            z0 = day_idx * 86_400 + zone["start_hour"] * 3_600 - self.start_offset
            z1 = day_idx * 86_400 + zone["end_hour"] * 3_600 - self.start_offset
            span = z1 - z0

            # — Internal traffic —
            for (abbr_o, abbr_d), frac in zone["internal_distribution"].items():
                cnt = round(self.P_int * frac)
                if cnt == 0:
                    continue
                o_type = Defaults.ABBR[abbr_o]
                d_type = Defaults.ABBR[abbr_d]
                origins = city.get_city_blocks_by_type(o_type)
                dests = city.get_city_blocks_by_type(d_type)
                if not origins or not dests:
                    continue
                for _ in range(cnt):
                    t = z0 + random.random() * span
                    oblk = random.choice(origins)
                    dblk = random.choice(dests)
                    o_cell = random.choice(oblk.get_entrances())
                    d_cell = random.choice(dblk.get_entrances())
                    trip = Trip(o_cell, d_cell, t, "internal")
                    self.pending.append(trip)
                    day_trips.append(trip)

            # — Service vehicles (uniform per zone) —
            Nf = food_quotas[idx]
            for j in range(1, Nf + 1):
                t = z0 + j * span / (Nf + 1)
                start_cell = random.choice(entrances)
                trip = Trip(start_cell, None, t, "service_food")
                self.pending.append(trip)
                day_trips.append(trip)

            Nw = waste_quotas[idx]
            for j in range(1, Nw + 1):
                t = z0 + j * span / (Nw + 1)
                start_cell = random.choice(entrances)
                trip = Trip(start_cell, None, t, "service_waste")
                self.pending.append(trip)
                day_trips.append(trip)

            # — Through traffic —
            thr = round(self.P_thr * zone["through_distribution"])
            if self.count_sv_as_through:
                thr = max(0, thr - (Nf + Nw))
            for _ in range(thr):
                t = z0 + random.random() * span
                ent_cell = random.choice(entrances)
                ex_cell = random.choice(exits)
                trip = Trip(ent_cell, ex_cell, t, "through")
                self.pending.append(trip)
                day_trips.append(trip)

        # 2) Index today's trips for O(1) lookup in pending_trips_today()
        self.pending_by_day[day_idx] = day_trips

    def _spawn(self, trip: Trip):
        city: CityModel = cast("CityModel", self.model)

        # Internal & Through traffic
        if trip.kind in ("internal", "through"):
            # increment before spawning
            if trip.kind == "internal":
                self.created_internal += 1
            else:
                self.created_through += 1

            vid = f"V_{int(trip.depart_secs):06d}_{random.randint(0, 9999):04d}"
            VehicleAgent(
                vid,
                city,
                trip.origin,
                cast(CellAgent, trip.destination),
                population_type=trip.kind
            )
            return

        # Service vehicles
        if trip.kind == "service_food":
            self.created_service_food += 1
            pool = self.sv_food_ids
            sv_type = "Food"
        else:  # service_waste
            self.created_service_waste += 1
            pool = self.sv_waste_ids
            sv_type = "Waste"

        vid = random.choice(pool)
        ServiceVehicleAgent(vid, city, trip.origin, sv_type)

    def _setup_results_dir(self):
        """Create ./Results/{run_ts}/ once if any saving is enabled."""
        if not (Defaults.SAVE_TOTAL_RESULTS or Defaults.SAVE_INDIVIDUAL_RESULTS):
            self.results_dir = None
            return
        base = os.path.join(os.getcwd(), "Results")
        os.makedirs(base, exist_ok=True)
        # 📌 Only call datetime.now() here, once per run
        self.run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(base, self.run_ts)
        os.makedirs(self.results_dir, exist_ok=True)

    def _initialize_total_saving(self):
        """Prepare the totals CSV and its schedule."""
        self.save_totals = True
        self.totals_path = os.path.join(
            self.results_dir,
            f"{self.run_ts}_total_statistics.csv"
        )
        # Write header
        with open(self.totals_path, "w") as f:
            f.write(
                "avg_duration_internal,avg_duration_through,"
                "avg_time_per_unit_internal,avg_time_per_unit_through,"
                "avg_daily_difference\n"
            )
        # Schedule first write
        unit = Defaults.RESULTS_TOTAL_INTERVAL_UNIT    # e.g. "hours"
        val  = Defaults.RESULTS_TOTAL_INTERVAL_VALUE   # e.g. 1
        secs_map = {"hours": 3600, "minutes": 60, "seconds": 1}
        self._total_interval = secs_map[unit] * val
        self._next_total_snapshot = self._total_interval

    def _initialize_individual_saving(self):
        """Set up the single snapshot‐CSV and its schedule."""
        self.save_individual = True
        # pull unit & value
        unit = Defaults.RESULTS_INDIVIDUAL_INTERVAL_UNIT    # e.g. "hours"
        val  = Defaults.RESULTS_INDIVIDUAL_INTERVAL_VALUE   # e.g. 2
        secs_map = {"hours": 3600, "minutes": 60, "seconds": 1}

        # how many seconds between snapshots
        self._indiv_interval = secs_map[unit] * val
        self._next_indiv_snapshot = self._indiv_interval

        # path for the one‐and‐only snapshot file
        self.snapshot_path = os.path.join(
            self.results_dir,
            f"{self.run_ts}_snapshot_statistics_{val}_{unit}.csv"
        )
        # write header: first column named after the unit
        headers = [unit,
                   "avg_duration_internal",
                   "avg_duration_through",
                   "avg_time_per_unit_internal",
                   "avg_time_per_unit_through",
                   "avg_daily_difference"]
        with open(self.snapshot_path, "w") as f:
            f.write(",".join(headers) + "\n")


    def _maybe_save_totals(self):
        """If due, overwrite the total‐statistics CSV."""
        if not getattr(self, "save_totals", False):
            return
        if self.elapsed >= self._next_total_snapshot:
            with open(self.totals_path, "w") as f:
                f.write(
                    "avg_duration_internal,avg_duration_through,"
                    "avg_time_per_unit_internal,avg_time_per_unit_through,"
                    "avg_daily_difference\n"
                )
                f.write(
                    f"{self.avg_duration_internal},"
                    f"{self.avg_duration_through},"
                    f"{self.avg_time_per_unit_internal},"
                    f"{self.avg_time_per_unit_through},"
                    f"{self.avg_daily_difference}\n"
                )
            self._next_total_snapshot += self._total_interval

    def _maybe_save_individual(self):
        """If due, append a new snapshot‐row to the same CSV."""
        if not getattr(self, "save_individual", False):
            return
        if self.elapsed >= self._next_indiv_snapshot:
            # compute unit‐value index (e.g. 2, 4, 6 for a 2-hour interval)
            # unit_secs is self._indiv_interval divided by interval value
            # so index = next_snapshot_time / unit_secs
            unit = Defaults.RESULTS_INDIVIDUAL_INTERVAL_UNIT
            secs_per_unit = {"hours":3600, "minutes":60, "seconds":1}[unit]
            idx = int(self._next_indiv_snapshot / secs_per_unit)

            # append row
            row = [
                str(idx),
                f"{self.avg_duration_internal}",
                f"{self.avg_duration_through}",
                f"{self.avg_time_per_unit_internal}",
                f"{self.avg_time_per_unit_through}",
                f"{self.avg_daily_difference}"
            ]
            with open(self.snapshot_path, "a") as f:
                f.write(",".join(row) + "\n")

            self._next_indiv_snapshot += self._indiv_interval
