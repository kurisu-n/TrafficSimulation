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

_STAT_HEADERS = [
    # timeline index placeholder written at runtime ↓
    #       unit | tick | day …
    # averages
    "avg_duration_internal_completed", "avg_duration_through_completed",
    "avg_time_per_unit_internal_completed", "avg_time_per_unit_through_completed",
    "avg_duration_internal_live",  "avg_duration_through_live",
    "avg_time_per_unit_internal_live",  "avg_time_per_unit_through_live",
    "avg_duration_internal_total", "avg_duration_through_total",
    "avg_time_per_unit_internal_total", "avg_time_per_unit_through_total",
    "avg_daily_difference",

    "created_through", "remaining_through", "live_through", ""
    "created_internal",  "remaining_internal", "live_internal",
    "collisions", "malfunctions", "parked", "overtaking", "stuck", "in_stuck_detour"
]

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
        self.errored_internal = 0

        self.total_duration_through = 0.0
        self.count_completed_through = 0
        self.total_distance_through = 0.0
        self.errored_through = 0

        self.daily_finished_internal = 0
        self.daily_finished_through = 0
        self.daily_difference_history: list[int] = []

        self.live_internal = 0
        self.live_through  = 0
        self.live_service_food  = 0
        self.live_service_waste = 0

        self.collisions   = 0
        self.malfunctions = 0
        self.parked       = 0
        self.overtaking   = 0
        self.stuck        = 0
        self.in_stuck_detour     = 0

        self.cached_stats: dict[str, int | float | None] = {}

        self._ticks_since_last_stats = 0
        self._cached_stats_dirty = True

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

        self._ticks_since_last_stats += 1
        if self._ticks_since_last_stats >= Defaults.STATISTICS_UPDATE_INTERVAL:
            self._update_cached_stats()
            self._ticks_since_last_stats = 0
            self._cached_stats_dirty = False
        else:
            self._cached_stats_dirty = True

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
        return self.cached_stats.get(f"live_{kind}", 0)


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
        headers = [unit] + _STAT_HEADERS
        with open(self.snapshot_path, "w") as f:
            f.write(",".join(headers) + "\n")

    def _maybe_save_totals(self):
        """If due, overwrite the total‐statistics CSV."""
        if not getattr(self, "save_totals", False):
            return

        if self.elapsed >= self._next_total_snapshot:

            if self._cached_stats_dirty:
                self._update_cached_stats(True)
                self._cached_stats_dirty = False

            with open(self.totals_path, "w") as f:
                f.write(",".join(_STAT_HEADERS) + "\n")
                f.write(",".join(str(self.cached_stats.get(k, 0.0)) for k in _STAT_HEADERS) + "\n")
            self._next_total_snapshot += self._total_interval

    def _maybe_save_individual(self):
        """If due, append a new snapshot‐row to the same CSV."""
        if not getattr(self, "save_individual", False):
            return

        if self.elapsed >= self._next_indiv_snapshot:

            if self._cached_stats_dirty:
                self._update_cached_stats(True)
                self._cached_stats_dirty = False

            unit = Defaults.RESULTS_INDIVIDUAL_INTERVAL_UNIT
            secs_per_unit = {"hours": 3600, "minutes": 60, "seconds": 1}[unit]
            idx = int(self._next_indiv_snapshot / secs_per_unit)

            row = [str(idx)] + [str(self.cached_stats.get(k, 0.0)) for k in _STAT_HEADERS]
            with open(self.snapshot_path, "a") as f:
                f.write(",".join(row) + "\n")

            self._next_indiv_snapshot += self._indiv_interval


    def _update_cached_stats(self, force: bool = False):
        # ─────────────────────────────────────────────
        # ➊  Gather “live” vehicles  ➜  durations & dist
        # ─────────────────────────────────────────────
        city = cast("CityModel", self.model)
        live_internal = []
        live_through = []

        dur_live_int = dur_live_thr = 0
        dist_live_int = dist_live_thr = 0
        n_live_int = n_live_thr = 0
        average_stuck_duration = 0
        live_max_stuck_duration = 0

        for ag in city.schedule.agents:
            if isinstance(ag, VehicleAgent):
                if ag.population_type == "internal":
                    live_internal.append(ag)
                    dur_live_int += self.elapsed - ag.depart_time
                    dist_live_int += ag.steps_traveled
                    n_live_int += 1
                elif ag.population_type == "through":
                    live_through.append(ag)
                    dur_live_thr += self.elapsed - ag.depart_time
                    dist_live_thr += ag.steps_traveled
                    n_live_thr += 1

                if ag.is_stuck:
                    average_stuck_duration += ag.stuck_ticks
                    live_max_stuck_duration = max(live_max_stuck_duration, ag.stuck_ticks)


        self.live_average_stuck_duration = (average_stuck_duration / self.stuck) if (self.stuck > 0) else 0.0
        self.live_max_stuck_duration = live_max_stuck_duration

        # pre-existing, completed-only tallies
        dur_comp_int, dur_comp_thr = self.total_duration_internal, self.total_duration_through
        dst_comp_int, dst_comp_thr = self.total_distance_internal, self.total_distance_through
        n_comp_int, n_comp_thr = self.count_completed_internal, self.count_completed_through

        # ─────────────────────────────────────────────
        # ➋  Store every flavour side-by-side
        # ─────────────────────────────────────────────
        def _safe(dividend: float, divisor: float) -> float:
            return dividend / divisor if divisor else 0.0

        # ➋a  average duration  (ticks / vehicle)
        self.cached_stats |= {
            # completed only
            "avg_duration_internal_completed": _safe(dur_comp_int, n_comp_int),
            "avg_duration_through_completed": _safe(dur_comp_thr, n_comp_thr),
            # live only
            "avg_duration_internal_live": _safe(dur_live_int, n_live_int),
            "avg_duration_through_live": _safe(dur_live_thr, n_live_thr),
            # combined
            "avg_duration_internal_total": _safe(dur_comp_int + dur_live_int,
                                                 n_comp_int + n_live_int),
            "avg_duration_through_total": _safe(dur_comp_thr + dur_live_thr,
                                                n_comp_thr + n_live_thr),
        }

        # ➋b  average time per unit distance  (ticks / cell-step)
        self.cached_stats |= {
            # completed only
            "avg_time_per_unit_internal_completed": _safe(dur_comp_int, dst_comp_int),
            "avg_time_per_unit_through_completed": _safe(dur_comp_thr, dst_comp_thr),
            # live only
            "avg_time_per_unit_internal_live": _safe(dur_live_int, dist_live_int),
            "avg_time_per_unit_through_live": _safe(dur_live_thr, dist_live_thr),
            # combined
            "avg_time_per_unit_internal_total": _safe(dur_comp_int + dur_live_int,
                                                      dst_comp_int + dist_live_int),
            "avg_time_per_unit_through_total": _safe(dur_comp_thr + dur_live_thr,
                                                     dst_comp_thr + dist_live_thr),
        }

        # ➋c  (optional) keep old keys pointing at the combined figure
        self.cached_stats["avg_duration_internal"] = self.cached_stats["avg_duration_internal_total"]
        self.cached_stats["avg_duration_through"] = self.cached_stats["avg_duration_through_total"]
        self.cached_stats["avg_time_per_unit_internal"] = self.cached_stats["avg_time_per_unit_internal_total"]
        self.cached_stats["avg_time_per_unit_through"] = self.cached_stats["avg_time_per_unit_through_total"]


        self.cached_stats["avg_daily_difference"] = (
            sum(self.daily_difference_history) / len(self.daily_difference_history)
            if self.daily_difference_history else 0.0
        )


        self.cached_stats["count_completed_internal"] = self.count_completed_internal

        self.cached_stats["live_internal"] = self.live_internal
        self.cached_stats["live_through"] = self.live_through
        self.cached_stats["live_service_food"] = self.live_service_food
        self.cached_stats["live_service_waste"] = self.live_service_waste

        self.cached_stats["collisions"] = self.collisions
        self.cached_stats["malfunctions"] = self.malfunctions
        self.cached_stats["parked"] = self.parked
        self.cached_stats["overtaking"] = self.overtaking
        self.cached_stats["stuck"] = self.stuck
        self.cached_stats["live_average_stuck_duration"] = self.live_average_stuck_duration
        self.cached_stats["live_max_stuck_duration"] = self.live_max_stuck_duration
        self.cached_stats["in_stuck_detour"] = self.in_stuck_detour

        # Daily trip statistics (direct access — no method call overhead)
        for kind in ("internal", "through", "service_food", "service_waste"):
            total = (
                self.P_int if kind == "internal" else
                self.P_thr if kind == "through" else
                getattr(self, f"created_{kind}") + len(self.pending_trips_today(kind))
            )
            created = getattr(self, f"created_{kind}")
            remaining = total - created
            percentage = (created / total * 100) if total else 0.0
            errored = getattr(self, f"errored_{kind}",0.0)
            eta = self.next_service_eta(kind)

            self.cached_stats[f"daily_total_{kind}"] = total
            self.cached_stats[f"created_{kind}"] = created
            self.cached_stats[f"remaining_{kind}"] = remaining
            self.cached_stats[f"percentage_created_{kind}"] = percentage
            self.cached_stats[f"errored_{kind}"] = errored
            self.cached_stats[f"eta_{kind}"] = eta


