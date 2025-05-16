from __future__ import annotations

from typing import TYPE_CHECKING
from mesa.visualization.modules import TextElement
from Simulation.config import Defaults
from Simulation.agents.vehicles.vehicle_base import VehicleAgent  # Added import to track vehicle statuses

if TYPE_CHECKING:
    from Simulation.city_model import CityModel


class TrafficStatistics(TextElement):
    """Responsive, horizontally-spanning traffic dashboard."""

    def render(self, model: CityModel) -> str:
        # — find the traffic agent —
        dta = next(
            (ag for ag in model.schedule.agents
             if ag.__class__.__name__ == "DynamicTrafficAgent"),
            None
        )
        if dta is None:
            return '<div style="color:red">DynamicTrafficAgent not found</div>'

        # —— current simulation time & elapsed in HH:MM:SS ——
        day, hh, mm, ss = dta.now_dhms()
        total_secs = dta.elapsed_seconds
        tot_h = int(total_secs // 3600)
        tot_m = int((total_secs % 3600) // 60)
        tot_s_mod = int(total_secs % 60)
        elapsed_s = f"{tot_h:02d}:{tot_m:02d}:{tot_s_mod:02d}"

        # —— THROUGH traffic ——
        total_thr = dta.daily_total("through")
        created_thr = dta.created_count("through")
        pct_thr = dta.percentage_created("through")
        rem_thr = dta.remaining("through")
        live_thr = dta.live_count("through")
        completed_thr = dta.count_completed_through

        # —— INSIDE traffic ——
        total_int = dta.daily_total("internal")
        created_int = dta.created_count("internal")
        pct_int = dta.percentage_created("internal")
        rem_int = dta.remaining("internal")
        live_int = dta.live_count("internal")
        completed_int = dta.count_completed_internal

        # —— SERVICE: Food ——
        total_food = dta.daily_total("service_food")
        created_food = dta.created_count("service_food")
        rem_food = dta.remaining("service_food")
        live_food = dta.live_count("service_food")
        eta_food = dta.next_service_eta("service_food")

        # —— SERVICE: Waste ——
        total_waste = dta.daily_total("service_waste")
        created_waste = dta.created_count("service_waste")
        rem_waste = dta.remaining("service_waste")
        live_waste = dta.live_count("service_waste")
        eta_waste = dta.next_service_eta("service_waste")

        # —— VEHICLE STATUS COUNTS ——
        vehicles = [ag for ag in model.schedule.agents if isinstance(ag, VehicleAgent)]
        collisions = sum(1 for v in vehicles if v.is_in_collision)
        malfunctions = sum(1 for v in vehicles if v.is_in_malfunction)
        parked = sum(1 for v in vehicles if v.is_parked)
        overtaking = sum(1 for v in vehicles if v.is_overtaking)

        # —— helper to format any seconds → “HH:MM:SS” or “MM:SS” ——
        def fmt(secs):
            if secs is None or secs < 0:
                return "—"
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            s = int(secs % 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        eta_food_s = fmt(eta_food)
        eta_waste_s = fmt(eta_waste)

        # —— PERFORMANCE METRICS ——
        avg_dur_thr_s = fmt(dta.avg_duration_through)
        avg_dur_int_s = fmt(dta.avg_duration_internal)
        avg_time_unit_thr_s = fmt(dta.avg_time_per_unit_through)
        avg_time_unit_int_s = fmt(dta.avg_time_per_unit_internal)
        avg_daily_diff = dta.avg_daily_difference

        # —— render HTML/CSS ——
        return f"""
    <style>
      #ts-card {{
        background: #000;
        color: #fff;
        padding: 14px;
        border-radius: 8px;
        font-family: sans-serif;
      }}
      #ts-top-info {{
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
        align-items: center;
      }}
      .time-block, .elapsed-block {{
        background: #111;
        padding: 8px 12px;
        border: 1px solid #444;
        border-radius: 6px;
        color: #fff;
        flex: initial;
      }}
      .time-block {{
        font-size: 18px;
        font-weight: bold;
        display: flex;
        flex-direction: row;
        gap: 8px;
      }}
      .elapsed-block {{
        font-size: 18px;
        font-weight: bold;
      }}
      #stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px;
      }}
      .stat-category {{
        background: #111;
        padding: 10px;
        border: 1px solid #444;
        border-radius: 6px;
        word-break: break-word;
        width: 100%;
      }}
      .stat-category h4 {{
        margin: 0 0 6px;
        font-size: 16px;
      }}
      .stat-subtitle {{
        font-size: 14px;
        margin-bottom: 6px;
        color: #aaa;
      }}
      .stat-block {{
        font-size: 14px;
        margin: 4px 0;
      }}
      .service-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 6px;
      }}
    </style>

    <div id="ts-card">
      <h3>Traffic Statistics</h3>
      <div id="ts-top-info">
        <div class="time-block">
          <div>Day: {day}</div>
          <div>Hour: {hh:02d}</div>
          <div>Minute: {mm:02d}</div>
          <div>Second: {ss:02d}</div>
        </div>
        <div class="elapsed-block">Elapsed: {elapsed_s}</div>
      </div>

      <div id="stats-grid">

        <!-- THROUGH -->
        <div class="stat-category">
          <h4>Through Traffic</h4>
          <div class="stat-subtitle">Daily Total: {total_thr}</div>
          <div class="stat-block">Created: {created_thr} ({pct_thr:.1f}%)</div>
          <div class="stat-block">Remaining: {rem_thr}</div>
          <div class="stat-block">Completed: {completed_thr}</div>
          <div class="stat-block">Live: {live_thr}</div>
        </div>

        <!-- INSIDE -->
        <div class="stat-category">
          <h4>Inside Traffic</h4>
          <div class="stat-subtitle">Daily Total: {total_int}</div>
          <div class="stat-block">Created: {created_int} ({pct_int:.1f}%)</div>
          <div class="stat-block">Remaining: {rem_int}</div>
          <div class="stat-block">Completed: {completed_int}</div>
          <div class="stat-block">Live: {live_int}</div>
        </div>

        <!-- SERVICE VEHICLES -->
        <div class="stat-category">
          <h4>Service Vehicles</h4>
          <div class="service-grid">
            <div>
              <div class="stat-subtitle">Food Total: {total_food}</div>
              <div class="stat-block">Created: {created_food}</div>
              <div class="stat-block">Remaining: {rem_food}</div>
              <div class="stat-block">Active: {live_food}</div>
              <div class="stat-block">ETA: {eta_food_s}</div>
            </div>
            <div>
              <div class="stat-subtitle">Waste Total: {total_waste}</div>
              <div class="stat-block">Created: {created_waste}</div>
              <div class="stat-block">Remaining: {rem_waste}</div>
              <div class="stat-block">Active: {live_waste}</div>
              <div class="stat-block">ETA: {eta_waste_s}</div>
            </div>
          </div>
        </div>

        <!-- VEHICLE STATUSES -->
        <div class="stat-category">
          <h4>Vehicle Statuses</h4>
          <div class="stat-block">Collisions: {collisions}</div>
          <div class="stat-block">Malfunctions: {malfunctions}</div>
          <div class="stat-block">Parked: {parked}</div>
          <div class="stat-block">Overtaking: {overtaking}</div>
        </div>
      </div>

      <div id="stats-grid">
          <div class="stat-category">
            <h4>Through Statistics</h4>
            <div class="stat-block">Avg Through Duration: {avg_dur_thr_s}</div>
            <div class="stat-block">Avg Time/Unit Through: {avg_time_unit_thr_s}</div>
          </div>
          <div class="stat-category">
          <h4>Inside Statistics</h4>
            <div class="stat-block">Avg Inside Duration: {avg_dur_int_s}</div>
            <div class="stat-block">Avg Time/Unit Inside: {avg_time_unit_int_s}</div>
          </div> 
          <div class="stat-category">
          <h4>Daily Statistics</h4>
            <div class="stat-block">Avg Daily Difference: {avg_daily_diff:.1f}</div>
          </div> 
      </div>
    </div>
        """
