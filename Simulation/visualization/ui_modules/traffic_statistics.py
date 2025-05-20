from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING
from mesa.visualization.modules import TextElement
from Simulation.config import Defaults
from Simulation.agents.vehicles.vehicle_base import VehicleAgent

if TYPE_CHECKING:
    from Simulation.city_model import CityModel


class TrafficStatistics(TextElement):
    """Responsive, horizontally-spanning traffic dashboard."""

    def __init__(self):
        super().__init__()
        # record the real wall-clock start time
        self._real_start = time.time()
        # prepare to track per-tick durations
        self._last_render = self._real_start
        self._last_tick_duration: float | None = None
        # exponential smoothing for tick and delta durations
        self._smoothed_tick_ms: float | None = None
        self._smoothed_delta_ms: float | None = None
        self._tick_alpha = 0.1  # smoothing factor for tick and delta
        self._last_tick_time = None
        self._tick_history = deque(maxlen=200)

    def render(self, model: CityModel) -> str:
        # measure real tick duration
        now = time.time()
        current_tick = now - self._last_render if self._last_render is not None else 0.0
        self._last_render = now

        # store the raw tick time
        self._tick_history.append(current_tick)

        # average tick duration
        avg_tick = sum(self._tick_history) / len(self._tick_history)
        smooth_tick_ms = avg_tick * 1000

        # compute deltas between ticks
        if len(self._tick_history) >= 2:
            deltas = [self._tick_history[i] - self._tick_history[i - 1] for i in range(1, len(self._tick_history))]
            avg_delta = sum(deltas) / len(deltas)
        else:
            avg_delta = 0.0

        smooth_delta_ms = avg_delta * 1000

        # formatting
        smooth_tick_str = f"{smooth_tick_ms:.1f}ms"
        smooth_delta_str = f"{smooth_delta_ms:+.1f}ms"

        # find dynamic traffic agent
        dta = next((ag for ag in model.schedule.agents
                    if ag.__class__.__name__ == "DynamicTrafficAgent"), None)
        if dta is None:
            return '<div style="color:red">DynamicTrafficAgent not found</div>'

        # helper for formatting
        def fmt_time(secs: float | None) -> str:
            if secs is None or secs < 0:
                return "—"
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            s = int(secs % 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        # conditional sections
        time_section = ''
        if Defaults.SHOW_TIME_STATISTICS:
            day, hh, mm, ss = dta.now_dhms()
            sim_elapsed = fmt_time(dta.elapsed_seconds)
            real_elapsed = fmt_time(time.time() - self._real_start)
            ticks_per_second = 1.0 / current_tick if current_tick > 0 else 0
            ratio_real_to_sim = (time.time() - self._real_start) / max(1.0, dta.elapsed_seconds)
            time_section = f'''
              <div id="ts-top-info">
                <div class="top-info-cell day-hour-block">
                  <div class="label">DAY</div>
                  <div class="label">HOUR</div>
                  <div class="label">MIN</div>
                  <div class="label">SEC</div>
                  <div class="value">{day:02d}</div>
                  <div class="value">{hh:02d}</div>
                  <div class="value">{mm:02d}</div>
                  <div class="value">{ss:02d}</div>
                </div>

                <div class="top-info-cell">
                  <div class="label">Sim Time</div>
                  <div class="value">{sim_elapsed}</div>
                </div>
                <div class="top-info-cell">
                  <div class="label">Real Time</div>
                  <div class="value">{real_elapsed}</div>
                </div>
                <div class="top-info-cell">
                  <div class="label">Tick Time</div>
                  <div class="value">{smooth_tick_str}</div>
                </div>
                <div class="top-info-cell">
                  <div class="label">Δ/tick</div>
                  <div class="value">{smooth_delta_str}</div>
                </div>
                <div class="top-info-cell">
                  <div class="label">Ticks/sec</div>
                  <div class="value">{ticks_per_second:.1f}</div>
                </div>
                <div class="top-info-cell">
                  <div class="label">Real/Sim Ratio</div>
                  <div class="value">{ratio_real_to_sim:.2f}</div>
                </div>
              </div>'''

        traffic_section = ''
        if Defaults.SHOW_TRAFFIC_STATISTICS:
            # compute metrics
            total_thr = dta.cached_stats.get("daily_total_through", 0)
            created_thr = dta.cached_stats.get("created_through", 0)
            pct_thr = dta.cached_stats.get("percentage_created_through", 0.0)
            rem_thr = dta.cached_stats.get("remaining_through", 0)
            live_thr = dta.live_count("through")
            errored_thr = dta.cached_stats.get("errored_through", 0)

            completed_thr = dta.count_completed_through
            total_int = dta.cached_stats.get("daily_total_internal", 0)
            created_int = dta.cached_stats.get("created_internal", 0)
            pct_int = dta.cached_stats.get("percentage_created_internal", 0.0)
            rem_int = dta.cached_stats.get("remaining_internal", 0)
            live_int = dta.live_count("internal")
            errored_int = dta.cached_stats.get("errored_internal", 0)
            completed_int = dta.cached_stats.get("count_completed_internal", 0)

            # Service: Food
            total_food = dta.cached_stats.get("daily_total_service_food", 0)
            created_food = dta.cached_stats.get("created_service_food", 0)
            rem_food = dta.cached_stats.get("remaining_service_food", 0)
            live_food = dta.cached_stats.get("live_service_food", 0)  # or use live_count("service_food")

            eta_food = fmt_time(dta.cached_stats.get("eta_service_food", 0.0))

            # Service: Waste
            total_waste = dta.cached_stats.get("daily_total_service_waste", 0)
            created_waste = dta.cached_stats.get("created_service_waste", 0)
            rem_waste = dta.cached_stats.get("remaining_service_waste", 0)
            live_waste = dta.cached_stats.get("live_service_waste", 0)  # or use live_count("service_waste")
            eta_waste = fmt_time(dta.cached_stats.get("eta_service_waste", 0.0))

            collisions = dta.cached_stats.get("collisions", 0)
            malfunctions = dta.cached_stats.get("malfunctions", 0)
            parked = dta.cached_stats.get("parked", 0)
            overtaking = dta.cached_stats.get("overtaking", 0)
            stuck = dta.cached_stats.get("stuck", 0)
            live_average_stuck_duration = dta.cached_stats.get("live_average_stuck_duration", 0.0)
            live_max_stuck_duration = dta.cached_stats.get("max_stuck_duration", 0.0)
            stuck_detouring = dta.cached_stats.get("stuck_detour", 0)

            traffic_section = f'''
            <div id="stats-grid">
              <div class="stat-category">
                <h4>Through Traffic</h4>
                <div class="stat-block">Daily Total {total_thr}</div>
                <div class="stat-block">Created {created_thr} ({pct_thr:.1f}%)</div>
                <div class="stat-block">Remaining {rem_thr}</div>
                <div class="stat-block">Completed {completed_thr}</div>
                <div class="stat-block">Errored {errored_thr}</div>
                <div class="stat-block">Live {live_thr}</div>
              </div>

              <div class="stat-category">
                <h4>Inside Traffic</h4>
                <div class="stat-block">Daily Total {total_int}</div>
                <div class="stat-block">Created {created_int} ({pct_int:.1f}%)</div>
                <div class="stat-block">Remaining {rem_int}</div>
                <div class="stat-block">Completed {completed_int}</div>
                <div class="stat-block">Errored {errored_int}</div>
                <div class="stat-block">Live {live_int}</div>
              </div>

              <div class="stat-category">
                <h4>Service Vehicles</h4>
                <div class="service-grid">
                  <div>
                    <div class="stat-block">Food Total {total_food}</div>
                    <div class="stat-block">Created {created_food}</div>
                    <div class="stat-block">Remaining {rem_food}</div>
                    <div class="stat-block">Active {live_food}</div>
                    <div class="stat-block">ETA {eta_food}</div>
                  </div>
                  <div>
                    <div class="stat-block">Waste Total {total_waste}</div>
                    <div class="stat-block">Created {created_waste}</div>
                    <div class="stat-block">Remaining {rem_waste}</div>
                    <div class="stat-block">Active {live_waste}</div>
                    <div class="stat-block">ETA {eta_waste}</div>
                  </div>
                </div>
              </div>

              <div class="stat-category">
                <h4>Vehicle Statuses</h4>
                <div class="stat-block">Collisions {collisions}</div>
                <div class="stat-block">Malfunctions {malfunctions}</div>
                <div class="stat-block">Parked {parked}</div>
                <div class="stat-block">Overtaking {overtaking}</div>
                <div class="stat-block">Stuck {stuck}</div>
                <div class="stat-block">Avg Stuck Duration {fmt_time(live_average_stuck_duration)}</div>
                <div class="stat-block">Max Stuck Duration {fmt_time(live_max_stuck_duration)}</div>
                <div class="stat-block">Detouring {stuck_detouring}</div>

                <h4>Weather Status</h4>
                <div class="stat-block">Rain Count Today {len(model.rains)}</div>
              </div>
            </div>'''

        metrics_section = ''
        if Defaults.SHOW_METRICS_STATISTICS:

            avg_dur_thr_completed = fmt_time(dta.cached_stats.get("avg_duration_through_completed", 0.0))
            avg_dur_thr_live = fmt_time(dta.cached_stats.get("avg_duration_through_live", 0.0))
            avg_dur_thr_total = fmt_time(dta.cached_stats.get("avg_duration_through_total", 0.0))

            avg_dur_int_completed = fmt_time(dta.cached_stats.get("avg_duration_internal_completed", 0.0))
            avg_dur_int_live = fmt_time(dta.cached_stats.get("avg_duration_internal_live", 0.0))
            avg_dur_int_total = fmt_time(dta.cached_stats.get("avg_duration_internal_total", 0.0))

            avg_tu_thr_completed = fmt_time(dta.cached_stats.get("avg_time_per_unit_through_completed", 0.0))
            avg_tu_thr_live = fmt_time(dta.cached_stats.get("avg_time_per_unit_through_live", 0.0))
            avg_tu_thr_total = fmt_time(dta.cached_stats.get("avg_time_per_unit_through_total", 0.0))

            avg_tu_int_completed = fmt_time(dta.cached_stats.get("avg_time_per_unit_internal_completed", 0.0))
            avg_tu_int_live = fmt_time(dta.cached_stats.get("avg_time_per_unit_internal_live", 0.0))
            avg_tu_int_total = fmt_time(dta.cached_stats.get("avg_time_per_unit_internal_total", 0.0))

            avg_daily_diff = dta.cached_stats.get("avg_daily_difference", 0.0)

            metrics_section = f'''
        <div id="stats-grid">
          <div class="stat-category">
            <h4>Total Through Statistics</h4>
            <div class="stat-block">Duration Completed: {avg_dur_thr_completed}</div>
            <div class="stat-block">Duration Live: {avg_dur_thr_live}</div>
            <div class="stat-block">Duration Total: {avg_dur_thr_total}</div>
            
            <div class="stat-block">Time/Block Completed: {avg_tu_thr_completed}</div>
            <div class="stat-block">Time/Block Live: {avg_tu_thr_live}</div>
            <div class="stat-block">Time/Block Total: {avg_tu_thr_total}</div>
          </div>
          <div class="stat-category">
            <h4>Total Inside Statistics</h4>
            <div class="stat-block">Duration Completed: {avg_dur_int_completed}</div>
            <div class="stat-block">Duration Live: {avg_dur_int_live}</div>
            <div class="stat-block">Duration Total: {avg_dur_int_total}</div>
            
            <div class="stat-block">Time/Block Completed: {avg_tu_int_completed}</div>
            <div class="stat-block">Time/Block Live: {avg_tu_int_live}</div>
            <div class="stat-block">Time/Block Total: {avg_tu_int_total}</div>
          </div>
          <div class="stat-category">
            <h4>Daily Statistics</h4>
            <div class="stat-block">Average Traffic Delta: {avg_daily_diff:.2f}</div>
          </div>
        </div>'''
        # assemble full HTML
        return f'''
            <style>
              #ts-card {{
                background: #000;
                color: #fff;
                padding: 14px;
                border-radius: 8px;
                font-family: sans-serif;
                display: flex;
                flex-direction: column;
                gap: 14px;
              }}
            
              #ts-top-info {{
                display: flex;
                flex-wrap: wrap;
                gap: 14px;
                align-items: flex-start;
              }}
            
              .top-info-cell {{
                display: grid;
                grid-template-rows: auto auto;
                justify-items: center;
                align-items: center;
                background: #111;
                padding: 5px 7px;
                border: 1px solid #444;
                border-radius: 2px;
                font-size: 14px;
                font-weight: bold;
                text-align: center;
                min-width: 40px;
                height: 100px;
                row-gap: 2px;
              }}
            
              .top-info-cell .label {{
                color: #aaa;
                font-size: 14px;
              }}
            
              .top-info-cell .value {{
                font-size: 16px;
              }}
            
              .day-hour-block {{
                grid-template-columns: repeat(4, auto);
                grid-template-rows: repeat(2, auto);
                display: grid;
                gap: 12px 16px;
                text-align: center;
              }}
            
              .stat-category {{
                margin-bottom: 12px;
              }}
            
              .stat-category h4 {{
                margin: 4px 0 6px 0;
                font-size: 14px;
                color: #ccc;
                font-weight: bold;
              }}
            
              .stat-block {{
                margin-bottom: 3px;
                font-size: 13px;
                color: #ddd;
              }}
            
              #metrics-info .label {{
                color: #aaa;
                font-size: 13px;
              }}
            
              #metrics-info .value {{
                font-size: 15px;
                font-weight: bold;
              }}
              
              #stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 12px;
                margin-bottom: 12px;
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
              {time_section}
              {traffic_section}
              {metrics_section}
            </div>
        '''
