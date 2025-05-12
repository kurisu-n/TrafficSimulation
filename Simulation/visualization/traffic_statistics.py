from __future__ import annotations

"""Live traffic statistics card for the Mesa web UI – **wide (horizontal)** layout.

Place in ``Simulation/visualization`` and append `TrafficStatistics()` to
`server.modules` to render a responsive dashboard stretching the full
width of the browser pane.
"""

from typing import TYPE_CHECKING, Dict, Tuple

from mesa.visualization.modules import TextElement

from Simulation.config import Defaults

if TYPE_CHECKING:  # avoid heavy imports at runtime
    from Simulation.city_model import CityModel
    from Simulation.agents.dynamic_traffic_generator import DynamicTrafficAgent
    from Simulation.agents.vehicles.vehicle_base import VehicleAgent
    from Simulation.agents.vehicles.vehicle_service import ServiceVehicleAgent


# ──────────────────────────────────────────────────────────────────────
class TrafficStatistics(TextElement):
    """Responsive, horizontally‑spanning traffic dashboard."""

    # ------------------------------------------------------------------
    def render(self, model: "CityModel") -> str:  # noqa: ANN001
        # —— locate DynamicTrafficAgent ——
        dta = next((ag for ag in model.schedule.agents if ag.__class__.__name__ == "DynamicTrafficAgent"), None)
        if dta is None:
            return (
                '<div style="color:red;margin:4px;padding:4px;border:1px solid#900;">'
                "DynamicTrafficAgent not found"
                "</div>"
            )

        # —— time info ——
        day, hour, minute, second = dta.now_dhms()  # type: ignore[attr-defined]
        elapsed_d, elapsed_h, elapsed_m, elapsed_s = (
            dta.elapsed_days,  # type: ignore[attr-defined]
            dta.elapsed_hours % 24,  # type: ignore[attr-defined]
            dta.elapsed_minutes % 60,  # type: ignore[attr-defined]
            dta.elapsed_seconds % 60,  # type: ignore[attr-defined]
        )

        # —— live vehicle counts ——
        from Simulation.agents.vehicles.vehicle_base import VehicleAgent  # local import to avoid circulars
        from Simulation.agents.vehicles.vehicle_service import ServiceVehicleAgent

        sched_agents = model.schedule.agents
        through_live = sum(
            1
            for ag in sched_agents
            if isinstance(ag, VehicleAgent) and getattr(ag, "population_type", None) == "through"
        )
        internal_live = sum(
            1
            for ag in sched_agents
            if isinstance(ag, VehicleAgent) and getattr(ag, "population_type", None) == "internal"
        )
        food_sv = sum(
            1 for ag in sched_agents if isinstance(ag, ServiceVehicleAgent) and ag.service_type == "Food"
        )
        waste_sv = sum(
            1 for ag in sched_agents if isinstance(ag, ServiceVehicleAgent) and ag.service_type == "Waste"
        )

        # —— percentages versus TOTAL populations (configured daily totals) ——
        pct_through_total = (
            through_live / Defaults.PASSING_POPULATION_TRAFFIC_PER_DAY * 100
            if Defaults.PASSING_POPULATION_TRAFFIC_PER_DAY else 0.0
        )
        pct_internal_total = (
            internal_live / Defaults.INTERNAL_POPULATION_TRAFFIC_PER_DAY * 100
            if Defaults.INTERNAL_POPULATION_TRAFFIC_PER_DAY else 0.0
        )

        # —— internal route breakdown ——
        breakdown: Dict[Tuple[str, str], int] = {}
        for ag in sched_agents:
            if isinstance(ag, VehicleAgent) and getattr(ag, "population_type", None) == "internal":
                o_bt = getattr(getattr(ag.start_cell, "block", None), "block_type", None)
                d_bt = getattr(getattr(ag.target, "block", None), "block_type", None)
                if o_bt and d_bt:
                    breakdown[(o_bt, d_bt)] = breakdown.get((o_bt, d_bt), 0) + 1

        def _fmt_route(kv):
            (ob, db), cnt = kv
            pct = cnt / internal_live * 100 if internal_live else 0.0
            return f"<li>{ob}→{db}: {cnt} ({pct:.1f}%)</li>"

        routes_html = (
            "".join(_fmt_route(kv) for kv in sorted(breakdown.items(), key=lambda kv: -kv[1]))
            or "<li>None</li>"
        )

        # —— pending trips ——
        pending = len(getattr(dta, "pending", []))

        # ------------------------------------------------------------------
        #  HTML & CSS – grid layout spanning full width
        # ------------------------------------------------------------------
        return f"""
        <style>
          #ts-card {{
            font:14px/1.4 system-ui,sans-serif; background:#202020; color:#eee;
            padding:10px 14px; margin:4px; border-radius:8px; width:100%;
          }}
          #ts-card h3 {{ margin:0 0 8px 0; font-size:17px; }}

          /* responsive grid for the small stats */
          #stats-grid {{
            display:grid;
            grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
            column-gap:14px; row-gap:6px;
          }}
          .stat-label {{ color:#aaa; margin-right:4px; }}
          .stat-block {{ white-space:nowrap; }}

          #routes {{ column-count:4; column-gap:12px; margin:6px 0 0 0; padding:0 0 0 16px; }}
          #routes li {{ break-inside:avoid; }}
        </style>
        <div id=\"ts-card\">
          <h3>Traffic Statistics</h3>
          <div id=\"stats-grid\">
            <div class=\"stat-block\"><span class=\"stat-label\">Sim Time</span>Day {day} {hour:02d}:{minute:02d}:{second:02d}</div>
            <div class=\"stat-block\"><span class=\"stat-label\">Elapsed</span>{elapsed_d}d {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}</div>
            <div class=\"stat-block\"><span class=\"stat-label\">Through</span>{through_live}/{Defaults.PASSING_POPULATION_TRAFFIC_PER_DAY} ({pct_through_total:.1f}%)</div>
            <div class=\"stat-block\"><span class=\"stat-label\">Internal</span>{internal_live}/{Defaults.INTERNAL_POPULATION_TRAFFIC_PER_DAY} ({pct_internal_total:.1f}%)</div>
            <div class=\"stat-block\"><span class=\"stat-label\">Pending&nbsp;Trips</span>{pending}</div>
            <div class=\"stat-block\"><span class=\"stat-label\">Food&nbsp;SV</span>{food_sv}</div>
            <div class=\"stat-block\"><span class=\"stat-label\">Waste&nbsp;SV</span>{waste_sv}</div>
          </div>
          <h4 style=\"margin:10px 0 4px 0;font-size:15px;\">Internal Routes</h4>
          <ul id=\"routes\">{routes_html}</ul>
        </div>"""
