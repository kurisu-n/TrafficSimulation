from mesa import Agent
from typing import List


class IntersectionLightGroup(Agent):
    """A **virtual** (off‑grid) agent that bundles together all
    TrafficLight *cell* agents protecting a single multi‑cell road
    intersection.  It lets you treat the set of lights as one unit.

    Parameters
    ----------
    unique_id : str
        Any unique identifier, e.g. "ILG_7".
    model : CityModel
        The model instance that owns this group.
    traffic_lights : list[Agent]
        The *TrafficLight* CellAgent objects that belong to this
        intersection (typically four – one per corner – but the list
        can be shorter on the map edge or if a light is missing).
    """

    def __init__(self, unique_id: str, model, traffic_lights: List[Agent]):
        super().__init__(unique_id, model)
        self.traffic_lights = traffic_lights

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_all_stop(self) -> None:
        """Force every member light to **Stop** (red)."""
        for tl in self.traffic_lights:
            tl.set_light_stop()

    def set_all_go(self) -> None:
        """Force every member light to **Pass** (green)."""
        for tl in self.traffic_lights:
            tl.set_light_go()

    # No autonomous behaviour for now
    def step(self):
        pass
