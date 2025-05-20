from mesa.visualization.modules import CanvasGrid
from mesa_viz_tornado.ModularVisualization import ModularServer

from Simulation.config import Defaults
from Simulation.utilities.general import cleanup_empty_results
from Simulation.visualization.model_parameters import model_params


class DynamicGridServer(ModularServer):
    def reset_model(self):

        cleanup_empty_results()
        # 1) Pull the current slider values
        new_w = Defaults.WIDTH
        new_h = Defaults.HEIGHT

        # 2) Update any CanvasGrid in-place
        for elem in self.visualization_elements:
            if isinstance(elem, CanvasGrid):
                elem.grid_width  = new_w
                elem.grid_height = new_h

        # 3) Now rebuild the model and re-reset each element
        super().reset_model()