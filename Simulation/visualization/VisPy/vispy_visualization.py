import numpy as np
import logging
# Install vispy package first using: pip install vispy
from vispy import app, scene
from vispy.color import Color

# Ensure glfw is installed: pip install glfw
app.use_app('glfw')

# Try multiple backends until one works
for backend in ('pyqt5', 'pyside2', 'glfw'):
    try:
        app.use_app(backend)
        break
    except Exception:
        continue

# Import your Mesa model and portrayal
from Simulation.city_model import CityModel
from Simulation.visualization.agent_portrayal import agent_portrayal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisPyGridVisualizer:
    def __init__(self, model: CityModel, interval: float = 0.1):
        self.model = model
        self.width = model.grid.width
        self.height = model.grid.height

        # Create a white-background canvas and PanZoom camera
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            show=True,
            title='CityModel VisPy',
            bgcolor='white'
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.flip = (False, True)  # Mesa’s origin is bottom-left
        self.view.camera.set_range(
            x=(0, self.width),
            y=(0, self.height)
        )

        # Image layer for cell fills
        initial_bg = np.zeros((self.height, self.width, 4), dtype=np.float32)
        self.bg = scene.visuals.Image(
            initial_bg,
            parent=self.view.scene,
            interpolation='nearest',
            origin='lower-left'
        )

        # Marker layer for point agents
        self.markers = scene.visuals.Markers(parent=self.view.scene)

        # Kick off the timer to step & redraw
        self.timer = app.Timer(interval, connect=self.on_timer, start=True)

    def on_timer(self, event):
        # Advance the model by one step
        self.model.step()

        # Build a fresh RGBA frame
        bg = np.zeros((self.height, self.width, 4), dtype=np.float32)
        coords, colors, sizes = [], [], []

        for agent in self.model.schedule.agents:
            pos = getattr(agent, 'pos', None)
            if pos is None:
                continue
            x, y = pos

            por = agent_portrayal(agent)
            shape = por.get('Shape', por.get('shape', '')).lower()
            rgba = Color(por.get('Color', por.get('color', 'black'))).rgba

            if shape in ('rect', 'rectangle', 'square'):
                bg[int(y), int(x)] = rgba
            else:
                coords.append((x + 0.5, y + 0.5))
                colors.append(rgba)
                sizes.append(por.get('r', 0.5) * 2)

        # Push data into VisPy visuals
        self.bg.set_data(bg)
        if coords:
            self.markers.set_data(
                pos=np.array(coords, dtype=np.float32),
                face_color=np.array(colors, dtype=np.float32),
                size=np.array(sizes, dtype=np.float32),
                symbol='o'
            )

        # Force a redraw
        self.canvas.update()

    def run(self):
        app.run()