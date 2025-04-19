# Cell 1: Import Statements & Constants

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.visualization import make_space_component, SolaraViz
import solara
import random
import colorsys
import matplotlib.colors as mcolors

# Constants from creation.py
GRID_WIDTH = 100
GRID_HEIGHT = 100
