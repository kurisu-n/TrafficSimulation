# Structured Urban Grid World Simulation

A procedural city generation and simulation built with Mesa (an agent‑based modeling framework). This project generates a grid‑based city with roads, sidewalks, blocks, traffic lights, and block entrances, and provides an interactive web interface to inspect individual cells and block entrances.

## Features

- **Procedural City Generation**: Creates walls, sidewalks, roads (multiple types), intersections, sub‑block roads, and block entrances.
- **Cell Agents**: Each grid cell is an agent (`CellAgent`) storing type, directions, status, and extra metadata (occupied, block ID/type, highway info, controlled blocks).
- **Interactive Visualization**: A left‑hand inspector pane displays cell details on click, alongside a CanvasGrid rendering of the city.
- **Traffic Control**: Traffic lights and controlled roads automatically placed at intersections; block entrances detected and displayed.

## Prerequisites

- Python 3.8+ (tested on 3.10)
- Conda (recommended) or virtualenv

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/city-grid-simulation.git
   cd city-grid-simulation
   ```

2. **Create and activate a conda environment**

   ```bash
   conda create -n city-sim python=3.10
   conda activate city-sim
   ```

3. **Install dependencies**

   - Install the Mesa agent‑based framework and visualization extras:
     ```bash
     pip install mesa[rec] pygments ipywidgets
     ```

   If you prefer conda, you can also do:
   ```bash
   conda install mesa
   ```

4. **Verify installation**

   ```bash
   python -c "import mesa; print(mesa.__version__)"
   ```

## Running the Simulation

1. **Launch the server**

   ```bash
   python server.py
   ```

2. **Open your browser** at the URL printed in the console (e.g., `http://localhost:8521`).
3. **Interact** with the grid:
   - Click on any cell to inspect its type, directions, status, and related metadata.
   - Click on a block entrance in the inspector pane to highlight its location on the grid.

## Project Structure

```
├─ server.py           # Starts the Mesa server and handles orchestrating modules
├─ creation.py         # Contains the StructuredCityModel building the grid and roads
├─ cell.py             # Defines CellAgent, portrayal functions, and utility methods
└─ README.md           # This file
```

## Customization

- **Grid size**: Modify `GRID_WIDTH` and `GRID_HEIGHT` in `creation.py` and `server.py`.
- **Road parameters**: Tweak constants (e.g. `ROAD_THICKNESS`, `MIN_BLOCK_SPACING`, `SUBBLOCK_CHANCE`) in `creation.py`.
- **Visualization**:
  - Swap between `cellinspector.py` (Python‑only inspector) or `infopane.py` (JS inspector) in `server.py`.


