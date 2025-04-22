# Structured Urban Grid World

A world-building engine for simulating a realistic urban environment using the **Mesa** agent-based modeling framework. This module focuses **only** on procedural generation of the city layout (walls, sidewalks, roads, intersections, blocks, and traffic-light infrastructure). Dynamic agent behaviors (vehicles, pedestrians, etc.) can be added separately.

---

## 🚀 Table of Contents

- [🔍 Features](#-features)
- [📦 Installation](#-installation)
  - [Git & pip](#git--pip)
  - [Conda Environment](#conda-environment)
- [▶️ Usage](#usage)
- [⚙️ Configuration](#-configuration)
- [📁 Code Structure](#-code-structure)
  - [World Building](#world-building)
  - [Visualization Parameters](#visualization-parameters)
  - [Traffic Light Control UI](#traffic-light-control-ui)
  - [Server Setup](#server-setup)
  - [Launcher](#launcher)
- [🛠 Extending with Agent Logic](#-extending-with-agent-logic)

---

## 🔍 Features

### Procedural City Layout
- Boundary walls & inner sidewalk rings
- Randomized network: highways (R1), major (R2), local (R3), and sub-block (R4) roads
- Optimized vs. full intersections
- Flood-fill zoning: Residential, Office, Market, Leisure, or Empty
- Block entrances automatically aligned to roads
- Optional L-shaped sub-block roads

### Traffic Infrastructure
- Traffic-light placement adjacent to intersections
- Controlled-road identification and status toggling

### Interactive Web UI
- Adjustable parameters via sliders/checkboxes
- CanvasGrid visualization & custom traffic-light controls

---

## 📦 Installation

**Prerequisites:** Python 3.10, Git, pip or Conda.

---

## ▶️ Usage

1. Launch the server:

   ```bash
   python run.py

---

## ⚙️ Configuration

On the left-side panel of the web UI, tweak parameters such as:

- **Grid Width / Height**
- **Wall Thickness**
- **Sidewalk Ring Width**
- **Road Types & Spacing**
- **Block Size & Empty-Block Chance**
- **Carve Sub-block Roads**
- **Traffic Light Range**

Changes apply upon resetting or restarting the model.

---

## 📁 Code Structure

### World Building

- `agents.py`: `CellAgent` defines each grid cell (walls, sidewalks, roads, intersections, zones, traffic lights). Includes utilities for positioning, directions, and portrayal.
- `model.py`: `CityModel` orchestrates environment generation:
  1. Place boundary walls & sidewalk rings
  2. Clear interior to “Nothing”
  3. Generate road bands (including forced highways)
  4. Create optimized/non-optimized intersections
  5. Flood-fill and zone interior blocks
  6. Carve optional sub-block (R4) roads
  7. Eliminate dead-ends
  8. Upgrade roads to intersections
  9. Place block entrances
  10. Validate & clean intersection directions
  11. Add traffic lights & link controlled roads

### Visualization Parameters

- `parameters.py`: Defines sliders, choices, and checkboxes for all tunable parameters in the Mesa UI.

### Traffic Light Control UI

- `ui_elements.py`: Custom Mesa `TextElement` that renders a dropdown of traffic lights plus **Go/Stop** buttons for individual or global control. Backed by Tornado request handlers.

### Server Setup

- `server.py`:
  - Configures a `CanvasGrid` with the cell portrayal function
  - Initializes `ModularServer` with `CityModel` and visualization modules
  - Adds HTTP routes for traffic-light endpoints
  - Dynamically binds to an available localhost port

### Launcher

- `run.py`: Imports and launches the Mesa server, printing the local URL.

---

## 🛠 Extending with Agent Logic

To simulate dynamics (vehicles, pedestrians):

1. **Create Agent Classes** (e.g., `VehicleAgent`, `PedestrianAgent`)
2. **Implement** `step()` **methods** for movement, pathfinding, and interactions
3. **Schedule & Place** agents in `CityModel.schedule` and on the grid
4. **Customize Visualization**: add new portrayal functions or modules

---