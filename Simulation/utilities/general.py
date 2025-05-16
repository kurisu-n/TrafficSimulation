import hashlib
import colorsys
import os
import shutil

import matplotlib.colors as mcolors

import numpy as np
from numba import njit


def str_to_unique_int(s: str) -> int:
    """Convert a string to a unique integer using MD5."""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def desaturate(color, sat_factor=0.5, light_factor=0.0):
    """
    Desaturate and adjust lightness of a color.

    Args:
        color: Matplotlib color name or hex string (e.g. "#aabbcc").
        sat_factor: Multiplier for saturation (0 = gray, 1 = original).
        light_factor: Additive lightness adjustment;
                      negative values darken, positives lighten.

    Returns:
        Hex string of the transformed color.
    """
    # parse input color
    if isinstance(color, str) and color.startswith('#'):
        r, g, b = hex_to_rgb(color)
    else:
        r, g, b = mcolors.to_rgb(color)

    # convert to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # adjust saturation
    s *= sat_factor

    # adjust lightness and clamp between 0 and 1
    l = max(0.0, min(1.0, l + light_factor))

    # convert back to RGB and hex
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return rgb_to_hex((r2, g2, b2))



def cleanup_empty_results():
    """Remove any Results/{run_ts} folder whose files have no data rows."""
    base = os.path.join(os.getcwd(), "Results")
    if not os.path.isdir(base):
        return

    for run_ts in os.listdir(base):
        folder = os.path.join(base, run_ts)
        if not os.path.isdir(folder):
            continue

        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        # if folder has no files at all, delete it
        if not files:
            shutil.rmtree(folder)
            continue

        # check each file: count non-blank, non-whitespace lines
        all_empty = True
        for path in files:
            if not os.path.isfile(path):
                continue
            with open(path, "r") as f:
                # strip out blank lines
                lines = [ln for ln in f if ln.strip()]
            # if more than one line, there's at least one data row beyond header
            if len(lines) > 1:
                all_empty = False
                break

        if all_empty:
            shutil.rmtree(folder)


@njit
def overlay_dynamic(grid: np.ndarray,
                    vehicle_positions: np.ndarray,
                    stop_positions: np.ndarray) -> np.ndarray:
    """
    JIT-compiled overlay of vehicles (1) and stops (2) onto the grid.
    Expects:
      – grid: the static mask as a 2D array
      – vehicle_positions: shape (n_vehicles, 2) array of (x,y)
      – stop_positions:    shape (n_stops,    2) array of (x,y)
    """
    # overlay vehicles
    for i in range(vehicle_positions.shape[0]):
        x = vehicle_positions[i, 0]
        y = vehicle_positions[i, 1]
        grid[x, y] = 1

    # overlay stops/red-lights
    for j in range(stop_positions.shape[0]):
        x = stop_positions[j, 0]
        y = stop_positions[j, 1]
        grid[x, y] = 2

    return grid

