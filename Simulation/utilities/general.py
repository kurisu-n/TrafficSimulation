import hashlib
import colorsys
import os
import shutil

import matplotlib.colors as mcolors

import numpy as np

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
    if isinstance(color, str) and color.startswith('#'):
        r, g, b = hex_to_rgb(color)
    else:
        r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s *= sat_factor
    l = min(1.0, l + light_factor)
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

