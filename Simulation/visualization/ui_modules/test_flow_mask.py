# test_flow_mask.py

import sys
from Simulation.city_model import CityModel

def assert_flow_mask(mask, x: int, y: int, k: int, expected: bool = True):
    """
    Raises an AssertionError if mask[x,y,k] != expected,
    otherwise does nothing.
    """
    actual = bool(mask[x, y, k])
    assert actual == expected, (
        f"mask[{x},{y},{k}] is {actual!r}, but expected {expected!r}"
    )

def interactive_check(mask):
    """
    Repeatedly prompt for `x y k [expected]` and check the mask.
    Enter blank line to exit.
    """
    prompt = "Enter x y k [expected:True/False] (blank to quit): "
    while True:
        line = input(prompt).strip()
        if not line:
            print("Exiting.")
            break
        parts = line.split()
        try:
            x, y, k = map(int, parts[:3])
            expected = True
            if len(parts) >= 4:
                expected = parts[3].lower() in ("1","true","t","yes","y")
            assert_flow_mask(mask, x, y, k, expected)
            print(f"✔ PASS: mask[{x},{y},{k}] == {expected}")
        except AssertionError as e:
            print(f"✘ FAIL: {e}")
        except Exception as e:
            print(f"Error parsing input or checking mask: {e}")