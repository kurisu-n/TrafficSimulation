from ...config import Defaults

if Defaults.PATHFINDING_METHOD == "CYTHON":
    try:
        from ._astar_cpp import astar_numba as astar_cpp   # built C++ version
        astar_numba = astar_cpp
        print("Requested CYTHON → Using C++ A* implementation")
    except Exception:                                      # fallback to Numba
        from .astar_numba import astar_numba
        print("Requested CYTHON → Using Numba A* implementation")
elif Defaults.PATHFINDING_METHOD == "NUMBA":
    from .astar_numba import astar_numba
    print("Requested NUMBA → Using Numba A* implementation")