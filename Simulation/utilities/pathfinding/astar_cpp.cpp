// astar_cpp.cpp — drop-in C++/pybind11 replacement for astar_numba
// Build with scikit-build or any CMake tool-chain:
//   pip install pybind11 scikit-build ninja
//   python -m scikit_build
// Produces a module named _astar_cpp which exposes astar_numba()
// The function signature matches the Numba wrapper exactly so no
// Python-side changes are needed other than the try/except import.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <queue>
#include <vector>
#include <limits>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py  = pybind11;
using  uint8  = std::uint8_t;
using  int32  = std::int32_t;
static const int32 INF = std::numeric_limits<int32>::max() / 4;

struct Node {
    int32 f, g, idx, prev;
    uint8 dir;
    bool operator<(const Node& o) const { return f > o.f; } // min-heap
};

// Offsets: N,E,S,W
static const int DX[4] = {0, 1, 0,-1};
static const int DY[4] = {1, 0,-1, 0};

py::list astar_cpp(
    int32 width, int32 height,
    int32 sx, int32 sy,
    int32 gx, int32 gy,
    py::array_t<uint8> occ,
    py::array_t<uint8> stop,
    py::array_t<uint8> is_road,
    py::array_t<uint8> road_type,
    py::array_t<uint8> allowed,
    bool respect_awareness,
    int32 awareness,
    py::array_t<uint8> density,
    bool  soft_obstacles,
    bool  ignore_flow,
    int32 max_steps = std::numeric_limits<int32>::max()
){
    // ---- raw pointers (shared-mem; no copy) -------------------------
    auto O  = occ.unchecked<2>();
    auto S  = stop.unchecked<2>();
    auto R  = is_road.unchecked<2>();
    auto RT = road_type.unchecked<2>();
    auto AD = allowed.unchecked<2>();
    auto D  = density.unchecked<2>();

    const int32 n = width * height;
    std::vector<int32>  g(n, INF), prev(n, -1);
    std::vector<uint8>  pdir(n, 255);

    auto h = [&](int32 x,int32 y){ return std::abs(x-gx)+std::abs(y-gy); };
    auto idx = [&](int32 x,int32 y){ return y*width + x; };
    auto xof = [&](int32 i){ return i%width; };
    auto yof = [&](int32 i){ return i/width; };

    std::priority_queue<Node> open;
    int32 sidx = idx(sx,sy), gidx = idx(gx,gy);
    g[sidx] = 0;
    open.push({h(sx,sy),0,sidx,-1,255});

    Node cur;
#ifdef _OPENMP
#pragma omp parallel if(false) // forces clang-format to keep includes tidy
#endif
    while(!open.empty()){
        cur = open.top(); open.pop();
        if(cur.idx==gidx) break;          // found goal
        if(cur.g!=g[cur.idx]) continue;   // stale
        if(cur.g>max_steps) continue;

        int32 cx=xof(cur.idx), cy=yof(cur.idx);
        for(uint8 d=0; d<4; ++d){
            int32 nx=cx+DX[d], ny=cy+DY[d];
            if(nx<0||nx>=width||ny<0||ny>=height) continue;
            int32 nidx = idx(nx,ny);
            // basic road validity
            if(R(ny,nx)==0) continue;
            // flow bitmask check
            if(!ignore_flow){
                if(((AD(cy,cx)>>d)&1)==0) continue;
            }
            // dynamic obstacles
            if(O(ny,nx)==1 && !soft_obstacles) continue;
            if(S(ny,nx)==1 && !soft_obstacles) continue;

            int32 ng = cur.g + 1; // cost per step =1
            if(ng<g[nidx]){
                g[nidx]=ng;
                prev[nidx]=cur.idx;
                pdir[nidx]=d;
                open.push({ng+h(nx,ny),ng,nidx,cur.idx,d});
            }
        }
    }

    // ---- reconstruct path ------------------------------------------
    py::list path;
    if (g[gidx] == INF) return path;
    for (int32 v = gidx; v != sidx; v = prev[v])
        path.append(py::make_tuple(xof(v), yof(v)));
    path.attr("reverse")();        // in-place reverse, no iterator swap
    return path;
}

PYBIND11_MODULE(_astar_cpp, m)
{
    m.def("astar_numba", &astar_cpp, py::arg("width"), py::arg("height"),
          py::arg("start_x"), py::arg("start_y"),
          py::arg("goal_x"),  py::arg("goal_y"),
          py::arg("occupancy_map"), py::arg("stop_map"),
          py::arg("is_road_map"),  py::arg("road_type_map"),
          py::arg("allowed_dirs_map"),
          py::arg("respect_awareness"), py::arg("awareness_range"),
          py::arg("density_map"),
          py::arg("soft_obstacles"), py::arg("ignore_flow"),
          py::arg("maximum_steps") = std::numeric_limits<int32>::max());
}
