from numba import cuda, int32, uint8

MAX_PATH      = 512          # longest path we actually return
MAX_FRONTIER  = 8192         # capacity of the OPEN list (was 512)

@cuda.jit
def batched_astar(starts, goals, flags,
                  grid,                  # uint8[width, height]
                  paths,                 # int32[n * MAX_PATH * 2]
                  path_len):             # int32[n]
    """
    One block per route, one active thread (tid==0) per block.
    Writes up to MAX_PATH (x, y) pairs into `paths`, and the length
    into `path_len`. 0 length means ‘no route found’.
    """
    bid = int(cuda.blockIdx.x)
    tid = int(cuda.threadIdx.x)
    n   = int(starts.shape[0])
    if bid >= n or tid != 0:          # only thread 0 does the work
        return

    # ── helpers ──────────────────────────────────────────────────────
    width, height = grid.shape
    def h(x, y, gx, gy):              # Manhattan heuristic
        return abs(gx - x) + abs(gy - y)

    sx, sy = starts[bid][0], starts[bid][1]
    gx, gy = goals [bid][0], goals [bid][1]
    blk_flags = flags[bid]            # not used yet but kept for parity

    # ── per-thread OPEN list in local memory ─────────────────────────
    open_x = cuda.local.array(shape=MAX_FRONTIER, dtype=int32)
    open_y = cuda.local.array(shape=MAX_FRONTIER, dtype=int32)
    open_f = cuda.local.array(shape=MAX_FRONTIER, dtype=int32)
    parent = cuda.local.array(shape=MAX_FRONTIER, dtype=int32)

    head = 0
    tail = 1
    open_x[0], open_y[0] = sx, sy
    open_f[0] = h(sx, sy, gx, gy)
    parent[0] = -1

    found = -1                         # index of the goal node in OPEN

    # ── A* search loop ───────────────────────────────────────────────
    while head < tail and tail < MAX_FRONTIER:

        # 1. pick node with smallest f in [head, tail)
        best = head
        best_f = open_f[best]
        for i in range(head + 1, tail):
            if open_f[i] < best_f:
                best, best_f = i, open_f[i]

        # 2. pop it (swap-remove to keep OPEN compact)
        if best != head:
            open_x[head], open_x[best] = open_x[best], open_x[head]
            open_y[head], open_y[best] = open_y[best], open_y[head]
            open_f[head], open_f[best] = open_f[best], open_f[head]
            parent[head], parent[best] = parent[best], parent[head]

        cx, cy = open_x[head], open_y[head]

        # goal reached?
        if cx == gx and cy == gy:
            found = head
            break

        g_here = open_f[head] - h(cx, cy, gx, gy)   # real cost so far
        head += 1                                   # ‘close’ this node

        # 3. expand four neighbours
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if grid[nx, ny] == 1:                   # wall / blocked
                continue
            if tail >= MAX_FRONTIER:                # OPEN full
                continue

            g_new = g_here + 1
            open_x[tail] = nx
            open_y[tail] = ny
            open_f[tail] = g_new + h(nx, ny, gx, gy)
            parent[tail] = head - 1                 # parent is cx,cy
            tail += 1

    # ── write result back to global memory ───────────────────────────
    base = bid * MAX_PATH * 2
    if found == -1:
        path_len[bid] = 0
        return

    length = 0
    i = found
    while i >= 0 and length < MAX_PATH:
        paths[base + length * 2    ] = open_x[i]
        paths[base + length * 2 + 1] = open_y[i]
        length += 1
        i = parent[i]

    path_len[bid] = length          # path is goal→start (reverse on host)
