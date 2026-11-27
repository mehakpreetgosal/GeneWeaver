"""
kernels.py - Numba CUDA kernel for GeneWeaver (Week 2: correct first, optimized in Week 4).

One GPU thread = one 23-base window start `w` of a chunk. Each thread checks BOTH strands:

    '+' strand : [20-mer][N G G]          -> hit position = chunk.start + w
    '-' strand : [C C N][revcomp 20-mer]  -> hit position = chunk.start + w + 3

Same rules as the CPU code: windows containing N are skipped, a hit needs mismatches <= max_mm,
and only windows w < n_windows (the chunk's owned, complete windows) are examined.

Output (filled with atomics; the host reads the first min(count, capacity) entries):
    out_win[k]    window start index inside the chunk (int32)
    out_strand[k] 0 = '+', 1 = '-'
    out_mm[k]     mismatch count (uint8)
    out_count[0]  total hits FOUND (can exceed capacity -> host detects overflow and retries)
"""
from numba import cuda

GUIDE_LEN = 20
WINDOW_LEN = 23
CODE_C = 1
CODE_G = 2
CODE_N = 4


@cuda.jit(device=True)
def _window_has_n(data, w):
    for k in range(WINDOW_LEN):
        if data[w + k] == CODE_N:
            return True
    return False


@cuda.jit
def search_kernel(data, n_windows, guide, guide_rc, max_mm,
                  out_win, out_strand, out_mm, out_count):
    w = cuda.grid(1)
    if w >= n_windows:
        return

    capacity = out_win.size

    # ---------------- forward strand: compare 20-mer, then PAM
    mm = 0
    for j in range(GUIDE_LEN):
        if data[w + j] != guide[j]:
            mm += 1
            if mm > max_mm:
                break
    if mm <= max_mm and data[w + 21] == CODE_G and data[w + 22] == CODE_G:
        if not _window_has_n(data, w):
            k = cuda.atomic.add(out_count, 0, 1)
            if k < capacity:
                out_win[k] = w
                out_strand[k] = 0
                out_mm[k] = mm

    # ---------------- reverse strand: compare revcomp 20-mer (offset 3), then CC
    mm = 0
    for j in range(GUIDE_LEN):
        if data[w + 3 + j] != guide_rc[j]:
            mm += 1
            if mm > max_mm:
                break
    if mm <= max_mm and data[w] == CODE_C and data[w + 1] == CODE_C:
        if not _window_has_n(data, w):
            k = cuda.atomic.add(out_count, 0, 1)
            if k < capacity:
                out_win[k] = w
                out_strand[k] = 1
                out_mm[k] = mm
