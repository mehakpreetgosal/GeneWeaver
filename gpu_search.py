"""
gpu_search.py - host-side driver for the CUDA kernel: memory transfers, launches, overflow handling.

Host (CPU RAM)  --H2D copy-->  Device (GPU VRAM)  --kernel-->  hits  --D2H copy-->  Host

Design choices (all about avoiding the classic CUDA mistakes):
  * The chunk buffer, guide and output buffers are allocated ONCE and reused for every chunk
    (allocating per chunk is slow and fragments VRAM).
  * Only the hits are copied back, never the whole chunk.
  * The output buffer has a fixed capacity. If a chunk produces more hits than that, the kernel
    still counts them, the host sees count > capacity, grows the buffer and re-runs that chunk.
    So hits are never silently dropped and VRAM use stays bounded.
  * Kernel launches are asynchronous: timing code must call cuda.synchronize() first.
  * The first launch includes JIT compile time: call warm_up() before timing anything.

Works with a real GPU, or with the CPU simulator for debugging:  NUMBA_ENABLE_CUDASIM=1
(the simulator is very slow and only meant for small tests).
"""
import numpy as np
from numba import cuda

from kernels import search_kernel
from parser import GUIDE_LEN, N_CODE, OVERLAP, WINDOW_LEN, Chunk, encode, reverse_complement

THREADS_PER_BLOCK = 256
MAX_CAPACITY = 50_000_000          # hard ceiling for hit buffer growth (entries)


def gpu_available() -> bool:
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def memory_info():
    """(free_bytes, total_bytes) of the current GPU, or None if unavailable (e.g. simulator)."""
    try:
        free, total = cuda.current_context().get_memory_info()
        return int(free), int(total)
    except Exception:
        return None


def suggest_chunk_size(vram_fraction: float = 0.25, cap: int = 256_000_000, floor: int = 1_000_000) -> int:
    """
    Pick a chunk size (in bases) from free VRAM. Each base costs 1 byte on the device, and the hit
    buffers are small, so we use only a fraction of free memory to leave headroom.
    """
    info = memory_info()
    if info is None:
        return min(cap, 10_000_000)
    free, _ = info
    return int(max(floor, min(cap, free * vram_fraction)))


class GPUSearcher:
    def __init__(self, guide: str, max_mm: int, max_chunk_bases: int = 10_000_000,
                 capacity: int = 100_000):
        guide = guide.upper()
        if len(guide) != GUIDE_LEN:
            raise ValueError(f"guide must be {GUIDE_LEN} letters")
        g = encode(guide)
        if (g == N_CODE).any():
            raise ValueError("guide must contain only A/C/G/T")

        self.guide = guide
        self.max_mm = int(max_mm)
        self.max_chunk_bases = int(max_chunk_bases)

        # one-time uploads / allocations
        self.d_guide = cuda.to_device(g)
        self.d_guide_rc = cuda.to_device(reverse_complement(g))
        self.d_data = cuda.device_array(self.max_chunk_bases + OVERLAP, dtype=np.uint8)
        self.d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
        self._alloc_outputs(capacity)

        # timings (seconds) accumulated over calls, for the benchmark
        self.t_h2d = self.t_kernel = self.t_d2h = 0.0

    # ------------------------------------------------------------------ buffers
    def _alloc_outputs(self, capacity: int):
        self.capacity = int(capacity)
        self.d_win = cuda.device_array(self.capacity, dtype=np.int32)
        self.d_strand = cuda.device_array(self.capacity, dtype=np.int8)
        self.d_mm = cuda.device_array(self.capacity, dtype=np.uint8)

    # ------------------------------------------------------------------ search
    def warm_up(self):
        """Trigger JIT compilation on a tiny input so it is not counted in timings."""
        tiny = Chunk("warmup", 0, np.zeros(WINDOW_LEN + 5, dtype=np.uint8), 5)
        self.search_chunk(tiny)
        self.t_h2d = self.t_kernel = self.t_d2h = 0.0

    def _launch(self, n_windows: int):
        self.d_count.copy_to_device(np.zeros(1, dtype=np.int32))
        blocks = (n_windows + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        search_kernel[blocks, THREADS_PER_BLOCK](
            self.d_data, n_windows, self.d_guide, self.d_guide_rc, self.max_mm,
            self.d_win, self.d_strand, self.d_mm, self.d_count)
        cuda.synchronize()                       # launches are async: wait before reading results
        return int(self.d_count.copy_to_host()[0])

    def search_chunk(self, chunk: Chunk):
        """Search one chunk on the GPU. Returns [(chrom, position, strand, mismatches)]."""
        import time
        data = chunk.data
        n = len(data)
        if n > self.d_data.size:
            raise ValueError(f"chunk has {n:,} bases but buffer holds {self.d_data.size:,}; "
                             f"create GPUSearcher with a larger max_chunk_bases")
        n_windows = min(chunk.n_owned, n - WINDOW_LEN + 1)
        if n_windows <= 0:
            return []

        # host -> device (only the used part of the buffer)
        t0 = time.perf_counter()
        self.d_data[:n].copy_to_device(np.ascontiguousarray(data))
        cuda.synchronize()
        self.t_h2d += time.perf_counter() - t0

        # kernel (retry with a bigger hit buffer if it overflowed)
        t0 = time.perf_counter()
        count = self._launch(n_windows)
        while count > self.capacity:
            if count > MAX_CAPACITY:
                raise MemoryError(f"{count:,} hits in one chunk exceeds MAX_CAPACITY; "
                                  f"use a smaller chunk or a smaller --max-mm")
            self._alloc_outputs(min(MAX_CAPACITY, int(count * 1.5)))
            count = self._launch(n_windows)
        self.t_kernel += time.perf_counter() - t0

        # device -> host (hits only)
        t0 = time.perf_counter()
        if count == 0:
            hits = []
        else:
            win = self.d_win[:count].copy_to_host()
            strand = self.d_strand[:count].copy_to_host()
            mm = self.d_mm[:count].copy_to_host()
            hits = [(chunk.chrom,
                     chunk.start + int(w) + (3 if s == 1 else 0),
                     "-" if s == 1 else "+",
                     int(m))
                    for w, s, m in zip(win, strand, mm)]
        self.t_d2h += time.perf_counter() - t0
        return hits

    def search_chunks(self, chunks):
        hits = []
        for ch in chunks:
            hits.extend(self.search_chunk(ch))
        return sorted(hits, key=lambda h: (h[0], h[1], h[2]))


def search_chunks(chunks, guide: str, max_mm: int, max_chunk_bases: int = 10_000_000):
    """Same call signature as cpu_baseline / numpy_search, so tests and benchmarks can swap them."""
    s = GPUSearcher(guide, max_mm, max_chunk_bases)
    return s.search_chunks(chunks)
