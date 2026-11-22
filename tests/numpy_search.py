"""
numpy_search.py - vectorized NumPy search: the honest "middle" baseline between pure Python and GPU.

Same rules and same output format as cpu_baseline.py:
    hits = [(chrom, position, strand, mismatches), ...]  sorted by (chrom, position, strand)

Trick: instead of looping over windows, loop over the 20 guide positions and compare the whole
chunk at once:  mm += (data[j : j+W] != guide[j]).  That is 20 vectorized passes per strand.
"""
import numpy as np

from parser import GUIDE_LEN, WINDOW_LEN, Chunk, chunk_genome, encode, reverse_complement

G, C, N = 2, 1, 4


def search_chunk(chunk: Chunk, guide: np.ndarray, guide_rc: np.ndarray, max_mm: int):
    data = chunk.data
    W = min(chunk.n_owned, len(data) - WINDOW_LEN + 1)   # number of owned, complete windows
    if W <= 0:
        return []

    # windows containing N: count Ns in every 23-base window with a cumulative sum
    cs = np.concatenate(([0], np.cumsum(data == N, dtype=np.int32)))
    has_n = (cs[WINDOW_LEN:WINDOW_LEN + W] - cs[:W]) > 0

    hits = []

    # ---- forward strand: [20-mer][N G G], position = window start
    mm = np.zeros(W, dtype=np.uint8)
    for j in range(GUIDE_LEN):
        mm += data[j:j + W] != guide[j]
    ok = (data[21:21 + W] == G) & (data[22:22 + W] == G) & (mm <= max_mm) & ~has_n
    for w in np.flatnonzero(ok):
        hits.append((chunk.chrom, chunk.start + int(w), "+", int(mm[w])))

    # ---- reverse strand: [C C N][revcomp 20-mer], position = window start + 3
    mm = np.zeros(W, dtype=np.uint8)
    for j in range(GUIDE_LEN):
        mm += data[3 + j:3 + j + W] != guide_rc[j]
    ok = (data[0:W] == C) & (data[1:W + 1] == C) & (mm <= max_mm) & ~has_n
    for w in np.flatnonzero(ok):
        hits.append((chunk.chrom, chunk.start + int(w) + 3, "-", int(mm[w])))

    return hits


def search_chunks(chunks, guide: str, max_mm: int):
    if len(guide) != GUIDE_LEN:
        raise ValueError(f"guide must be {GUIDE_LEN} letters")
    g = encode(guide.upper())
    if (g == N).any():
        raise ValueError("guide must contain only A/C/G/T")
    g_rc = reverse_complement(g)
    hits = []
    for ch in chunks:
        hits.extend(search_chunk(ch, g, g_rc, max_mm))
    return sorted(hits, key=lambda h: (h[0], h[1], h[2]))


def search_genome(fasta_path: str, guide: str, max_mm: int,
                  chunk_size: int = 5_000_000, region=None):
    return search_chunks(chunk_genome(fasta_path, chunk_size, region=region), guide, max_mm)
