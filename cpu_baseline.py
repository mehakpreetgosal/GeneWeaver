"""
cpu_baseline.py - pure-Python CRISPR off-target search (the "before" number for GeneWeaver).

Deliberately simple: one window at a time, nested loops, no NumPy in the hot path.

Rules (identical to parser.py / find_targets.py):
  * 0-based forward-strand coordinates.
  * A window is 23 bases: [20-mer][NGG] on '+', [CCN][revcomp 20-mer] on '-'.
  * '+' hit position = window start; '-' hit position = window start + 3.
  * Any window containing N (code 4) is skipped.
  * A chunk only reports windows whose start index is < chunk.n_owned (no duplicate hits).
  * A hit is kept if mismatches <= max_mm.

Usage:
  python cpu_baseline.py --fasta data/chr21.fa --guide GACGCATAAAGATGAGACGC --max-mm 3 \
         --start 20000000 --end 21000000 [--key answer_key.json]
"""
import argparse
import json
import time

from parser import (GUIDE_LEN, WINDOW_LEN, Chunk, chunk_genome, encode,
                    reverse_complement)

G = 2  # code for 'G'
C = 1  # code for 'C'
N = 4  # code for 'N'


def search_chunk(chunk: Chunk, guide: bytes, guide_rc: bytes, max_mm: int):
    """Return [(chrom, position, strand, mismatches)] for one chunk."""
    data = chunk.data.tobytes()          # indexing bytes gives plain ints (fast in pure Python)
    hits = []
    last_start = min(chunk.n_owned, len(data) - WINDOW_LEN + 1)
    for w in range(last_start):
        window = data[w:w + WINDOW_LEN]
        if N in window:
            continue

        # ---- forward strand: [20-mer][N G G]
        if window[21] == G and window[22] == G:
            mm = 0
            for j in range(GUIDE_LEN):
                if window[j] != guide[j]:
                    mm += 1
                    if mm > max_mm:
                        break
            if mm <= max_mm:
                hits.append((chunk.chrom, chunk.start + w, "+", mm))

        # ---- reverse strand: [C C N][revcomp 20-mer]
        if window[0] == C and window[1] == C:
            mm = 0
            for j in range(GUIDE_LEN):
                if window[3 + j] != guide_rc[j]:
                    mm += 1
                    if mm > max_mm:
                        break
            if mm <= max_mm:
                hits.append((chunk.chrom, chunk.start + w + 3, "-", mm))
    return hits


def search_chunk_naive(chunk: Chunk, guide: bytes, guide_rc: bytes, max_mm: int):
    """
    Textbook version: for EVERY window, run the 20-letter mismatch loop first (early exit),
    and only check the PAM afterwards. No PAM pre-filter, no N-slice shortcut: this is what
    "standard Python loops" means in the project brief, and it is also the order the GPU kernel
    works in. Results are identical to search_chunk(); only the speed differs.
    """
    data = chunk.data.tobytes()
    hits = []
    last_start = min(chunk.n_owned, len(data) - WINDOW_LEN + 1)
    for w in range(last_start):
        # forward: compare 20-mer, then PAM, then N-check
        mm = 0
        for j in range(GUIDE_LEN):
            if data[w + j] != guide[j]:
                mm += 1
                if mm > max_mm:
                    break
        if mm <= max_mm and data[w + 21] == G and data[w + 22] == G \
                and N not in data[w:w + WINDOW_LEN]:
            hits.append((chunk.chrom, chunk.start + w, "+", mm))
        # reverse: compare revcomp 20-mer, then CC, then N-check
        mm = 0
        for j in range(GUIDE_LEN):
            if data[w + 3 + j] != guide_rc[j]:
                mm += 1
                if mm > max_mm:
                    break
        if mm <= max_mm and data[w] == C and data[w + 1] == C \
                and N not in data[w:w + WINDOW_LEN]:
            hits.append((chunk.chrom, chunk.start + w + 3, "-", mm))
    return hits


def search_chunks(chunks, guide: str, max_mm: int, naive: bool = False):
    """Search an iterable of Chunk objects; returns a sorted hit list. naive=True uses the textbook loop."""
    if len(guide) != GUIDE_LEN:
        raise ValueError(f"guide must be {GUIDE_LEN} letters")
    g = encode(guide.upper())
    if (g == N).any():
        raise ValueError("guide must contain only A/C/G/T")
    guide_b = g.tobytes()
    guide_rc_b = reverse_complement(g).tobytes()
    fn = search_chunk_naive if naive else search_chunk
    hits = []
    for ch in chunks:
        hits.extend(fn(ch, guide_b, guide_rc_b, max_mm))
    return sorted(hits, key=lambda h: (h[0], h[1], h[2]))


def search_genome(fasta_path: str, guide: str, max_mm: int,
                  chunk_size: int = 1_000_000, region=None):
    return search_chunks(chunk_genome(fasta_path, chunk_size, region=region), guide, max_mm)


def check_against_key(hits, key_path: str) -> bool:
    """Every hit in the answer key must appear in our results with the same mismatch count."""
    key = json.load(open(key_path))
    found = {(h[1], h[2]): h[3] for h in hits}
    missing = [k for k in key["hits"]
               if found.get((k["position"], k["strand"])) != k["mismatches"]]
    print(f"Answer key: {len(key['hits']) - len(missing)}/{len(key['hits'])} found correctly")
    for m in missing[:10]:
        print("  MISSING/WRONG:", m)
    return not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--guide", required=True)
    ap.add_argument("--max-mm", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--key", default=None, help="answer_key.json from find_targets.py")
    a = ap.parse_args()

    region = (a.start, a.end) if (a.start or a.end) else None
    t0 = time.perf_counter()
    hits = search_genome(a.fasta, a.guide, a.max_mm, a.chunk_size, region)
    dt = time.perf_counter() - t0

    size = (a.end - a.start) if a.end else None
    print(f"{len(hits)} hits in {dt:.2f}s" + (f" ({size / dt:,.0f} bases/s)" if size else ""))
    for h in hits[:20]:
        print("  ", h)
    if a.key:
        check_against_key(hits, a.key)


if __name__ == "__main__":
    main()

