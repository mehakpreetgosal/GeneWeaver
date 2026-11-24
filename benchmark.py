"""
benchmark.py - Week 1 benchmark: pure Python vs NumPy on REAL genome sequence, extrapolated to 3.2 billion bases.

What is timed:   chunking + searching an already-encoded array (FASTA reading/encoding is timed
                 separately, once, and reported on its own).
What is counted: "bases" = non-N bases in the searched region, so long N gaps do not inflate speed.

Usage:
  python benchmark.py --fasta data/chr21.fa --guide GACGCATAAAGATGAGACGC
  python benchmark.py --fasta data/chr21.fa --guide ... --python-sizes 1000000 10000000 --max-mm 3
  python benchmark.py --fasta data/chr21.fa --guide ... --skip-python      # NumPy only (quick)

Output: a table on screen plus benchmark_results.json (reuse it in your report and in Week 2).
"""
import argparse
import json
import platform
import time

import numpy as np
from Bio import SeqIO

import cpu_baseline
import numpy_search
from parser import N_CODE, chunk_array, encode

GENOME_SIZE = 3_200_000_000


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def find_clean_start(arr: np.ndarray, length: int, max_n_frac: float = 0.01) -> int:
    """First offset (stepping by 1 MB) where a window of `length` bases has < max_n_frac N."""
    step = 1_000_000
    for s in range(0, max(1, len(arr) - length + 1), step):
        if (arr[s:s + length] == N_CODE).mean() < max_n_frac:
            return s
    raise RuntimeError(f"no clean {length:,}-base region found; use a larger chromosome or smaller size")


def timed(fn, repeats: int = 1):
    best, out = None, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def run_case(label, impl, arr, guide, max_mm, chunk_size, repeats, **kw):
    n_bases = int((arr != N_CODE).sum())
    dt, hits = timed(lambda: impl.search_chunks(chunk_array("bench", arr, chunk_size), guide, max_mm, **kw),
                     repeats)
    bps = n_bases / dt
    return {"method": label, "bases": n_bases, "seconds": dt, "bases_per_sec": bps,
            "hits": len(hits), "est_3_2B_seconds": GENOME_SIZE / bps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--guide", required=True)
    ap.add_argument("--max-mm", type=int, default=3)
    ap.add_argument("--python-sizes", type=int, nargs="+", default=[1_000_000, 10_000_000],
                    help="slice sizes (bases) for pure Python")
    ap.add_argument("--numpy-slice", type=int, default=None,
                    help="also time NumPy on a clean slice of this size (default: full chromosome only)")
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--out", default="benchmark_results.json")
    a = ap.parse_args()
    guide = a.guide.upper()

    # ---- load + encode once (timed separately)
    t0 = time.perf_counter()
    rec = next(SeqIO.parse(a.fasta, "fasta"))
    arr = encode(str(rec.seq))
    parse_s = time.perf_counter() - t0
    real = int((arr != N_CODE).sum())
    print(f"{rec.id}: {len(arr):,} bases ({real:,} non-N). Parse+encode: {parse_s:.2f}s "
          f"({real / parse_s:,.0f} bases/s)\n")

    # ---- warm-up (imports, caches) on a tiny clean slice
    s0 = find_clean_start(arr, min(200_000, len(arr) // 2), 0.5)
    numpy_search.search_chunks(chunk_array("w", arr[s0:s0 + 100_000], 100_000), guide, a.max_mm)

    results = []

    # ---- pure Python on clean slices
    if not a.skip_python:
        for size in a.python_sizes:
            start = find_clean_start(arr, size)
            sl = arr[start:start + size]
            print(f"pure Python on {size:,} bases (offset {start:,}) ...", flush=True)
            for naive in (False, True):
                tag = "naive" if naive else "PAM-first"
                r = run_case(f"pure Python {tag} ({size / 1e6:g} MB)", cpu_baseline, sl, guide,
                             a.max_mm, a.chunk_size, 1, naive=naive)
                results.append(r)
                print(f"  {tag}: {r['seconds']:.2f}s, {r['hits']} hits")

    # ---- NumPy: optional clean slice, then full chromosome (best of 3)
    if a.numpy_slice:
        start = find_clean_start(arr, a.numpy_slice)
        sl = arr[start:start + a.numpy_slice]
        results.append(run_case(f"NumPy ({a.numpy_slice / 1e6:g} MB slice)", numpy_search, sl, guide,
                                a.max_mm, a.chunk_size, 3))
    print(f"NumPy on full {rec.id} ...", flush=True)
    r = run_case(f"NumPy (full {rec.id})", numpy_search, arr, guide, a.max_mm, a.chunk_size, 3)
    results.append(r)
    print(f"  {r['seconds']:.2f}s, {r['hits']} hits\n")

    # ---- cross-check: Python and NumPy must agree on the smallest slice
    if not a.skip_python:
        size = min(a.python_sizes)
        sl = arr[find_clean_start(arr, size):][:size]
        h1 = cpu_baseline.search_chunks(chunk_array("x", sl, a.chunk_size), guide, a.max_mm)
        h3 = cpu_baseline.search_chunks(chunk_array("x", sl, a.chunk_size), guide, a.max_mm, naive=True)
        assert h1 == h3, "naive and PAM-first Python disagree"
        h2 = numpy_search.search_chunks(chunk_array("x", sl, a.chunk_size), guide, a.max_mm)
        print("Python == NumPy on", f"{size:,}-base slice:", h1 == h2, "\n")
        assert h1 == h2, "implementations disagree - fix before trusting any timing"

    # ---- report
    hdr = f"{'Method':38} {'Bases':>13} {'Time':>10} {'Bases/s':>14} {'Est. for 3.2B bases':>20}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['method']:38} {r['bases']:>13,} {r['seconds']:>9.2f}s {r['bases_per_sec']:>14,.0f} "
              f"{human_time(r['est_3_2B_seconds']):>20}")
    py = [r for r in results if r["method"].startswith("pure Python naive")]
    npf = [r for r in results if r["method"].startswith("NumPy")]
    if py and npf:
        print(f"\nNumPy vs naive pure Python speedup: {npf[-1]['bases_per_sec'] / py[-1]['bases_per_sec']:,.0f}x")

    meta = {"fasta": a.fasta, "guide": guide, "max_mm": a.max_mm, "chunk_size": a.chunk_size,
            "parse_encode_seconds": parse_s, "python": platform.python_version(),
            "numpy": np.__version__, "machine": platform.processor() or platform.machine()}
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
