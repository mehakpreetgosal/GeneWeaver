"""
parser.py - GeneWeaver genomic parser: FASTA -> encoded, overlapping, globally-addressed chunks.

Encoding:  A=0  C=1  G=2  T=3  N=4 (anything else, e.g. R/Y/K/M, is also treated as N).
           Lowercase (soft-masked repeats) is real DNA and maps to the same codes as uppercase.

Conventions (same as find_targets.py):
  * Coordinates are 0-based, forward strand.
  * A "window" is GUIDE_LEN + PAM_LEN = 23 bases starting at window-start w.
        + strand: [20-mer][NGG]  -> hit position = w
        - strand: [CCN][revcomp 20-mer] -> hit position = w + 3
  * Windows containing N are skipped by the search code (the parser only supplies code 4).

Chunking and ownership:
  * Each chunk holds chunk_size bases + OVERLAP extra bases from the next chunk.
  * OVERLAP = WINDOW_LEN - 1 = 22, so any 23-base window that straddles a boundary lies fully
    inside one chunk.
  * To avoid reporting the same hit twice, a chunk "owns" only the windows whose START index is
    in [0, n_owned). The search code must ignore window starts >= n_owned.
  * Chunks that are entirely N are skipped (chr21 begins with millions of N).

Memory note: SeqIO.parse loads one whole record (chromosome) into RAM. That is fine for chr21/22
(~50 MB). For chr1 (~250 MB) it still works with a few GB of RAM. For the full genome, records
are processed one chromosome at a time, so peak memory is the largest chromosome.
"""
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from Bio import SeqIO

GUIDE_LEN = 20
PAM_LEN = 3
WINDOW_LEN = GUIDE_LEN + PAM_LEN          # 23
OVERLAP = WINDOW_LEN - 1                  # 22
N_CODE = 4

# 256-entry lookup table: ASCII byte -> code. Default everything to N.
_LUT = np.full(256, N_CODE, dtype=np.uint8)
for _code, _base in enumerate("ACGT"):
    _LUT[ord(_base)] = _code
    _LUT[ord(_base.lower())] = _code

# Decode table and complement table (complement of N stays N)
_DECODE = np.frombuffer(b"ACGTN", dtype=np.uint8)
_COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


@dataclass
class Chunk:
    chrom: str
    start: int            # global 0-based coordinate of data[0]
    data: np.ndarray      # uint8 codes, length <= chunk_size + OVERLAP
    n_owned: int          # window starts 0..n_owned-1 belong to this chunk


def encode(seq: str) -> np.ndarray:
    """String of DNA letters -> uint8 array of codes."""
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode(arr: np.ndarray) -> str:
    """uint8 codes -> string of uppercase letters."""
    return _DECODE[arr].tobytes().decode("ascii")


def reverse_complement(arr: np.ndarray) -> np.ndarray:
    """Reverse complement of an encoded array."""
    return _COMP[arr][::-1].copy()


def chunk_array(chrom: str, arr: np.ndarray, chunk_size: int,
                skip_all_n: bool = True) -> Iterator[Chunk]:
    """Cut one encoded chromosome into overlapping chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    total = len(arr)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size + OVERLAP, total)
        data = arr[start:end]
        n_owned = min(chunk_size, total - start)
        if skip_all_n and not (data != N_CODE).any():
            continue
        yield Chunk(chrom, start, data, n_owned)


def chunk_genome(fasta_path: str, chunk_size: int = 10_000_000,
                 skip_all_n: bool = True, region: tuple[int, int] | None = None
                 ) -> Iterator[Chunk]:
    """
    Stream chunks from a FASTA file, one record (chromosome) at a time.
    region=(start, end) restricts to that slice of each record (handy for small tests);
    returned chunk.start values remain GLOBAL coordinates.
    """
    for rec in SeqIO.parse(fasta_path, "fasta"):
        arr = encode(str(rec.seq))
        offset = 0
        if region is not None:
            offset = region[0]
            arr = arr[region[0]:region[1]]
        for ch in chunk_array(rec.id, arr, chunk_size, skip_all_n):
            ch.start += offset
            yield ch


def stitch(chunks) -> np.ndarray:
    """Rebuild the sequence from the owned part of each chunk (used for testing)."""
    parts = []
    for ch in chunks:
        parts.append(ch.data[:ch.n_owned])
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)


def parse_summary(fasta_path: str) -> None:
    """Print per-record length, N fraction and lowercase fraction (quick sanity check on real data)."""
    for rec in SeqIO.parse(fasta_path, "fasta"):
        s = str(rec.seq)
        arr = encode(s)
        n_frac = float((arr == N_CODE).mean())
        lower = sum(c.islower() for c in s[:5_000_000]) / max(1, min(len(s), 5_000_000))
        print(f"{rec.id}: {len(s):,} bases | N: {n_frac:.1%} | lowercase (first 5M): {lower:.1%}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/chr21.fa"
    parse_summary(path)
    n = total = 0
    for ch in chunk_genome(path, chunk_size=5_000_000):
        n += 1
        total += ch.n_owned
    print(f"{n} non-empty chunks, {total:,} owned bases")
