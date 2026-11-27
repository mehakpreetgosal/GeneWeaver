"""
GPU correctness tests: the CUDA kernel must return EXACTLY the same hits as the CPU code.
Runs on a real GPU, or on Numba's CUDA simulator if no GPU is present (see conftest.py).
Run:  pytest tests/test_gpu.py -v
"""
import random

import numpy as np
import pytest

import cpu_baseline
import gpu_search
from parser import chunk_array, encode

GUIDE = "GACGCATAAAGATGAGACGC"
COMP = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s.translate(COMP)[::-1]


def mutate(s, positions):
    s = list(s)
    for p in positions:
        s[p] = "A" if s[p] != "A" else "C"
    return "".join(s)


def filler(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice("AT") for _ in range(n))


def both(seq, max_mm, chunk_size=None):
    arr = encode(seq)
    cs = chunk_size or len(arr)
    cpu = cpu_baseline.search_chunks(chunk_array("t", arr, cs), GUIDE, max_mm)
    gpu = gpu_search.search_chunks(chunk_array("t", arr, cs), GUIDE, max_mm, max_chunk_bases=cs)
    return cpu, gpu


def test_perfect_forward_hit():
    cpu, gpu = both(filler(60) + GUIDE + "TGG" + filler(60), 0)
    assert gpu == cpu and [(h[1], h[2], h[3]) for h in gpu] == [(60, "+", 0)]


def test_reverse_strand_hit():
    cpu, gpu = both(filler(60) + "CCA" + rc(GUIDE) + filler(60), 0)
    assert gpu == cpu and [(h[1], h[2], h[3]) for h in gpu] == [(63, "-", 0)]


def test_mismatch_limit():
    seq = filler(60) + mutate(GUIDE, [2, 9]) + "AGG" + filler(60)
    assert both(seq, 2)[1] != []
    assert both(seq, 1)[1] == []


def test_no_pam_no_hit_and_N_skipped():
    assert both(filler(60) + GUIDE + "TAA" + filler(60), 4)[1] == []
    assert both(filler(60) + "N" + GUIDE[1:] + "TGG" + filler(60), 4)[1] == []


@pytest.mark.parametrize("chunk_size", [40, 61, 64, 65, 500])
def test_chunk_boundaries_match_cpu(chunk_size):
    seq = filler(60) + GUIDE + "TGG" + filler(40) + "CCA" + rc(GUIDE) + filler(30)
    cpu, gpu = both(seq, 0, chunk_size)
    assert len(cpu) == 2 and gpu == cpu


def test_hit_buffer_overflow_is_recovered():
    """Tiny initial capacity forces the retry path; no hit may be lost."""
    seq = "".join(GUIDE + "TGG" + filler(10, i) for i in range(20))
    arr = encode(seq)
    s = gpu_search.GPUSearcher(GUIDE, 0, max_chunk_bases=len(arr), capacity=3)
    hits = s.search_chunks(chunk_array("t", arr, len(arr)))
    assert hits == cpu_baseline.search_chunks(chunk_array("t", arr, len(arr)), GUIDE, 0)
    assert len(hits) == 20 and s.capacity >= 20


@pytest.mark.parametrize("seed", range(3))
def test_random_dna_identical_to_cpu(seed):
    rng = random.Random(seed)
    seq = [rng.choice("ACGTacgtN") for _ in range(4000)]
    for k, pos in enumerate([200, 1500, 3000]):
        site = mutate(GUIDE, list(range(k + 1)))
        seq[pos:pos + 23] = site + "TGG"
        seq[pos + 400:pos + 423] = "CCT" + rc(site)
    cpu, gpu = both("".join(seq), 4, 700)
    assert len(cpu) >= 6 and gpu == cpu
