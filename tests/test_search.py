"""
pytest tests for GeneWeaver Week 1. Run from the project root:   pytest -v
Every test runs against BOTH the pure-Python and the NumPy implementation.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cpu_baseline
import numpy_search
from parser import (Chunk, chunk_array, chunk_genome, encode, reverse_complement,
                    decode, stitch)

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
    """Random DNA with no accidental CC or GG so tests are not polluted by random PAMs."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(rng.choice("AT"))
    return "".join(out)


def run(impl, seq, max_mm, chunk_size=None):
    arr = encode(seq)
    cs = chunk_size or len(arr)
    return impl.search_chunks(chunk_array("t", arr, cs), GUIDE, max_mm)


def summary(hits):
    return [(h[1], h[2], h[3]) for h in hits]


IMPLS = [cpu_baseline, numpy_search]


@pytest.mark.parametrize("impl", IMPLS)
def test_perfect_match_found(impl):
    seq = filler(50) + GUIDE + "TGG" + filler(50)
    assert summary(run(impl, seq, 0)) == [(50, "+", 0)]


@pytest.mark.parametrize("impl", IMPLS)
def test_two_mismatches_depends_on_max_mm(impl):
    seq = filler(50) + mutate(GUIDE, [3, 10]) + "AGG" + filler(50)
    assert summary(run(impl, seq, 2)) == [(50, "+", 2)]
    assert run(impl, seq, 1) == []


@pytest.mark.parametrize("impl", IMPLS)
def test_reverse_strand_found(impl):
    seq = filler(50) + "CCA" + rc(GUIDE) + filler(50)
    assert summary(run(impl, seq, 0)) == [(53, "-", 0)]


@pytest.mark.parametrize("impl", IMPLS)
def test_no_pam_no_hit(impl):
    seq = filler(50) + GUIDE + "TAA" + filler(50)
    assert run(impl, seq, 4) == []


@pytest.mark.parametrize("impl", IMPLS)
def test_window_with_N_is_skipped(impl):
    bad = "N" + GUIDE[1:]
    seq = filler(50) + bad + "TGG" + filler(50)
    assert run(impl, seq, 4) == []
    seq2 = filler(50) + GUIDE + "TNG" + filler(50)       # N inside PAM region
    assert run(impl, seq2, 4) == []


@pytest.mark.parametrize("impl", IMPLS)
def test_lowercase_same_as_uppercase(impl):
    seq = filler(50) + GUIDE + "TGG" + filler(50)
    assert run(impl, seq.lower(), 0) == run(impl, seq, 0)
    assert len(run(impl, seq.lower(), 0)) == 1


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("chunk_size", [10, 33, 50, 51, 52, 60, 1000])
def test_hit_across_chunk_boundary_found_exactly_once(impl, chunk_size):
    seq = filler(50) + GUIDE + "TGG" + filler(50) + "CCA" + rc(GUIDE) + filler(20)
    expected = summary(run(impl, seq, 0))
    assert len(expected) == 2
    assert summary(run(impl, seq, 0, chunk_size)) == expected


def test_stitched_chunks_equal_original():
    seq = filler(1000, 3).replace("A", "N", 40) + "acgtn" * 20
    arr = encode(seq)
    chunks = list(chunk_array("t", arr, 97, skip_all_n=False))
    assert (stitch(chunks) == arr).all()


def test_encode_decode_and_revcomp():
    assert decode(encode("acgtnACGTN")) == "ACGTNACGTN"
    assert decode(reverse_complement(encode("AACGN"))) == "NCGTT"


def test_all_N_chunks_skipped():
    # chunk 100 covers bases 100..172 (all N, including its overlap) -> skipped.
    # chunk 150 is kept because its overlap reaches real bases starting at 200.
    arr = encode("N" * 200 + filler(100))
    chunks = list(chunk_array("t", arr, 50))
    assert [c.start for c in chunks] == [150, 200, 250]


@pytest.mark.parametrize("seed", range(5))
def test_python_and_numpy_identical_on_random_dna(seed):
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGTacgtN") for _ in range(20000))
    # plant some targets with various mismatches on both strands
    seq = list(seq)
    for k, pos in enumerate([500, 3000, 9000, 15000]):
        site = mutate(GUIDE, list(range(k)))
        seq[pos:pos + 23] = site + "TGG"
        seq[pos + 100:pos + 123] = "CCT" + rc(site)
    seq = "".join(seq)
    a = run(cpu_baseline, seq, 4, 1234)
    b = run(numpy_search, seq, 4, 1234)
    assert a == b
    assert len(a) >= 8
