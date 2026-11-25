# GeneWeaver

GPU-accelerated CRISPR off-target search. Given a 20-letter guide RNA sequence, GeneWeaver scans a
genome for every site that matches it with up to *k* mismatches (and a valid `NGG` PAM), on both strands.

The tool only **finds and reports** sites. It does not edit anything, and it uses no machine learning:
the search is exact counting, and the goal is speed.

**Status:** Week 1 complete (parser, pure-Python baseline, NumPy baseline, tests, benchmark).
Week 2 onward adds Numba CUDA kernels, a Dask scheduler, scoring and a Textual dashboard.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is required.

## Get a real genome

Genome files are large and are **not** stored in this repo (`data/` is git-ignored).

```bash
mkdir -p data && cd data
curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz
gunzip chr21.fa.gz
```

chr21 is about 46 MB. It starts with several million `N` bases (unknown sequence), which the code skips.
Lowercase letters (soft-masked repeats) are real DNA and are treated like uppercase.

## Conventions

All modules use these. Mixing them up produces hits that don't line up.

| Item | Rule |
|---|---|
| Coordinates | 0-based, forward strand |
| Encoding | A=0, C=1, G=2, T=3, N=4 (any other letter is treated as N) |
| Window | 23 bases = 20-letter guide + 3-letter PAM |
| `+` strand | forward text reads `[20-mer][NGG]`; hit position = window start |
| `-` strand | forward text reads `[CCN][reverse complement of 20-mer]`; hit position = window start + 3 |
| `N` bases | any window containing `N` is skipped |
| Mismatches | hits are kept if mismatches <= `max_mm` |
| Chunks | overlap by 22 bases; a chunk reports only windows starting before `n_owned` (no duplicates) |

## Project layout

```
geneweaver/
├── parser.py            FASTA -> encoded, overlapping chunks with global coordinates
├── find_targets.py      builds a trusted answer key with exact string matching
├── cpu_baseline.py      pure-Python search (PAM-first and naive versions)
├── numpy_search.py      vectorized NumPy search
├── benchmark.py         timing + extrapolation to 3.2 billion bases
├── tests/test_search.py pytest suite (runs against both implementations)
├── data/                genomes (not committed)
└── requirements.txt
```

## Usage

**1. Pick a guide** from a region with real sequence (not the leading `N`s):

```bash
python find_targets.py pick --fasta data/chr21.fa --start 20000000 --end 21000000
```

**2. Build an answer key** for that guide (known hits found by exact matching):

```bash
python find_targets.py key --fasta data/chr21.fa --guide YOUR20LETTERGUIDE \
    --start 20000000 --end 22000000 --out answer_key.json
```

The key is a lower bound: the search must find at least these hits. On real DNA it will find more.

**3. Run the pure-Python search** and check it against the key:

```bash
python cpu_baseline.py --fasta data/chr21.fa --guide YOUR20LETTERGUIDE --max-mm 3 \
    --start 20000000 --end 22000000 --key answer_key.json
```

**4. Run the tests:**

```bash
pytest -v
```

**5. Benchmark:**

```bash
python benchmark.py --fasta data/chr21.fa --guide YOUR20LETTERGUIDE --max-mm 3
```

This writes `benchmark_results.json`.

## Benchmark results (Week 1)

Fill this in from your own run. Report the hardware, because the numbers depend on it.

- CPU / RAM: _your machine here_
- Python / NumPy versions: _from benchmark_results.json_
- Guide: _your guide_, `max_mm` = _k_

| Method | Bases | Time | Bases/s | Est. time for 3.2B bases |
|---|---|---|---|---|
| Pure Python, naive | | | | |
| Pure Python, PAM-first | | | | |
| NumPy | | | | |

Notes:
- Extrapolation is `3.2e9 / bases_per_second`, a rough linear estimate.
- "Bases" counts non-`N` bases only.
- Search time is linear in the number of guides, so multi-guide scans are where a GPU matters most.

## Roadmap

- [x] Week 1: parser, CPU baselines, tests, benchmark
- [ ] Week 2: Numba CUDA kernel, host/device memory handling, Textual scaffold
- [ ] Mid-project: GPU vs CPU audit (target 100x), out-of-memory check
- [ ] Week 3: Dask scheduler across GPUs, MIT/CFD scoring
- [ ] Week 4: shared-memory optimization, final TUI with red mismatch highlighting
