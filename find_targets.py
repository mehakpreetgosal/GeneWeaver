"""
find_targets.py - build a trusted "answer key" for GeneWeaver using ONLY exact string matching.

Idea: exact matching with str.find() is simple and fast, so we can trust it.
To get known hits WITH mismatches, we deliberately mutate the guide (1, 2, 3 letters),
then search the genome for those mutated versions exactly. Any exact hit of a variant
is a real site that differs from the guide by exactly that many mismatches.

Conventions (write these in your README, keep them identical in every module):
  * Positions are 0-based, on the FORWARD strand of the chromosome.
  * Position = index of the first base of the 20-letter protospacer (as seen on the forward strand).
  * strand '+': forward text reads  [20-mer][N G G]
  * strand '-': forward text reads  [C C N][revcomp of 20-mer]; position is the start of the
                revcomp 20-mer (i.e. just after the CCN).
  * Windows (including the PAM) containing 'N' are skipped.
  * Lowercase (soft-masked) bases are treated as normal bases.

Usage:
  # 1) pick a guide from a region of the chromosome
  python find_targets.py pick --fasta data/chr21.fa --start 20000000 --end 21000000

  # 2) build the answer key for that guide
  python find_targets.py key --fasta data/chr21.fa --guide GACGCATAAAGATGAGACGC \
        --start 20000000 --end 22000000 --out answer_key.json
"""
import argparse
import json
import random

from Bio import SeqIO

COMP = str.maketrans("ACGT", "TGCA")
GUIDE_LEN = 20


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def load_chromosome(fasta_path: str, start: int = 0, end: int | None = None):
    """Return (chrom_name, uppercase sequence slice, offset). Uses the first record."""
    rec = next(SeqIO.parse(fasta_path, "fasta"))
    seq = str(rec.seq).upper()
    end = len(seq) if end is None else min(end, len(seq))
    return rec.id, seq[start:end], start


def find_all(text: str, pattern: str):
    """Yield every (possibly overlapping) exact match index of pattern in text."""
    i = text.find(pattern)
    while i != -1:
        yield i
        i = text.find(pattern, i + 1)


# ---------------------------------------------------------------- pick a guide
def pick_guide(seq: str, offset: int, n: int = 5, seed: int = 1):
    """Find n 20-mers followed by NGG with no N and 40-60% GC. They exist at least once (here)."""
    rng = random.Random(seed)
    found = []
    tries = 0
    while len(found) < n and tries < 200_000:
        tries += 1
        i = rng.randrange(0, len(seq) - GUIDE_LEN - 3)
        g, pam = seq[i:i + GUIDE_LEN], seq[i + GUIDE_LEN:i + GUIDE_LEN + 3]
        if set(g + pam) - set("ACGT"):
            continue
        if pam[1:] != "GG":
            continue
        gc = (g.count("G") + g.count("C")) / GUIDE_LEN
        if 0.40 <= gc <= 0.60:
            found.append((g, pam, offset + i))
    return found


# ---------------------------------------------------------------- answer key
def make_variants(guide: str, n_mm: int, how_many: int, rng: random.Random):
    """Return up to how_many distinct variants of guide with exactly n_mm substitutions."""
    variants = set()
    attempts = 0
    while len(variants) < how_many and attempts < 1000:
        attempts += 1
        v = list(guide)
        for pos in rng.sample(range(GUIDE_LEN), n_mm):
            v[pos] = rng.choice([b for b in "ACGT" if b != guide[pos]])
        variants.add("".join(v))
    return sorted(variants)


def search_both_strands(seq: str, offset: int, variant: str):
    """Exact-search one variant on both strands, enforcing the NGG PAM and no-N rule."""
    hits = []
    # forward strand: [variant][NGG]
    for i in find_all(seq, variant):
        pam = seq[i + GUIDE_LEN:i + GUIDE_LEN + 3]
        if len(pam) == 3 and pam[1:] == "GG" and "N" not in pam:
            hits.append((offset + i, "+"))
    # reverse strand: forward text is [CCN][revcomp(variant)]
    rc = revcomp(variant)
    for p in find_all(seq, rc):
        pre = seq[p - 3:p] if p >= 3 else ""
        if len(pre) == 3 and pre[:2] == "CC" and "N" not in pre:
            hits.append((offset + p, "-"))
    return hits


def build_key(seq: str, offset: int, guide: str, max_mm: int = 3,
              per_level: int = 30, seed: int = 7):
    rng = random.Random(seed)
    found = {}  # (pos, strand) -> mismatches
    for n_mm in range(0, max_mm + 1):
        variants = [guide] if n_mm == 0 else make_variants(guide, n_mm, per_level, rng)
        for v in variants:
            for pos, strand in search_both_strands(seq, offset, v):
                # recompute true mismatches against the guide (a site could match several variants)
                local = pos - offset
                if strand == "+":
                    site = seq[local:local + GUIDE_LEN]
                else:
                    site = revcomp(seq[local:local + GUIDE_LEN])
                found[(pos, strand)] = hamming(guide, site)
    return [{"position": p, "strand": s, "mismatches": m}
            for (p, s), m in sorted(found.items())]


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pick", help="pick candidate guides from a region")
    p1.add_argument("--fasta", required=True)
    p1.add_argument("--start", type=int, default=0)
    p1.add_argument("--end", type=int, default=None)
    p1.add_argument("-n", type=int, default=5)

    p2 = sub.add_parser("key", help="build answer key for a guide")
    p2.add_argument("--fasta", required=True)
    p2.add_argument("--guide", required=True)
    p2.add_argument("--start", type=int, default=0)
    p2.add_argument("--end", type=int, default=None)
    p2.add_argument("--max-mm", type=int, default=3)
    p2.add_argument("--per-level", type=int, default=30,
                    help="how many mutated variants to try per mismatch count")
    p2.add_argument("--out", default="answer_key.json")

    a = ap.parse_args()
    chrom, seq, offset = load_chromosome(a.fasta, a.start, a.end)

    if a.cmd == "pick":
        print(f"Candidates from {chrom}:{offset}-{offset + len(seq)}  (guide, PAM, position)")
        for g, pam, pos in pick_guide(seq, offset, a.n):
            print(f"  {g}  {pam}  pos={pos}")
    else:
        guide = a.guide.upper()
        assert len(guide) == GUIDE_LEN and set(guide) <= set("ACGT"), "guide must be 20 letters of ACGT"
        hits = build_key(seq, offset, guide, a.max_mm, a.per_level)
        out = {"chromosome": chrom, "guide": guide, "region": [offset, offset + len(seq)],
               "max_mm": a.max_mm, "hits": hits}
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        by_mm = {}
        for h in hits:
            by_mm[h["mismatches"]] = by_mm.get(h["mismatches"], 0) + 1
        print(f"Saved {len(hits)} hits to {a.out}; counts by mismatches: {dict(sorted(by_mm.items()))}")


if __name__ == "__main__":
    main()
