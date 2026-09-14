#!/usr/bin/env python3
"""Difficulty strata for every utterance — axis 3 (README §4).

Splits the evaluation by how hard the signs in an utterance are to *read*, which
is what separates decoder credit from encoder credit in the transfer table. See
`tsl.strata` for why these two measures and not others.

    python3 scripts/12_strata.py                    # build data/strata.json
    python3 scripts/12_strata.py --audit            # + coverage and reliability
    python3 scripts/12_strata.py --audit-citation   # reproduce the rejected measure

Building the strata is seconds and needs nothing but the two databases.
`--audit-citation` is the slow one: it decodes ~1,200 dictionary clips to
reproduce the citation-duration measure that was tried and rejected, and caches
the result in `data/citation_ms.json`.

CPU only. Nothing here touches the GPU or any trained model, and every number is
computed before a model runs, so splitting results by these strata cannot be
circular.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl import strata as S  # noqa: E402
from tsl.db import Corpus  # noqa: E402


def corpus_tokens() -> dict[str, list[tuple[str, int]]]:
    """utterance key -> [(gloss, duration_ms)], tier 1 only."""
    import sqlite3

    from tsl.db import DB_PATH

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    out: dict[str, list[tuple[str, int]]] = {}
    for uuid, seq, word, t1, t2 in con.execute(
        "SELECT uuid, seq, word, t1, t2 FROM tokens "
        "WHERE hand_tier = 1 AND t1 IS NOT NULL AND t2 > t1"
    ):
        out.setdefault(f"{uuid}:{seq:03d}", []).append(((word or "").strip(), t2 - t1))
    return out


def build_citation(glosses: set[str], phon: S.Phonology, cache: Path,
                   rel_thr: float, workers: int) -> dict[str, float]:
    """Active-span citation duration per gloss, measured once and cached."""
    have: dict[str, float] = {}
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("rel_thr") == rel_thr:
            have = blob["ms"]
    todo = sorted(g for g in glosses if g not in have and g in phon.clip)
    if todo:
        print(f"measuring citation duration for {len(todo)} signs "
              f"({sum(len(phon.clip[g]) for g in todo)} clips, rel_thr={rel_thr})",
              flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for g, ms in zip(todo, ex.map(
                    lambda g: S.citation_ms(g, phon, rel_thr=rel_thr), todo)):
                if ms:
                    have[g] = ms
        cache.write_text(json.dumps({"rel_thr": rel_thr, "ms": have},
                                    ensure_ascii=False, indent=0))
    return have


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, without pulling in scipy for one number."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):  # average ranks within ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--rel-thr", type=float, default=S.REL_THR)
    ap.add_argument("--min-covered", type=int, default=S.MIN_COVERED)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--audit-citation", action="store_true",
                    help="decode the dictionary clips and reproduce the "
                         "citation-duration measure that was rejected (slow)")
    a = ap.parse_args()
    out = a.out or a.data / "strata.json"
    cache = a.cache or a.data / "citation_ms.json"

    phon = S.Phonology()
    toks = corpus_tokens()
    records = {json.loads(l)["key"] for l in (a.data / "records.jsonl").open()}
    toks = {k: v for k, v in toks.items() if k in records}
    print(f"twtsl: {len(phon.by_gloss)} coded signs, "
          f"{len(phon.size)} distinct slot-1 signatures, "
          f"neighbourhood median {st.median(phon.size.values()):.0f} max {max(phon.size.values())}")

    ref = S.type_medians(toks)
    strata = S.record_strata(toks, phon, ref, a.min_covered)
    cuts = [S.add_classes(strata, "density"), S.add_classes(strata, "reduction")]

    n_tok = sum(len(v) for v in toks.values())
    cov_d = sum(r["n_density"] for r in strata.values())
    cov_r = sum(r["n_reduction"] for r in strata.values())
    print(f"records {len(strata)}  tokens {n_tok}  "
          f"reference types {len(ref)} (>= {S.MIN_TOKENS_PER_TYPE} tokens each)")
    print(f"  density   covers {cov_d} tokens ({cov_d/n_tok:.1%}), "
          f"{sum(r['density_ok'] for r in strata.values())} records classed")
    print(f"  reduction covers {cov_r} tokens ({cov_r/n_tok:.1%}), "
          f"{sum(r['reduction_ok'] for r in strata.values())} records classed")
    for c in cuts:
        if c["bins"]:
            inside = f" within {c['within']} bins" if c["within"] else ""
            print(f"  {c['field']:9s} terciles{inside}: " + "  ".join(
                f"[{b['n']}] {b['cuts'][0]:.2f}/{b['cuts'][1]:.2f}" for b in c["bins"]))
    for field in ("density", "reduction"):
        by = {}
        for r in strata.values():
            k = r.get(f"{field}_class")
            if k:
                by.setdefault(k, []).append(r["dur_ms"])
        if by:
            print(f"  {field:9s} mean utterance duration by class: " + "  ".join(
                f"{k} {sum(v)/len(v):.0f}ms n={len(v)}"
                for k, v in sorted(by.items())))

    if a.audit:
        audit(strata)

    for r in strata.values():
        r.pop("_rel", None)
    out.write_text(json.dumps(
        {"min_covered": a.min_covered, "min_tokens_per_type": S.MIN_TOKENS_PER_TYPE,
         "cuts": cuts, "strata": strata}, ensure_ascii=False))
    print(f"-> {out}")

    if a.audit_citation:
        glosses = {S.normalise_gloss(g) for v in toks.values() for g, _ in v}
        audit_citation(toks, phon,
                       build_citation(glosses, phon, cache, a.rel_thr, a.workers))
    return 0


def audit(strata: dict, min_tokens: int = 4, seed: int = 0) -> None:
    """Split-half reliability of the utterance-level reduction measure.

    An utterance's tokens are split at random into two halves and the two means
    are correlated: if the measure is a property of the utterance rather than of
    whichever tokens happened to be counted, the halves agree. Spearman-Brown
    steps that up to the reliability of the full-length measure, which is the
    number that matters — a stratum built on an unreliable measure produces
    strata that differ by nothing, and any effect found in it is noise.
    """
    import random

    print("\naudit: gloss normalisation is idempotent")
    phon = S.Phonology()
    bad = [g for g in phon.by_gloss if S.normalise_gloss(g) != g]
    print(f"  {len(bad)} of {len(phon.by_gloss)} twtsl keys change under a second "
          f"pass{': ' + repr(bad[:4]) if bad else ' — none, so every stored key is reachable'}")
    unreachable = [g for g in phon.by_gloss if phon.density(g) is None]
    print(f"  {len(unreachable)} stored signs that a lookup cannot reach")

    print("\naudit: split-half reliability of the reduction measure")
    rng = random.Random(seed)
    xs, ys = [], []
    for r in strata.values():
        v = r.get("_rel")
        if not v or len(v) < min_tokens:
            continue
        w = list(v)
        rng.shuffle(w)
        h = len(w) // 2
        xs.append(st.mean(w[:h]))
        ys.append(st.mean(w[h: 2 * h]))
    if len(xs) < 10:
        print("  not enough multi-token utterances to test")
        return
    rho = spearman(xs, ys)
    print(f"  n={len(xs)} utterances with >= {min_tokens} covered tokens")
    print(f"  Spearman {rho:+.3f}  ->  Spearman-Brown {2*rho/(1+rho):+.3f}")


def audit_citation(toks, phon: S.Phonology, citation: dict[str, float]) -> None:
    """Reproduce the rejected measure, so the rejection is checkable.

    A sign's citation duration and its median duration in running discourse are
    two independent recordings of the same sign, so a measure that finds the sign
    must correlate across signs and one that finds clip padding will not. Both
    the single-burst measure and the raw clip length it replaced are reported.
    """
    import subprocess

    running: dict[str, list[int]] = {}
    for v in toks.values():
        for g, dur in v:
            running.setdefault(S.normalise_gloss(g), []).append(dur)
    shared = sorted(g for g in citation if len(running.get(g, [])) >= 3)
    x = [citation[g] for g in shared]
    y = [st.median(running[g]) for g in shared]
    print(f"\naudit-citation: {len(shared)} signs with a citation measure "
          f"and >=3 running tokens")
    print(f"  citation burst ms: median {st.median(x):.0f}   "
          f"running median ms: {st.median(y):.0f}   ratio {st.median(y)/st.median(x):.2f}")
    print(f"  Spearman(citation burst,  running median) = {spearman(x, y):+.3f}")

    raw = {}
    for g in shared[:400]:
        p = S.TWTSL_MEDIA / f"{phon.clip[g][0]}.mp4"
        try:
            raw[g] = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(p)],
                capture_output=True, text=True, timeout=20).stdout.strip()) * 1000
        except Exception:
            pass
    if raw:
        sub = sorted(raw)
        print(f"  Spearman(raw clip length, running median) = "
              f"{spearman([raw[g] for g in sub], [st.median(running[g]) for g in sub]):+.3f}"
              f"   (n={len(sub)})")
    print("  -> rejected; the built strata use within-corpus relative duration")


if __name__ == "__main__":
    raise SystemExit(main())
