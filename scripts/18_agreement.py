#!/usr/bin/env python3
"""Do the collapsed models share an error pattern, or one and the same solution?

The paper says the three ASL rows "converged on one and the same solution", and
supports it with an agreement figure of 92.0-93.7%. That figure is computed on
*hit/miss*: two systems agree on an item when both are right or both are wrong.
It is a real observation and it is weaker than the sentence it carries -- three
systems that find the same items hard will share a hit/miss pattern while
choosing different wrong candidates, which is a shared difficulty profile, not a
shared solution.

The stronger statistic costs no GPU, because `05_eval.py` now records which of
the four candidates each model picked (`pick`) and the four scores behind it:

    hit agreement       both right or both wrong          -- what the paper had
    pick agreement      the SAME candidate, right or wrong -- what it claimed
    kappa               pick agreement corrected for the marginals, since a pair
                        of systems that both answer "candidate 0" most of the
                        time agree often for a reason that is not agreement
    wrong-pick match    of the items where BOTH are wrong, the share where they
                        chose the same wrong candidate. This is the sharpest of
                        the four: distractors are item-specific, so two systems
                        landing on the same one are running the same function of
                        the same input, and it cannot be produced by both being
                        at chance
    score rho           mean Spearman correlation between the two systems'
                        four-candidate score vectors, item by item

    python3 scripts/18_agreement.py --data data/lex --out data/agreement.json

Reads `eval_clip.json` (which carries `pick`), pools folds -- every item is held
out exactly once, so pooling gives one answer per item per system -- and refuses
to compare two cells that do not cover the same items.
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = re.compile(r"^f[012]_local_.+_full_k4_s\d+$")


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation of two four-element score vectors."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        # Average ties, which matter here only for degenerate score vectors.
        seen: dict[float, list[int]] = collections.defaultdict(list)
        for i, x in enumerate(v):
            seen[x].append(i)
        for idxs in seen.values():
            if len(idxs) > 1:
                m = sum(r[i] for i in idxs) / len(idxs)
                for i in idxs:
                    r[i] = m
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def load(fname: str, condition: str, seed: int) -> dict[str, dict[str, dict]]:
    """init -> key -> row, pooled over folds. One row per item per system."""
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for p in sorted(glob.glob(str(ROOT / "runs" / "f[012]_local_*" / fname))):
        d = json.loads(Path(p).read_text())
        run = d.get("run", Path(p).parent.name)
        if not CANONICAL.match(run) or d.get("seed", 0) != seed:
            continue
        init = d["init"]
        if d["cfg"].get("init_parts", "all") != "all":
            init = f"{init}-{d['cfg']['init_parts']}"
        if d["cfg"].get("init_mt5"):
            init = f"{init}+{d['cfg']['init_mt5']}-{d['cfg'].get('init_mt5_parts', 'mt5')}"
        for r in d["contrastive"]:
            if r["condition"] != condition or "pick" not in r:
                continue
            out[init][r["key"]] = r
    return out


def pair_stats(A: dict[str, dict], B: dict[str, dict]) -> dict:
    keys = sorted(set(A) & set(B))
    n = len(keys)
    if not n:
        return {"n": 0}
    hit_agree = sum(bool(A[k]["hit"]) == bool(B[k]["hit"]) for k in keys)
    pick_agree = sum(A[k]["pick"] == B[k]["pick"] for k in keys)

    # Cohen's kappa over the four-way pick. The marginals are what makes this
    # necessary: a system answering candidate 0 on 60% of items agrees with
    # another such system 36% of the time for no shared reason at all.
    ca = collections.Counter(A[k]["pick"] for k in keys)
    cb = collections.Counter(B[k]["pick"] for k in keys)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    po = pick_agree / n
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    both_wrong = [k for k in keys if not A[k]["hit"] and not B[k]["hit"]]
    wrong_match = sum(A[k]["pick"] == B[k]["pick"] for k in both_wrong)

    rhos = [spearman(A[k]["scores"], B[k]["scores"]) for k in keys
            if A[k].get("scores") and B[k].get("scores")
            and len(A[k]["scores"]) == len(B[k]["scores"]) >= 2]
    return {
        "n": n,
        "hit_agree": 100 * hit_agree / n,
        "pick_agree": 100 * po,
        "pick_agree_chance": 100 * pe,
        "kappa": kappa,
        "both_wrong_n": len(both_wrong),
        "wrong_pick_match": 100 * wrong_match / len(both_wrong) if both_wrong else None,
        "score_rho": sum(rhos) / len(rhos) if rhos else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval_clip.json")
    ap.add_argument("--condition", default="local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "agreement.json")
    a = ap.parse_args()

    cells = load(a.file, a.condition, a.seed)
    if not cells:
        raise SystemExit(f"no {a.file} carries condition {a.condition!r} with a "
                         "`pick` field — re-run 05_eval.py")
    print(f"{len(cells)} cells: {', '.join(sorted(cells))}\n")

    pairs = {}
    for x, y in itertools.combinations(sorted(cells), 2):
        s = pair_stats(cells[x], cells[y])
        if s["n"]:
            pairs[f"{x}|{y}"] = s

    # The groups the paper's sentences are about.
    groups = {
        "asl_intact": ["how2sign", "openasl", "wlasl"],
        "csl_intact": ["csl_daily", "csl_stage1"],
        "asl_vs_csl": None,   # filled below: every cross-group pair
    }
    summary = {}
    for g, members in groups.items():
        if members is None:
            continue
        got = [m for m in members if m in cells]
        ps = [pairs[f"{x}|{y}"] for x, y in itertools.combinations(sorted(got), 2)
              if f"{x}|{y}" in pairs]
        if ps:
            summary[g] = {
                "members": got,
                "pairs": len(ps),
                **{k: (min(p[k] for p in ps), max(p[k] for p in ps))
                   for k in ("hit_agree", "pick_agree", "pick_agree_chance",
                             "kappa")
                   if all(p.get(k) is not None for p in ps)},
                "wrong_pick_match": (
                    min(p["wrong_pick_match"] for p in ps),
                    max(p["wrong_pick_match"] for p in ps))
                if all(p.get("wrong_pick_match") is not None for p in ps) else None,
            }
    # Written as a loop, not a comprehension: the comprehension that did this
    # rebound its own loop variables while sorting the pair, so after the first
    # inner iteration the "ASL" name had become a CSL one and the CSL-CSL pair
    # was counted as cross-group -- which showed up as an implausibly high
    # cross-group maximum, identical to the within-CSL figure.
    cross = []
    for x0 in groups["asl_intact"]:
        for y0 in groups["csl_intact"]:
            k = "|".join(sorted((x0, y0)))
            if k in pairs:
                cross.append(pairs[k])
    if cross:
        summary["asl_vs_csl"] = {
            "pairs": len(cross),
            **{k: (min(p[k] for p in cross), max(p[k] for p in cross))
               for k in ("hit_agree", "pick_agree", "pick_agree_chance",
                         "kappa")},
            "wrong_pick_match": (min(p["wrong_pick_match"] for p in cross),
                                 max(p["wrong_pick_match"] for p in cross)),
        }

    for name, s in summary.items():
        print(f"{name}: {s.get('pairs')} pairs")
        for k in ("hit_agree", "pick_agree", "pick_agree_chance", "kappa",
                  "wrong_pick_match"):
            if s.get(k):
                lo, hi = s[k]
                print(f"    {k:18s} {lo:6.3f} - {hi:6.3f}")
    a.out.write_text(json.dumps({
        "file": a.file, "condition": a.condition, "seed": a.seed,
        "pairs": pairs, "summary": summary,
    }, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
