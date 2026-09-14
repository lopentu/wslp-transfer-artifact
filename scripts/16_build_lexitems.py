#!/usr/bin/env python3
"""Build the lexical contrastive diagnostic into a parallel data directory.

    python3 scripts/16_build_lexitems.py --out data/lex

Writes `<out>/records.jsonl` in exactly the schema `05_eval.py` already reads,
and links `folds.json` from the main data directory rather than copying it — the
folds must be the *same* folds, or a lexical result could differ from a pronoun
result because the two used different held-out paragraphs. Nothing downstream
needs changing: evaluate with `--data <out>`.

Reports the composition on the way out, because the two numbers that decide
whether this instrument is usable — how many items survive the safety guards,
and whether the frequency prior can answer them — are properties of the build,
not of any model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.asllex import iconicity_by_zh  # noqa: E402
from tsl.db import Corpus  # noqa: E402
from tsl.lexitems import (  # noqa: E402
    add_iconicity_class, build_lexical_records, form_convergence, summarise,
)
from tsl.splits import load as load_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/lex"))
    ap.add_argument("--ctx-k", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--restratify", action="store_true",
                    help="update only the iconicity fields of an existing "
                         "records.jsonl, leaving items and candidates alone. Use "
                         "after 17_lm_match_distractors.py has re-chosen "
                         "distractors, so a rebuild does not discard that pass.")
    a = ap.parse_args()

    if a.restratify:
        path = a.out / "records.jsonl"
        recs = [json.loads(l) for l in path.open()]
        icon, conv = iconicity_by_zh(), form_convergence()
        for r in recs:
            w = r.get("gold_word")
            if r.get("item_type") != "lexical" or not w:
                continue
            ic, cv = icon.get(w, {}), conv.get(w, {})
            r["icon"] = ic.get("icon")
            r["icon_en"] = ic.get("en")
            r["icon_n_en"] = ic.get("n_en")
            r["asl_z"] = cv.get("ase")
            r["csl_z"] = cv.get("csl")
            r.pop("conv_class", None)   # the rejected stratifier's label
    else:
        recs = build_lexical_records(Corpus(), ctx_k=a.ctx_k)
    cuts = add_iconicity_class(recs)
    items = [r for r in recs if r.get("item_type") == "lexical"]

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "records.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    link = a.out / "folds.json"
    src = (a.data / "folds.json").resolve()
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src)

    s = summarise(recs)
    print(f"wrote {a.out}/records.jsonl  ({s['records']} records, "
          f"{s['items']} lexical items)")
    print(f"  folds.json -> {src}")
    print(f"  with context      {s['items_with_context']}")
    prior = s["prior_would_answer"]
    print(f"  frequency prior answers  {prior}/{s['items']} = "
          f"{prior / max(1, s['items']):.1%}   (chance 25%)")
    print("  by POS:  " + "  ".join(
        f"{k.split(':')[1]} {v}" for k, v in sorted(s.items()) if k.startswith("pos:")))
    print("  gold length: " + "  ".join(
        f"{k.split(':')[1]}字 {v}" for k, v in sorted(s.items()) if k.startswith("len:")))
    print("  ASL-LEX iconicity stratum: " + "  ".join(
        f"{k.split(':')[1]} {v}" for k, v in sorted(s.items()) if k.startswith("icon:"))
        + (f"   (rating cuts {cuts[0]:.2f} / {cuts[1]:.2f} on 1-7)" if cuts else ""))

    folds = load_folds(a.data / "folds.json")
    for i, f in enumerate(folds["folds"]):
        test = set(f["test"])
        n = sum(1 for r in items if r["uuid"] in test)
        print(f"  fold {i}: {n} test items")
    print(f"  pooled over folds: {sum(1 for _ in items)}")

    gold = Counter(r["gold_word"] for r in items)
    print(f"  distinct gold types {len(gold)}; commonest "
          f"{gold.most_common(6)}")
    top = gold.most_common(1)[0][1] / max(1, len(items))
    print(f"  most frequent single gold type is {top:.1%} of items "
          f"(pronoun set: 我 was 54.2%)")

    print(f"\n{a.samples} sample items:")
    for r in random.Random(0).sample(items, min(a.samples, len(items))):
        print(f"\n  [{r['key']}] {r['gold_pos']} {r['gold_word']!r} "
              f"(freq {r['gold_freq']}, {r['n_slots']} eligible slots)")
        for j, c in enumerate(r["candidates"]):
            print(f"    {'*' if j == 0 else ' '} {c}")


if __name__ == "__main__":
    main()
