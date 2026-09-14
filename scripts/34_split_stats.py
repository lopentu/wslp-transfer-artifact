#!/usr/bin/env python3
"""Exact split sizes and signer allocation, for the reproducibility appendix.

    ../.venv/bin/python scripts/34_split_stats.py   # -> data/split_stats.json

The camera-ready reviewer's last concern: the manuscript reports paragraph-level
folds "but not exact train/dev/test paragraph counts, exact fold sizes, or signer
allocation across splits", and therefore reads the evaluation as held-out
paragraph rather than cross-signer. That reading is correct, and the paper should
state it as a fact rather than leave a reviewer to infer it -- so measure it.

Signer identity is read from the corpus's own paragraph attributes and reported
ONLY in aggregate: how many distinct signers a split touches, and how many test
signers also appear in training. No per-signer counts, no identifiers, nothing
that would let a reader re-identify a signer from the paper, which is the same
line the Ethics section draws around the pose data. The point of the number is
the scope of the generalization claim, and an aggregate carries that entirely.

Note that `speaker` on a *sentence* is the L/R dialogue side within a two-person
film, not a person: a monologue paragraph has one signer and a dialogue has two,
and the side label distinguishes them within a film rather than across the
corpus. Reading signer identity off `speaker` would report two signers for the
whole corpus, so this script goes to the paragraph attributes instead.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsl.db import Corpus  # noqa: E402

# The corpus stores performers as free-text attributes: `演繹者:5` on a monologue
# and `演繹者1:6` / `演繹者2:7` on a dialogue. Anchored so a stray attribute that
# merely contains a digit cannot become a signer.
SIGNER = re.compile(r"^演繹者\d*:(\d+)$")


def signers_by_paragraph(uuids: set[str]) -> dict[str, set[str]]:
    con = Corpus().con
    out: dict[str, set[str]] = {}
    for r in con.execute("SELECT uuid, attr FROM paragraphs"):
        if r["uuid"] not in uuids:
            continue
        try:
            attrs = json.loads(r["attr"] or "[]")
        except json.JSONDecodeError:
            attrs = []
        out[r["uuid"]] = {m.group(1) for a in attrs if (m := SIGNER.match(a))}
    return out


def main() -> None:
    folds = json.loads((ROOT / "data" / "folds.json").read_text())
    recs = [json.loads(l) for l in (ROOT / "data" / "records.jsonl").open()]
    lex = [json.loads(l) for l in (ROOT / "data" / "lex" / "records.jsonl").open()]

    by_para: dict[str, list] = collections.defaultdict(list)
    for r in recs:
        by_para[r["uuid"]].append(r)
    lex_items = collections.Counter(r["uuid"] for r in lex if r.get("candidates"))
    pron_items = collections.Counter(r["uuid"] for r in recs if r.get("candidates"))

    sig = signers_by_paragraph(set(by_para))
    all_signers = {s for v in sig.values() for s in v}
    n_dialogue = sum(1 for v in sig.values() if len(v) == 2)

    out = {
        "paragraphs": len(by_para),
        "utterances": len(recs),
        "signers": len(all_signers),
        "monologue_paragraphs": len(by_para) - n_dialogue,
        "dialogue_paragraphs": n_dialogue,
        "lex_items": sum(lex_items.values()),
        "pron_items": sum(pron_items.values()),
        "k": folds["k"],
        "folds": [],
    }

    for fo in folds["folds"]:
        rec = {"fold": fo["fold"]}
        for split in ("train", "dev", "test"):
            ps = fo[split]
            rec[split] = {
                "paragraphs": len(ps),
                "utterances": sum(len(by_para[p]) for p in ps),
                "lex_items": sum(lex_items[p] for p in ps),
                "pron_items": sum(pron_items[p] for p in ps),
                "signers": len({s for p in ps for s in sig.get(p, ())}),
            }
        tr = {s for p in fo["train"] for s in sig.get(p, ())}
        te = {s for p in fo["test"] for s in sig.get(p, ())}
        # The number the generalization claim actually rests on. Folds are cut at
        # the paragraph and signer identity is not a splitting variable, so this
        # is expected to be near-total; reporting it is what turns "held-out
        # paragraph, not cross-signer" from an inference into a measurement.
        rec["test_signers_seen_in_train"] = len(te & tr)
        rec["test_signers"] = len(te)
        out["folds"].append(rec)

    out["test_signers_unseen_max"] = max(
        f["test_signers"] - f["test_signers_seen_in_train"] for f in out["folds"]
    )

    dest = ROOT / "data" / "split_stats.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{out['paragraphs']} paragraphs, {out['utterances']} utterances, "
          f"{out['signers']} signers "
          f"({out['monologue_paragraphs']} monologue / "
          f"{out['dialogue_paragraphs']} dialogue)")
    for f in out["folds"]:
        print(f"  fold {f['fold']}: "
              + "  ".join(f"{s} {f[s]['paragraphs']}p/{f[s]['utterances']}u"
                          for s in ("train", "dev", "test"))
              + f"   test signers seen in train "
                f"{f['test_signers_seen_in_train']}/{f['test_signers']}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
