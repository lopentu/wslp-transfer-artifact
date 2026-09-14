#!/usr/bin/env python3
"""Data audit — the go/no-go numbers for the paper.

Prints, for the 文化部 corpus: annotation fill rates, the pointing-sign inventory,
how often a gloss surfaces in the Chinese translation, and how many diagnostic
items survive each filter. Run this first; every design decision downstream is
justified by one of these numbers.

    python3 scripts/00_audit.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.db import ANAPHORIC, DEICTIC, Corpus, agreeing_verbs  # noqa: E402
from tsl.items import build_records, summarise  # noqa: E402
from tsl.splits import make_folds  # noqa: E402


def rule(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> None:
    c = Corpus()

    rule("corpus")
    paras = c.paragraphs
    types = Counter(p["type"] for p in paras.values())
    print(f"paragraphs           {len(paras)}  (篇章={types['1']}, 對話={types['2']})")
    sents = c.sentences
    print(f"sentences            {len(sents)}")
    dur = sorted(s.dur_ms for s in sents)
    print(
        f"utterance duration   median {dur[len(dur)//2]/1000:.2f}s  "
        f"p10 {dur[len(dur)//10]/1000:.2f}s  p90 {dur[9*len(dur)//10]/1000:.2f}s  "
        f"max {dur[-1]/1000:.1f}s"
    )
    print(f"annotated video      {sum(dur)/3.6e6:.2f} h")
    print(f"signers              {len({p['signer'] for p in paras.values()})}")

    rule("annotation fill rate (layers we depend on)")
    for layer, n in c.con.execute(
        """SELECT layer, COUNT(*) FROM annotations
           WHERE TRIM(COALESCE(value,'')) <> '' GROUP BY layer ORDER BY 2 DESC"""
    ):
        total = c.con.execute(
            "SELECT COUNT(*) FROM annotations WHERE layer=?", (layer,)
        ).fetchone()[0]
        print(f"  {layer:8s} {n:6d}/{total:6d}  {n/total:5.1%}")
    print("  動詞類型      0/23699   0.0%   <- empty; verb class is borrowed from tsldict")

    rule("pointing-sign inventory (詞類 = 代名詞)")
    tok = Counter()
    hit = Counter()
    for s in sents:
        for _, w, pos in s.tokens:
            if pos.startswith("代名詞") and w:
                tok[w] += 1
                if w in s.text:
                    hit[w] += 1
    print(f"  {'gloss':10s} {'n':>5s} {'in zh':>6s} {'rate':>6s}  class")
    for w, n in tok.most_common(16):
        cls = "deictic" if w in DEICTIC else "anaphoric" if w in ANAPHORIC else "-"
        print(f"  {w:10s} {n:5d} {hit[w]:6d} {hit[w]/n:6.0%}  {cls}")
    dn = sum(n for w, n in tok.items() if w in DEICTIC)
    an = sum(n for w, n in tok.items() if w in ANAPHORIC)
    print(f"\n  deictic total {dn}   anaphoric total {an}")
    print("  Note: the anaphoric gloss surfaces less often precisely because the")
    print("  annotator resolved the pointing sign into an explicit NP — those become")
    print("  the `resolved_referent` items.")

    rule("agreeing / spatial verbs borrowed from tsldict")
    agree = agreeing_verbs()
    print(f"  dictionary glosses: {len(agree)}")
    matched = Counter()
    for s in sents:
        for _, w, pos in s.tokens:
            if "動詞" in pos and w in agree:
                matched[w] += 1
    print(f"  corpus tokens matched: {sum(matched.values())} over {len(matched)} types")
    print(f"  top: {matched.most_common(8)}")

    rule("diagnostic items after every filter")
    recs = build_records(c, ctx_k=8)
    s = summarise(recs)
    for k in sorted(s):
        print(f"  {k:24s} {s[k]}")
    n_ana = s.get("item:anaphoric", 0)
    print("\n  anaphoric items, reach of the context window (person-NP heuristic):")
    for k in (1, 2, 4, 8):
        got = s.get(f"anaphoric:antecedent_within_{k}", 0)
        print(f"    antecedent within {k}: {got:3d}/{n_ana} ({got/max(1,n_ana):5.1%})")
    print(f"    no candidate in the paragraph: {s.get('anaphoric:no_antecedent_found',0)}")
    print(f"    >=2 distinct candidates: {s.get('anaphoric:ambiguous_2plus_candidates',0)}"
          "  <- the referent-resolution task's item pool")
    print(
        "\n  filters applied: has >=1 preceding utterance in the same paragraph;"
        "\n  the pointing sign is the FIRST referring expression in the utterance"
        "\n  (no lexical NP fixes the reference clip-internally); the gloss occurs"
        "\n  exactly once in the Chinese translation."
    )

    rule("grouped 3-fold cross-evaluation")
    folds = make_folds(recs, k=3)
    for f in folds["folds"]:
        cc = f["counts"]
        print(
            f"  fold {f['fold']}: train {len(f['train'])} paras / {cc['train_sentences']} sents"
            f" | dev {len(f['dev'])} | test {len(f['test'])} paras / {cc['test_sentences']} sents"
            f"  -> anaphoric {cc['test_anaphoric']}, deictic {cc['test_deictic']}"
        )
    tot_a = sum(f["counts"]["test_anaphoric"] for f in folds["folds"])
    tot_d = sum(f["counts"]["test_deictic"] for f in folds["folds"])
    print(f"\n  pooled held-out items: anaphoric {tot_a}, deictic {tot_d}")

    rule("video")
    films = list(Path("/mnt/md0/corpus/sign/tslcorpus/films").glob("*.mp4"))
    print(f"  films on disk: {len(films)}")
    print(f"  dialogue films hold TWO signers side by side; crop by `speaker` (L/R).")
    print(f"  G4C26 is truncated at source — spans past 46.1 s are dropped.")


if __name__ == "__main__":
    main()
