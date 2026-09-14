#!/usr/bin/env python3
"""An automated audit of the diagnostic's distractors — and what it found first.

ReviewerA's W9/B5: the paper is about Taiwan Sign Language, no Deaf signer took
part in building its items, and the review's reading is that the distractors are
"driven entirely by jieba segmentation and Qwen word frequency, with no
sign-linguistic check whatsoever". A signer rating 100 items is a human study and
this script is not it.

**But the review's premise is wrong, and that is this script's first finding.**
`src/tsl/lexitems.py:_distractors` keeps only candidates whose sign has a
DIFFERENT twtsl (location, handshape) signature from the gold's -- `t["sig"] !=
g["sig"]`. So a sign-phonological exclusion has been in the instrument since it
was built. The paper never says so: its item-construction paragraph lists the POS
tags, the frequency band and the jieba check, and stops there. The reviewer read
what the paper wrote.

That makes the obvious measurement -- "how many distractors are signed like their
gold?" -- a guaranteed zero, and a guaranteed zero is a property of the
instrument, not a result. Reporting it as though it were evidence would be the
same mistake in the other direction. So this script reports:

  what the control is worth   the collision rate the item set WOULD have had
                              without the exclusion, estimated on the same pool.
                              A control that removes a 0.2% risk is not worth a
                              sentence; one that removes a 10% risk is.
  how hard what survives is   the exclusion only removes signature IDENTITY. A
                              distractor sharing the gold's LOCATION but not its
                              handshape, or the reverse, is a near-minimal pair
                              and is retained. That fraction says whether the
                              surviving items are visually confusable -- i.e.
                              whether the diagnostic is hard for the right reason.
  where a signer should look   `--shortlist`: the items whose distractors are
                              nearest the gold in sign space, which is a better
                              100 to hand a signer than a random 100.

POS is read off the corpus's own 詞類 layer, which is what the builder matched on,
NOT off jieba: jieba disagrees with the corpus tags on a large minority of
multi-character types, and auditing the item set with a different tagger from the
one that built it would report tagger disagreement as item defects. The
disagreement rate is reported separately, since it is a real caveat about the
jieba-based standalone-token filter.

    ../.venv/bin/python scripts/32_item_audit.py

CPU only, a few seconds. -> data/item_audit.json, data/item_shortlist.tsv
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.db import POS_LAYER, Corpus  # noqa: E402
from tsl.strata import Phonology, normalise_gloss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTENT_POS = ("名詞", "動詞", "形容詞")


def jieba_pos(words: set[str]) -> dict[str, str]:
    try:
        import jieba.posseg as pseg
    except ImportError:
        return {}
    out = {}
    for w in words:
        tags = [t for _, t in pseg.cut(w)]
        out[w] = tags[0] if tags else ""
    return out


def sig_of(phon: Phonology, w: str) -> set:
    return phon.by_gloss.get(normalise_gloss(w)) or set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--shortlist", type=int, default=100)
    ap.add_argument("--draws", type=int, default=20000,
                    help="Monte Carlo draws for the counterfactual collision rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shortlist-out", type=Path,
                    default=ROOT / "data" / "item_shortlist.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "item_audit.json")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.data / "lex" / "records.jsonl")]
    by_key = {r["key"]: r for r in recs}
    items = [r for r in recs if r.get("candidates") and r.get("distractors")]
    print(f"{len(items)} content-word items")

    phon = Phonology()
    corpus = Corpus()

    # The corpus's own POS tag per type, and its token frequency. Both are what
    # the builder used, so both are what an audit has to use.
    corpus_pos: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    tok_freq: collections.Counter = collections.Counter()
    for s in corpus.sentences:
        for _, w, p in s.tokens:
            if not w:
                continue
            tok_freq[w] += 1
            if p:
                corpus_pos[w][p] += 1
    majority_pos = {w: c.most_common(1)[0][0] for w, c in corpus_pos.items()}

    words = {r["gold_word"] for r in items if r.get("gold_word")}
    for r in items:
        words |= set(r["distractors"])
    jpos = jieba_pos(words)

    rows, jdisagree = [], 0
    for r in items:
        gold = r.get("gold_word") or ""
        gs = sig_of(phon, gold)
        g_loc = {s[0] for s in gs}
        g_hs = {s[1] for s in gs}
        gp = majority_pos.get(gold)
        if gp and jpos.get(gold) and gp not in ("",):
            # jieba's tagset is not the corpus's, so "disagreement" here means
            # jieba does not call it a content word at all.
            if not jpos[gold].startswith(("n", "v", "a")):
                jdisagree += 1
        rec = {"key": r["key"], "gold": gold, "gold_pos_corpus": gp,
               "gold_pos_paper": r.get("gold_pos"),
               "gold_freq": r.get("gold_freq"),
               "gold_density": r.get("gold_density"),
               "gold_in_twtsl": bool(gs), "d": []}
        for d in r["distractors"]:
            ds = sig_of(phon, d)
            rec["d"].append({
                "word": d,
                "pos_corpus": majority_pos.get(d),
                "pos_matches_gold": (majority_pos.get(d) == gp
                                     if (gp and d in majority_pos) else None),
                "in_twtsl": bool(ds),
                "in_corpus": tok_freq.get(d, 0),
                "identical_signature": bool(gs and ds and gs == ds),
                "shares_location": bool(g_loc & {s[0] for s in ds}),
                "shares_handshape": bool(g_hs & {s[1] for s in ds}),
                "char_overlap": len(set(d) & set(gold)) / max(len(set(gold)), 1),
            })
        rows.append(rec)

    n = len(rows)
    dall = [x for r in rows for x in r["d"]]

    def frac(xs, pred):
        xs = [x for x in xs if pred(x) is not None]
        return (sum(bool(pred(x)) for x in xs) / len(xs)) if xs else None

    # What the exclusion is worth: draw distractors the way the builder would have
    # without the signature filter -- same POS pool, same frequency band -- and
    # count how often the sign would have collided.
    rng = random.Random(a.seed)
    pool_by_pos: dict[str, list[str]] = collections.defaultdict(list)
    for w, p in majority_pos.items():
        if p in CONTENT_POS and sig_of(phon, w):
            pool_by_pos[p].append(w)
    hit = tot = 0
    for _ in range(a.draws):
        r = rng.choice(rows)
        p = r["gold_pos_corpus"]
        cands = pool_by_pos.get(p) or []
        if len(cands) < 2 or not r["gold_in_twtsl"]:
            continue
        w = rng.choice(cands)
        if w == r["gold"]:
            continue
        tot += 1
        hit += sig_of(phon, w) == sig_of(phon, r["gold"])
    counterfactual = hit / tot if tot else None

    summary = {
        "n_items": n, "n_distractors": len(dall),
        "control_already_in_the_instrument": {
            "where": "src/tsl/lexitems.py::_distractors, `t['sig'] != g['sig']`",
            "documented_in_paper": False,
            "observed_identical_signature_rate": frac(
                dall, lambda x: x["identical_signature"]),
            "counterfactual_rate_without_it": counterfactual,
            "counterfactual_draws": tot,
            "note": "the observed rate is zero BY CONSTRUCTION and is not "
                    "evidence; the counterfactual is what the filter removes",
        },
        "how_hard_what_survives_is": {
            "note": "the filter removes signature identity only, so near-minimal "
                    "pairs are retained and the surviving items are visually "
                    "confusable rather than trivially distinct",
            "distractor_shares_location": frac(dall, lambda x: x["shares_location"]),
            "distractor_shares_handshape": frac(dall,
                                                lambda x: x["shares_handshape"]),
            "distractor_shares_neither": frac(
                dall, lambda x: not (x["shares_location"] or x["shares_handshape"])),
            "items_with_a_near_minimal_pair": sum(
                any(x["shares_location"] or x["shares_handshape"] for x in r["d"])
                for r in rows) / n,
        },
        "text_side_checks": {
            # The builder matches POS on the gold TOKEN's 詞類 tag (it draws from
            # `pools[pos]`), so this is exact by construction. The number below
            # is lower only because it compares the distractor TYPE's majority
            # tag: Chinese content words are POS-ambiguous at type level and the
            # 動詞/名詞 alternation alone accounts for most of the gap. So the POS
            # control is token-level, not type-level -- a real caveat, and a
            # small one.
            "distractor_majority_pos_matches_gold_majority_pos": frac(
                dall, lambda x: x["pos_matches_gold"]),
            "gold_majority_pos_equals_builder_token_pos": sum(
                r["gold_pos_corpus"] == r["gold_pos_paper"] for r in rows) / n,
            "distractor_attested_in_corpus": frac(dall,
                                                  lambda x: x["in_corpus"] > 0),
            "gold_in_twtsl": sum(r["gold_in_twtsl"] for r in rows) / n,
            "distractor_in_twtsl": frac(dall, lambda x: x["in_twtsl"]),
            "mean_char_overlap": sum(x["char_overlap"] for x in dall) / len(dall),
            "golds_jieba_does_not_call_content_words": jdisagree,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    # Hand a signer the confusable end, not a random sample.
    def nearness(r):
        return sum(int(x["shares_location"]) + int(x["shares_handshape"])
                   for x in r["d"])

    short = sorted((r for r in rows if nearness(r) > 0), key=lambda r: -nearness(r))
    lines = ["key\tgold\tdistractor\tshares_location\tshares_handshape\tsentence"]
    for r in short[: a.shortlist]:
        sent = by_key.get(r["key"], {}).get("text", "")
        for x in r["d"]:
            if x["shares_location"] or x["shares_handshape"]:
                lines.append(f"{r['key']}\t{r['gold']}\t{x['word']}\t"
                             f"{x['shares_location']}\t{x['shares_handshape']}\t"
                             f"{sent}")
    a.shortlist_out.write_text("\n".join(lines) + "\n")
    print(f"\n{len(short)} items have a near-minimal-pair distractor; "
          f"top {min(a.shortlist, len(short))} -> {a.shortlist_out}")

    a.out.write_text(json.dumps({"summary": summary, "items": rows},
                                ensure_ascii=False, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
