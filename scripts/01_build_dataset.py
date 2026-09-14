#!/usr/bin/env python3
"""Materialise the dataset: records, diagnostic items, folds.

    python3 scripts/01_build_dataset.py --out data --ctx 2 --folds 3

Writes
  data/records.jsonl   one row per utterance (target + up to `ctx` preceding)
  data/folds.json      grouped k-fold paragraph assignment
  data/items.tsv       the diagnostic items, human-readable, for manual validation
  data/spans.json      what 03_extract_features.py has to decode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.db import Corpus  # noqa: E402
from tsl.items import build_records, summarise  # noqa: E402
from tsl.splits import make_folds, save as save_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--ctx", type=int, default=8,
                    help="preceding utterances to STORE; the window used at train/eval "
                         "time is TSLDataset(ctx_k=...)")
    ap.add_argument("--folds", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    c = Corpus()
    recs = build_records(c, ctx_k=a.ctx)

    with (a.out / "records.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")

    folds = make_folds(recs, k=a.folds)
    save_folds(folds, a.out / "folds.json")

    # Every span any condition can ask for: the target, and each context clip.
    spans: dict[str, dict] = {}
    for r in recs:
        spans[r.key] = {"uuid": r.uuid, "type": r.para_type, "t1": r.t1, "t2": r.t2,
                        "speaker": r.speaker}
        for cx in r.ctx:
            k = f"{r.uuid}:{cx['seq']:03d}"
            spans.setdefault(k, {"uuid": r.uuid, "type": r.para_type, "t1": cx["t1"],
                                 "t2": cx["t2"], "speaker": cx["speaker"]})
    (a.out / "spans.json").write_text(json.dumps(spans, ensure_ascii=False))

    # Human-checkable dump. The paper can only call this a gold discourse
    # benchmark if a fluent signer signs off on these rows (plan §4.4).
    items = [r for r in recs if r.item_type and r.candidates]
    with (a.out / "items.tsv").open("w") as f:
        f.write("key\ttype\tgloss\tn_ctx\tsame_speaker\tcontext\tgold\tdistractors\n")
        for r in sorted(items, key=lambda r: (r.item_type, r.key)):
            ctx = " ⏎ ".join(cx["text"] for cx in r.ctx)
            same = all(cx["same_speaker"] for cx in r.ctx)
            f.write(
                f"{r.key}\t{r.item_type}\t{r.pron_gloss}\t{len(r.ctx)}\t{same}\t"
                f"{ctx}\t{r.candidates[0]}\t{' | '.join(r.candidates[1:])}\n"
            )

    s = summarise(recs)
    print(f"records        {s['sentences']}  ({s['with_context']} with context)")
    print(f"spans to cut   {len(spans)}")
    print(f"items          anaphoric {s.get('item:anaphoric',0)}, "
          f"deictic {s.get('item:deictic',0)}")
    print(f"resolved-ref   anaphoric {s.get('resolved:anaphoric',0)}, "
          f"deictic {s.get('resolved:deictic',0)}")
    print(f"agreeing verb  {s.get('agreeing_verb',0)} utterances")
    n_ana = s.get("item:anaphoric", 0)
    print(f"\nanaphoric items — does the context window reach the antecedent?")
    for k in (1, 2, 4, 8):
        got = s.get(f"anaphoric:antecedent_within_{k}", 0)
        print(f"  within {k} preceding utterances: {got:3d}/{n_ana}  ({got/max(1,n_ana):5.1%})")
    print(f"  no candidate found in the paragraph: "
          f"{s.get('anaphoric:no_antecedent_found', 0)}")
    print(f"  >=2 distinct candidates (a real disambiguation): "
          f"{s.get('anaphoric:ambiguous_2plus_candidates', 0)}")
    for fd in folds["folds"]:
        print(f"  fold {fd['fold']}: {fd['counts']}")
    print(f"\nwrote {a.out}/records.jsonl, folds.json, items.tsv, spans.json")


if __name__ == "__main__":
    main()
