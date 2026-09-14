"""Paragraph-level splits.

Two constraints shape this:

1. **Leakage.** Adjacent sentences share signer, theme, referents and often
   whole clauses, so any split must cut at the paragraph, never the sentence.

2. **The diagnostic set is small.** There are ~200 bare anaphoric items in the
   whole corpus. A single 20% test split would hold ~40 of them — too few to
   separate conditions. So the default is *grouped 2-fold cross-evaluation*:
   train two models on complementary halves, evaluate each on its held-out half,
   and pool. Every anaphoric item is then scored out-of-training, and the paper
   reports ~200 items instead of ~40, for 2x the training cost.

Folds are balanced on (paragraph type, anaphoric item count) by greedy
largest-first assignment, which keeps both halves comparable on the quantity the
main result is measured on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .items import Record


def _para_stats(records: list[Record]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        st = out.setdefault(
            r.uuid, {"uuid": r.uuid, "type": r.para_type, "n": 0, "ana": 0, "dei": 0}
        )
        st["n"] += 1
        if r.item_type == "anaphoric" and r.candidates:
            st["ana"] += 1
        if r.item_type == "deictic" and r.candidates:
            st["dei"] += 1
    return out


def _stable_key(uuid: str) -> int:
    return int(hashlib.md5(uuid.encode()).hexdigest()[:8], 16)


def make_folds(records: list[Record], k: int = 2) -> dict:
    """Greedy balanced assignment of paragraphs to k folds, in two phases.

    Phase 1 spreads the paragraphs that actually carry anaphoric items, largest
    first, onto the fold currently holding the fewest — that is the quantity the
    main result is measured on, so it is balanced first. Phase 2 then spreads the
    remaining paragraphs by sentence count, which is what training size depends
    on. Doing it in one pass instead lets every zero-item paragraph land on
    whichever fold has the lowest item count, which never changes, so one fold
    swallows the corpus.

    Ties break on a hash of the uuid, so folds are deterministic and independent
    of DB row order.
    """
    stats = _para_stats(records)
    folds: list[list[str]] = [[] for _ in range(k)]
    ana = [0] * k
    sents = [0] * k

    carriers = [s for s in stats.values() if s["ana"] > 0]
    rest = [s for s in stats.values() if s["ana"] == 0]

    for st in sorted(carriers, key=lambda s: (-s["ana"], -s["n"], _stable_key(s["uuid"]))):
        j = min(range(k), key=lambda i: (ana[i], sents[i]))
        folds[j].append(st["uuid"])
        ana[j] += st["ana"]
        sents[j] += st["n"]

    for st in sorted(rest, key=lambda s: (-s["n"], _stable_key(s["uuid"]))):
        j = min(range(k), key=lambda i: (sents[i], ana[i]))
        folds[j].append(st["uuid"])
        sents[j] += st["n"]

    out = {"k": k, "folds": []}
    for i in range(k):
        test = set(folds[i])
        # dev comes out of the training side, so test stays untouched by tuning
        train_paras = [u for j in range(k) if j != i for u in folds[j]]
        train_paras.sort(key=_stable_key)
        n_dev = max(8, len(train_paras) // 10)
        dev, train = train_paras[:n_dev], train_paras[n_dev:]
        out["folds"].append(
            {
                "fold": i,
                "train": train,
                "dev": dev,
                "test": sorted(test),
                "counts": {
                    "test_sentences": sum(stats[u]["n"] for u in test),
                    "test_anaphoric": sum(stats[u]["ana"] for u in test),
                    "test_deictic": sum(stats[u]["dei"] for u in test),
                    "train_sentences": sum(stats[u]["n"] for u in train),
                },
            }
        )
    return out


def assign(records: list[Record], folds: dict, fold: int) -> dict[str, list[Record]]:
    f = folds["folds"][fold]
    where = {u: "train" for u in f["train"]}
    where.update({u: "dev" for u in f["dev"]})
    where.update({u: "test" for u in f["test"]})
    out: dict[str, list[Record]] = {"train": [], "dev": [], "test": []}
    for r in records:
        split = where.get(r.uuid)
        if split:
            out[split].append(r)
    return out


def save(folds: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(folds, ensure_ascii=False, indent=2))


def load(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
