#!/usr/bin/env python3
"""A Chinese-output mT5 head that never saw a sign language.

Limitations says: "no Chinese-output text component adapted without a
sign-language checkpoint is available". A review's answer is that this is a
statement about what we *did*, not about what exists -- we can build one. That
turns written-language match from a property the released checkpoints happen to
carry into a factor we set.

What the control has to be, and what it must not be. The paper's claim is about
the output projection's written language. Every Chinese-output `lm_head` it has
comes from a checkpoint that was also fine-tuned on CSL video, so "Chinese" and
"CSL-pretrained" are the same column. This script produces an `lm_head` that is
Chinese-adapted and nothing else:

  * mT5-base, exactly the initialization every `-pose` row and RAND-VIS uses,
  * fine-tuned on TSL Chinese *translations only*, no pose input at all,
  * from the target fold's TRAIN split, so the diagnostic items stay unseen,
  * as span-denoising -- mT5's own pretraining objective -- so the adaptation is
    to the text distribution and not to a task the sign models never had.

It sees no video, no sign language, and no held-out sentence. Cross-loading its
`lm_head` into How2Sign therefore asks the question the released checkpoints
cannot: does a Chinese-adapted output projection rescue an ASL visual half, or
does only a *CSL-checkpoint* one?

    ../.venv/bin/python scripts/23_text_lm.py --fold 0

Writes `data/external/tsl_text_lm.pth` in Uni-Sign key order, so it drops into
the existing machinery as `--init-mt5 tsl_text --init-mt5-parts lm_head`.
A few minutes on one card; the model is 966.6M parameters and the corpus is a
few thousand short sentences.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.conditions import load_records, split_records  # noqa: E402
from tsl.splits import load as load_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def spans(text: str, tok, rng: random.Random, rate: float = 0.15,
          mean_len: float = 3.0) -> tuple[str, str] | None:
    """mT5's span-corruption pair for one sentence, in sentinel form.

    Reimplemented rather than imported because the T5 data pipeline lives in
    mesh-tensorflow. What matters for our purpose is only that the objective is
    the *same kind* of thing mT5 was pretrained with, so the adaptation moves the
    model along its own training manifold rather than teaching it a new task.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < 4:
        return None
    n_noise = max(1, int(round(len(ids) * rate)))
    n_spans = max(1, int(round(n_noise / mean_len)))
    starts = sorted(rng.sample(range(len(ids)), min(n_spans, len(ids))))
    picked: list[tuple[int, int]] = []
    for st in starts:
        ln = max(1, int(rng.gauss(mean_len, 1)))
        if picked and st <= picked[-1][1]:
            continue
        picked.append((st, min(len(ids), st + ln)))
    if not picked:
        return None
    src: list[int] = []
    tgt: list[int] = []
    prev = 0
    for i, (st, en) in enumerate(picked):
        sent = tok.convert_tokens_to_ids(f"<extra_id_{i}>")
        src += ids[prev:st] + [sent]
        tgt += [sent] + ids[st:en]
        prev = en
    src += ids[prev:]
    tgt += [tok.convert_tokens_to_ids(f"<extra_id_{len(picked)}>")]
    return (tok.decode(src, skip_special_tokens=False),
            tok.decode(tgt, skip_special_tokens=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0,
                    help="use this fold's TRAIN split only; the diagnostic items "
                         "live in its test split and must stay unseen")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="the same mT5 learning rate the sign cells use, so the "
                         "amount of movement is comparable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "external" / "tsl_text_lm.pth")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "data" / "text_lm.json")
    a = ap.parse_args()

    from transformers import MT5ForConditionalGeneration, T5Tokenizer

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    recs = load_records(a.data / "records.jsonl")
    folds = load_folds(a.data / "folds.json")
    sp = split_records(recs, folds, a.fold)
    train_txt = [r["text"] for r in sp["train"] if r["text"].strip()]
    dev_txt = [r["text"] for r in sp["dev"] if r["text"].strip()]
    print(f"fold {a.fold}: {len(train_txt)} train sentences, {len(dev_txt)} dev")

    tok = T5Tokenizer.from_pretrained("google/mt5-base", legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base").cuda()
    head0 = model.lm_head.weight.detach().clone()

    def pairs(texts: list[str]) -> list[tuple[str, str]]:
        out = [spans(t, tok, rng) for t in texts]
        return [p for p in out if p]

    dev_pairs = pairs(dev_txt)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    def batches(ps: list[tuple[str, str]], shuffle: bool):
        idx = list(range(len(ps)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), a.batch):
            chunk = [ps[j] for j in idx[i:i + a.batch]]
            x = tok([c[0] for c in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=96)
            y = tok([c[1] for c in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=96)["input_ids"]
            y[y == tok.pad_token_id] = -100
            yield ({k: v.cuda() for k, v in x.items()}, y.cuda())

    @torch.no_grad()
    def dev_loss() -> float:
        model.eval()
        tot = n = 0.0
        for x, y in batches(dev_pairs, False):
            tot += float(model(**x, labels=y).loss) * len(y)
            n += len(y)
        return tot / max(n, 1)

    log = [{"epoch": 0, "dev": dev_loss()}]
    print(f"  epoch 0 dev {log[0]['dev']:.4f}")
    best = log[0]["dev"]
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(1, a.epochs + 1):
        # A fresh corruption per epoch: the corpus is a few thousand sentences and
        # a fixed mask would be memorized rather than learned from.
        tr = pairs(train_txt)
        model.train()
        tot = n = 0.0
        for x, y in batches(tr, True):
            loss = model(**x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss) * len(y)
            n += len(y)
        d = dev_loss()
        log.append({"epoch": ep, "train": tot / max(n, 1), "dev": d})
        flag = ""
        if d < best:
            best, flag = d, "  *"
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
        print(f"  epoch {ep} train {tot/max(n,1):.4f} dev {d:.4f}{flag}")

    head = best_sd["lm_head.weight"]
    moved = float((head - head0.cpu()).norm() / head0.cpu().norm())
    cos = float(torch.nn.functional.cosine_similarity(
        head.double().flatten(), head0.cpu().double().flatten(), dim=0))
    sd = {f"mt5_model.{k}": v for k, v in best_sd.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd}, a.out)
    rep = {"fold": a.fold, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
           "n_train": len(train_txt), "n_dev": len(dev_txt),
           "best_dev": best, "log": log,
           "lm_head_rel_l2_from_mt5base": moved,
           "lm_head_cos_to_mt5base": cos}
    a.report.write_text(json.dumps(rep, indent=1))
    print(f"\nlm_head moved {moved:.4f} rel L2 from mT5-base (cos {cos:.5f})")
    print(f"-> {a.out}  ({a.out.stat().st_size/2**30:.2f} GiB)")
    print(f"-> {a.report}")


if __name__ == "__main__":
    main()
