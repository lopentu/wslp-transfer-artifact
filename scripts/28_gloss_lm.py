#!/usr/bin/env python3
"""Gloss->Chinese mT5: a strong written-language instrument, and a regime that works.

This one script answers two of the sharpest objections in the 8/22 reviews,
because they turn out to need the same model.

**ReviewerA B2 -- a real Chinese-output projection with no sign video in it.**
Appendix O's `tsl_text` head is span-denoising on 3,255 Chinese sentences, and it
is a weak instrument by its own measurement: `lm_head` moves 0.026 in relative L2
against 0.92-0.98 for every released sign fine-tune, so its null says nothing.
The reason is the objective, not the budget -- mT5 already models Chinese, so the
gradients are small. The corpus's 23,699 hand-tier-1 sign tokens give a *task*
whose loss does not start near zero: gloss sequence -> Chinese sentence, the same
seq2seq-into-Chinese objective the sign checkpoints were fine-tuned with, with
the video removed and nothing else changed. If that moves `lm_head` as far as a
sign fine-tune does, cross-loading it separates "Chinese-output specialisation"
from "CSL-checkpoint provenance" -- which is the confound the whole §3.3 rests on.

**ReviewerA B4 / ReviewerO W5 -- does any of this survive outside a collapsed
regime?** Every cell in the paper has BLEU <= 2.18, and the fair objection is
that a component-level ordering measured where nothing works may not hold where
something does. We have no CSL-Daily or PHOENIX14T video locally, so the paper's
own regime cannot be re-run at higher resource. But gloss->Chinese on this corpus
IS a non-degenerate regime on the same sentences, same target language, same mT5,
same optimiser: gold glosses are a strong enough input that the model produces
real translations. So run the head factorial again there --

    --init-head mt5_base | how2sign | csl_daily | rand_head_nm

-- and ask whether an ASL-specialised output projection also destroys a system
that otherwise works. Either answer is worth reporting: if it collapses here too,
the projection effect is not an artefact of the collapsed regime; if it does not,
§3's ordering is regime-specific and the paper must say so.

    ../.venv/bin/python scripts/28_gloss_lm.py --fold 0                  # B2 donor
    ../.venv/bin/python scripts/28_gloss_lm.py --fold 0 --init-head how2sign
    ../.venv/bin/python scripts/28_gloss_lm.py --fold 0 --init-head csl_daily
    ../.venv/bin/python scripts/28_gloss_lm.py --fold 0 --init-head rand_head_nm

The first writes `data/external/gloss_lm.pth` (a donor, like `tsl_text_lm.pth`);
every run writes a report to `data/gloss_lm[_<head>].json` with dev CE, test
BLEU/chrF, output diversity, and how far `lm_head` travelled. Text only, no pose
features: a few minutes a run on one card.
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
from tsl.db import Corpus  # noqa: E402
from tsl.metrics import corpus_bleu_chrf  # noqa: E402
from tsl.splits import load as load_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"

# Where an `--init-head` name comes from. The released checkpoints and the
# synthetic donors of scripts/27_head_donors.py are addressed the same way, so
# this arm can use any level of the head factor the sign cells can.
HEADS = {
    "mt5_base": None,  # leave the published projection in place
    "how2sign": EXT / "how2sign_pose_only_slt.pth",
    "openasl": EXT / "openasl_pose_only_slt.pth",
    "csl_daily": EXT / "csl_daily_pose_only_slt.pth",
    "csl_stage1": EXT / "csl_stage1_weight.pth",
    "rand_head": EXT / "head_donors" / "rand_head.pth",
    "rand_head_nm": EXT / "head_donors" / "rand_head_nm.pth",
}


def glosses(fold_records: list[dict], corpus: Corpus) -> dict[str, str]:
    """key -> space-joined hand-tier-1 gloss string.

    Joined on `key`, never on order: `tsl.db` documents that the annotation
    layers are not index-aligned, and the records file has already dropped the
    sentences with no usable span.
    """
    want = {r["key"] for r in fold_records}
    out = {}
    for s in corpus.sentences:
        if s.key not in want:
            continue
        g = " ".join(w for _, w, _ in s.tokens if w)
        if g.strip():
            out[s.key] = g
    return out


def pairs(recs: list[dict], gl: dict[str, str]) -> list[tuple[str, str]]:
    return [(gl[r["key"]], r["text"]) for r in recs
            if r["key"] in gl and r["text"].strip()]


def load_head(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = obj
    for k in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    for k, v in sd.items():
        if k.replace("module.", "", 1) in ("mt5_model.lm_head.weight",
                                           "core.mt5_model.lm_head.weight",
                                           "lm_head.weight"):
            return v.detach().float().clone()
    raise SystemExit(f"{path} has no lm_head tensor")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--epochs", type=int, default=12,
                    help="the sign cells' budget, so movement is comparable")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="the sign cells' mT5 learning rate")
    ap.add_argument("--label-smoothing", type=float, default=0.2,
                    help="upstream's value, as the sign cells use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-head", default="mt5_base", choices=sorted(HEADS),
                    help="which output projection to START from (B4 arm)")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--max-target-len", type=int, default=64)
    ap.add_argument("--save-donor", action="store_true",
                    help="write the trained mT5 out as an --init-mt5 donor")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()

    tag = "" if a.init_head == "mt5_base" else f"_{a.init_head}"
    out = a.out or EXT / f"gloss_lm{tag}.pth"
    report = a.report or a.data / f"gloss_lm{tag}.json"
    if a.init_head == "mt5_base" and a.out is None:
        a.save_donor = True  # the B2 donor is this arm's whole point

    from transformers import MT5ForConditionalGeneration, T5Tokenizer

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    recs = load_records(a.data / "records.jsonl")
    folds = load_folds(a.data / "folds.json")
    sp = split_records(recs, folds, a.fold)
    corpus = Corpus()
    gl = glosses([r for v in sp.values() for r in v], corpus)
    tr_pairs = pairs(sp["train"], gl)
    dv_pairs = pairs(sp["dev"], gl)
    te_pairs = pairs(sp["test"], gl)
    print(f"fold {a.fold}: {len(tr_pairs)} train / {len(dv_pairs)} dev / "
          f"{len(te_pairs)} test gloss-text pairs "
          f"({sum(len(g.split()) for g in gl.values())} gloss tokens)")

    tok = T5Tokenizer.from_pretrained("google/mt5-base", legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    base_head = model.lm_head.weight.detach().float().clone()
    if HEADS[a.init_head] is not None:
        w = load_head(HEADS[a.init_head])
        if tuple(w.shape) != tuple(model.lm_head.weight.shape):
            raise SystemExit(f"donor head {tuple(w.shape)} != "
                             f"{tuple(model.lm_head.weight.shape)}")
        with torch.no_grad():
            model.lm_head.weight.copy_(w)
        print(f"init lm_head <- {a.init_head} ({HEADS[a.init_head].name})")
    head0 = model.lm_head.weight.detach().float().clone()
    model = model.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    def batches(ps, shuffle: bool):
        idx = list(range(len(ps)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), a.batch):
            chunk = [ps[j] for j in idx[i:i + a.batch]]
            x = tok([c[0] for c in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=96)
            y = tok([c[1] for c in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=a.max_target_len)["input_ids"]
            y[y == tok.pad_token_id] = -100
            yield ({k: v.cuda() for k, v in x.items()}, y.cuda())

    @torch.no_grad()
    def dev_ce() -> float:
        """Plain cross-entropy, never the smoothed objective — the sign cells'
        `best.pt` is selected on this quantity and the two must be comparable."""
        model.eval()
        tot = n = 0.0
        for x, y in batches(dv_pairs, False):
            tot += float(model(**x, labels=y).loss) * len(y)
            n += len(y)
        return tot / max(n, 1)

    def train_step(x, y):
        out_ = model(**x, labels=y)
        if a.label_smoothing:
            lg = out_.logits.float()
            loss = torch.nn.functional.cross_entropy(
                lg.view(-1, lg.size(-1)), y.view(-1), ignore_index=-100,
                label_smoothing=a.label_smoothing)
        else:
            loss = out_.loss
        return loss

    log = [{"epoch": 0, "dev": dev_ce()}]
    print(f"  epoch 0 dev CE {log[0]['dev']:.4f}")
    best = log[0]["dev"]
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = n = 0.0
        for x, y in batches(tr_pairs, True):
            loss = train_step(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss) * len(y)
            n += len(y)
        d = dev_ce()
        log.append({"epoch": ep, "train": tot / max(n, 1), "dev": d})
        flag = ""
        if d < best:
            best, flag = d, "  *"
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
        print(f"  epoch {ep} train {tot/max(n,1):.4f} dev CE {d:.4f}{flag}",
              flush=True)

    # ------------------------------------------------ the non-degenerate check
    model.load_state_dict({k: v.cuda() for k, v in best_sd.items()})
    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for i in range(0, len(te_pairs), a.batch):
            chunk = te_pairs[i:i + a.batch]
            x = tok([c[0] for c in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=96)
            g = model.generate(**{k: v.cuda() for k, v in x.items()},
                               num_beams=a.beams, max_new_tokens=a.max_target_len)
            hyps += tok.batch_decode(g, skip_special_tokens=True)
            refs += [c[1] for c in chunk]
    bleu = corpus_bleu_chrf(hyps, refs)
    distinct = len(set(hyps)) / max(len(hyps), 1)
    modal = max((hyps.count(h) for h in set(hyps)), default=0) / max(len(hyps), 1)
    print(f"\ntest {len(hyps)} hyps: {bleu}  distinct {distinct:.3%}  "
          f"modal {modal:.3%}")

    head = best_sd["lm_head.weight"].float()
    moved_from_init = float((head - head0.cpu()).norm() / head0.cpu().norm())
    moved_from_base = float((head - base_head).norm() / base_head.norm())
    cos_base = float(torch.nn.functional.cosine_similarity(
        head.double().flatten(), base_head.double().flatten(), dim=0))

    if a.save_donor:
        sd = {f"mt5_model.{k}": v for k, v in best_sd.items()}
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": sd}, out)
        print(f"-> {out}  ({out.stat().st_size/2**30:.2f} GiB)")

    rep = {"fold": a.fold, "init_head": a.init_head, "epochs": a.epochs,
           "lr": a.lr, "label_smoothing": a.label_smoothing, "seed": a.seed,
           "n_train": len(tr_pairs), "n_dev": len(dv_pairs),
           "n_test": len(te_pairs), "best_dev_ce": best, "log": log,
           "test": {**bleu, "n": len(hyps), "distinct": distinct,
                    "modal": modal},
           "lm_head_rel_l2_from_init": moved_from_init,
           "lm_head_rel_l2_from_mt5base": moved_from_base,
           "lm_head_cos_to_mt5base": cos_base,
           "donor": str(out) if a.save_donor else None,
           "sample": [{"gloss": g, "hyp": h, "ref": r}
                      for (g, r), h in list(zip(te_pairs, hyps))[:20]]}
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"lm_head moved {moved_from_init:.4f} rel L2 from its init, "
          f"{moved_from_base:.4f} from mT5-base (cos {cos_base:.5f})")
    print(f"-> {report}")


if __name__ == "__main__":
    main()
