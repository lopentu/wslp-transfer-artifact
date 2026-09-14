#!/usr/bin/env python3
"""Does the wrong clip move the encoder, or only the output?

§4 leaves two readings of the `lm_head` result open and says two controls would
settle it. This is a third, and it needs no training at all.

The paper's wrong-clip evidence is read off the *output*: replace the target clip
and How2Sign's accuracy does not move and its score margin moves 0.003 nats. Two
things produce that number:

  the clip is not read      the pose branch's output carries nothing about which
                            utterance this is, so nothing downstream can differ
  the clip is read and not  the encoder state moves as much as a matched model's,
  expressible               and the output projection maps both states to the
                            same distribution

They are distinguishable inside one forward pass, because the two live at
different depths. For every item, run the model twice -- own clip, wrong clip --
and measure the displacement at two points:

    encoder     mean-pooled mT5 encoder state over the POSE positions only,
                excluding the text prefix, which is identical in both passes and
                would otherwise dilute the displacement toward zero
    output      the first-step next-token distribution over the 250112-token
                vocabulary, as a Jensen-Shannon divergence in nats

If How2Sign's encoder displacement is comparable to CSL-Daily's while its output
displacement is not, the second reading is supported and the paper can say so. If
its encoder barely moves either, the first is, and the wrong-clip result stands as
the paper already reads it. Either way "two readings remain open" becomes a
measurement.

    ../.venv/bin/python scripts/25_encoder_shift.py \
        --runs f0_local_csl_daily_full_k4_s0 f0_local_how2sign_full_k4_s0 \
               f0_local_random_full_k4_s0 f0_local_how2sign+csl_daily-lm_head_full_k4_s0

-> data/encoder_shift.json. Forward passes only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl import unisign as us  # noqa: E402
from tsl.conditions import (  # noqa: E402
    assign_donors, assign_target_swaps, assign_within_swaps, load_records,
    split_records,
)
from tsl.dataset import TSLDataset  # noqa: E402
from tsl.splits import load as load_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def to_cuda(b):
    if torch.is_tensor(b):
        return b.cuda()
    if isinstance(b, dict):
        return {k: to_cuda(v) for k, v in b.items() if k != "meta"}
    return b


@torch.no_grad()
def states(model, src, n_prefix_probe: bool = False):
    """Pooled encoder state over the pose block, and the first-step distribution.

    Reimplements `encoder_inputs` rather than calling it, because the quantity
    needed is the boundary between the text prefix and the pose block and that
    function returns only the concatenation. The prefix is the same string for
    every row here (no context text), so its length is one number per batch.
    """
    pose = model.pose_embeds(src)
    mask = src["attention_mask"]
    pre_emb, pre_mask = model._prefix(pose.shape[0], None, pose.device)
    n_pre = pre_emb.shape[1]
    embeds = torch.cat([pre_emb, pose], dim=1)
    full = torch.cat([pre_mask, mask], dim=1)
    enc = model.mt5.encoder(inputs_embeds=embeds, attention_mask=full,
                            return_dict=True).last_hidden_state
    m = full[:, n_pre:].unsqueeze(-1).to(enc.dtype)
    pooled = (enc[:, n_pre:] * m).sum(1) / m.sum(1).clamp(min=1e-6)

    # One decoder step from the start token: the model's opinion about the first
    # output token given this clip, which is the shallowest place the output
    # projection is involved at all.
    dec_in = torch.zeros(enc.shape[0], 1, dtype=torch.long, device=enc.device)
    out = model.mt5(encoder_outputs=(enc,), attention_mask=full,
                    decoder_input_ids=dec_in, return_dict=True)
    logp = torch.log_softmax(out.logits[:, 0].float(), -1)
    return pooled.float(), logp


def js(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in nats between two log-probability rows."""
    m = torch.logaddexp(a, b) - np.log(2.0)
    ka = (a.exp() * (a - m)).sum(-1)
    kb = (b.exp() * (b - m)).sum(-1)
    return 0.5 * (ka + kb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "lex")
    ap.add_argument("--folds", type=Path, default=ROOT / "data" / "folds.json")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--draws", type=int, default=3,
                    help="wrong-clip assignments to average over, matching the "
                         "three the accuracy tables use")
    ap.add_argument("--within", action="store_true",
                    help="also the within-paragraph wrong clip")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N items only; for checking the code path without "
                         "waiting for a full pass")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "encoder_shift.json")
    a = ap.parse_args()

    records = load_records(a.data / "records.jsonl")
    folds = load_folds(a.folds)
    out: dict[str, dict] = {}

    for name in a.runs:
        ckpt = ROOT / "runs" / name / "best.pt"
        if not ckpt.exists():
            print(f"{name}: no best.pt, skipped")
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        fold = ck["args"]["fold"]
        feat = Path(ck["args"]["feat"])
        model, _ = us.load_run(ckpt)
        model = model.cuda().eval()
        cfg = ck["cfg"]
        test = split_records(records, folds, fold)["test"]
        items = [r for r in test if r["item_type"] and r["candidates"]]
        if a.limit:
            items = items[: a.limit]
        donors = assign_donors(records, seed=ck["args"].get("seed", 0))

        def pooled_for(cond, swaps=None):
            ds = TSLDataset(items, feat, condition=cond, ctx_k=ck["args"].get("ctx_k", 4),
                            donors=donors, all_records=records, swaps=swaps,
                            max_frames=cfg.get("max_frames", 192),
                            ctx_max_frames=cfg.get("ctx_max_frames", 192))
            dl = DataLoader(ds, batch_size=a.batch, shuffle=False,
                            num_workers=a.workers, collate_fn=model_collate)
            P, L, K = [], [], []
            for batch in dl:
                b = to_cuda(batch)
                p, l = states(model, b["src"])
                P.append(p.cpu())
                L.append(l.cpu())
                K += [m["key"] for m in batch["meta"]]
            return torch.cat(P), torch.cat(L), K

        def model_collate(bat):
            return us.collate_unisign(bat, cfg.get("max_frames", 192),
                                      cfg.get("ctx_max_frames", 192))

        base_p, base_l, keys = pooled_for("local")
        row: dict = {"run": name, "fold": fold, "n": len(keys),
                     "init": cfg.get("init"),
                     "init_mt5": cfg.get("init_mt5"),
                     "init_mt5_parts": cfg.get("init_mt5_parts")}
        families = [("swap_plain", assign_target_swaps)]
        if a.within:
            families.append(("swap_within", assign_within_swaps))
        for cond, fn in families:
            cos, rel, jsd = [], [], []
            for d in range(a.draws):
                sw = fn(test, seed=0, perm=d, pool=test)
                p, l, k = pooled_for(cond, swaps=sw)
                assert k == keys, "item order changed between conditions"
                cos.append(torch.nn.functional.cosine_similarity(base_p, p, dim=1))
                rel.append((base_p - p).norm(dim=1) / base_p.norm(dim=1))
                jsd.append(js(base_l, l))
            f = lambda t: float(torch.stack(t).mean())  # noqa: E731
            md = lambda t: float(torch.stack(t).mean(0).median())  # noqa: E731
            row[cond] = {"enc_cos": f(cos), "enc_cos_median": md(cos),
                         "enc_rel_l2": f(rel), "enc_rel_l2_median": md(rel),
                         "out_js": f(jsd), "out_js_median": md(jsd),
                         "draws": a.draws}
            print(f"{name}\n   {cond}: encoder cos {row[cond]['enc_cos']:.4f}  "
                  f"rel L2 {row[cond]['enc_rel_l2']:.4f}   output JS "
                  f"{row[cond]['out_js']:.4f} nats", flush=True)
        out[name] = row
        del model
        torch.cuda.empty_cache()

    a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
