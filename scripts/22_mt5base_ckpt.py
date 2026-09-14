#!/usr/bin/env python3
"""Write `google/mt5-base` out as a Uni-Sign-shaped checkpoint.

Two reviews ask for the same missing cell, and neither is reachable with the five
released files. Table 1c currently reads

    How2Sign + CSL-Daily lm_head    48.5
    How2Sign + CSL-Daily body only  25.5
    CSL-Daily + How2Sign lm_head    24.3

which identifies "the output projection carries the difference" but *not* which
side of it is doing the work. Two readings survive:

  adding a Chinese head helps          the paper's current framing
  removing a damaged English head is enough   weaker, and more likely a priori,
                                       since RAND-VIS -- whose head is exactly
                                       this file's -- already scores 40.1

Telling them apart needs a *neutral* third level of the head factor: the output
projection every one of these checkpoints started from, before any sign-language
fine-tuning specialized it to one written language. That is `google/mt5-base`'s
own `lm_head`, and it is not in any released checkpoint because the released
checkpoints overwrote it.

So dump it. The keys are prefixed `mt5_model.` to match Uni-Sign's own naming,
which is the only thing `unisign.PART_SETS` cares about, so the resulting file
drops into the existing machinery unchanged:

    --init how2sign --init-parts all --init-mt5 mt5_base --init-mt5-parts lm_head

and `21_lineage.py` can measure distances to it like any other checkpoint.

    ../.venv/bin/python scripts/22_mt5base_ckpt.py     # -> data/external/mt5_base.pth

CPU only, about a minute, 2.3 GB on disk. Note this is NOT a sixth pretrained
model: it is the published mT5-base, which is already the initialization of every
`-pose` row and of RAND-VIS. Writing it to a file only makes one *part* of it
addressable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "mt5_base.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default="google/mt5-base")
    a = ap.parse_args()

    from transformers import MT5ForConditionalGeneration

    m = MT5ForConditionalGeneration.from_pretrained(a.model)
    sd = {f"mt5_model.{k}": v.detach().clone()
          for k, v in m.state_dict().items()}
    n_head = sum(v.numel() for k, v in sd.items() if k.startswith("mt5_model.lm_head"))
    if not n_head:
        raise SystemExit("no mt5_model.lm_head.* in the state dict — mT5-base ties "
                         "its output embedding in this transformers version, and "
                         "the whole point of the file is that the head is separable")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd}, a.out)
    print(f"{len(sd)} tensors, {sum(v.numel() for v in sd.values())/1e6:.1f}M params, "
          f"lm_head {n_head/1e6:.1f}M")
    print(f"-> {a.out}  ({a.out.stat().st_size/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
