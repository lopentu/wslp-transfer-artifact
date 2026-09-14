#!/usr/bin/env python3
"""How independent are the five released checkpoints, measured rather than read?

Uni-Sign's own recipe fine-tunes every downstream model FROM its stage-1 pose-only
pre-training on CSL-News (`script/train_stage3.sh`: `--finetune
out/stage1_pretraining/best_checkpoint.pth`). If that is what the released files
are, then two of the paper's sentences need changing:

  * "two Chinese-emitting halves never trained together" -- they share an
    ancestor, so co-adaptation is not excluded, only attenuated;
  * "three ASL checkpoints from three corpora and two tasks agree on 92-94% of
    items" -- some of that agreement could be shared ancestry rather than
    convergence.

And one gets sharper: an ASL encoder is not a branch that never saw CSL. It is a
CSL-News-pretrained branch that was then fine-tuned on ASL, so what the transfer
result measures is what that fine-tuning *undid*.

Reading a recipe is not evidence about the files on disk, so this measures the
weights. For every checkpoint, against the stage-1 file and against the published
`google/mt5-base`:

    rel L2      ||W - W_ref|| / ||W_ref||, summed over the half in question
    cos         cosine between the flattened halves

A downstream fine-tune of stage 1 sits close to it; an independently trained model
does not. mt5-base is the second reference because it is where every one of these
models' decoders started, and it says how far each has moved from a general
multilingual LM -- which is the axis the paper's decoder claim is about.

    python3 scripts/21_lineage.py            # -> data/lineage.json

CPU only, and streamed one checkpoint at a time: this box has 31 GB and the sweep
is using most of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
FILES = {
    "csl_stage1": "csl_stage1_weight.pth",
    "csl_daily": "csl_daily_pose_only_slt.pth",
    "how2sign": "how2sign_pose_only_slt.pth",
    "openasl": "openasl_pose_only_slt.pth",
    "wlasl": "wlasl_pose_only_islr.pth",
}
HALVES = {
    "pose": lambda k: not k.startswith("mt5_model"),
    "mt5": lambda k: k.startswith("mt5_model"),
    "mt5_encoder": lambda k: k.startswith("mt5_model") and ".encoder." in k,
    "mt5_decoder": lambda k: k.startswith("mt5_model") and ".decoder." in k,
    "lm_head": lambda k: k.startswith("mt5_model.lm_head"),
}


def state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = obj
    for key in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    return {k.replace("module.", "", 1): v for k, v in sd.items()
            if torch.is_tensor(v) and v.is_floating_point()}


def compare(a: dict, b: dict, keep) -> dict | None:
    keys = sorted(k for k in set(a) & set(b) if keep(k) and a[k].shape == b[k].shape)
    if not keys:
        return None
    dot = num = da = db = 0.0
    n = 0
    for k in keys:
        # float64: the halves being compared run to 10^8 parameters, and a
        # float32 accumulation of that many terms puts a cosine of 1 at 1.00002,
        # which is not a number this script may print.
        x = a[k].double().flatten()
        y = b[k].double().flatten()
        dot += float(torch.dot(x, y))
        num += float(torch.sum((x - y) ** 2))
        da += float(torch.sum(x * x))
        db += float(torch.sum(y * y))
        n += x.numel()
    return {"tensors": len(keys), "params_M": n / 1e6,
            "rel_l2": (num ** 0.5) / (db ** 0.5) if db else float("nan"),
            "cos": dot / ((da ** 0.5) * (db ** 0.5)) if da and db else float("nan")}


def mt5_base_state() -> dict[str, torch.Tensor]:
    """`google/mt5-base` under the checkpoints' own key names."""
    from transformers import MT5ForConditionalGeneration
    m = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    return {f"mt5_model.{k}": v.detach()
            for k, v in m.state_dict().items() if v.is_floating_point()}


def pairwise(names: list[str], get) -> dict:
    """Every pair, every half. The group structure is the point: if a fine-tune
    of stage 1 moves one half and leaves the others, the pairwise matrix says so
    without anyone having to trust a README."""
    import itertools
    out = {}
    for x, y in itertools.combinations(names, 2):
        sx, sy = get(x), get(y)
        for h, keep in HALVES.items():
            c = compare(sx, sy, keep)
            if c:
                out[f"{x}|{y}|{h}"] = c
        del sx, sy
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "lineage.json")
    ap.add_argument("--skip-mt5-base", action="store_true")
    ap.add_argument("--pairwise", action="store_true",
                    help="every pair of checkpoints, not only each against stage 1")
    a = ap.parse_args()

    ref = state(EXT / FILES["csl_stage1"])
    base = None if a.skip_mt5_base else mt5_base_state()

    out: dict[str, dict] = {}
    for name, fn in FILES.items():
        if name == "csl_stage1":
            continue
        sd = state(EXT / fn)
        row = {f"vs_stage1_{h}": compare(sd, ref, keep) for h, keep in HALVES.items()}
        if base is not None:
            row.update({f"vs_mt5base_{h}": compare(sd, base, keep)
                        for h, keep in HALVES.items() if h != "pose"})
        out[name] = row
        del sd
        print(f"{name}:")
        for k, v in row.items():
            if v:
                print(f"   {k:22s} rel_l2 {v['rel_l2']:.4f}  cos {v['cos']:.5f}"
                      f"  ({v['params_M']:.1f}M)")
    if base is not None:
        out["csl_stage1"] = {f"vs_mt5base_{h}": compare(ref, base, keep)
                             for h, keep in HALVES.items() if h != "pose"}
        print("csl_stage1:")
        for k, v in out["csl_stage1"].items():
            if v:
                print(f"   {k:22s} rel_l2 {v['rel_l2']:.4f}  cos {v['cos']:.5f}")

    if a.pairwise:
        cache: dict[str, dict] = {}

        def get(n):
            if n not in cache:
                cache.clear()   # one at a time; the box is shared
                cache[n] = mt5_base_state() if n == "mt5_base" else state(EXT / FILES[n])
            return cache[n]

        names = list(FILES) + ([] if a.skip_mt5_base else ["mt5_base"])
        pw = pairwise(names, get)
        out["pairwise"] = pw
        print("\npairwise (cos):")
        for h in HALVES:
            print(f"  -- {h}")
            for k, v in pw.items():
                if k.endswith(f"|{h}"):
                    x, y, _ = k.split("|")
                    print(f"     {x:12s} {y:12s} cos {v['cos']:+.5f}  rel_l2 {v['rel_l2']:.4f}")

    a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
