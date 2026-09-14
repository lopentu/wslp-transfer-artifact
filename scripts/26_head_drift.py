#!/usr/bin/env python3
"""How far a run's output projection ends up from where it started, and from
another run's.

Written 2026-08-22 to close the one gap left in the post-hoc argument. §3.3 shows
that swapping `lm_head` before fine-tuning is worth ~24 points and swapping the
same tensor after fine-tuning is worth nothing, and concludes that what a
mismatched projection costs is the adaptation rather than the readout. A reader
can still object that the two interventions did not really install the same
tensor: the initialization-time cell trained its projection for twelve epochs
afterwards, so perhaps it ended up somewhere better than the donor the post-hoc
graft installs.

They did install nearly the same tensor. This measures it. If the
initialization-swapped model's FINAL projection is close to the donor's, then two
models carrying essentially the same output projection differ by the whole effect,
and the difference has to live in the rest of the network.

    ../.venv/bin/python scripts/26_head_drift.py     # -> data/head_drift.json

CPU only. `best.pt` is 7.8 GB and memory-mapped, and only one 192.1M-parameter
tensor is materialized per run, so the peak is a few gigabytes rather than the
file size.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
KEY = "core.mt5_model.lm_head.weight"

# The runs whose projections the argument compares. Keyed by the name the paper
# uses, so `make_numbers.py` does not have to know about run directories.
RUNS = {
    "how2sign": "f0_local_how2sign_full_k4_s0",
    "csl_daily": "f0_local_csl_daily_full_k4_s0",
    "init_swapped": "f0_local_how2sign+csl_daily-lm_head_full_k4_s0",
    "base_head": "f0_local_how2sign+mt5_base-lm_head_full_k4_s0",
    "csl_base_head": "f0_local_csl_daily+mt5_base-lm_head_full_k4_s0",
    "how2sign_seed1": "f0_local_how2sign_full_k4_s1",
    "csl_daily_seed1": "f0_local_csl_daily_full_k4_s1",
}


def head(run: str) -> torch.Tensor | None:
    p = ROOT / "runs" / run / "best.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    w = ck["state"].get(KEY)
    return None if w is None else w.double().flatten().clone()


def pair(x: torch.Tensor, y: torch.Tensor) -> dict:
    return {"cos": float(torch.dot(x, y) / (x.norm() * y.norm())),
            "rel_l2": float((x - y).norm() / y.norm())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "head_drift.json")
    a = ap.parse_args()

    got: dict[str, torch.Tensor] = {}
    for name, run in RUNS.items():
        w = head(run)
        if w is None:
            print(f"{name}: {run} absent, skipped")
            continue
        got[name] = w

    # Distance from each run's FINAL projection to the released tensor it was
    # initialized from. This is what says whether fine-tuning moves the thing at
    # all -- and it barely does, which is why the seed-to-seed grafts in
    # Appendix M cost nothing and why "initialization" and "final state" are
    # nearly the same object for this tensor.
    out: dict = {"pairs": {}, "vs_released": {}}
    ext = ROOT / "data" / "external"
    released = {"how2sign": "how2sign_pose_only_slt.pth",
                "csl_daily": "csl_daily_pose_only_slt.pth",
                "mt5_base": "mt5_base.pth"}
    rel: dict[str, torch.Tensor] = {}
    for name, fn in released.items():
        p = ext / fn
        if not p.exists():
            continue
        obj = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        sd = obj.get("model", obj) if isinstance(obj, dict) else obj
        for k, v in sd.items():
            if k.replace("module.", "", 1) == "mt5_model.lm_head.weight":
                rel[name] = v.double().flatten().clone()
                break
    # Which released tensor each run's projection was initialized from.
    INIT_OF = {"how2sign": "how2sign", "how2sign_seed1": "how2sign",
               "csl_daily": "csl_daily", "csl_daily_seed1": "csl_daily",
               "init_swapped": "csl_daily", "base_head": "mt5_base",
               "csl_base_head": "mt5_base"}
    for name, w in got.items():
        src = INIT_OF.get(name)
        if src in rel:
            out["vs_released"][name] = {"init_from": src, **pair(w, rel[src])}
            print(f"{name:16s} vs its own init ({src:9s}): "
                  f"cos {out['vs_released'][name]['cos']:+.4f}  "
                  f"rel L2 {out['vs_released'][name]['rel_l2']:.4f}")

    print()
    for x, y in itertools.combinations(sorted(got), 2):
        d = pair(got[x], got[y])
        out["pairs"][f"{x}|{y}"] = d
        print(f"{x:16s} vs {y:16s}  cos {d['cos']:+.4f}  rel L2 {d['rel_l2']:.4f}")

    a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
