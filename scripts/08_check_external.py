#!/usr/bin/env python3
"""Verify external pretrained sign-language weights before betting the paper on them.

The cross-sign-language transfer axis needs somebody else's checkpoint. This
script answers, without training anything:

  1. Do the weights actually download?
  2. How big are they really, in parameters and in bf16/fp32 GPU memory?
  3. **What pose format do they expect?** — read off the shape of the first
     projection in the pose encoder, which pins down the keypoint count that the
     docs do not state. Our MediaPipe pipeline emits 57 points; if Uni-Sign wants
     COCO-WholeBody 133, that is a preprocessing job to schedule, not a surprise
     to discover on day 8.
  4. Which checkpoints exist, so the transfer matrix can be planned.

Run it before `03_extract_features.py`, because the answer to (3) may change what
we extract.

    python3 scripts/08_check_external.py                  # metadata only, no download
    python3 scripts/08_check_external.py --download csl_daily_pose_only_slt.pth
    python3 scripts/08_check_external.py --download all --out data/external
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = "ZechengLi19/Uni-Sign"

# What each checkpoint contributes to the transfer matrix. The point of the
# comparison is to separate "shares the written output language" from "shares the
# sign-language visual task" — plan §A argues these are confounded in the
# literature and must be ablated apart.
CHECKPOINTS = {
    "csl_daily_pose_only_slt.pth": dict(
        sign="CSL", text="zh", task="sentence SLT", pose_only=True,
        role="shares the OUTPUT LANGUAGE with our target (Chinese)"),
    "csl_stage1_weight.pth": dict(
        sign="CSL", text="zh", task="stage-1 pose pretraining", pose_only=True,
        role="representation before SLT fine-tuning; cleanest encoder transfer"),
    "csl_stage2_weight.pth": dict(
        sign="CSL", text="zh", task="stage-2 RGB-pose pretraining", pose_only=False,
        role="as above, with RGB"),
    "how2sign_pose_only_slt.pth": dict(
        sign="ASL", text="en", task="sentence SLT", pose_only=True,
        role="shares the SIGN-LANGUAGE TASK but not the output language"),
    "openasl_pose_only_slt.pth": dict(
        sign="ASL", text="en", task="sentence SLT", pose_only=True,
        role="second ASL source; checks that ASL results are not dataset-specific"),
    "wlasl_pose_only_islr.pth": dict(
        sign="ASL", text="-", task="isolated sign recognition", pose_only=True,
        role="pure sign-form encoder; no sentence, no output language"),
    "csl_daily_rgb_pose_slt.pth": dict(
        sign="CSL", text="zh", task="sentence SLT", pose_only=False,
        role="RGB+pose variant; only if the pose-only row is promising"),
    "wlasl_rgb_pose_islr.pth": dict(
        sign="ASL", text="-", task="isolated sign recognition", pose_only=False,
        role="RGB+pose ISLR; candidate for the dictionary retrieval probe"),
}

# Keypoint counts of the formats these models are built on, so a first-layer
# shape can be named rather than merely reported.
KNOWN_FORMATS = {
    133: "COCO-WholeBody (MMPose/RTMPose): body 17 + feet 6 + face 68 + hands 42",
    134: "COCO-WholeBody + 1",
    76: "CoSign-style reduced whole-body set",
    57: "our MediaPipe set: 15 body + 21 + 21 hands",
    27: "upper body + hands, coarse",
    17: "COCO body only",
}


def plan() -> None:
    print(f"transfer matrix available from {REPO}\n")
    print(f"{'checkpoint':34s} {'sign':5s} {'text':5s} {'pose-only':10s} task")
    print("-" * 100)
    for name, m in CHECKPOINTS.items():
        print(f"{name:34s} {m['sign']:5s} {m['text']:5s} "
              f"{str(m['pose_only']):10s} {m['task']}")
    print("\nrole in the experiment:")
    for name, m in CHECKPOINTS.items():
        print(f"  {name:34s} {m['role']}")
    print("\nThe comparison that matters: if the CSL checkpoint beats the ASL ones on")
    print("TSL only because it already speaks Chinese, the WLASL encoder — which has")
    print("no output language at all — should transfer as well at the form level and")
    print("worse at the sentence level. That separates decoder credit from encoder")
    print("credit, which plan §A says the literature conflates.")


def inspect(path: Path) -> dict:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj
    for key in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    n = sum(v.numel() for v in tensors.values())

    info = {
        "file": path.name,
        "file_gb": path.stat().st_size / 2**30,
        "n_tensors": len(tensors),
        "n_params_M": n / 1e6,
        "fp32_gb": n * 4 / 2**30,
        "bf16_gb": n * 2 / 2**30,
        "adamw_full_ft_gb": n * (2 + 4 + 4 + 4) / 2**30,  # bf16 wts + fp32 master+m+v
        "top_prefixes": {},
        "pose_input_candidates": [],
        "text_backbone_hint": [],
    }
    pref: dict[str, int] = {}
    for k, v in tensors.items():
        pref[k.split(".")[0]] = pref.get(k.split(".")[0], 0) + v.numel()
    info["top_prefixes"] = {k: round(v / 1e6, 1) for k, v in
                            sorted(pref.items(), key=lambda x: -x[1])[:12]}

    # Pose format: look for small leading dims that plausibly encode keypoints.
    for k, v in tensors.items():
        if any(t in k.lower() for t in ("pose", "gcn", "joint", "skeleton", "keypoint")):
            if v.ndim >= 2 and min(v.shape) <= 200:
                info["pose_input_candidates"].append([k, list(v.shape)])
    info["pose_input_candidates"] = info["pose_input_candidates"][:20]

    for k in tensors:
        for t in ("mt5", "mbart", "t5", "bert", "llama", "qwen", "gpt", "xglm", "byt5"):
            if t in k.lower():
                info["text_backbone_hint"].append(t)
                break
    info["text_backbone_hint"] = sorted(set(info["text_backbone_hint"]))
    # Vocabulary size is the other tell for which tokenizer/decoder it carries.
    for k, v in tensors.items():
        if "embed" in k.lower() and v.ndim == 2 and v.shape[0] > 5000:
            info.setdefault("vocab_candidates", []).append([k, list(v.shape)])
    return info


def report(info: dict) -> None:
    print(f"\n=== {info['file']} ===")
    print(f"  on disk        {info['file_gb']:.2f} GB, {info['n_tensors']} tensors")
    print(f"  parameters     {info['n_params_M']:.1f} M")
    print(f"  weights        fp32 {info['fp32_gb']:.2f} GB | bf16 {info['bf16_gb']:.2f} GB")
    print(f"  full FT AdamW  ~{info['adamw_full_ft_gb']:.2f} GB of optimiser+weights "
          f"(24 GB budget: {'fits' if info['adamw_full_ft_gb'] < 16 else 'use LoRA'})")
    print(f"  parameter mass by top-level module (M): {info['top_prefixes']}")
    if info["text_backbone_hint"]:
        print(f"  text backbone hint: {info['text_backbone_hint']}")
    if info.get("vocab_candidates"):
        print(f"  embedding tables: {info['vocab_candidates'][:4]}")
    if info["pose_input_candidates"]:
        print("  POSE-SHAPED tensors (read the keypoint count off these):")
        for k, shp in info["pose_input_candidates"][:10]:
            named = KNOWN_FORMATS.get(max(shp) if shp else 0, "")
            print(f"    {k:52s} {shp}  {named}")
        print("    Our MediaPipe features are 57 points (15 body + 21 + 21).")
        print("    A mismatch means re-extracting poses in their format — schedule it.")
    else:
        print("  no obviously pose-shaped tensors; inspect `top_prefixes` by hand")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", default="",
                    help="checkpoint filename(s), comma-separated, or 'all'")
    ap.add_argument("--out", type=Path, default=Path("data/external"))
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    plan()
    if not a.download:
        print("\n(no --download given; nothing fetched)")
        print("Next: python3 scripts/08_check_external.py --download csl_daily_pose_only_slt.pth")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("\nhuggingface_hub not installed — `pip install huggingface_hub`")
        sys.exit(1)

    a.out.mkdir(parents=True, exist_ok=True)
    names = (list(CHECKPOINTS) if a.download == "all"
             else [x.strip() for x in a.download.split(",") if x.strip()])
    out = []
    for name in names:
        try:
            p = Path(hf_hub_download(REPO, name, local_dir=str(a.out)))
        except Exception as e:
            print(f"\n=== {name} ===\n  DOWNLOAD FAILED: {type(e).__name__}: {e}")
            continue
        try:
            info = inspect(p)
        except Exception as e:
            print(f"\n=== {name} ===\n  LOADED BUT UNREADABLE: {type(e).__name__}: {e}")
            continue
        info["meta"] = CHECKPOINTS.get(name, {})
        report(info)
        out.append(info)

    if len(out) > 1:
        print("\n=== comparability across initialisations ===")
        print("A transfer comparison is only fair if the rows differ ONLY in what the")
        print("weights were pretrained on. Same architecture, same parameter count.")
        print(f"\n{'checkpoint':34s} {'params (M)':>11s} {'tensors':>8s}  same shape as first?")
        ref = None
        for info in out:
            if ref is None:
                ref = info
                same = "(reference)"
            else:
                same = ("YES" if abs(info["n_params_M"] - ref["n_params_M"]) < 0.5
                        and info["n_tensors"] == ref["n_tensors"] else "NO -- NOT COMPARABLE")
            print(f"{info['file']:34s} {info['n_params_M']:11.1f} {info['n_tensors']:8d}  {same}")
        spread = max(i["n_params_M"] for i in out) - min(i["n_params_M"] for i in out)
        print(f"\nparameter-count spread: {spread:.1f} M")
        if spread > 1.0:
            print("Rows differ in size. Report the counts in the paper and say why, or")
            print("a reviewer will read the difference as capacity, not as transfer.")
        else:
            print("Rows are size-matched: any difference is attributable to pretraining.")

    if a.json and out:
        a.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n-> {a.json}")
    if not out:
        print("\nNothing loaded. The transfer axis is not available on this machine;")
        print("fall back to the in-corpus rows (TSL dictionary -> discourse, north/south)")
        print("and drop the cross-sign-language column, as agreed.")


if __name__ == "__main__":
    main()
