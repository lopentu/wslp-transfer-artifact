#!/usr/bin/env python3
"""Reconcile the feature store against the record set, and sanity-check its contents.

    python3 scripts/11_verify_features.py
    python3 scripts/11_verify_features.py --feat data/feat_mediapipe

Run this after **every** extraction. `03_extract_features.py` can finish cleanly
having silently dropped spans: it crops a dialogue frame by speaker side, and a
span whose side is not a key of that dict matches no branch and is skipped with no
warning. That is how 17 spans — three of them diagnostic items — went missing on
the first full run. Exit status is non-zero if anything is absent, so this is
usable as a gate in front of training.

The other checks are cheap and catch the failures that do not announce themselves
either: an all-zero frame means the detector found nothing, a dimension mismatch
means the keypoint layout drifted, and the frame total should agree with the
corpus's annotated duration once divided by the extraction frame rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.features import FPS, RTMPOSE_DIM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=Path, default=Path("data/feat_rtmpose"))
    ap.add_argument("--records", type=Path, default=Path("data/records.jsonl"))
    ap.add_argument("--dim", type=int, default=RTMPOSE_DIM)
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()

    records = [json.loads(l) for l in a.records.open()]
    by_key = {r["key"]: r for r in records}

    store: dict[str, tuple[str, tuple]] = {}
    unreadable = []
    for f in sorted(a.feat.glob("*.npz")):
        try:
            z = np.load(f)
            for k in z.files:
                store[k] = (f.name, z[k].shape)
        except Exception as e:
            unreadable.append(f"{f.name}: {type(e).__name__}: {e}")

    missing = [k for k in by_key if k not in store]
    extra = [k for k in store if k not in by_key]

    print(f"records {len(records)}   spans on disk {len(store)}   "
          f"films {len(list(a.feat.glob('*.npz')))}")
    if unreadable:
        print(f"\nUNREADABLE {len(unreadable)}")
        for u in unreadable[:10]:
            print("  ", u)
    if missing:
        print(f"\nMISSING {len(missing)} — extraction dropped these silently")
        for k in missing[:20]:
            r = by_key[k]
            print(f"   {k:14s} spk={str(r['speaker']):4s} para={r['para_type']} "
                  f"item={str(r['item_type']):9s} {r['text'][:26]}")
        n_item = sum(1 for k in missing if by_key[k]["item_type"])
        print(f"   of which diagnostic items: {n_item}")
    if extra:
        print(f"\nEXTRA {len(extra)} on disk with no record: {extra[:10]}")

    # Contents. Only worth reporting over the spans that are actually present.
    bad_dim, empty, zero_frames = [], [], []
    total_frames = 0
    conf, lens = [], []
    for f in sorted(a.feat.glob("*.npz")):
        try:
            z = np.load(f)
        except Exception:
            continue
        for k in z.files:
            arr = z[k]
            if arr.ndim != 2 or arr.shape[0] == 0:
                empty.append(k)
                continue
            if arr.shape[1] != a.dim:
                bad_dim.append((k, arr.shape))
                continue
            total_frames += arr.shape[0]
            lens.append(arr.shape[0])
            kp = arr.reshape(arr.shape[0], -1, 3)
            if (kp[..., :2].sum(-1) == 0).any():
                zero_frames.append(k)
            conf.append(float(np.median(kp[..., 2])))

    L = np.array(lens) if lens else np.array([0])
    print(f"\nframes {total_frames:,}  = {total_frames / a.fps / 3600:.2f} h at {a.fps} fps")
    print(f"frames/span  min {L.min()}  p05 {np.percentile(L, 5):.0f}  "
          f"med {np.median(L):.0f}  p95 {np.percentile(L, 95):.0f}  max {L.max()}")
    print(f"keypoint confidence: median of medians {np.median(conf):.3f}")
    print(f"dimension {a.dim}: mismatches {len(bad_dim)} {bad_dim[:3]}")
    print(f"empty arrays {len(empty)} {empty[:3]}")
    print(f"spans containing an all-zero keypoint {len(zero_frames)} {zero_frames[:3]}")

    short = [k for k, (_, s) in store.items() if s[0] < 8]
    short_items = [k for k in short if by_key.get(k, {}).get("item_type")]
    print(f"spans under 8 frames {len(short)}; diagnostic items among them "
          f"{len(short_items)} {short_items[:5]}")

    ok = not (missing or extra or unreadable or bad_dim or empty)
    print("\nOK" if ok else "\nFAILED — do not train on this store")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
