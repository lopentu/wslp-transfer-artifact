#!/usr/bin/env python3
"""Precompute frozen visual features, once, for every span any condition needs.

    python3 scripts/03_extract_features.py --check-sides        # verify L/R first
    python3 scripts/03_extract_features.py --backend pose --workers 12
    python3 scripts/03_extract_features.py --backend dino --workers 1

Output: one .npz per paragraph under `--out`, keyed by record key. Per-paragraph
grouping means a target and its context clips load with one file open.

`--check-sides` measures which half of a dialogue frame is moving while each
speaker holds the floor, instead of assuming L means screen-left. It costs about
a minute and protects every downstream number, because a wrong mapping silently
feeds the model the listener.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.db import FILMS_DIR  # noqa: E402
from tsl.features import BACKENDS, FPS, crops_for, decode, extract_film  # noqa: E402

_STATE: dict = {}


def _init(backend_name: str, device: str, threads: int) -> None:
    # Cap per-worker threading before the ONNX session is built: N workers each
    # spawning 32 OpenMP threads would spend all their time in contention.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    kw = ({"device": device, "threads": threads} if backend_name == "rtmpose"
          else {"device": device} if backend_name == "dino" else {})
    _STATE["backend"] = BACKENDS[backend_name](**kw)


def _one_film(job: tuple[str, str, dict, Path, int]) -> tuple[str, int, float]:
    uuid, para_type, spans, out, fps = job
    t0 = time.time()
    dest = out / f"{uuid}.npz"
    if dest.exists():
        return (uuid, -1, 0.0)
    feats = extract_film(
        FILMS_DIR / f"{uuid}.mp4", para_type,
        {k: (v["t1"], v["t2"], v["speaker"]) for k, v in spans.items()},
        _STATE["backend"], fps=fps,
    )
    # The temp name must itself end in .npz: savez_compressed silently appends
    # ".npz" to any path that does not, so a ".npz.part" temp lands as
    # ".npz.part.npz" and the rename below fails on a file that was never created.
    tmp = dest.with_suffix(".part.npz")
    np.savez_compressed(tmp, **feats)
    tmp.replace(dest)
    return (uuid, len(feats), time.time() - t0)


def check_sides(spans: dict, n_films: int = 8) -> None:
    """Which screen half moves while speaker L / speaker R holds the floor?"""
    per: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for k, v in spans.items():
        if v["type"] == "2" and v["speaker"] in ("L", "R"):
            per[v["uuid"]][v["speaker"]].append((v["t1"], v["t2"]))
    uuids = [u for u in sorted(per) if len(per[u]) == 2][:n_films]
    print(f"checking L/R on {len(uuids)} dialogue films\n")
    print(f"{'film':10s} {'speaker':8s} {'motion left':>12s} {'motion right':>13s}  verdict")
    agree = 0
    total = 0
    for u in uuids:
        halves = {}
        film = FILMS_DIR / f"{u}.mp4"
        for side, crop in crops_for("2", film).items():
            frames = list(decode(film, crop, fps=5, out_h=180))
            if len(frames) < 3:
                halves[side] = None
                continue
            g = np.stack([f.mean(-1) for f in frames]).astype(np.float32)
            halves[side] = np.abs(np.diff(g, axis=0)).mean((1, 2))  # per-frame motion
        if halves["L"] is None or halves["R"] is None:
            continue
        for speaker, sp in per[u].items():
            idx = np.concatenate([np.arange(int(t1 * 5 / 1000), int(t2 * 5 / 1000))
                                  for t1, t2 in sp])
            idx = idx[idx < min(len(halves["L"]), len(halves["R"]))]
            if len(idx) == 0:
                continue
            ml, mr = halves["L"][idx].mean(), halves["R"][idx].mean()
            ok = (speaker == "L" and ml > mr) or (speaker == "R" and mr > ml)
            agree += ok
            total += 1
            print(f"{u:10s} {speaker:8s} {ml:12.3f} {mr:13.3f}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n{agree}/{total} consistent with L = screen-left.")
    if total and agree / total < 0.9:
        print("STOP: the mapping is not what features.crops_for() assumes. Fix it "
              "before extracting, or the model is fed the listener.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", type=Path, default=Path("data/spans.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="rtmpose")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = pick by backend: 4 for rtmpose (one GPU ONNX session "
                         "each), 12 for the CPU-bound mediapipe backend")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="rtmpose/dino only; cpu is the faster choice here while "
                         "the GPU is shared (see features.RTMPoseBackend)")
    ap.add_argument("--threads", type=int, default=2, help="ONNX threads per worker")
    ap.add_argument("--check-sides", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="films, for a smoke test")
    a = ap.parse_args()

    spans = json.loads(a.spans.read_text())
    if a.check_sides:
        check_sides(spans)
        return

    if a.workers == 0:
        # This machine is shared. workers x threads is the real core footprint;
        # keep it well under nproc so other users still get the box.
        a.workers = {"rtmpose": 6, "dino": 1}.get(a.backend, 8)
    out = a.out or Path(f"data/feat_{a.backend}")
    out.mkdir(parents=True, exist_ok=True)

    by_film: dict[str, dict] = defaultdict(dict)
    for k, v in spans.items():
        by_film[v["uuid"]][k] = v
    jobs = [
        (u, next(iter(s.values()))["type"], s, out, a.fps)
        for u, s in sorted(by_film.items())
    ]
    if a.limit:
        jobs = jobs[: a.limit]

    print(f"backend={a.backend} device={a.device} films={len(jobs)} "
          f"spans={len(spans)} fps={a.fps} workers={a.workers}x{a.threads}thr -> {out}")
    done = t_sum = 0
    t_start = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(a.backend, a.device, a.threads)) as ex:
        for uuid, n, dt in ex.map(_one_film, jobs, chunksize=1):
            done += 1
            if n < 0:
                continue
            t_sum += dt
            el = time.time() - t_start
            eta = el / done * (len(jobs) - done)
            print(f"[{done}/{len(jobs)}] {uuid} {n} spans {dt:5.1f}s  "
                  f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    print(f"\ndone in {(time.time()-t_start)/60:.1f} min "
          f"({t_sum/60:.1f} min of worker time)")


if __name__ == "__main__":
    main()
