#!/usr/bin/env python3
"""MediaPipe Holistic poses for the twtsl citation clips, in SignCLIP's format.

Stage 1 of the JSL probe (`experiment-suggestion-first.md` §3). SignCLIP takes
MediaPipe Holistic (543 landmarks) as a `.pose` file, whereas
`tsl.features.PoseBackend` keeps a 57-point subset for Uni-Sign, so the probe
needs its own extraction rather than a reuse of the RTMPose store.

Why this can be an entirely separate, CPU-only job: SignCLIP was pretrained on
Spreadthesign, whose language inventory includes Japanese, Chinese (China) and
American ("English (United States)") sign languages but **not** Taiwanese. TSL
is therefore genuinely unseen data for it, which is what makes the probe clean
instead of circular. And because SignCLIP accepts language-tagged text prompts
(`<en> <ase> house`), the cross-language comparison needs no non-TSL video at
all — only these TSL clips against per-language prompts.

Runs in `.venv_signclip`, NOT the training venv: `pose_format` pins
`mediapipe<0.10.30`, and installing that next to the training stack would
downgrade a dependency out from under a running sweep.

    .venv_signclip/bin/python scripts/13_signclip_poses.py --workers 8

Deliberately modest defaults. The card is shared and so is the CPU: the
training dataloaders already push this 32-core box past a load of 60, and a
32-worker extraction here would slow whoever is on the GPU rather than just
using idle capacity.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

TWTSL = Path("/mnt/md0/corpus/sign/twtsl")
OUT = Path("data/signclip/pose")


def entries() -> list[dict]:
    """One row per sign: the Chinese name, the English name, and the clip.

    The English gloss is what SignCLIP's text prompts need, since its text side
    is spoken-language text.

    **The join is on `clip`, not on `id`.** `entries` is keyed
    `(lang, id, row_id)`, so `id` is scoped per language and the Chinese and
    English rows for one sign do not share it. Joining on `id` -- as this
    function originally did -- silently pairs 蕃茄醬 (ketchup) with "LARGE
    AMOUNT", 洋蔥 (onion) with "PARIS" and 衣服 (clothes) with "COLD", which
    reads as a working join and makes every downstream text prompt meaningless.
    Both rows of a real pair point at the same clip; 3,436 of 3,519 have both.
    """
    con = sqlite3.connect(TWTSL / "twtsl.db")
    rows = sorted(con.execute(
        "select lang, id, row_id, name, clip from entries "
        "where clip is not null and clip != ''"))
    con.close()
    byclip: dict[str, dict] = {}
    for lang, i, _row_id, name, clip in rows:
        byclip.setdefault(clip, {}).setdefault(lang, (i, name))
    out = []
    for clip, d in sorted(byclip.items()):
        if "zh" not in d:
            continue
        path = TWTSL / "media" / f"{clip}.mp4"
        if not path.exists():
            continue
        zid, zh_name = d["zh"]
        out.append({"id": zid, "zh": zh_name,
                    "en": d["en"][1] if "en" in d else None, "clip": str(path)})
    return out


def extract(row: dict) -> tuple[int, str]:
    dst = OUT / f"{row['id']}.pose"
    if dst.exists() and dst.stat().st_size > 0:
        return row["id"], "skip"
    try:
        import cv2
        from pose_format.utils.holistic import load_holistic

        cap = cv2.VideoCapture(row["clip"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return row["id"], "empty-video"
        h, w = frames[0].shape[:2]
        pose = load_holistic(frames, fps=fps, width=w, height=h,
                             depth=0, progress=False)
        tmp = dst.with_suffix(".pose.part")
        with open(tmp, "wb") as fh:
            pose.write(fh)
        tmp.rename(dst)          # atomic, so an interrupted run leaves no half file
        return row["id"], "ok"
    except Exception as exc:                                  # noqa: BLE001
        return row["id"], f"error: {type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8,
                    help="kept well below nproc on purpose; the GPU tenant needs cores too")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = entries()
    if a.limit:
        rows = rows[: a.limit]
    todo = [r for r in rows
            if not (OUT / f"{r['id']}.pose").exists()]
    print(f"twtsl citation clips: {len(rows)}; already done: {len(rows) - len(todo)}; "
          f"to extract: {len(todo)}  (workers={a.workers})", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    os.nice(19)
    t0 = time.time()
    counts: dict[str, int] = {}
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(extract, r): r for r in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            sid, status = fut.result()
            key = status if status in ("ok", "skip", "empty-video") else "error"
            counts[key] = counts.get(key, 0) + 1
            if key == "error" and len(errors) < 10:
                errors.append(f"  id={sid}: {status}")
            if n % 200 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el
                print(f"  {n}/{len(todo)}  {el/60:.1f} min elapsed, "
                      f"~{(len(todo)-n)/rate/60:.1f} min left  {counts}", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min: {counts}")
    if errors:
        print("first errors:")
        print("\n".join(errors))
    # A partial extraction is usable, so this is not fatal -- but say so plainly
    # rather than exiting 0 and letting a short store look complete.
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
