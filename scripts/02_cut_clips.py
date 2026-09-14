#!/usr/bin/env python3
"""Cut review clips for the diagnostic items, for human validation.

The paper cannot call this a gold TSL discourse benchmark unless a fluent signer
confirms, per item, that (a) the pointing sign really does refer to what the
Chinese translation says it does, and (b) the target utterance really is
under-determined without the preceding context (plan §4.4). That check needs
video, not a TSV.

For each item this writes ONE mp4 containing the context utterances followed by
the target, cropped to the active signer, with the target segment marked. A
validation sheet lists the questions to answer per clip.

    python3 scripts/02_cut_clips.py --types anaphoric --out data/review
    python3 scripts/02_cut_clips.py --types anaphoric,deictic --sample 120
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.db import FILMS_DIR  # noqa: E402
from tsl.features import crops_for  # noqa: E402

PAD_MS = 120  # a little air so the first sign is not clipped


def build(job: tuple[dict, Path]) -> str | None:
    r, outdir = job
    film = FILMS_DIR / f"{r['uuid']}.mp4"
    if not film.exists():
        return None
    crop = crops_for(r["para_type"], film)[
        r["speaker"] if r["para_type"] == "2" else None]

    spans = [(c["t1"], c["t2"], "ctx") for c in r["ctx"]]
    spans.append((r["t1"], r["t2"], "target"))
    t0 = max(0, spans[0][0] - PAD_MS)
    t1 = spans[-1][1] + PAD_MS
    # Where the target starts, relative to the clip — drives the on-screen marker.
    mark = (r["t1"] - t0) / 1000.0

    out = outdir / f"{r['key'].replace(':', '_')}.mp4"
    vf = (
        f"{crop.ffmpeg},scale=-2:480,"
        f"drawbox=x=0:y=0:w=iw:h=6:color=red@0.9:t=fill:enable='gte(t,{mark:.3f})',"
        f"drawtext=text='TARGET':x=10:y=14:fontcolor=red:fontsize=22"
        f":enable='gte(t,{mark:.3f})'"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{t0/1000:.3f}", "-to", f"{t1/1000:.3f}", "-i", str(film),
        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "26", "-preset", "veryfast",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return f"FAIL {r['key']}: {e.stderr.decode()[:160]}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("data/records.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/review"))
    ap.add_argument("--types", default="anaphoric")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    want = set(a.types.split(","))
    recs = [json.loads(l) for l in a.records.open()]
    items = [r for r in recs if r["item_type"] in want and r["candidates"] and r["ctx"]]
    if a.sample and a.sample < len(items):
        random.Random(a.seed).shuffle(items)
        items = items[: a.sample]
    items.sort(key=lambda r: r["key"])
    print(f"{len(items)} items -> {a.out}")

    with (a.out / "validation_sheet.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "clip", "key", "type", "gloss", "context_zh", "gold_zh",
            "Q1_referent_correct(y/n/unsure)",
            "Q2_needs_context(y/n)",           # is the target ambiguous alone?
            "Q3_distractor_plausible(y/n)",    # would the distractor be a real reading?
            "notes",
        ])
        for r in items:
            w.writerow([
                f"{r['key'].replace(':', '_')}.mp4", r["key"], r["item_type"],
                r["pron_gloss"], " ⏎ ".join(c["text"] for c in r["ctx"]),
                r["candidates"][0], "", "", "", "",
            ])

    fails = []
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(build, [(r, a.out) for r in items], chunksize=4):
            if msg:
                fails.append(msg)
    for m in fails:
        print(m)
    print(f"done: {len(items) - len(fails)} clips, {len(fails)} failures")
    print(f"validation sheet: {a.out}/validation_sheet.tsv")


if __name__ == "__main__":
    main()
