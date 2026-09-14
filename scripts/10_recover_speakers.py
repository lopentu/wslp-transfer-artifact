#!/usr/bin/env python3
"""Recover the speaker side for dialogue sentences the corpus fails to label.

    python3 scripts/10_recover_speakers.py            # measure and write sidecar
    python3 scripts/10_recover_speakers.py --dry-run  # measure only

**The corpus bug.** In 17 of the 3,620 dialogue sentences the `sentences.speaker`
column does not hold 'L' or 'R'. Three are empty; the other fourteen hold a
*transcription of the sentence itself* — seven byte-identical to `sentences.text`
and seven an earlier draft of it ("，" where text reads
"，"). That is a column-shift during annotation, not a judgement
that the speaker was unknown.

Downstream this was silent: `crops_for()` returns only {'L', 'R'} for a dialogue
film, so a span with side None matched no crop and was dropped without a warning.
17 spans, 3 of them diagnostic items, never reached the feature store.

**The recovery.** Which half of the frame is moving? That is the same measurement
`03_extract_features.py --check-sides` uses to verify the L/R convention, and it
is calibrated per film here: every *labelled* span in the same film is scored the
same way, and the script reports how many of them the measure gets right. If the
measure cannot reproduce the annotator on spans where the annotator spoke, it has
no business filling in the ones where they did not.

Output `data/speaker_inferred.json` is a sidecar, not an edit: `tsl.db` merges it
and stamps `speaker_src='motion'` so an inferred side is never mistaken for an
annotated one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.db import FILMS_DIR, SPEAKER_SIDES, Corpus  # noqa: E402
from tsl.features import crops_for, decode  # noqa: E402

PROBE_FPS = 5   # enough for gross motion; a full-rate decode buys nothing here
PROBE_H = 180


def motion(film: Path) -> dict[str, np.ndarray]:
    """Per-frame mean absolute luma difference for each half of the frame."""
    out = {}
    for side, crop in crops_for("2", film).items():
        frames = list(decode(film, crop, fps=PROBE_FPS, out_h=PROBE_H))
        if len(frames) < 3:
            return {}
        g = np.stack([f.mean(-1) for f in frames]).astype(np.float32)
        out[side] = np.abs(np.diff(g, axis=0)).mean((1, 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/speaker_inferred.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    corpus = Corpus()
    dialogue = {u for u, t in corpus.con.execute("SELECT uuid, type FROM paragraphs")
                if t == "2"}
    rows = [dict(r) for r in corpus.con.execute(
        "SELECT uuid, seq, t1, t2, text, speaker FROM sentences")
        if r["uuid"] in dialogue]

    unlabelled = [r for r in rows if r["speaker"] not in SPEAKER_SIDES]
    print(f"dialogue sentences {len(rows)}; without an L/R label {len(unlabelled)} "
          f"in {len({r['uuid'] for r in unlabelled})} films\n")

    hdr = (f"{'key':13s} {'motion L':>9s} {'motion R':>9s} {'ratio':>6s} {'call':>5s} "
           f"{'film calib':>12s} {'weakest':>8s}  flag")
    print(hdr)
    print("-" * len(hdr))

    out: dict[str, dict] = {}
    n_weak = 0
    for uuid in sorted({r["uuid"] for r in unlabelled}):
        film = FILMS_DIR / f"{uuid}.mp4"
        half = motion(film)
        if not half:
            print(f"{uuid}: could not decode, skipped")
            continue
        n = min(len(half["L"]), len(half["R"]))

        def energy(t1: int, t2: int) -> tuple[float, float]:
            lo = int(t1 * PROBE_FPS / 1000)
            i = np.arange(lo, max(int(t2 * PROBE_FPS / 1000), lo + 1))
            i = i[i < n]
            if not len(i):
                return (float("nan"), float("nan"))
            return (float(half["L"][i].mean()), float(half["R"][i].mean()))

        # Calibrate: replay the measure on this film's labelled spans.
        cal = []
        for r in rows:
            if r["uuid"] != uuid or r["speaker"] not in SPEAKER_SIDES:
                continue
            l, rr = energy(r["t1"], r["t2"])
            if np.isnan(l) or min(l, rr) <= 0:
                continue
            cal.append((l / rr) if r["speaker"] == "L" else (rr / l))
        hits = sum(c > 1 for c in cal)
        weakest = min(cal) if cal else float("nan")

        for r in sorted((x for x in unlabelled if x["uuid"] == uuid),
                        key=lambda x: x["seq"]):
            key = f"{uuid}:{r['seq']:03d}"
            l, rr = energy(r["t1"], r["t2"])
            if np.isnan(l) or min(l, rr) <= 0:
                print(f"{key:13s} no usable frames")
                continue
            call = "L" if l > rr else "R"
            ratio = max(l, rr) / min(l, rr)
            # "Confident" means the margin is at least as clear as the weakest
            # span in this film that the measure got right. A film-relative bar,
            # because signing intensity varies by signer and by camera framing.
            conf = bool(cal) and hits == len(cal) and ratio >= weakest
            n_weak += not conf
            out[key] = {
                "speaker": call,
                "ratio": round(ratio, 3),
                "motion_L": round(l, 4), "motion_R": round(rr, 4),
                "calib_n": len(cal), "calib_correct": hits,
                "calib_weakest_ratio": round(weakest, 3) if cal else None,
                "confident": conf,
                "db_speaker_value": r["speaker"],
                "text": r["text"],
            }
            print(f"{key:13s} {l:9.3f} {rr:9.3f} {ratio:6.2f} {call:>5s} "
                  f"{hits:5d}/{len(cal):<6d} {weakest:8.2f}  "
                  f"{'' if conf else 'LOW MARGIN'}")

    tot_cal = sum(v["calib_n"] for v in out.values() if v["calib_n"])
    # Each film's calibration is counted once, not once per recovered span.
    per_film = {}
    for v in out.values():
        per_film.setdefault((v["calib_n"], v["calib_correct"]), 0)
    cal_n = sum(n for n, _ in per_film)
    cal_ok = sum(c for _, c in per_film)
    print(f"\nrecovered {len(out)} spans; calibration {cal_ok}/{cal_n} labelled "
          f"spans in these films classified correctly by the same measure")
    if n_weak:
        print(f"{n_weak} below the film's weakest correct margin — flagged "
              f"confident=false, check before relying on them")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nwrote {a.out}")
    print("next: rerun 01_build_dataset.py, delete the affected .npz, rerun 03")


if __name__ == "__main__":
    main()
