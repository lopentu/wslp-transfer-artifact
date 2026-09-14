#!/usr/bin/env python3
"""SignCLIP embeddings for the twtsl citation clips, and for language-tagged text.

Stage 2 of the JSL probe (`experiment-suggestion-first.md` §3). Stage 1
(`13_signclip_poses.py`) produced `data/signclip/pose/*.pose`; this turns those
into vectors, and produces the text side they are compared against.

What the text side is for
-------------------------
SignCLIP was trained on SpreadTheSign with the prompt
``f"<{language}> <{videoLanguage}> {text}"`` (their
`dsprocessor_sign.py:432`), i.e. ``<en> <jsl> house`` is the text that was
contrastively paired with *Japanese* videos of "house". So the embedding of
``<en> <jsl> house`` is a description of **how JSL signs that concept**, and
its similarity to a TSL clip of the same concept is a measure of TSL--JSL form
similarity that needs no JSL video and no Chinese text at all.

Their training language inventory is `data_stat_sp_lan_dis.csv`, and it decides
whether the probe is clean:

    jsl  9,238   csl 13,406   ase 12,490   gsg 15,657   ...   tss  absent

Japanese, Chinese, American and German sign languages are all in; **Taiwanese
(`tss`) is not** — `tsm` in that file is Turkish. TSL is therefore genuinely
unseen data for this model, which is what makes the comparison a measurement
rather than a lookup. Note also that JSL has the *fewest* training clips of the
four, so a JSL advantage cannot be an artefact of training volume (a
disadvantage could be, which `15_signclip_similarity.py` controls for).

Two traps this script exists to avoid
-------------------------------------
1. **`demo_sign.embed_pose` mis-masks batches.** `MMPTModel.forward` builds
   `vmasks` from `video_frames.size(1)` — the *batch-padded* length, shared by
   the whole batch. Batch a 40-frame clip with a 200-frame one and the short
   one is pooled as if 160 frames of zero-padding were signal. We call
   `model.forward_video` with per-clip masks instead, which is the same code
   path the model uses internally and is correct for any batch. `--check`
   verifies the result against the unbatched public API.

2. **The zh/en join is on `clip`, not on `id`.** `entries` is keyed
   `(lang, id, row_id)`, so `id` is scoped *per language*: joining zh to en on
   `id` pairs 蕃茄醬 (ketchup) with "LARGE AMOUNT" and 洋蔥 (onion) with
   "PARIS". Both rows of a real pair point at the same `clip`, which is the
   actual key. 3,436 of 3,519 clips have both.

CPU only, in `.venv_signclip` (see `scripts/README_signclip.md`):

    .venv_signclip/bin/python scripts/14_signclip_embed.py --check
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MMPT = ROOT / "third_party" / "signclip_fairseq" / "examples" / "MMPT"
TWTSL = Path("/mnt/md0/corpus/sign/twtsl")
POSE = ROOT / "data" / "signclip" / "pose"
OUT = ROOT / "data" / "signclip"
MAX_VIDEO_LEN = 256          # dataset.max_video_len in baseline_temporal.yaml

# The four languages the probe is about, then a wider panel. The panel is what
# turns "TSL is close to JSL" into "TSL is closer to JSL than to 37 other sign
# languages", and it is what makes the training-volume control possible: with
# only four points, similarity and corpus size cannot be told apart.
FOCUS = ["jsl", "csl", "ase", "gsg"]


def clean_gloss(gloss: str) -> str:
    """twtsl's English headword -> a SpreadTheSign-style concept string.

    twtsl marks lexical variants with `_A`/`_B`/`_S` suffixes and carries
    disambiguating parentheticals; SpreadTheSign's `text` field is a bare
    lowercase phrase. Normalising is what lets `--vocab` check a gloss against
    the concepts the model was actually trained on.
    """
    g = re.sub(r"\([^)]*\)", " ", gloss)              # "(place name)"
    g = re.sub(r"_[A-Z]{1,2}(?![A-Za-z])", " ", g)    # "_A", "_S"
    g = re.sub(r"\s+", " ", g).strip()
    g = re.sub(r"[\s_;,.]+$", "", g)
    return g.lower().strip()


def concepts() -> list[dict]:
    """One row per citation clip that has both a Chinese and an English headword."""
    con = sqlite3.connect(TWTSL / "twtsl.db")
    rows = sorted(con.execute(
        "select lang, id, row_id, name, clip from entries "
        "where clip is not null and clip != ''"))
    con.close()
    byclip: dict[str, dict] = collections.defaultdict(dict)
    for lang, i, _row, name, clip in rows:
        byclip[clip].setdefault(lang, (i, name))
    out = []
    for clip, d in sorted(byclip.items()):
        if "zh" not in d or "en" not in d:
            continue
        zid, zname = d["zh"]
        out.append({"id": zid, "clip": clip, "zh": zname,
                    "en_raw": d["en"][1], "gloss": clean_gloss(d["en"][1])})
    return [r for r in out if r["gloss"]]


def sp_vocabulary() -> set[str]:
    """The concepts SignCLIP actually saw, from their own dataset statistics."""
    p = MMPT / "data_stat_sp_concept_dis.csv"
    with open(p) as fh:
        return {r["text"].strip().lower() for r in csv.DictReader(fh)}


def sp_languages() -> dict[str, int]:
    """Sign language code -> number of training clips."""
    p = MMPT / "data_stat_sp_lan_dis.csv"
    with open(p) as fh:
        return {r["videoLanguage"]: int(r["count"]) for r in csv.DictReader(fh)}


def load_model(threads: int):
    """The pretrained SignCLIP, on CPU, with its own directory as cwd.

    `MMPTModel.from_pretrained` resolves the checkpoint against
    `config.eval.save_path`, which is relative — so the working directory has to
    be theirs, and every path of ours has to be absolute.
    """
    import torch

    torch.set_num_threads(threads)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path.insert(0, str(MMPT))
    os.chdir(MMPT)
    import demo_sign

    info = demo_sign.get_model("default")
    assert not next(info["model"].parameters()).is_cuda, \
        "SignCLIP reached the GPU; this probe must stay on CPU (README_signclip.md)"
    return demo_sign, info


def pose_frames(demo_sign, path: Path):
    """A `.pose` file -> (T, 609) float32, truncated to the model's window."""
    from pose_format import Pose

    with open(path, "rb") as fh:
        pose = Pose.read(fh.read())
    feat = demo_sign.preprocess_pose(pose)          # (1, T, 609)
    feat = feat[0]
    if feat.shape[0] > MAX_VIDEO_LEN:
        feat = feat[:MAX_VIDEO_LEN]
    return feat


def embed_videos(demo_sign, info, rows: list[dict], batch: int) -> np.ndarray:
    """Pooled video embeddings, batched with per-clip masks.

    Deliberately not `demo_sign.embed_pose`: see trap 1 in the module docstring.
    """
    import torch

    model = info["model"]
    caps, cmasks = demo_sign.preprocess_text("", "default")
    out = np.zeros((len(rows), 768), np.float32)
    t0 = time.time()
    done = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        feats = [pose_frames(demo_sign, POSE / f"{r['id']}.pose") for r in chunk]
        b = len(feats)
        vfeats = torch.zeros(b, MAX_VIDEO_LEN, 609)
        vmasks = torch.zeros(b, MAX_VIDEO_LEN, dtype=torch.bool)
        for i, f in enumerate(feats):
            t = f.shape[0]
            vfeats[i, :t] = f
            vmasks[i, :t] = True
        with torch.inference_mode():
            pooled = model.model.forward_video(
                vfeats, vmasks, caps.repeat(b, 1), cmasks.repeat(b, 1))
        out[start:start + b] = pooled.cpu().numpy()
        done += b
        if start // batch % 10 == 0 or done == len(rows):
            el = time.time() - t0
            print(f"  video {done}/{len(rows)}  {el/60:.1f} min elapsed, "
                  f"~{(len(rows)-done)*el/max(done,1)/60:.1f} min left", flush=True)
    return out


def embed_texts(demo_sign, info, prompts: list[str], batch: int) -> np.ndarray:
    """Pooled text embeddings.

    Calls `forward_text` directly. `demo_sign.embed_text` feeds a `torch.randn`
    dummy video through the *video* tower on every call, which costs the whole
    video encoder per batch and makes the call non-deterministic-looking;
    `MMFusionShare.forward` shows `forward_text` never sees `vfeats`.

    The sequences are also trimmed to the longest real prompt in the batch. A
    concept prompt is ~8 tokens and the aligner pads to 128, so BERT is
    otherwise run over 94% padding. This is exact, not an approximation:
    `forward_text` hands `cmasks` to the encoder as the attention mask and pools
    under the same mask, so positions after the last unmasked one cannot affect
    any output — and BERT's position embeddings are absolute, so dropping
    *trailing* columns does not move the real tokens.
    """
    import torch

    model = info["model"]
    out = np.zeros((len(prompts), 768), np.float32)
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        caps, cmasks = zip(*(demo_sign.preprocess_text(p, "default") for p in chunk))
        caps, cmasks = torch.cat(caps), torch.cat(cmasks)
        keep = int(cmasks.any(0).nonzero().max()) + 1
        with torch.inference_mode():
            pooled = model.model.forward_text(caps[:, :keep], cmasks[:, :keep])
        out[start:start + len(chunk)] = pooled.cpu().numpy()
    return out


def check_batching(demo_sign, info, rows: list[dict], emb: np.ndarray) -> None:
    """Our masked batch path against their unbatched public API.

    The point of the check is that it must hold for clips of *different*
    lengths in one batch, which is exactly what the public batch helper gets
    wrong.
    """
    import torch

    model = info["model"]
    caps, cmasks = demo_sign.preprocess_text("", "default")
    idx = [i for i in range(min(4, len(rows)))]   # all in the first batch together
    worst = 0.0
    for i in idx:
        f = pose_frames(demo_sign, POSE / f"{rows[i]['id']}.pose")
        with torch.inference_mode():
            single = model(f[None], caps, cmasks, return_score=False)
        v = single["pooled_video"].cpu().numpy()[0]
        a, b = emb[i], v
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        worst = max(worst, abs(1 - cos))
        print(f"  check id={rows[i]['id']} frames={f.shape[0]} cos(batched, single)={cos:.6f}")
    if worst > 1e-4:
        raise SystemExit(f"batched embeddings disagree with the single-clip path "
                         f"(worst 1-cos = {worst:.2e}) — masking is wrong")
    print(f"  batching check passed (worst 1-cos = {worst:.2e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--text-batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6,
                    help="kept well below nproc: the card's tenant needs cores too")
    ap.add_argument("--langs", default="panel",
                    help="'focus' (jsl,csl,ase,gsg), 'panel' (every language "
                         "SignCLIP was trained on), or a comma-separated list")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="verify batched video embeddings against the single-clip path")
    a = ap.parse_args()

    rows = concepts()
    have = {int(p.stem) for p in POSE.glob("*.pose")}
    rows = [r for r in rows if r["id"] in have]
    if a.limit:
        rows = rows[: a.limit]

    vocab = sp_vocabulary()
    langs_all = sp_languages()
    for r in rows:
        r["in_sp_vocab"] = r["gloss"] in vocab
    print(f"{len(rows)} clips with a pose file and an English headword; "
          f"{sum(r['in_sp_vocab'] for r in rows)} of their glosses are concepts "
          f"SignCLIP was trained on")

    if a.langs == "focus":
        langs = list(FOCUS)
    elif a.langs == "panel":
        langs = FOCUS + [l for l in langs_all if l not in FOCUS]
    else:
        langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    unknown = [l for l in langs if l not in langs_all]
    if unknown:
        # A code the model never saw yields an embedding of the tag's subwords,
        # which would look like a language rather than announcing itself as
        # noise. Refuse rather than silently report it as a row.
        raise SystemExit(f"not in SignCLIP's training inventory: {unknown}")
    print(f"{len(langs)} languages: {' '.join(langs)}")

    demo_sign, info = load_model(a.threads)

    print("embedding video...", flush=True)
    vid = embed_videos(demo_sign, info, rows, a.batch)
    if a.check:
        check_batching(demo_sign, info, rows, vid)

    # Text is embedded per *distinct* gloss, not per clip. twtsl lists lexical
    # variants of one concept as separate clips (`_A`, `_B`), which collapse to
    # the same cleaned gloss — and retrieval has to rank over concepts anyway,
    # or a variant pair would put two equally correct answers in the candidate
    # set and score as an error whichever one came second.
    glosses = [r["gloss"] for r in rows]
    uniq = sorted(set(glosses))
    at = {g: i for i, g in enumerate(uniq)}
    gloss_index = np.array([at[g] for g in glosses])
    print(f"embedding text: {len(uniq)} distinct glosses x {len(langs)} languages "
          f"({len(rows)} clips)", flush=True)

    txt = np.zeros((len(langs), len(uniq), 768), np.float32)
    t0 = time.time()
    for i, lang in enumerate(langs):
        txt[i] = embed_texts(demo_sign, info,
                             [f"<en> <{lang}> {g}" for g in uniq], a.text_batch)
        el = time.time() - t0
        print(f"  text {i+1}/{len(langs)} [{lang}]  {el/60:.1f} min elapsed, "
              f"~{(len(langs)-i-1)*el/(i+1)/60:.1f} min left", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "embeddings.npz",
        video=vid, text=txt,
        langs=np.array(langs),
        lang_train_clips=np.array([langs_all[l] for l in langs]),
        ids=np.array([r["id"] for r in rows]),
        gloss=np.array(glosses),
        uniq_gloss=np.array(uniq),
        gloss_index=gloss_index,
        zh=np.array([r["zh"] for r in rows]),
        in_sp_vocab=np.array([r["in_sp_vocab"] for r in rows]),
    )
    print(f"-> {OUT / 'embeddings.npz'}  video {vid.shape}, text {txt.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
