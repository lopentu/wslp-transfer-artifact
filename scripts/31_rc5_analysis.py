#!/usr/bin/env python3
"""Every contrast the 8/22 reviews ask for, in one file.

Reads whatever has landed under `runs/` and writes `data/rc5_analysis.json`. Safe
to run repeatedly while the queues are still going: a contrast whose cells are not
on disk yet is recorded as null rather than skipped silently, so the report can
say what is still missing instead of quietly omitting it.

Statistics are 19_bootstrap.py's, imported rather than reimplemented -- the
paragraph-clustered bootstrap and the sign-flip permutation test are the ones the
paper's intervals already use, and a second implementation of them would be a
second thing to keep correct.

Sections, keyed to the reviews:

  cells        every cell's accuracy with a paragraph-clustered CI
  seeds        O-P1: mean +/- SD across training seeds for the head-reset cells,
               which is the interval the reviews actually want and the paper's
               item-resampling CI is not
  posthoc      A-A2: the post-hoc graft at three seeds, both directions
  synthead     A-A3/W2: the rescale ladder and the post-hoc mT5-base head
  headgrid     the (mT5 body x lm_head) grid, with the CI for ReviewerA's Q2
  video        O-M5: the local-minus-wrong-clip margin as a grounding metric
  lastepoch    A-B6: best-dev-CE against last-epoch selection
  lr           A-B1: the learning-rate sweep
  gen          O-P0/A5: BLEU/chrF/distinct/modal for the rescued cells
  params       A-Q1: what the 966.6M actually counts
  gloss        A-B2/B4: the gloss->Chinese instrument and the working regime

    ../.venv/bin/python scripts/31_rc5_analysis.py
    ../.venv/bin/python scripts/31_rc5_analysis.py --boot 2000   # quick pass
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_bootstrap():
    """Import scripts/19_bootstrap.py, whose name is not a legal module name."""
    spec = importlib.util.spec_from_file_location(
        "bootstrap19", ROOT / "scripts" / "19_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B19 = _load_bootstrap()
RUN_LABEL = re.compile(r"^f\d+_local_(?P<label>.+)_full_k\d+_s(?P<seed>\d+)$")

# The two cells Kevin flagged and all three reviews ask about, plus their
# controls. Written out rather than globbed so the report's tables have a fixed
# row order and a missing cell is visible as a gap.
HEAD_CELLS = [
    ("how2sign", "intact ASL checkpoint"),
    ("how2sign+mt5_base-lm_head", "+ unadapted mT5-base projection"),
    ("how2sign+csl_daily-lm_head", "+ CSL-Daily projection"),
    ("how2sign+rand_head_nm-lm_head", "+ fresh random projection, norm-matched"),
    ("how2sign+rand_head-lm_head", "+ fresh random projection, library init"),
    ("how2sign+how2sign_cjk-lm_head", "+ own projection, CJK rows rescaled"),
    ("how2sign+how2sign_gl-lm_head", "+ own projection, global radius matched"),
    ("how2sign+how2sign_cjkgl-lm_head", "+ own projection, both rescalings"),
    ("how2sign+gloss_lm-lm_head", "+ gloss->Chinese projection (no video)"),
    ("how2sign+tsl_text-lm_head", "+ span-denoised Chinese projection"),
    ("how2sign+tsl_text_strong-lm_head",
     "+ the same, drift-matched to a released fine-tune"),
    ("how2sign+csl_stage1-lm_head", "+ CSL-News projection"),
    ("how2sign+csl_daily-mt5_nohead", "+ CSL-Daily mT5 body, own projection"),
    ("csl_daily", "intact CSL checkpoint"),
    ("csl_daily+mt5_base-lm_head", "+ unadapted mT5-base projection"),
    ("csl_daily+how2sign-lm_head", "+ How2Sign projection"),
    ("csl_daily+rand_head_nm-lm_head", "+ fresh random projection, norm-matched"),
    ("openasl", "intact OpenASL checkpoint"),
    ("openasl+mt5_base-lm_head", "+ unadapted mT5-base projection"),
    ("openasl+csl_daily-lm_head", "+ CSL-Daily projection"),
    ("wlasl", "intact WLASL checkpoint"),
    ("wlasl+mt5_base-lm_head", "+ unadapted mT5-base projection"),
    ("wlasl+csl_daily-lm_head", "+ CSL-Daily projection"),
    ("random", "RAND-VIS (untrained pose branch)"),
    # The mT5-base arm. Appendix H's 16.6-point seed spread is on the POINTING
    # diagnostic in these cells; ReviewerA's Q5 asks what the content-word spread
    # is, and the answer needs these rows in the seeds table.
    ("how2sign-pose", "How2Sign visual half on mT5-base"),
    ("wlasl-pose", "WLASL visual half on mT5-base"),
    ("openasl-pose", "OpenASL visual half on mT5-base"),
    ("csl_stage1-pose", "CSL-News visual half on mT5-base"),
    ("how2sign-pose+csl_daily-mt5", "How2Sign visual half + whole CSL mT5 half"),
    ("csl_daily-pose", "CSL-Daily visual half on mT5-base"),
    ("csl_daily-pose+csl_daily-lm_head", "... + its own projection back"),
    ("csl_daily-pose+gloss_lm-lm_head", "... + gloss->Chinese projection"),
]

# Contrasts the reviews name explicitly. (a, b, label, why)
CONTRASTS = [
    ("how2sign+mt5_base-lm_head", "how2sign",
     "neutral head rescue (ASL)", "the headline, O-P1/A-A1"),
    ("how2sign+csl_daily-lm_head", "how2sign",
     "CSL head rescue (ASL)", "the paper's original rescue"),
    ("how2sign+mt5_base-lm_head", "how2sign+csl_daily-lm_head",
     "neutral vs Chinese head", "G's question: why are these the same?"),
    ("how2sign+rand_head_nm-lm_head", "how2sign",
     "random head rescue (ASL)", "O-P3: is it a reset?"),
    ("how2sign+rand_head_nm-lm_head", "how2sign+mt5_base-lm_head",
     "random vs neutral head", "O-P3: is mT5-base's geometry special?"),
    ("how2sign+how2sign_cjkgl-lm_head", "how2sign",
     "own head, rescaled both ways", "A-W2: is it only logit scale?"),
    ("how2sign+how2sign_cjk-lm_head", "how2sign",
     "own head, CJK rows rescaled", "A-W2 / A-B3"),
    ("how2sign+gloss_lm-lm_head", "how2sign",
     "gloss->Chinese head rescue", "A-B2: the strong instrument"),
    ("how2sign+gloss_lm-lm_head", "how2sign+mt5_base-lm_head",
     "gloss->Chinese vs neutral", "A-B2: does Chinese-ness add anything?"),
    ("how2sign+tsl_text_strong-lm_head", "how2sign",
     "drift-matched Chinese head rescue", "G: the head effect at matched drift"),
    ("how2sign+tsl_text_strong-lm_head", "how2sign+mt5_base-lm_head",
     "drift-matched vs neutral",
     "G: does matching the displacement buy anything over mT5-base?"),
    ("how2sign+tsl_text_strong-lm_head", "how2sign+tsl_text-lm_head",
     "drift-matched vs undertrained donor",
     "G: was the tsl_text null an artefact of a donor that barely moved?"),
    ("how2sign+csl_daily-lm_head", "how2sign+tsl_text_strong-lm_head",
     "real sign donor vs drift-matched text donor",
     "G: what is left once displacement is held constant?"),
    ("csl_daily+mt5_base-lm_head", "csl_daily",
     "neutral head on a matched checkpoint", "the paper's strongest applied claim"),
    ("csl_daily+rand_head_nm-lm_head", "csl_daily+mt5_base-lm_head",
     "random vs neutral head (CSL)", "O-P3 on the CSL side"),
    ("csl_daily+how2sign-lm_head", "csl_daily",
     "ASL head into the working model",
     "the abolition cell: does a collapsed head destroy grounding?"),
    ("openasl+mt5_base-lm_head", "openasl",
     "neutral head rescue (OpenASL)", "O-P2/M3: does it replicate?"),
    ("wlasl+mt5_base-lm_head", "wlasl",
     "neutral head rescue (WLASL)", "O-P2/M3: does it replicate?"),
    ("how2sign+csl_daily-lm_head", "how2sign+csl_daily-mt5_nohead",
     "CSL head only vs CSL body only", "which half of the mT5 half matters"),
    ("how2sign+csl_daily-lm_head", "how2sign-pose+csl_daily-mt5",
     "lm_head-only swap vs whole-mT5-half swap",
     "A-Q2: 48.5 > 43.8, real reversal or noise?"),
    ("how2sign", "random",
     "intact ASL vs no sign-language pretraining",
     "S3.1's central claim, and A-Q4 asks how to read it"),
    ("how2sign+mt5_base-lm_head", "random",
     "rescued ASL vs RAND-VIS", "does the rescue clear the no-pretraining floor?"),
    ("csl_daily+mt5_base-lm_head", "random",
     "best cell vs RAND-VIS", "the same question for the best configuration"),
    ("csl_daily-pose+csl_daily-lm_head", "csl_daily-pose",
     "is the projection sufficient?", "the mirror of the necessity result"),
    ("csl_daily-pose+gloss_lm-lm_head", "csl_daily-pose",
     "sufficiency with the strong instrument", "A-B2"),
]


def norm(x):
    """JSON-safe: numpy scalars and NaN both break json.dump downstream."""
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return norm(o)


# --------------------------------------------------------------------- loaders


def load_synthead(pattern: str = "eval_synthead_*.json") -> dict:
    """The synthetic-donor grafts, labelled by FILENAME.

    19_bootstrap.py labels a graft by its donor checkpoint's parent directory,
    which is `head_donors` for every one of these -- twelve different tensors
    would collapse into one row. The filename is the only thing that
    distinguishes them, so it is what gets used.
    """
    out: dict = {}
    for p in sorted(glob.glob(str(ROOT / "runs" / "f[012]_local_*" / pattern))):
        path = Path(p)
        run = path.parent.name
        m = RUN_LABEL.match(run)
        if not m:
            continue
        d = json.loads(path.read_text())
        tag = path.stem.replace("eval_synthead_", "")
        key = f"{m.group('label')}@synth-{tag}"
        seed = int(m.group("seed"))
        rows = out.setdefault(key, {}).setdefault(seed, {})
        for r in d["contrastive"]:
            rows.setdefault(r["condition"], {})[r["key"]] = r
        out[key].setdefault("meta", {})[seed] = {
            "run": run, "donor": (d.get("posthoc_head") or {}).get("donor"),
            "cos_recipient_donor": (d.get("posthoc_head") or {}).get(
                "cos_recipient_donor")}
    return out


def cells_for_seed(seed: int, fold: int | None, fname: str, boot: int):
    return B19.load(fname, seed=seed, fold=fold)


def acc_of(cells, init, cond="local", field="hit"):
    rows = cells.get(init, {}).get(cond)
    if not rows:
        return None
    return 100.0 * float(np.mean([float(r[field]) for r in rows.values()]))


# ------------------------------------------------------------------- sections


def sec_cells(boot: int) -> dict:
    """Accuracy + CI for every cell, pooled over folds at seed 0."""
    cells = B19.load("eval_lex.json", seed=0)
    out = {}
    for label, why in HEAD_CELLS:
        ci = B19.accuracy_ci(cells, label, B=boot)
        n_folds = len({k.split(":")[0][:4] for k in
                       cells.get(label, {}).get("local", {})}) or None
        out[label] = {"note": why, "acc": None if ci is None else ci["acc"],
                      "lo": None if ci is None else ci["lo"],
                      "hi": None if ci is None else ci["hi"],
                      "n": None if ci is None else ci["n"],
                      "paragraphs": None if ci is None else ci["clusters"],
                      "folds_seen": n_folds}
    return out


def sec_contrasts(boot: int) -> dict:
    cells = B19.load("eval_lex.json", seed=0)
    out = {}
    for a, b, label, why in CONTRASTS:
        r = B19.cell_contrast(cells, a, b, B=boot)
        m = B19.cell_contrast(cells, a, b, field="margin", scale=1.0, B=boot)
        out[label] = {"a": a, "b": b, "why": why, "accuracy": r, "margin": m}
    return out


def sec_seeds(boot: int) -> dict:
    """O-P1. Fold 0 only: seeds 1 and 2 exist there and nowhere else, so pooling
    folds would compare a three-fold seed-0 number against a one-fold seed-1 one
    and call the difference a seed effect."""
    per_seed = {s: B19.load("eval_lex.json", seed=s, fold=0) for s in (0, 1, 2)}
    labels = [c for c, _ in HEAD_CELLS]
    out = {}
    for label in labels:
        vals = {s: acc_of(per_seed[s], label) for s in (0, 1, 2)}
        got = [v for v in vals.values() if v is not None]
        out[label] = {
            "per_seed": vals, "n_seeds": len(got),
            "mean": statistics.mean(got) if got else None,
            "sd": statistics.stdev(got) if len(got) > 1 else None,
            "range": (max(got) - min(got)) if len(got) > 1 else None}
    # The contrasts that matter, computed inside each seed and then summarised
    # across them -- which is the quantity "retrain and is the effect there?"
    pairs = [("how2sign+mt5_base-lm_head", "how2sign"),
             ("how2sign+csl_daily-lm_head", "how2sign"),
             ("csl_daily+mt5_base-lm_head", "csl_daily"),
             ("csl_daily+how2sign-lm_head", "csl_daily"),
             ("how2sign+mt5_base-lm_head", "how2sign+csl_daily-lm_head")]
    deltas = {}
    for a, b in pairs:
        per = {}
        for s in (0, 1, 2):
            r = B19.cell_contrast(per_seed[s], a, b, B=boot)
            per[s] = None if r is None else r["delta"]
        got = [v for v in per.values() if v is not None]
        deltas[f"{a} - {b}"] = {
            "per_seed": per, "n_seeds": len(got),
            "mean": statistics.mean(got) if got else None,
            "sd": statistics.stdev(got) if len(got) > 1 else None,
            "min": min(got) if got else None, "max": max(got) if got else None}
    return {"accuracy": out, "deltas": deltas}


def sec_posthoc(boot: int) -> dict:
    """A-A2. The post-hoc graft rows, per seed, fold 0."""
    out = {}
    for s in (0, 1, 2):
        cells = B19.load("eval_lex.json", seed=s, fold=0)
        graft = B19.load("eval_posthoc_csldaily.json", seed=s, fold=0)
        graft.update(B19.load("eval_posthoc_how2sign.json", seed=s, fold=0))
        graft.update(B19.load("eval_posthoc_rescued.json", seed=s, fold=0))
        graft.update(B19.load("eval_posthoc_selfseed.json", seed=s, fold=0))
        merged = {**cells, **graft}
        row = {}
        for name in sorted(graft):
            base = name.split("@")[0]
            r = B19.cell_contrast(merged, name, base, B=boot)
            m = B19.cell_contrast(merged, name, base, field="margin",
                                  scale=1.0, B=boot)
            row[name] = {"acc": acc_of(merged, name),
                         "acc_base": acc_of(merged, base),
                         "accuracy_delta": r, "margin_delta": m}
        out[s] = row
    # Across-seed summary of the one contrast the abstract leans on.
    key = "how2sign@post-csl_daily"
    per = {s: (out[s].get(key, {}).get("accuracy_delta") or {}).get("delta")
           for s in (0, 1, 2)}
    got = [v for v in per.values() if v is not None]
    return {"per_seed": out,
            "headline": {"cell": key, "per_seed": per, "n_seeds": len(got),
                         "mean": statistics.mean(got) if got else None,
                         "sd": statistics.stdev(got) if len(got) > 1 else None,
                         "min": min(got) if got else None,
                         "max": max(got) if got else None}}


def sec_synthead(boot: int) -> dict:
    """A-A3/W2, plus the post-hoc mT5-base head the paper is missing."""
    synth = load_synthead()
    donors = {}
    hd = ROOT / "data" / "head_donors.json"
    if hd.exists():
        donors = json.loads(hd.read_text()).get("donors", {})
    out = {}
    for key, per_seed in sorted(synth.items()):
        base = key.split("@")[0]
        for seed, conds in per_seed.items():
            if seed == "meta":
                continue
            cells = B19.load("eval_lex.json", seed=seed, fold=0)
            merged = {**cells, key: conds}
            r = B19.cell_contrast(merged, key, base, B=boot)
            m = B19.cell_contrast(merged, key, base, field="margin",
                                  scale=1.0, B=boot)
            meta = per_seed.get("meta", {}).get(seed, {})
            tag = Path(meta.get("donor") or "").stem.replace(".run", "")
            # Whether the graft's accuracy gain READS THE CLIP. This is the
            # decisive column for the rescale ladder: a per-script rescaling
            # inflates the whole Chinese logit block, so it can raise accuracy on
            # a Chinese contrastive set without the model having become any more
            # grounded. `local` minus the wrong clip cannot move for that reason,
            # because no text prior changes when only the video is replaced.
            vid = None
            loc, sw = conds.get("local"), None
            draws = [c for c in conds if c.startswith("swap_plain@")]
            if loc and draws:
                sw = {k: float(np.mean([conds[d][k]["hit"] for d in draws]))
                      for k in set.intersection(*[set(conds[d]) for d in draws])}
                keys = sorted(set(loc) & set(sw))
                d = np.array([float(loc[k]["hit"]) - sw[k] for k in keys])
                vid = {"local": 100.0 * float(np.mean(
                           [float(loc[k]["hit"]) for k in keys])),
                       "swap": 100.0 * float(np.mean([sw[k] for k in keys])),
                       **B19.cluster_stats(keys, d, B=boot, scale=100.0)}
            out[f"{key}#s{seed}"] = {
                "recipient": base, "seed": seed,
                "acc": acc_of(merged, key, cond="local"),
                "acc_base": acc_of(merged, base, cond="local"),
                "accuracy_delta": r, "margin_delta": m, "video": vid,
                "cos_recipient_donor": meta.get("cos_recipient_donor"),
                "donor_stats": donors.get(tag)}
    return out


def sec_video(boot: int) -> dict:
    """O-M5. The wrong-clip drop as the grounding metric, per cell.

    ReviewerO's point is that the diagnostic's ABSOLUTE level is contaminated by
    text structure (unigram prior 38.3%, contextual text-only 66.2%) while the
    local-minus-wrong-clip difference is not: no text prior changes when only the
    video is replaced. So this is the number that should carry the validity
    argument, and it needs to exist per cell rather than for two cells.
    """
    cells = B19.load("eval_clip.json", seed=0)
    out = {}
    for label, _ in HEAD_CELLS:
        if label not in cells:
            out[label] = None
            continue
        row: dict = {}
        for field, scale in (("hit", 100.0), ("margin", 1.0)):
            loc = cells[label].get("local")
            sw = B19.swap_mean(cells, label, field=field)
            if not loc or not sw:
                row[field] = None
                continue
            # local minus the wrong clip, averaged over the three draws inside
            # each item first so the contrast stays paired. There is no
            # single-cell version of this in 19_bootstrap.py -- swap_delta_contrast
            # is a difference of two cells' deltas -- so it is assembled here from
            # the same cluster_stats the paper's other intervals use.
            keys = sorted(set(loc) & set(sw))
            d = np.array([float(loc[k][field]) - sw[k][field] for k in keys])
            row[field] = {
                "local": scale * float(np.mean([float(loc[k][field])
                                                for k in keys])),
                "swap": scale * float(np.mean([sw[k][field] for k in keys])),
                **B19.cluster_stats(keys, d, B=boot, scale=scale)}
            for other in ("blank_plain", "shuffle_frames"):
                row[f"{field}:{other}"] = B19.condition_contrast(
                    cells, label, "local", other, field=field, scale=scale,
                    B=boot)
        out[label] = row
    return out


def sec_videodelta(boot: int) -> dict:
    """Differences BETWEEN two cells' clip dependence, paired item by item.

    Kept separate from `video` so that section stays a flat label->row map for
    its consumers. Needed because "this cell has no grounding" and "this cell
    has LESS grounding than that one" are different claims, and the second is
    the one the abolition cell makes: two per-cell intervals, one covering zero
    and one not, are not a test of the difference between them.
    """
    cells = B19.load("eval_clip.json", seed=0)
    pairs = [("csl_daily+how2sign-lm_head", "csl_daily"),
             ("csl_daily+how2sign-lm_head", "random"),
             ("csl_daily+how2sign-lm_head", "how2sign"),
             ("how2sign+mt5_base-lm_head", "how2sign"),
             ("csl_daily+mt5_base-lm_head", "csl_daily")]
    return {f"{a} - {b}": B19.swap_delta_contrast(cells, a, b, B=boot)
            for a, b in pairs}


def sec_lastepoch(boot: int) -> dict:
    """A-B6. Same trajectory, selected at its last epoch instead of best dev CE."""
    out = {}
    for p in sorted(glob.glob(str(ROOT / "runs" / "f*" / "eval_lex_last.json"))):
        run = Path(p).parent.name
        d = json.loads(Path(p).read_text())
        best = Path(p).parent / "eval_lex.json"
        rows_last = [r for r in d["contrastive"] if r["condition"] == "local"]
        acc_last = 100.0 * np.mean([float(r["hit"]) for r in rows_last])
        acc_best = None
        if best.exists():
            rb = [r for r in json.loads(best.read_text())["contrastive"]
                  if r["condition"] == "local"]
            acc_best = 100.0 * np.mean([float(r["hit"]) for r in rb])
        log = json.loads((Path(p).parent / "log.json").read_text())
        devs = [e["dev"] for e in log["log"]]
        out[run] = {"acc_best_dev": acc_best, "acc_last_epoch": acc_last,
                    "delta": None if acc_best is None else acc_last - acc_best,
                    "best_epoch": int(np.argmin(devs)) + 1,
                    "epochs": len(devs), "best_dev": min(devs),
                    "last_dev": devs[-1]}
    return out


def sec_lr() -> dict:
    """A-B1. Dev CE and diagnostic accuracy across the mT5 learning rate."""
    # Only the sweep rows and their three references. `cfg` does not carry the
    # learning rate, so the fallback is to read `args` out of best.pt — which is a
    # 7.8 GB file, so the filter has to come BEFORE the load, not after it.
    keep = ("f0_local_how2sign_full_k4_s0",
            "f0_local_how2sign+mt5_base-lm_head_full_k4_s0",
            "f0_local_csl_daily_full_k4_s0")
    out = {}
    for p in sorted(glob.glob(str(ROOT / "runs" / "f0_local_*" / "log.json"))):
        run = Path(p).parent.name
        if "_lr" not in run and run not in keep:
            continue
        d = json.loads(Path(p).read_text())
        lr = d["cfg"].get("lr_mt5")
        if lr is None:
            bp = Path(p).parent / "best.pt"
            if bp.exists():
                ck = torch.load(bp, map_location="cpu", weights_only=False,
                                mmap=True)
                lr = (ck.get("args") or {}).get("lr_mt5")
        lex = Path(p).parent / "eval_lex.json"
        acc = None
        if lex.exists():
            rows = [r for r in json.loads(lex.read_text())["contrastive"]
                    if r["condition"] == "local"]
            acc = 100.0 * np.mean([float(r["hit"]) for r in rows])
        out[run] = {"lr_mt5": lr, "best_dev": d.get("best_dev"),
                    "acc": acc, "epochs": d.get("epochs_done"),
                    "minutes": d.get("minutes")}
    return out


def sec_gen() -> dict:
    """O-P0 / A-A5. Free decoding for every cell that has it."""
    from tsl.metrics import corpus_bleu_chrf
    out = {}
    for p in sorted(glob.glob(str(ROOT / "runs" / "f*" / "eval_gen*.json"))):
        path = Path(p)
        d = json.loads(path.read_text())
        rows = [r for r in d.get("generation", []) if r["condition"] == "local"]
        if not rows:
            continue
        hyps = [r["hyp"] for r in rows]
        refs = [r["ref"] for r in rows]
        uniq = set(hyps)
        modal = max((hyps.count(h) for h in uniq), default=0) / len(hyps)
        graft = d.get("posthoc_head") or {}
        out[f"{path.parent.name}/{path.name}"] = {
            "run": path.parent.name,
            "posthoc_donor": graft.get("donor_run"),
            "n": len(hyps),
            **corpus_bleu_chrf(hyps, refs),
            "distinct": len(uniq) / len(hyps),
            "modal": modal,
            "mean_len": float(np.mean([len(h) for h in hyps])),
            "empty": sum(1 for h in hyps if not h.strip()) / len(hyps),
            "sample": hyps[:8]}
    return out


def sec_params() -> dict:
    """A-Q1. What 966.6M counts, and why lm_head moves while the embedding does not."""
    ext = ROOT / "data" / "external"
    files = {"mt5_base": "mt5_base.pth", "csl_daily": "csl_daily_pose_only_slt.pth",
             "how2sign": "how2sign_pose_only_slt.pth",
             "csl_stage1": "csl_stage1_weight.pth",
             "openasl": "openasl_pose_only_slt.pth",
             "wlasl": "wlasl_pose_only_islr.pth"}
    out: dict = {}
    base_emb = base_head = None
    for name, f in files.items():
        obj = torch.load(ext / f, map_location="cpu", weights_only=False, mmap=True)
        sd = obj
        for k in ("model", "state_dict", "module"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        mt5 = {k: v for k, v in sd.items() if k.startswith("mt5_model")}
        pose = {k: v for k, v in sd.items() if not k.startswith("mt5_model")}
        emb_keys = [k for k in mt5 if k.endswith("shared.weight")
                    or k.endswith("embed_tokens.weight")]
        head_keys = [k for k in mt5 if "lm_head" in k]
        emb = mt5[emb_keys[0]].float() if emb_keys else None
        head = mt5[head_keys[0]].float() if head_keys else None
        aliases = all(torch.equal(mt5[emb_keys[0]], mt5[k]) for k in emb_keys[1:])
        dup = sum(mt5[k].numel() for k in emb_keys[1:])
        rec = {
            "tensors": len(sd),
            "mt5_state_dict_M": sum(v.numel() for v in mt5.values()) / 1e6,
            "pose_M": sum(v.numel() for v in pose.values()) / 1e6,
            "embedding_keys": emb_keys, "embedding_aliases_identical": aliases,
            "embedding_M": emb.numel() / 1e6 if emb is not None else None,
            "duplicate_embedding_M": dup / 1e6,
            "lm_head_M": head.numel() / 1e6 if head is not None else None,
            "mt5_distinct_M": (sum(v.numel() for v in mt5.values()) - dup) / 1e6,
            "lm_head_tied_to_embedding": (
                bool(torch.equal(head, emb)) if (head is not None
                                                 and emb is not None) else None),
            "lm_head_cos_to_embedding": (
                float(torch.nn.functional.cosine_similarity(
                    head.double().flatten(), emb.double().flatten(), dim=0))
                if (head is not None and emb is not None) else None),
        }
        if name == "mt5_base":
            base_emb, base_head = emb, head
        elif base_emb is not None:
            # A-Q1's second half: the projection specialises and the embedding
            # "differs very little". Quantified, since the asymmetry is itself a
            # finding worth stating.
            rec["embedding_rel_l2_from_mt5base"] = float(
                (emb - base_emb).norm() / base_emb.norm())
            rec["lm_head_rel_l2_from_mt5base"] = float(
                (head - base_head).norm() / base_head.norm())
        out[name] = rec
    return out


def sec_gloss() -> dict:
    """A-B2 / A-B4 / O-W5. The instrument, and the regime that works."""
    out = {}
    for p in sorted(glob.glob(str(ROOT / "data" / "gloss_lm*.json"))):
        d = json.loads(Path(p).read_text())
        out[d.get("init_head", Path(p).stem)] = {
            "n_train": d.get("n_train"), "best_dev_ce": d.get("best_dev_ce"),
            "test": d.get("test"),
            "lm_head_rel_l2_from_init": d.get("lm_head_rel_l2_from_init"),
            "lm_head_rel_l2_from_mt5base": d.get("lm_head_rel_l2_from_mt5base"),
            "sample": (d.get("sample") or [])[:5]}
    tl = ROOT / "data" / "text_lm.json"
    if tl.exists():
        d = json.loads(tl.read_text())
        out["_tsl_text_reference"] = {
            "best_dev": d.get("best_dev"),
            "lm_head_rel_l2_from_mt5base": d.get("lm_head_rel_l2_from_mt5base")}
    return out


def sec_interface() -> dict:
    p = ROOT / "data" / "interface_align.json"
    return json.loads(p.read_text()) if p.exists() else {}


def sec_ladder() -> dict:
    """Reviewer G's drift ladder. One row per rung, sorted by the displacement
    reached, so the write-up cannot quote the chosen rung without the two it
    climbed past -- and cannot quote the displacement without `best_dev`, which
    is the price the ladder paid for it."""
    rows = []
    tl = ROOT / "data" / "text_lm.json"
    if tl.exists():
        rows.append((json.loads(tl.read_text()), " (the original donor)"))
    for q in ROOT.glob("data/text_lm_strong_lr*.json"):
        rows.append((json.loads(q.read_text()), ""))
    # By displacement, not by filename: the ladder IS an ordering in drift, and
    # `sorted(glob)` puts lr=1e-3 above lr=3e-4.
    rows.sort(key=lambda t: t[0].get("lm_head_rel_l2_from_mt5base") or 0.0)
    out = {}
    for d, note in rows:
        out[f"lr={d['lr']:g}, {d['epochs']} ep{note}"] = {
            "lr": d["lr"], "epochs": d["epochs"], "n_train": d.get("n_train"),
            "best_dev": d.get("best_dev"),
            "drift_from_mt5base": d.get("lm_head_rel_l2_from_mt5base"),
            "cos_to_mt5base": d.get("lm_head_cos_to_mt5base")}
    return out


def sec_grad() -> dict:
    p = ROOT / "data" / "grad_probe.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
            for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--only", default="",
                    help="comma-separated section names, for a fast re-run")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rc5_analysis.json")
    a = ap.parse_args()

    sections = {
        "params": lambda: sec_params(),
        "cells": lambda: sec_cells(a.boot),
        "contrasts": lambda: sec_contrasts(a.boot),
        "seeds": lambda: sec_seeds(a.boot),
        "posthoc": lambda: sec_posthoc(a.boot),
        "synthead": lambda: sec_synthead(a.boot),
        "video": lambda: sec_video(a.boot),
        "videodelta": lambda: sec_videodelta(a.boot),
        "lastepoch": lambda: sec_lastepoch(a.boot),
        "lr": lambda: sec_lr(),
        "gen": lambda: sec_gen(),
        "gloss": lambda: sec_gloss(),
        "interface": lambda: sec_interface(),
        "grad": lambda: sec_grad(),
        "ladder": lambda: sec_ladder(),
    }
    want = [s.strip() for s in a.only.split(",") if s.strip()] or list(sections)
    out = {}
    if a.out.exists():
        out = json.loads(a.out.read_text())
    for name in want:
        print(f"=== {name}", flush=True)
        try:
            out[name] = clean(sections[name]())
        except Exception as e:  # a half-landed queue must not lose the rest
            print(f"    FAILED: {type(e).__name__}: {e}")
            out.setdefault("_errors", {})[name] = f"{type(e).__name__}: {e}"
            continue
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
