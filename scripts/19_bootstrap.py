#!/usr/bin/env python3
"""Confidence intervals and tests that respect how the items are clustered.

Every interval in the paper so far is a Wilson interval over pooled items, and
two things are wrong with that here. The items are not independent -- 940 lexical
items sit inside 341 of the corpus's 407 paragraphs, several items to a
paragraph, sharing a signer,
a topic and often a vocabulary -- so an interval that assumes independence is too
narrow. And the pooled figure spans three folds, i.e. three *different trained
models*, which a within-model interval does not acknowledge either.

This script reports, for each contrast the paper leans on:

    delta       the observed difference in accuracy, in points
    CI          95% percentile interval from a paired bootstrap that resamples
                PARAGRAPHS with replacement, not items
    p           a paired permutation test at the same level: the sign of a
                paragraph's whole block of per-item differences is flipped with
                probability 1/2, which is the cluster-level analogue of the
                exact test and makes no distributional assumption
    n, clusters how many items and how many paragraphs the contrast rests on

Two kinds of contrast, both paired item by item:

    condition   two readouts of the SAME trained model (local vs blank, local vs
                the wrong-clip control, ...). The strongest form of pairing: same
                weights, same items, one factor changed.
    cell        the same condition on two different trained models (a Chinese
                decoder against an English one, a CSL pose branch against an ASL
                one). Paired on items, which is what makes the comparison sharper
                than two independent intervals.

And one contrast that is neither, because the paper's central claim is neither:

    interaction the 2x2's difference of differences, paired across all FOUR
                cells at once. "Output language gates visual transfer" is a
                statement about two simple effects differing, which no pair of
                separate intervals tests. Reported in points and, because two of
                the four cells sit on the chance floor where points compress, as
                a ratio of odds ratios as well.

    .venv/bin/python scripts/19_bootstrap.py            # content words -> data/bootstrap.json
    .venv/bin/python scripts/19_bootstrap.py \
        --file eval_clip_pron.json --cell-file eval.json \
        --out data/bootstrap_pron.json                  # referential set

There is no `--data` flag: which item set is read is decided by WHICH EVAL FILES
are named, because the two sets were scored into differently-named files rather
than into one file with a column.

No GPU and no retraining: this reads the eval files that already exist.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = re.compile(r"^f[012]_local_.+_full_k4_s\d+$")


RUN_LABEL = re.compile(r"^f\d+_local_(?P<label>.+)_full_k\d+_s\d+$")


def run_label(run: str, cfg: dict, init: str) -> str:
    """The cell's name, taken from the run directory rather than rebuilt from cfg.

    They agree for every cell 04_train.py names itself, because that is the name
    it builds. They must not be conflated where a cell carries a `--tag`: as of
    8/22 there is a cell that differs from an existing one only in a
    hyperparameter (`how2sign_ls00`, label smoothing 0 instead of upstream's 0.2),
    and a cfg-derived label would call it `how2sign` and merge its 330 items into
    that row's. Reading the directory name keeps tagged cells separate by
    construction, and the assertion below is what stops the two conventions
    drifting apart silently.
    """
    m = RUN_LABEL.match(run)
    label = m.group("label") if m else init
    rebuilt = init
    if cfg.get("init_parts", "all") != "all":
        rebuilt = f"{rebuilt}-{cfg['init_parts']}"
    if cfg.get("init_mt5"):
        rebuilt = f"{rebuilt}+{cfg['init_mt5']}-{cfg.get('init_mt5_parts', 'mt5')}"
    if label != rebuilt and not label.startswith(f"{rebuilt}_"):
        raise SystemExit(
            f"run directory {run!r} implies cell {label!r} but its cfg implies "
            f"{rebuilt!r}. One of them is wrong, and guessing which would put a "
            "mislabelled row in a table.")
    return label


def load(fname: str, seed: int = 0, fold: int | None = None,
         suffix: str = "") -> dict[str, dict[str, dict[str, dict]]]:
    """init -> condition -> key -> row, pooled over folds.

    `fold` restricts to one fold, which the seed comparison needs: seeds 1 and 2
    exist on fold 0 only, so pooling folds would compare a three-fold seed-0
    figure against a one-fold seed-1 one and call the difference a seed effect.

    A file written with `05_eval.py --posthoc-head` is the same trained weights
    with one tensor replaced after the fact, so it is a different cell and gets a
    suffixed label: `how2sign@post-csl_daily` is How2Sign's fine-tuned model
    wearing CSL-Daily's fine-tuned output projection. Without the suffix it would
    merge into `how2sign` and quietly average an intervention with its control.
    """
    out: dict[str, dict[str, dict[str, dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    glb = "f[012]_local_*" if fold is None else f"f{fold}_local_*"
    for p in sorted(glob.glob(str(ROOT / "runs" / glb / fname))):
        d = json.loads(Path(p).read_text())
        run = d.get("run", Path(p).parent.name)
        if not CANONICAL.match(run) or d.get("seed", 0) != seed:
            continue
        init = run_label(run, d["cfg"], d["init"])
        graft = d.get("posthoc_head")
        if graft:
            donor = graft.get("donor_run", "")
            m = RUN_LABEL.match(donor)
            init = f"{init}@post-{m.group('label') if m else donor}"
        elif suffix:
            init = f"{init}{suffix}"
        for r in d["contrastive"]:
            out[init][r["condition"]][r["key"]] = r
    return out


def cluster_stats(keys: list[str], diff: np.ndarray, B: int = 10000,
                  rng_seed: int = 0, scale: float = 100.0,
                  cluster_of: dict[str, str] | None = None) -> dict:
    """Paired bootstrap CI and permutation p, both clustered by paragraph.

    `scale` is 100 for a difference of proportions and 1 for a quantity already
    in its own units -- the score margin, which is in nats and has no floor, so
    multiplying it by a hundred would report centinats as if they were points.

    `cluster_of` replaces the resampling unit. Paragraph is the default because
    that is what the items are nested in, but a reviewer asked whether the
    intervals survive clustering by GOLD TYPE instead: 238 types under 940
    items, so a type appears in several paragraphs and a paragraph carries
    several types, and neither nesting contains the other.
    """
    para = (np.array([cluster_of[k] for k in keys]) if cluster_of
            else np.array([k.split(":")[0] for k in keys]))
    uniq, inv = np.unique(para, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(rng_seed)
    obs = float(diff.mean())

    # Bootstrap: resample paragraphs, keeping each paragraph's items together.
    boots = np.empty(B)
    sizes = np.array([len(g) for g in groups])
    sums = np.array([diff[g].sum() for g in groups])
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = sums[pick].sum() / sizes[pick].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # Permutation: flip the sign of a whole paragraph's differences at once.
    signs = rng.choice([-1.0, 1.0], size=(B, len(groups)))
    perm = (signs * sums).sum(1) / sizes.sum()
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return {"delta": scale * obs, "lo": scale * float(lo), "hi": scale * float(hi),
            "p": max(p, 1.0 / B), "n": len(keys), "clusters": len(uniq)}


def accuracy_ci(cells, init, cond="local", B: int = 10000,
                rng_seed: int = 0, field: str = "hit",
                scale: float = 100.0) -> dict | None:
    """A cell's own accuracy, with a paragraph-clustered interval.

    Everything else here is a CONTRAST. But S3.1 makes a one-sample claim -- the
    ASL rows' intervals include chance -- and the only intervals available for it
    were Wilson intervals over pooled items, which assume the independence these
    items do not have and are therefore too narrow. That made the appendix's
    "all confidence intervals in the main text resample paragraphs" false for
    exactly the sentence a reader is most likely to check.

    Same resampling as `cluster_stats`, minus the permutation: there is no null to
    permute against for a single accuracy, and "does the interval cover chance"
    is the question anyway.
    """
    rows = cells.get(init, {}).get(cond)
    if not rows:
        return None
    keys = sorted(rows)
    hit = np.array([float(rows[k][field]) for k in keys])
    para = np.array([k.split(":")[0] for k in keys])
    uniq, inv = np.unique(para, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    sizes = np.array([len(g) for g in groups])
    sums = np.array([hit[g].sum() for g in groups])
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = sums[pick].sum() / sizes[pick].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"kind": "accuracy", "init": init, "condition": cond,
            "field": field, "acc": scale * float(hit.mean()),
            "lo": scale * float(lo), "hi": scale * float(hi),
            "n": len(keys), "clusters": len(uniq)}


def condition_contrast(cells, init, a, b, field: str = "hit",
                       scale: float = 100.0, **kw) -> dict | None:
    ca, cb = cells.get(init, {}).get(a), cells.get(init, {}).get(b)
    if not ca or not cb:
        return None
    keys = sorted(set(ca) & set(cb))
    if not keys:
        return None
    d = np.array([float(ca[k][field]) - float(cb[k][field]) for k in keys])
    return {"kind": "condition", "init": init, "a": a, "b": b, "field": field,
            "acc_a": scale * float(np.mean([ca[k][field] for k in keys])),
            "acc_b": scale * float(np.mean([cb[k][field] for k in keys])),
            **cluster_stats(keys, d, scale=scale, **kw)}


def cell_contrast(cells, init_a, init_b, cond="local", field: str = "hit",
                  scale: float = 100.0, **kw) -> dict | None:
    ca = cells.get(init_a, {}).get(cond)
    cb = cells.get(init_b, {}).get(cond)
    if not ca or not cb:
        return None
    keys = sorted(set(ca) & set(cb))
    if not keys:
        return None
    d = np.array([float(ca[k][field]) - float(cb[k][field]) for k in keys])
    return {"kind": "cell", "a": init_a, "b": init_b, "condition": cond,
            "field": field,
            "acc_a": scale * float(np.mean([ca[k][field] for k in keys])),
            "acc_b": scale * float(np.mean([cb[k][field] for k in keys])),
            **cluster_stats(keys, d, scale=scale, **kw)}


def swap_mean(cells, init, field: str = "hit",
              prefix: str = "swap_plain@") -> dict | None:
    """The wrong-clip condition averaged over its three draws, item by item.

    Averaging inside the item -- rather than reporting three separate accuracies
    -- is what makes `local` minus `swap` a paired quantity, and it stops one
    unlucky donor assignment from carrying the contrast.

    `prefix` selects which wrong clip. `swap_plain@` draws the donor from another
    paragraph, `swap_within@` from the target's own; the second holds signer and
    session fixed and matches duration less tightly, so the two bracket the
    effect rather than one superseding the other.
    """
    draws = [c for c in cells.get(init, {}) if c.startswith(prefix)]
    if not draws:
        return None
    keys = set.intersection(*[set(cells[init][d]) for d in draws])
    return {k: {field: float(np.mean([cells[init][d][k][field] for d in draws]))}
            for k in keys}


def swap_delta_contrast(cells, init_a, init_b, field: str = "hit",
                        scale: float = 100.0, **kw) -> dict | None:
    """Difference of differences: (local - wrong clip) for one cell minus the same
    for another, paired item by item.

    This is the quantity the encoder claim needs and it is NOT what two overlapping
    intervals tell you. `csl_stage1-pose+csl_daily-mt5` and
    `how2sign-pose+csl_daily-mt5` share an mT5 half, so subtracting their
    Delta_swap isolates what the pose branch's source is worth in clip-reading,
    with everything downstream of it held fixed.
    """
    da, db = {}, {}
    for init, out in ((init_a, da), (init_b, db)):
        loc = cells.get(init, {}).get("local")
        sw = swap_mean(cells, init, field=field)
        if not loc or not sw:
            return None
        for k in set(loc) & set(sw):
            out[k] = float(loc[k][field]) - sw[k][field]
    keys = sorted(set(da) & set(db))
    if not keys:
        return None
    d = np.array([da[k] - db[k] for k in keys])
    return {"kind": "swap_delta", "a": init_a, "b": init_b, "field": field,
            "delta_a": scale * float(np.mean([da[k] for k in keys])),
            "delta_b": scale * float(np.mean([db[k] for k in keys])),
            **cluster_stats(keys, d, scale=scale, **kw)}


def cross_swap_delta(cells, init_a, cond_a, init_b, cond_b,
                     field: str = "hit", scale: float = 100.0, **kw) -> dict | None:
    """(local - cond_a) on cell A minus (local - cond_b) on cell B, paired by item.

    Needed because the two wrong-clip families live in different eval files and
    therefore under different cell labels: `X@within` carries `local` and
    `swap_within`, `X` carries `local` and `swap_plain`, and a single cell holding
    both would have had one file's `local` block overwrite the other's. The
    question this answers is whether the between-paragraph drop is larger than the
    within-paragraph one, which is a difference of two deltas and not a difference
    of two accuracies -- an interval on each separately would not test it.

    The two `local` blocks are separate forward passes over the same weights and
    differ on a few items, so a small part of this quantity is scoring noise; the
    magnitude of that is reported as `LocalJitter` in the paper.
    """
    da, db = {}, {}
    for init, cond, out in ((init_a, cond_a, da), (init_b, cond_b, db)):
        loc = cells.get(init, {}).get("local")
        sw = cells.get(init, {}).get(cond)
        if not loc or not sw:
            return None
        for k in set(loc) & set(sw):
            out[k] = float(loc[k][field]) - float(sw[k][field])
    keys = sorted(set(da) & set(db))
    if not keys:
        return None
    d = np.array([db[k] - da[k] for k in keys])
    return {"kind": "cross_swap_delta", "a": f"{init_a}:{cond_a}",
            "b": f"{init_b}:{cond_b}", "field": field,
            "delta_a": scale * float(np.mean([da[k] for k in keys])),
            "delta_b": scale * float(np.mean([db[k] for k in keys])),
            **cluster_stats(keys, d, scale=scale, **kw)}


def _quad_keys(cells, quad, cond):
    """The items every one of the four cells scored, or None if a cell is missing.

    Intersected across all four rather than pairwise: a difference of differences
    computed on two different item sets is not a difference of differences.
    """
    got = [cells.get(init, {}).get(cond) for init in quad]
    if any(not g for g in got):
        return None, None
    keys = sorted(set.intersection(*[set(g) for g in got]))
    return (keys, got) if keys else (None, None)


def interaction_contrast(cells, quad, cond="local", field: str = "hit",
                         scale: float = 100.0, **kw) -> dict | None:
    """The 2x2 interaction itself, paired item by item.

    `quad` is the factorial flattened as (pp, pm, mp, mm) -- CSL visual half with
    a Chinese decoder, CSL visual with an English one, ASL visual with Chinese,
    ASL visual with English -- and the quantity is

        d_i = (pp_i - pm_i) - (mp_i - mm_i)

    what replacing the decoder half costs on a CSL-pretrained visual half, minus
    what the same replacement costs on an ASL-pretrained one. Differencing along
    the other axis gives the identical number, and that identity is the point:
    the four legs the abstract quotes have to come out of one table. They did not
    before 8/19, because the encoder leg was read off a fifth cell (CSL-News
    rather than CSL-Daily on the visual side), so the pair of differences the
    prose called an interaction did not subtract to the interaction.

    It is also the missing test. `csl_daily::vs::how2sign-pose+csl_daily-mt5`
    excluding zero while `csl_daily-pose+how2sign-mt5::vs::how2sign` contains it
    is not evidence that the two differ -- that inference is the one this
    contrast actually licenses.
    """
    keys, got = _quad_keys(cells, quad, cond)
    if not keys:
        return None
    pp, pm, mp, mm = got
    d = np.array([(float(pp[k][field]) - float(pm[k][field]))
                  - (float(mp[k][field]) - float(mm[k][field])) for k in keys])
    accs = [scale * float(np.mean([g[k][field] for k in keys])) for g in got]
    return {"kind": "interaction", "cells": list(quad), "condition": cond,
            "field": field, "acc": [round(x, 4) for x in accs],
            # Both marginal readings, so a caller can check the closure the
            # comment above is about instead of taking it on trust.
            "simple_hi": accs[0] - accs[1], "simple_lo": accs[2] - accs[3],
            "simple_hi_alt": accs[0] - accs[2], "simple_lo_alt": accs[1] - accs[3],
            **cluster_stats(keys, d, scale=scale, **kw)}


def interaction_logit(cells, quad, cond="local", B: int = 10000,
                      rng_seed: int = 0, field: str = "hit") -> dict | None:
    """The same interaction on the logit scale: log of the ratio of odds ratios.

    A difference of differences in points is scale-dependent, and these cells are
    badly placed for it: two of the four sit on the 25% chance floor the item set
    was rank-balanced to produce, where no effect has room to show. Points cannot
    tell a gate from a floor. The odds-ratio ratio is not bounded that way, so
    reporting both says which of the two readings survives a change of scale --
    and neither *removes* the floor, which stays a wording problem, not a
    statistical one.

    Not a per-item quantity, so it is bootstrapped over paragraphs directly
    instead of going through `cluster_stats`, and the p is the bootstrap's
    two-sided tail rather than a sign-flip permutation. The points-scale
    permutation above already tests the same null without that assumption; this
    interval is here for the effect size.
    """
    keys, got = _quad_keys(cells, quad, cond)
    if not keys:
        return None
    hits = np.array([[float(g[k][field]) for k in keys] for g in got])
    para = np.array([k.split(":")[0] for k in keys])
    uniq, inv = np.unique(para, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    sizes = np.array([len(g) for g in groups])
    # Per-paragraph hit counts per cell: everything below is a ratio of sums, so
    # a bootstrap draw is an index into these rather than a rescoring.
    sums = np.stack([np.array([h[g].sum() for g in groups]) for h in hits])

    def lor2(s: np.ndarray, n: float) -> float:
        # Haldane-Anscombe +0.5, which here is insurance rather than a
        # correction: no cell is near 0 or 1, but a bootstrap draw of 341
        # paragraphs is not guaranteed to keep it that way.
        p = (s + 0.5) / (n + 1.0)
        l = np.log(p / (1 - p))
        return float((l[0] - l[1]) - (l[2] - l[3]))

    obs = lor2(sums.sum(1), float(sizes.sum()))
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = lor2(sums[:, pick].sum(1), float(sizes[pick].sum()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tail = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {"kind": "interaction_logit", "cells": list(quad), "condition": cond,
            "field": field, "delta": obs, "lo": float(lo), "hi": float(hi),
            "or_ratio": float(np.exp(obs)),
            "or_ratio_lo": float(np.exp(lo)), "or_ratio_hi": float(np.exp(hi)),
            "p": max(min(tail, 1.0), 1.0 / B), "p_kind": "bootstrap two-sided",
            "n": len(keys), "clusters": len(uniq)}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval_clip.json",
                    help="source for the CONDITION contrasts (local vs blank vs "
                         "swap): only eval_clip.json has the wrong-clip rows")
    ap.add_argument("--cell-file", default="eval_lex.json",
                    help="source for the CELL contrasts. Deliberately a "
                         "different file: eval_lex.json covers three folds for "
                         "every established cell where eval_clip.json is still "
                         "landing, and an interval computed on one fold does not "
                         "belong next to a point estimate pooled over three.")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "bootstrap.json")
    a = ap.parse_args()

    cells = load(a.file)
    if not cells:
        raise SystemExit(f"no {a.file} found under runs/")
    cell_cells = load(a.cell_file) if a.cell_file != a.file else cells

    # The 8/22 additions arrive in their own files, because both are readouts of
    # weights that already existed rather than new cells: a post-hoc `lm_head`
    # graft (05_eval.py --posthoc-head) and the within-paragraph wrong clip.
    # Merged in here so every contrast helper below can reach them, and merged
    # into BOTH dicts because the post-hoc cells are compared with their own
    # ungrafted parents (a cell contrast) and with their own `local` (a condition
    # contrast).
    extra_files = sorted(
        {Path(p).name for p in glob.glob(str(ROOT / "runs" / "f[012]_local_*"
                                             / "eval_posthoc_*.json"))}
        | ({"eval_within.json"}
           if glob.glob(str(ROOT / "runs" / "f[012]_local_*" / "eval_within.json"))
           else set())
        # Added 8/23 for A-B6. `--save-last` writes a second checkpoint and
        # `eval_lex_last.json` scores it, so one training run yields two
        # readouts of the SAME weights trajectory at two selection points. The
        # suffix is what makes the pair a contrast rather than an overwrite:
        # without it the last-epoch rows would land on the best-epoch cell's
        # label and silently replace the number the tables quote.
        | ({"eval_lex_last.json"}
           if glob.glob(str(ROOT / "runs" / "f[012]_local_*"
                            / "eval_lex_last.json"))
           else set()))
    for fn in extra_files:
        # `eval_within.json` re-scores an EXISTING cell, so its rows arrive under
        # that cell's own label and its `local` block would overwrite the one the
        # tables are built from. That is not a cosmetic collision: the pose
        # branch's convolutions are not run deterministically, so the two `local`
        # blocks differ by a few items, and on 8/22 the overwrite moved the
        # headline interaction by 0.11 points and tripped make_numbers.py's
        # closure guard. Suffixing the label keeps the within-paragraph readout
        # self-contained, which is also what the contrast needs: `local` minus
        # `swap_within` has to be paired inside one forward pass.
        #
        # A post-hoc graft file is already a distinct cell by its `@post-` label,
        # so it needs no suffix.
        sfx = ("@within" if fn == "eval_within.json" else
               "@last" if fn == "eval_lex_last.json" else "")
        got = load(fn, suffix=sfx)
        for init, conds in got.items():
            for cond, rows in conds.items():
                cells[init][cond].update(rows)
                cell_cells[init][cond].update(rows)
        if got:
            print(f"merged {fn}: {', '.join(sorted(got))}")

    for init in list(cells):
        # All three readouts of the same three draws: the accuracy the body
        # quotes, the same accuracy under summed log probability, and the score
        # margin, which the review's floor question is asked on.
        for synth, prefix in (("swap_mean", "swap_plain@"),
                              ("within_mean", "swap_within@")):
            m = swap_mean(cells, init, prefix=prefix)
            if not m:
                continue
            for extra in ("hit_sum", "margin"):
                other = swap_mean(cells, init, field=extra, prefix=prefix) or {}
                for k, row in m.items():
                    if k in other:
                        row[extra] = other[k][extra]
            cells[init][synth] = m
            cell_cells[init][synth] = m
    print(f"{len(cells)} cells: {', '.join(sorted(cells))}\n")

    res: dict[str, dict] = {}

    # 1. Within a model: what the target clip is worth, three ways of removing it.
    for init in sorted(cells):
        for a_, b_ in (("local", "blank_plain"), ("local", "swap_mean"),
                       ("local", "shuffle_frames"), ("swap_mean", "blank_plain"),
                       # 8/22: the wrong clip drawn from the target's own
                       # paragraph. The two families against each other cannot be
                       # a condition contrast, because they live on different cell
                       # labels by design -- see `cross_swap_delta` below.
                       ("local", "within_mean")):
            r = condition_contrast(cells, init, a_, b_, B=a.boot)
            if r:
                res[f"{init}::{a_}-{b_}"] = r

    # 2. Between models, on the same items: the factorial's two axes.
    AXES = [
        # decoder axis, VISUAL half held exactly constant -- this is the pair the
        # body quotes, so it is the pair the interval has to be computed on.
        ("csl_daily", "csl_daily-pose+how2sign-mt5"),
        ("how2sign-pose+csl_daily-mt5", "how2sign"),
        ("csl_stage1-pose+csl_daily-mt5", "csl_daily-pose+how2sign-mt5"),
        # 8/19: the encoder leg of the 2x2 proper. The ladder's version of this
        # contrast starts from CSL-News, which is the right cell for pricing two
        # instruments against each other but the wrong one for a factorial whose
        # other three legs are CSL-Daily's -- quoted as an interaction leg it did
        # not subtract to the interaction.
        ("csl_daily", "how2sign-pose+csl_daily-mt5"),
        ("csl_daily-pose+csl_stage1-mt5", "csl_daily-pose+how2sign-mt5"),
        ("csl_daily-pose+csl_stage1-mt5", "csl_daily-pose+openasl-mt5"),
        ("csl_daily-pose+csl_stage1-mt5", "csl_daily-pose+wlasl-mt5"),
        ("csl_daily-pose+openasl-mt5", "csl_daily-pose+how2sign-mt5"),
        # 8/19: a third English-output half. Two rows landing together is a pair;
        # three is a replication, and §3.3's claim is about the class.
        ("csl_daily-pose+wlasl-mt5", "csl_daily-pose+how2sign-mt5"),
        # 8/20: the same replication on the OTHER axis. §3.3 called the three
        # ASL-visual rows indistinguishable "as do[es]" the English trio, but the
        # English trio was the only one with intervals behind it -- this pair was
        # never computed, so that half of the sentence rested on three accuracies
        # looking close, which is the inference this file exists to replace.
        ("wlasl-pose+csl_daily-mt5", "how2sign-pose+csl_daily-mt5"),
        # encoder axis, decoder held at CSL-Daily's
        ("csl_stage1-pose+csl_daily-mt5", "how2sign-pose+csl_daily-mt5"),
        ("csl_stage1-pose+csl_daily-mt5", "wlasl-pose+csl_daily-mt5"),
        ("csl_stage1-pose+csl_daily-mt5", "openasl-pose+csl_daily-mt5"),
        ("how2sign-pose+csl_daily-mt5", "csl_daily-mt5"),
        ("openasl-pose+csl_daily-mt5", "how2sign-pose+csl_daily-mt5"),
        # Camera-ready reviewer 2's third concern. Every released Uni-Sign visual
        # state inherits the same CSL-News stage-1 pose pretraining, so the
        # factorial's "released CSL-Daily visual vs untrained" leg bundles that
        # shared stage with CSL-Daily's own adaptation on top of it and cannot
        # say which one the visual benefit comes from. The stage-1 visual state
        # is on disk, so it can be a THIRD level of the visual factor instead of
        # a caveat: the first pair is CSL-Daily adaptation with the shared stage
        # held fixed, and the next two are what the shared stage alone is worth
        # under each mT5 state.
        ("csl_daily", "csl_stage1-pose+csl_daily-mt5"),
        ("csl_stage1-pose+csl_daily-mt5", "csl_daily-mt5"),
        ("csl_stage1-pose", "random"),
        ("csl_daily-pose", "csl_stage1-pose"),
        # where the decoder factor lives: one tensor against its complement
        ("how2sign+csl_daily-lm_head", "how2sign"),
        ("how2sign+csl_daily-mt5_nohead", "how2sign"),
        ("how2sign-pose+csl_daily-mt5", "how2sign+csl_daily-lm_head"),
        ("csl_daily", "csl_daily+how2sign-lm_head"),
        ("openasl+csl_daily-lm_head", "openasl"),
        ("how2sign+csl_stage1-lm_head", "how2sign"),
        # 8/20, from review: the 2x2 that has no cell on the floor. Both of
        # these legs were quoted in S3.3 as bare accuracies with no interval,
        # which is what let the body rest its gate on the factorial whose
        # English-output half sits at chance.
        ("csl_daily", "csl_daily-mt5"), ("csl_daily-pose", "random"),
        ("csl_daily", "csl_daily-pose"), ("csl_daily-mt5", "random"),
        # 8/22, from review: the head factor's missing NEUTRAL level. Table 1c
        # shows a Chinese head beats an English one, which two readings fit
        # equally well -- adding a Chinese output projection helps, or removing an
        # English-specialized one is enough. mT5-base's own head is the level that
        # separates them, and it is the level every released checkpoint started
        # from before its fine-tuning overwrote it.
        ("how2sign+mt5_base-lm_head", "how2sign"),
        ("how2sign+csl_daily-lm_head", "how2sign+mt5_base-lm_head"),
        ("csl_daily+mt5_base-lm_head", "csl_daily"),
        ("csl_daily+mt5_base-lm_head", "csl_daily+how2sign-lm_head"),
        # ... and whether written-language match survives being SET rather than
        # inherited: a Chinese-adapted head that never saw a sign language.
        ("how2sign+tsl_text-lm_head", "how2sign"),
        ("how2sign+tsl_text-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+csl_daily-lm_head", "how2sign+tsl_text-lm_head"),
        # Added 8/23. Reviewer G's objection to the `tsl_text` null: that donor's
        # head barely moved from mt5-base (rel L2 0.026), so "a Chinese head with
        # no sign language does nothing" could just be "that head is mt5-base".
        # `tsl_text_strong` is the same recipe trained up an LR ladder until its
        # head drifts as far as a real sign-language donor's does. If the null
        # survives at MATCHED drift the confound is closed; if it breaks, the
        # original null was an artefact of an undertrained donor.
        ("how2sign+tsl_text_strong-lm_head", "how2sign"),
        ("how2sign+tsl_text_strong-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+tsl_text_strong-lm_head", "how2sign+tsl_text-lm_head"),
        ("how2sign+csl_daily-lm_head", "how2sign+tsl_text_strong-lm_head"),
        ("how2sign+gloss_lm-lm_head", "how2sign+tsl_text_strong-lm_head"),
        # Sufficiency, the mirror of the existing necessity result: the head alone
        # on an mT5-base body, against that body with no head transplant and
        # against the intact checkpoint.
        ("csl_daily-pose+csl_daily-lm_head", "csl_daily-pose"),
        ("csl_daily", "csl_daily-pose+csl_daily-lm_head"),
        # The post-hoc graft. Same trained weights, one tensor replaced after
        # fine-tuning, no retraining -- the control §4 named and did not run.
        ("how2sign@post-csl_daily", "how2sign"),
        ("csl_daily@post-how2sign", "csl_daily"),
        ("how2sign@post-how2sign", "how2sign"),
        ("how2sign@post-how2sign+csl_daily-lm_head", "how2sign"),
        # ... and against the initialization-time swap, which is the comparison
        # that adjudicates: init-time works, so does post-hoc close the gap?
        ("how2sign+csl_daily-lm_head", "how2sign@post-csl_daily"),
        # Added 8/23. Reviewer O's P3 in the video regime, which is the branch
        # the reframing turns on: is the rescue a RESET of the projection or
        # mT5-base's projection specifically? `rand_head_nm` carries mT5-base's
        # per-row norms, so a null against the neutral head cannot be blamed on
        # scale. The third pair is the one that matters for the paper's own
        # sentence about RAND-VIS: a random head at init lands BELOW no
        # sign-language pretraining at all.
        ("how2sign+rand_head_nm-lm_head", "how2sign"),
        ("how2sign+rand_head_nm-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+rand_head_nm-lm_head", "random"),
        ("csl_daily+rand_head_nm-lm_head", "csl_daily"),
        ("csl_daily+rand_head_nm-lm_head", "csl_daily+mt5_base-lm_head"),
        # And the pair that corrects the first reading of finding 0f: a RANDOM
        # head and a FOREIGN FINE-TUNED head do the same thing to the working
        # model. If this comes back null, "the donated head is collapsed" is not
        # the mechanism -- distance from mT5-base is.
        ("csl_daily+rand_head_nm-lm_head", "csl_daily+how2sign-lm_head"),
        ("csl_daily+rand_head_nm-lm_head", "random"),
        # The library-init random head against the norm-matched one. Row norm
        # 27.7 against 12.746 -- a factor of 2.2 -- and BOTH are at
        # initialization, so unlike the post-hoc rescale rows this pair tests
        # "is scale the variable?" with no timing confound at all.
        ("how2sign+rand_head-lm_head", "how2sign+rand_head_nm-lm_head"),
        ("how2sign+rand_head-lm_head", "how2sign"),
        ("how2sign+rand_head-lm_head", "how2sign+mt5_base-lm_head"),
        # Camera-ready reviewer 2. Every donor above varies the projection's
        # CONTENTS, so "pretrained structure" and "rows against the tokens they
        # were trained for" move together and neither can be credited alone. A
        # row permutation holds the first exactly -- same rows, same Frobenius
        # norm, same row-norm distribution, same singular values -- and destroys
        # only the second. The second pair is the one that answers the question:
        # a null there means token correspondence is not what the recipient
        # needs, and a large gap means it is. The third asks whether a permuted
        # pretrained head is simply a random one, which is the reading the
        # existing `rand_head_nm` row would otherwise leave open.
        ("how2sign+mt5_base_perm-lm_head", "how2sign"),
        ("how2sign+mt5_base_perm-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+mt5_base_perm-lm_head", "how2sign+rand_head_nm-lm_head"),
        ("how2sign+mt5_base_perm-lm_head", "random"),
        # The within-script permutation keeps each row inside its own CJK/Latin/
        # other group, so per-script norm structure survives too. It separates
        # "the rows must match their tokens" from "the scripts must match their
        # regions", which the CJK/global rescaling donors only bound indirectly.
        ("how2sign+mt5_base_permws-lm_head", "how2sign"),
        ("how2sign+mt5_base_permws-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+mt5_base_permws-lm_head", "how2sign+mt5_base_perm-lm_head"),
        # A-B3. The per-script rescale as an INITIALIZATION. Its post-hoc twin
        # (`how2sign@synth-cjk1p6`) is worth a significant +4.5, so if this comes
        # back null the rescale is a test-time calibration and not a starting
        # point -- which is finding 0's pattern running the other way.
        ("how2sign+how2sign_cjk-lm_head", "how2sign"),
        ("how2sign+how2sign_cjk-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+how2sign_cjk-lm_head", "how2sign+rand_head_nm-lm_head"),
        ("how2sign+how2sign_cjkgl-lm_head", "how2sign"),
        ("how2sign+how2sign_cjkgl-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+how2sign_gl-lm_head", "how2sign"),
        ("how2sign+how2sign_gl-lm_head", "how2sign+mt5_base-lm_head"),
        # O-P2/O-W3: does the neutral-head rescue replicate on ASL checkpoints
        # that are not How2Sign? OpenASL is a second ASL->English translation
        # model; WLASL is isolated-sign RECOGNITION, a different task, which is
        # the harder replication of the two.
        ("openasl+mt5_base-lm_head", "openasl"),
        ("wlasl+mt5_base-lm_head", "wlasl"),
        ("openasl+mt5_base-lm_head", "how2sign+mt5_base-lm_head"),
        ("wlasl+mt5_base-lm_head", "how2sign+mt5_base-lm_head"),
        # The Chinese donor on the same three recipients, so the 3x2 grid is
        # complete and the neutral-vs-Chinese ordering can be read per recipient
        # rather than only on How2Sign.
        ("openasl+csl_daily-lm_head", "openasl"),
        ("wlasl+csl_daily-lm_head", "wlasl"),
        ("openasl+mt5_base-lm_head", "openasl+csl_daily-lm_head"),
        ("wlasl+mt5_base-lm_head", "wlasl+csl_daily-lm_head"),
        # A-B1/A-W6. The learning rate on the INTACT checkpoint -- no surgery at
        # all. At 3e-4 this is the one result in the set that cuts against S3.1's
        # negative claim, so it needs the same interval machinery as everything
        # else rather than a dev-CE column and a shrug.
        ("how2sign_lr1e4", "how2sign"),
        ("how2sign_lr3e4", "how2sign"),
        ("how2sign_lr3e4", "how2sign+mt5_base-lm_head"),
        ("how2sign_lr3e4", "random"),
        # The 2x3 interaction. If the head effect holds at 3e-4 the readout result
        # is not an undertuned-baseline artefact; if it collapses, it is.
        # A-B2's downstream cell: a Chinese-output head with no video anywhere in
        # its history, against every other donor on the same recipient. The nulls
        # against mt5_base / csl_daily / tsl_text ARE the result.
        ("how2sign+gloss_lm-lm_head", "how2sign"),
        ("how2sign+gloss_lm-lm_head", "how2sign+mt5_base-lm_head"),
        ("how2sign+gloss_lm-lm_head", "how2sign+csl_daily-lm_head"),
        ("how2sign+gloss_lm-lm_head", "how2sign+tsl_text-lm_head"),
        ("how2sign+gloss_lm-lm_head", "how2sign+rand_head_nm-lm_head"),
        ("how2sign+mt5_base-lm_head_lr1e4", "how2sign_lr1e4"),
        ("how2sign+mt5_base-lm_head_lr3e4", "how2sign_lr3e4"),
        ("how2sign+mt5_base-lm_head_lr3e4", "how2sign+mt5_base-lm_head"),
        ("how2sign+mt5_base-lm_head_lr3e4", "csl_daily+mt5_base-lm_head"),
        # A-B6. Model selection. A review asks whether the dev-CE early stop is
        # doing the work the surgery is credited with: if the head effect is an
        # artefact of WHICH epoch each arm is read at, it should shrink or vanish
        # when both arms are read at the last epoch instead of the best one. The
        # `@last` rows are the same run's final weights, so the within-cell pair
        # is not a new training run and carries no seed noise at all.
        ("how2sign_sl@last", "how2sign_sl"),
        ("csl_daily_sl@last", "csl_daily_sl"),
        ("how2sign+mt5_base-lm_head_sl@last", "how2sign+mt5_base-lm_head_sl"),
        # And the head effect itself, read at both selection points. These two
        # are the answer to the question; the three above only say how far each
        # arm moves.
        ("how2sign+mt5_base-lm_head_sl", "how2sign_sl"),
        ("how2sign+mt5_base-lm_head_sl@last", "how2sign_sl@last"),
        # The retrains against the cells they reproduce. Same config, same seed,
        # a fresh run: nonzero differences here are the floor on how much of any
        # small effect above is run-to-run noise rather than the intervention.
        ("how2sign_sl", "how2sign"),
        ("csl_daily_sl", "csl_daily"),
        ("how2sign+mt5_base-lm_head_sl", "how2sign+mt5_base-lm_head"),
        # Label smoothing. A review's sharpest alternative reading is that the
        # ASL rows are an optimization collapse rather than a transfer result, and
        # names 0.2 (upstream's value) as a candidate cause.
        ("how2sign_ls00", "how2sign"),
        ("how2sign_ls00", "random"),
        # intact rows
        ("csl_daily", "how2sign"), ("csl_daily", "random"),
        ("random", "how2sign"), ("random", "openasl"), ("random", "wlasl"),
    ]
    for x, y in AXES:
        r = cell_contrast(cell_cells, x, y, B=a.boot)
        if r:
            r["source"] = a.cell_file
            res[f"{x}::vs::{y}"] = r

    # 3. How much of a cell's clip-reading the ENCODER source is worth, mT5 half
    #    held at CSL-Daily's throughout. Reported as its own interval because the
    #    claim is that the two item sets price this differently, and a difference
    #    of differences read off two overlapping CIs is a weaker, different test.
    ENC = [("csl_stage1-pose+csl_daily-mt5", "how2sign-pose+csl_daily-mt5"),
           ("csl_stage1-pose+csl_daily-mt5", "wlasl-pose+csl_daily-mt5"),
           ("csl_stage1-pose+csl_daily-mt5", "openasl-pose+csl_daily-mt5"),
           # Added 8/23. The same difference-of-differences argument, applied to
           # the output projection instead of the encoder. "This cell has no
           # clip dependence" and "this cell has LESS clip dependence than that
           # one" are different claims; the abolition result needs the second,
           # and the two comparisons against a cell that has none by
           # construction (`random`) and one that has none by collapse
           # (`how2sign`) are what make the equivalence a measurement rather
           # than a shared failure to reject.
           ("csl_daily+how2sign-lm_head", "csl_daily"),
           ("csl_daily+how2sign-lm_head", "random"),
           ("csl_daily+how2sign-lm_head", "how2sign"),
           ("csl_daily+rand_head_nm-lm_head", "csl_daily"),
           ("csl_daily+rand_head_nm-lm_head", "csl_daily+how2sign-lm_head"),
           ("how2sign+rand_head_nm-lm_head", "how2sign"),
           ("how2sign+rand_head_nm-lm_head", "how2sign+mt5_base-lm_head"),
           ("how2sign+rand_head-lm_head", "how2sign+rand_head_nm-lm_head"),
           ("how2sign+how2sign_cjk-lm_head", "how2sign"),
           ("how2sign+how2sign_cjk-lm_head", "how2sign+mt5_base-lm_head"),
           ("how2sign+how2sign_cjkgl-lm_head", "how2sign"),
           ("how2sign+how2sign_cjkgl-lm_head", "how2sign+mt5_base-lm_head"),
           ("how2sign+how2sign_gl-lm_head", "how2sign"),
           ("how2sign+gloss_lm-lm_head", "how2sign"),
           ("how2sign+gloss_lm-lm_head", "how2sign+mt5_base-lm_head"),
           ("openasl+mt5_base-lm_head", "openasl"),
           ("wlasl+mt5_base-lm_head", "wlasl"),
           ("openasl+csl_daily-lm_head", "openasl"),
           ("wlasl+csl_daily-lm_head", "wlasl"),
           ("how2sign_lr3e4", "how2sign"),
           ("how2sign_lr3e4", "how2sign+mt5_base-lm_head"),
           ("how2sign_lr1e4", "how2sign"),
           # Added 8/23, once run_rc5m.sh landed the clip files for the two
           # head x LR cells. Until then the interaction could only be stated in
           # accuracy; these three put it in clip dependence, which is the
           # quantity S3.1's claim is actually about.
           ("how2sign+mt5_base-lm_head_lr1e4", "how2sign_lr1e4"),
           ("how2sign+mt5_base-lm_head_lr3e4", "how2sign_lr3e4"),
           ("how2sign+mt5_base-lm_head_lr3e4", "how2sign+mt5_base-lm_head"),
           # The drift-matched donor in clip dependence: an accuracy null and a
           # grounding null are different claims, and G's objection is about
           # whether the donor could have moved anything at all.
           ("how2sign+tsl_text_strong-lm_head", "how2sign"),
           ("how2sign+tsl_text_strong-lm_head", "how2sign+mt5_base-lm_head"),
           ("how2sign+tsl_text_strong-lm_head", "how2sign+tsl_text-lm_head"),
           ("how2sign+mt5_base-lm_head", "how2sign"),
           ("csl_daily+mt5_base-lm_head", "csl_daily")]
    for x, y in ENC:
        r = swap_delta_contrast(cells, x, y, B=a.boot)
        if r:
            res[f"{x}::dswap::{y}"] = r

    # 3a'. How much bigger the between-paragraph wrong clip is than the
    #      within-paragraph one. A review's objection to `swap_plain` is that its
    #      donor changes signer, session and topic as well as the clip-sentence
    #      correspondence; this is the size of everything the within-paragraph
    #      version removes, as a paired difference of the two deltas rather than
    #      as two intervals side by side.
    for base in sorted({i for i in cells if i.endswith("@within")}):
        canon = base[: -len("@within")]
        r = cross_swap_delta(cells, base, "within_mean", canon, "swap_mean",
                             B=a.boot)
        if r:
            res[f"{canon}::within_vs_swap"] = r
        rm = cross_swap_delta(cells, base, "within_mean", canon, "swap_mean",
                              field="margin", scale=1.0, B=a.boot)
        if rm:
            res[f"{canon}::within_vs_swap::margin"] = rm

    # 3b. Every cell's own accuracy at the same clustering, so the one-sample
    #     sentences in the body stop leaning on Wilson intervals.
    for init in sorted(cell_cells):
        r = accuracy_ci(cell_cells, init, B=a.boot)
        if r:
            r["source"] = a.cell_file
            res[f"{init}::acc"] = r

    # 4. The interaction the factorial claim is actually about, on both scales.
    #    Everything above is a simple effect; a gate is a difference between two
    #    of them, and until this contrast existed the paper inferred one from a
    #    significant CI sitting beside a non-significant one.
    QUAD = ("csl_daily", "csl_daily-pose+how2sign-mt5",
            "how2sign-pose+csl_daily-mt5", "how2sign")
    r = interaction_contrast(cell_cells, QUAD, B=a.boot)
    if r:
        r["source"] = a.cell_file
        res["interaction::decoder-x-encoder"] = r
        # The identity that makes it one table and not two pairs of numbers.
        closure = abs((r["simple_hi"] - r["simple_lo"])
                      - (r["simple_hi_alt"] - r["simple_lo_alt"]))
        if closure > 1e-6:
            raise SystemExit(f"2x2 does not close: {closure:.6f} points")
        print(f"2x2 cells {QUAD[0]}/{QUAD[1]}/{QUAD[2]}/{QUAD[3]}: "
              + "/".join(f"{x:.1f}" for x in r["acc"]))
        print(f"  simple effects: decoder {r['simple_hi']:+.1f} on CSL vs "
              f"{r['simple_lo']:+.1f} on ASL; encoder {r['simple_hi_alt']:+.1f} "
              f"on Chinese vs {r['simple_lo_alt']:+.1f} on English\n")
    rl = interaction_logit(cell_cells, QUAD, B=a.boot)
    if rl:
        rl["source"] = a.cell_file
        res["interaction::decoder-x-encoder::logit"] = rl
        print(f"interaction log OR ratio {rl['delta']:+.2f} "
              f"[{rl['lo']:+.2f},{rl['hi']:+.2f}] "
              f"OR ratio {rl['or_ratio']:.2f} "
              f"[{rl['or_ratio_lo']:.2f},{rl['or_ratio_hi']:.2f}] "
              f"p={rl['p']:.4f} ({rl['p_kind']})\n")

    # 4b. Is the interaction a property of the seed? Fold 0 carries seeds 0/1/2
    #     for all four cells, so the whole 2x2 can be recomputed inside each
    #     seed. Fold 0 only, hence ~1/3 the items and visibly wider intervals:
    #     the question here is whether the sign and rough size hold, not a
    #     second estimate of the effect.
    for sd in (0, 1, 2):
        cs = load(a.cell_file, seed=sd, fold=0)
        r = interaction_contrast(cs, QUAD, B=a.boot)
        if r:
            r["source"] = f"{a.cell_file} fold 0 seed {sd}"
            r["seed"] = sd
            res[f"interaction::decoder-x-encoder::seed{sd}"] = r

    # 4c. The same difference of differences on a 2x2 with NO cell on the floor.
    #     Reviewers of the third draft made the same objection twice: the
    #     factorial above puts two of its four cells at the \ChanceLevel-point
    #     level the item set was built to produce, so its encoder leg is bounded
    #     by the instrument rather than measured by it. This quad replaces the
    #     English-output mT5 half with an UNADAPTED mT5-base one, which is the
    #     other way to switch the text side off, and every cell then sits at
    #     least fourteen points clear of chance. It tests a narrower claim --
    #     text-side adaptation, not written language -- and tests it cleanly.
    QUAD_ADAPT = ("csl_daily", "csl_daily-mt5", "csl_daily-pose", "random")
    r = interaction_contrast(cell_cells, QUAD_ADAPT, B=a.boot)
    if r:
        r["source"] = a.cell_file
        res["interaction::adapt-x-encoder"] = r
        closure = abs((r["simple_hi"] - r["simple_lo"])
                      - (r["simple_hi_alt"] - r["simple_lo_alt"]))
        if closure > 1e-6:
            raise SystemExit(f"adapt 2x2 does not close: {closure:.6f} points")
        print(f"floor-free 2x2 {'/'.join(QUAD_ADAPT)}: "
              + "/".join(f"{x:.1f}" for x in r["acc"]))
        print(f"  visual source {r['simple_hi']:+.1f} on a Chinese mT5 half vs "
              f"{r['simple_lo']:+.1f} on mT5-base\n")
    rl = interaction_logit(cell_cells, QUAD_ADAPT, B=a.boot)
    if rl:
        rl["source"] = a.cell_file
        res["interaction::adapt-x-encoder::logit"] = rl

    # 4c''. The same interaction with the SHARED stage-1 visual state in place of
    #       the CSL-Daily one. Camera-ready reviewer 2 points out that every
    #       released visual checkpoint inherits Uni-Sign's CSL-News pose
    #       pretraining, so QUAD_ADAPT's visual leg is "stage 1 + CSL-Daily
    #       adaptation" against "nothing" and cannot separate the two. This quad
    #       is "stage 1 alone" against "nothing". If it reproduces the
    #       interaction, the conditional visual benefit is already present in the
    #       shared stage and the paper must say so.
    QUAD_STAGE1 = ("csl_stage1-pose+csl_daily-mt5", "csl_daily-mt5",
                   "csl_stage1-pose", "random")
    r = interaction_contrast(cell_cells, QUAD_STAGE1, B=a.boot)
    if r:
        r["source"] = a.cell_file
        res["interaction::stageone-x-encoder"] = r
        closure = abs((r["simple_hi"] - r["simple_lo"])
                      - (r["simple_hi_alt"] - r["simple_lo_alt"]))
        if closure > 1e-6:
            raise SystemExit(f"stage-1 2x2 does not close: {closure:.6f} points")
        print(f"stage-1 2x2 {'/'.join(QUAD_STAGE1)}: "
              + "/".join(f"{x:.1f}" for x in r["acc"]))
        print(f"  stage-1 visual {r['simple_hi']:+.1f} on a Chinese mT5 half vs "
              f"{r['simple_lo']:+.1f} on mT5-base\n")

    # 4c'. The headline interaction inside each seed. Two reviews ask for exactly
    #      this and rank it above any new checkpoint: a pooled 15.5 with a narrow
    #      interval says nothing about whether re-training reproduces the sign,
    #      because the interval holds the weights fixed. Fold 0 only, so ~1/3 the
    #      items; the question is the sign and rough size, not a second estimate.
    for sd in (0, 1, 2):
        cs = load(a.cell_file, seed=sd, fold=0)
        r = interaction_contrast(cs, QUAD_ADAPT, B=a.boot)
        if r:
            r["source"] = f"{a.cell_file} fold 0 seed {sd}"
            r["seed"] = sd
            res[f"interaction::adapt-x-encoder::seed{sd}"] = r
            print(f"floor-free 2x2 seed {sd}: "
                  + "/".join(f"{x:.1f}" for x in r["acc"])
                  + f"  interaction {r['delta']:+.1f} [{r['lo']:+.1f},{r['hi']:+.1f}]")
        # Every cell of it, per seed, so Limitations can quote a spread per cell
        # rather than one number for the whole factorial.
        for init in QUAD_ADAPT:
            rc = accuracy_ci(cs, init, B=a.boot)
            if rc:
                rc["source"] = f"{a.cell_file} fold 0 seed {sd}"
                rc["seed"] = sd
                res[f"{init}::acc::seed{sd}"] = rc

    # 4d. Both interactions on the score MARGIN, which has no floor at all.
    #     Accuracy is a threshold on this quantity: margin_i is the gold
    #     candidate's mean token log probability minus the best distractor's, in
    #     nats, and it keeps moving after accuracy has bottomed out at chance. If
    #     the interaction survives here, the points-scale version is not an
    #     artefact of two cells sitting on a bound; if it does not, no amount of
    #     rewording rescues the points-scale one.
    for name, quad in (("decoder-x-encoder", QUAD),
                       ("adapt-x-encoder", QUAD_ADAPT)):
        r = interaction_contrast(cell_cells, quad, field="margin", scale=1.0,
                                 B=a.boot)
        if r:
            r["source"] = a.cell_file
            res[f"interaction::{name}::margin"] = r
            print(f"margin interaction {name}: {r['delta']:+.3f} nats "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.4f}")
    for init in sorted(set(QUAD) | set(QUAD_ADAPT)):
        r = accuracy_ci(cell_cells, init, field="margin", scale=1.0, B=a.boot)
        if r:
            r["source"] = a.cell_file
            res[f"{init}::margin"] = r
    for x, y in (("how2sign+csl_daily-lm_head", "how2sign"),
                 ("csl_daily", "csl_daily+how2sign-lm_head"),
                 ("csl_daily", "how2sign-pose+csl_daily-mt5"),
                 # 8/22. The margin matters more for these than for anything
                 # else in the paper: the post-hoc rows are the ones a reviewer
                 # would expect to move a score without moving an accuracy.
                 ("how2sign@post-csl_daily", "how2sign"),
                 ("csl_daily@post-how2sign", "csl_daily"),
                 ("how2sign@post-how2sign", "how2sign"),
                 ("how2sign+csl_daily-lm_head", "how2sign@post-csl_daily"),
                 ("how2sign+mt5_base-lm_head", "how2sign"),
                 ("how2sign+csl_daily-lm_head", "how2sign+mt5_base-lm_head"),
                 ("csl_daily+mt5_base-lm_head", "csl_daily"),
                 ("how2sign+tsl_text-lm_head", "how2sign+mt5_base-lm_head"),
                 ("how2sign+tsl_text_strong-lm_head",
                  "how2sign+mt5_base-lm_head"),
                 ("csl_daily-pose+csl_daily-lm_head", "csl_daily-pose"),
                 ("how2sign_ls00", "how2sign")):
        r = cell_contrast(cell_cells, x, y, field="margin", scale=1.0, B=a.boot)
        if r:
            r["source"] = a.cell_file
            res[f"{x}::vs::{y}::margin"] = r

    # 4e. The three headline effects under the OTHER contrastive scoring rule.
    #     Appendix A shows the 25% reference level is specific to mean token log
    #     probability; a reviewer asked, fairly, whether the effects are too. Same
    #     trained weights, same items, summed log probability instead.
    for name, quad in (("decoder-x-encoder", QUAD),
                       ("adapt-x-encoder", QUAD_ADAPT)):
        r = interaction_contrast(cell_cells, quad, field="hit_sum", B=a.boot)
        if r:
            r["source"] = a.cell_file
            res[f"interaction::{name}::sum"] = r
    r = cell_contrast(cell_cells, "how2sign+csl_daily-lm_head", "how2sign",
                      field="hit_sum", B=a.boot)
    if r:
        r["source"] = a.cell_file
        res["how2sign+csl_daily-lm_head::vs::how2sign::sum"] = r
    for init in ("csl_daily", "how2sign"):
        r = condition_contrast(cells, init, "local", "swap_mean",
                               field="hit_sum", B=a.boot)
        if r:
            res[f"{init}::local-swap_mean::sum"] = r

    # 4f. Does the wrong clip move the SCORE when it does not move the accuracy?
    #     The mismatched rows lose nothing measurable in points when the target
    #     clip is replaced, but points are a threshold: a model whose
    #     representation still tracked the clip could show it here and nowhere
    #     else. Reported for every cell that has the wrong-clip condition.
    for init in sorted(cells):
        r = condition_contrast(cells, init, "local", "swap_mean",
                               field="margin", scale=1.0, B=a.boot)
        if r:
            res[f"{init}::local-swap_mean::margin"] = r

    # 4g. Clustering sensitivity. Paragraph is the nesting the items have, but a
    #     reviewer asked what happens if the resampling unit is the gold word
    #     type instead -- 238 types, each recurring across paragraphs.
    lex_rec = ROOT / "data" / "lex" / "records.jsonl"
    if lex_rec.exists():
        gold_of = {}
        for line in lex_rec.open():
            d = json.loads(line)
            if d.get("item_type") == "lexical" and d.get("gold_word"):
                gold_of[d["key"]] = d["gold_word"]
        for name, quad in (("decoder-x-encoder", QUAD),
                           ("adapt-x-encoder", QUAD_ADAPT)):
            keys, _ = _quad_keys(cell_cells, quad, "local")
            if keys and all(k in gold_of for k in keys):
                r = interaction_contrast(cell_cells, quad, B=a.boot,
                                         cluster_of=gold_of)
                if r:
                    r["source"] = f"{a.cell_file} clustered by gold type"
                    res[f"interaction::{name}::bytype"] = r

    for k, r in res.items():
        if r["kind"] == "accuracy":
            covers = " covers chance" if r["lo"] <= 25.0 <= r["hi"] else ""
            print(f"{k:64s} {r['acc']:6.1f} [{r['lo']:5.1f},{r['hi']:5.1f}] "
                  f"{'':>15s} n={r['n']} clusters={r['clusters']}{covers}")
            continue
        star = "" if r["p"] >= 0.05 else " *"
        print(f"{k:64s} {r['delta']:+6.1f} [{r['lo']:+5.1f},{r['hi']:+5.1f}] "
              f"p={r['p']:.4f} n={r['n']} clusters={r['clusters']}{star}")

    a.out.write_text(json.dumps({"file": a.file, "boot": a.boot,
                                 "contrasts": res}, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
