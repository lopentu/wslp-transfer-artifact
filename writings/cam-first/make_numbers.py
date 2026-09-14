#!/usr/bin/env python3
"""Every number in the paper, regenerated from the run outputs.

    ../../.venv/bin/python make_numbers.py        # writes numbers.tex

The paper cites `\\CsldailyLocal`, never `77.6`. Runs are still landing — the
split rows for folds 1 and 2 among them — and a paper with hand-copied numbers
would either go stale silently or have to be re-read line by line every time a
cell finishes. Re-run this and recompile instead.

Macros that depend on runs which have not finished emit `\\TODO{...}`, which
renders as a visible marker rather than a wrong number, and `--check` lists
them. Anything still marked at submission time is a bug in the schedule, not a
typo to fix in the text.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# RELEASE: free-decoding statistics, precomputed. See the block that
# reads it for why the raw decodes are not in the public artifact.
_GM = ROOT / "data" / "gen_metrics.json"
GEN_METRICS = json.loads(_GM.read_text()) if _GM.exists() else {}

# CJK unified ideographs, for the "is the Chinese script reachable at all"
# check on the free-decoding outputs.
HAN = re.compile(r"[\u4e00-\u9fff]")
CHANCE = 25.0

# Macro-safe names: LaTeX control sequences are letters only.
NAME = {
    "csl_daily": "Csldaily", "csl_stage1": "Cslstageone",
    "how2sign": "Howtwosign", "openasl": "Openasl", "wlasl": "Wlasl",
    "random": "Randominit",
    "csl_daily-pose": "Csldailypose", "csl_daily-mt5": "Csldailymtfive",
    "csl_stage1-pose": "Cslstageonepose", "csl_stage1-mt5": "Cslstageonemtfive",
    "how2sign-pose": "Howtwosignpose", "openasl-pose": "Openaslpose",
    "wlasl-pose": "Wlaslpose",
    # Cross-pairing: both halves pretrained and both Chinese-capable, but taken
    # from checkpoints that were never trained together.
    "csl_daily-pose+csl_stage1-mt5": "CrossDailyNews",
    "csl_stage1-pose+csl_daily-mt5": "CrossNewsDaily",
    # The 2x2 over (encoder sign language) x (decoder written language). These
    # are the rows that separate the two factors by design rather than by
    # inference from the released checkpoints, in which they always co-vary.
    "how2sign-pose+csl_daily-mt5": "CrossHowCsl",
    "wlasl-pose+csl_daily-mt5": "CrossWlaslCsl",
    "csl_daily-pose+how2sign-mt5": "CrossCslHow",
    # Added 8/18. Three English-output mT5 halves, not one: identifying the
    # decoder factor with a single checkpoint is what the reviews called out.
    "csl_daily-pose+openasl-mt5": "CrossCslOpen",
    "csl_daily-pose+wlasl-mt5": "CrossCslWlasl",
    "openasl-pose+csl_daily-mt5": "CrossOpenCsl",
    # The single-tensor cross-loads: `lm_head` is where the released checkpoints
    # actually differ (scripts/21_lineage.py), so these ask whether the decoder
    # effect is that tensor. `mt5_nohead` is the complement and the control.
    "how2sign+csl_daily-lm_head": "HeadHowCsl",
    "csl_daily+how2sign-lm_head": "HeadCslHow",
    "how2sign+csl_daily-mt5_nohead": "BodyHowCsl",
    # Added 8/19. The generality probes: a second ASL recipient and a second
    # CSL donor, one fold each, so every macro of theirs carries a dagger.
    "openasl+csl_daily-lm_head": "HeadOpenCsl",
    "how2sign+csl_stage1-lm_head": "HeadHowNews",
    # Added 8/22, and the block that decides the paper's framing. Table 1c had
    # only two levels of the output-projection factor -- a Chinese head and an
    # English one -- so "a Chinese head lets visual transfer be used" and
    # "removing an English-specialized head is enough" fitted it equally well.
    # `mt5_base` is the neutral third level: the projection every one of these
    # checkpoints started from. `tsl_text` is Chinese-adapted with no sign
    # language anywhere in it, which is the one cell where written-language match
    # is set rather than inherited.
    "how2sign+mt5_base-lm_head": "HeadHowBase",
    "csl_daily+mt5_base-lm_head": "HeadCslBase",
    "how2sign+tsl_text-lm_head": "HeadHowText",
    # ... and the same donor pushed up a learning-rate ladder until its
    # projection has travelled as far from mT5-base as a released sign
    # fine-tune's has (Reviewer G). Matched on DRIFT, not on language-model
    # quality -- the ladder bought the displacement by overtraining.
    "how2sign+tsl_text_strong-lm_head": "HeadHowTextStrong",
    # Sufficiency: the head alone on an unadapted body, against Table 1c's
    # necessity result.
    "csl_daily-pose+csl_daily-lm_head": "HeadPoseCsl",
    # The post-hoc grafts. Not trained cells at all -- the same fine-tuned
    # weights with the output projection replaced afterwards, which is the
    # control S4 named and did not run.
    "how2sign@post-csl_daily": "PostHowCsl",
    "csl_daily@post-how2sign": "PostCslHow",
    "how2sign@post-how2sign": "PostHowSelf",
    "how2sign@post-how2sign+csl_daily-lm_head": "PostHowRescued",
    "csl_daily@post-csl_daily": "PostCslSelf",
    # Label smoothing 0 instead of upstream's 0.2, on the intact ASL checkpoint:
    # the optimization-collapse reading's own prediction.
    "how2sign_ls00": "HowtwosignLsZero",
    # A-B1/A-W6. The mT5 learning rate on the intact checkpoint. `LrThreeEFour`
    # rather than `Lr3e4`: a LaTeX control sequence is letters only.
    "how2sign_lr1e4": "HowtwosignLrOneEFour",
    "how2sign_lr3e4": "HowtwosignLrThreeEFour",
    "how2sign+mt5_base-lm_head_lr1e4": "HeadHowBaseLrOneEFour",
    "how2sign+mt5_base-lm_head_lr3e4": "HeadHowBaseLrThreeEFour",
    # A-B6. The three `--save-last` retrains. Same config and same seed as the
    # cells they reproduce; the point of them is the SECOND checkpoint each one
    # writes, so the head effect can be read at the last epoch as well as at the
    # dev-CE best one. The best-epoch readout of these three is a reproducibility
    # check on the cells above, and is reported as one.
    # Five cells that reached the report through bootstrap contrasts alone and
    # so had no accuracy macro of their own until 8/23. The contrast is the
    # result in each case, but a report that quotes a difference and cannot
    # quote either arm is not checkable, and `folds per system` prints the raw
    # label for exactly these -- which is how the omission was found.
    "how2sign+gloss_lm-lm_head": "HeadHowGloss",
    "csl_daily-pose+gloss_lm-lm_head": "HeadPoseGloss",
    "openasl+mt5_base-lm_head": "HeadOpenBase",
    "wlasl+mt5_base-lm_head": "HeadWlaslBase",
    "wlasl+csl_daily-lm_head": "HeadWlaslCsl",
    "how2sign_sl": "HowtwosignSl",
    "csl_daily_sl": "CsldailySl",
    "how2sign+mt5_base-lm_head_sl": "HeadHowBaseSl",
    # Added 8/23. Reviewer O's P3, the branch the whole reframing turns on: a
    # projection that is merely NOT ASL-specialized against mT5-base's actual
    # one. `rand_head_nm` carries mT5-base's per-row norms so that "reset" is
    # separated from "rescale"; `rand_head` is the library default and keeps the
    # scale confound, which is why both exist.
    # A-B3: the rescaled ASL head as an INITIALIZATION rather than a post-hoc
    # graft. The post-hoc versions live in eval_synthead_*.json and are named by
    # donor; these are trained cells and need row prefixes of their own.
    "how2sign+how2sign_cjk-lm_head": "HeadHowCjk",
    "how2sign+how2sign_gl-lm_head": "HeadHowGl",
    "how2sign+how2sign_cjkgl-lm_head": "HeadHowCjkgl",
    "how2sign+rand_head_nm-lm_head": "HeadHowRandNm",
    "csl_daily+rand_head_nm-lm_head": "HeadCslRandNm",
    "how2sign+rand_head-lm_head": "HeadHowRand",
    # Camera-ready reviewer 2. mT5-base's own projection with its vocabulary
    # rows reordered: every global statistic of the matrix preserved exactly,
    # only the row-to-token correspondence destroyed. `permws` draws the
    # permutation inside each CJK/Latin/other group, so per-script structure
    # survives too (scripts/27_head_donors.py).
    "how2sign+mt5_base_perm-lm_head": "HeadHowPerm",
    "how2sign+mt5_base_permws-lm_head": "HeadHowPermWs",
    # Where inside the checkpoint the pairing lives.
    "csl_daily-proj": "Csldailyproj",
    "csl_daily-pose_noproj": "Csldailyposenoproj",
    "csl_daily-mt5_proj": "Csldailymtfiveproj",
}

# Rows re-run under more than one seed. Every other number in the paper is
# seed 0, and these say how much of a difference that choice can carry.
SEED_ROWS = ["csl_daily", "how2sign", "random",
             "how2sign-pose", "wlasl-pose", "csl_daily-pose",
             # Added 8/20. Limitations said "other cells use a single seed", and
             # the cells that covered were the two headline claims: the fourth
             # 2x2 cell (the other three already had seeds 0/1/2 on fold 0, and
             # one missing cell blocks the interaction at any other seed) and
             # the whole lm_head decomposition.
             "csl_daily-pose+how2sign-mt5",
             "how2sign+csl_daily-lm_head",
             "how2sign+csl_daily-mt5_nohead",
             "csl_daily+how2sign-lm_head",
             # Added 8/22. Two reviews rank this above any new checkpoint: it is
             # the fourth cell of the FLOOR-FREE 2x2, which v4 promoted to the
             # headline while leaving it at one seed, so the 15.5-point
             # interaction could not be quoted at any seed but 0.
             "csl_daily-mt5",
             # Added 8/22 (RC5). O-P1 and A-A2 both ask for seeds on the two
             # neutral-head cells specifically, which v5 introduced at seed 0
             # only while making them the framing-deciding rows. Blocks 1 of
             # run_rc5.sh fills seeds 1 and 2 for both.
             "how2sign+mt5_base-lm_head",
             "csl_daily+mt5_base-lm_head"]

# Rows of the encoder-only axis: the pose branch from each source on an
# untouched mT5, so written-language handling is identical across them and the
# only thing varying is which sign language the encoder saw.
ENCODER_ONLY = ["csl_daily-pose", "csl_stage1-pose", "how2sign-pose",
                "openasl-pose", "wlasl-pose"]


def todo(label: str) -> str:
    """A visible placeholder that is also valid LaTeX.

    Run names carry underscores (`csl_stage1-pose`), and `\\TODO` typesets its
    argument, so an unescaped label aborts the build with "Missing $ inserted"
    — i.e. an unfinished run would break compilation rather than show up as a
    marker, which is the opposite of what the placeholder is for.
    """
    return "\\TODO{" + label.replace("_", "\\_") + "}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def paired(a: list, b: list) -> tuple[list, list] | None:
    """The two conditions restricted to items both actually scored.

    Returns None unless they cover the *same* items. A condition added in a
    later sweep lands one cell at a time, so for most of an afternoon `local`
    holds three folds and `blank_plain` holds one; subtracting those two pooled
    accuracies gives a difference between two different item sets that looks
    like a result. The generator's job is to emit a placeholder there instead.
    """
    ka = {r["key"] for r in a}
    kb = {r["key"] for r in b}
    if not ka or ka != kb:
        return None
    return a, b


def mcnemar(a: list, b: list) -> float:
    """Exact McNemar p for condition A vs B on the same items.

    Delegated to `tsl.metrics.paired_delta` rather than reimplemented: it is the
    same test `06_report.py` prints, and two copies of a sign test in one
    repository is one copy too many for a number that goes in a table.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from tsl.metrics import paired_delta
    return paired_delta(a, b)["p_exact"]


def fmt_p(p: float) -> str:
    """`<0.001` rather than `0.000`, which claims a precision the test lacks.

    Math-mode content that carries its own *relation*, to be cited as
    `$p\\SomethingP$` — no `=` at the call site. The relation has to live in
    here because it changes with the value: a caller that supplies its own `=`
    is right for `0.381` and prints `p = <0.001` for a bound, which is what it
    did on page 3 until 8/19.

    Bare `<` and `=` rather than `{<}`: the braces make the relation an
    ordinary atom and drop the spacing TeX puts around a relation, so a braced
    `p<0.001` did not match the `p = 0.381` in the next clause. No `$` in here
    either — a macro carrying its own would close the caller's math group the
    first time a p-value fell below the threshold, a failure that hides until
    the data changes.
    """
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def logit(p: float, n: int = 313) -> float:
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)
    return math.log(p / (1 - p))


# A cell trained under the paper's protocol. Everything the tables compare is
# 12 epochs at a matched budget, and the run directory is the only place the
# budget appears — `cfg` records the checkpoint and the tuning mode but not the
# epoch count, so two runs of the same (init, seed, fold) at different budgets
# are indistinguishable once the file is open.
#
# They are not indistinguishable by key, though: they collide on it. The 8/17
# sweep added two deliberately off-protocol 20-epoch cells, and because
# `load_cells` extends the bucket for a key it already has, the 20-epoch fold-1
# run silently pooled with the 12-epoch one and turned that cell's 57.3% into
# 68.7% — the mean of two runs, over 614 items, in a column captioned 307. It was
# caught only because the number moved in a direction the sweep could not
# explain. Anything not matching this pattern is therefore skipped by name and
# the skip is announced, since a silent drop is the other way to get this wrong.
CANONICAL_RUN = re.compile(r"^f[012]_local_.+_(?:full|lora)_k\d+_s\d+$")
_SKIPPED: set[str] = set()


RUN_LABEL = re.compile(r"^f\d+_local_(?P<label>.+)_(?:full|lora)_k\d+_s\d+$")


def row_label(d: dict) -> str:
    """The name of the cell this eval.json is, as the paper thinks of it.

    Rebuilt from `cfg` and then checked against the run directory, which is where
    a `--tag` cell announces itself. Two additions on 8/22 make the check
    load-bearing rather than decorative:

      `how2sign_ls00`   the same checkpoint at label smoothing 0 instead of
                        upstream's 0.2. `cfg` cannot tell it apart from
                        `how2sign` -- it records the checkpoint and the tuning
                        mode, not the objective -- so a cfg-only label would pool
                        an ablation into the row it is an ablation of, which is
                        exactly the failure CANONICAL_RUN exists to prevent one
                        version of.
      `@post-...`       an `05_eval.py --posthoc-head` file: the same fine-tuned
                        weights with the output projection replaced afterwards.
                        Same run directory, different model.
    """
    init = d["init"]
    if d["cfg"].get("init_parts", "all") != "all":
        init = f"{init}-{d['cfg']['init_parts']}"
    if d["cfg"].get("init_mt5"):
        # The part matters in the label: an lm_head-only cross-load and a
        # whole-mT5 one are different experiments and must not share a row.
        init = f"{init}+{d['cfg']['init_mt5']}-{d['cfg'].get('init_mt5_parts', 'mt5')}"
    run = d.get("run", "")
    m = RUN_LABEL.match(run)
    if m and m.group("label") != init:
        if not m.group("label").startswith(f"{init}_"):
            raise SystemExit(
                f"run {run!r} implies cell {m.group('label')!r} but its cfg "
                f"implies {init!r}; refusing to guess which row it belongs in")
        init = m.group("label")
    graft = d.get("posthoc_head")
    if graft:
        donor = graft.get("donor_run", "")
        dm = RUN_LABEL.match(donor)
        init = f"{init}@post-{dm.group('label') if dm else donor}"
    return init


def load_cells(*names: str) -> dict:
    """(init, seed, fold) -> {condition: [rows]}, one entry per trained cell.

    Keyed on all three parts because collapsing any of them hides something.
    Folds pool by design — every item is held out exactly once, so pooling them
    is what makes 975 items. A *seed* is a rerun of the same cell on the same
    items, so pooling it into the same bucket would silently turn a 975-item
    table into a 1,950-item one whose accuracy is an average over seeds, and
    every confidence interval in the paper would shrink for no reason. Seeds are
    therefore separated here and recombined only by `seed_stats`.

    Several files per cell, because conditions were added after the first sweep
    and re-running a finished eval to append one would put working numbers at
    risk for no gain: `eval.json` holds local and text_only, `eval_blank.json`
    the blank_plain added later. Pass only files over the *same* item set — the
    lexical set is 940 different items and is pooled separately, since merging
    it here would silently average two diagnostics into one accuracy.
    """
    cells: dict[tuple, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for name in names:
        for p in sorted(glob.glob(str(ROOT / "runs" / "f[012]_local_*" / name))):
            d = json.loads(Path(p).read_text())
            run = d.get("run", Path(p).parent.name)
            if not CANONICAL_RUN.match(run):
                # Off-protocol by name (see CANONICAL_RUN). Announced once per
                # run so it can be checked against what the sweep meant to do,
                # never dropped in silence.
                if run not in _SKIPPED:
                    _SKIPPED.add(run)
                    print(f"off-protocol, not pooled: {run}", file=sys.stderr)
                continue
            key = (row_label(d), d["seed"], d["fold"])
            for r in d["contrastive"]:
                cells[key][r["condition"]].append(r)
    return cells


def acc(rows: list) -> float | None:
    return 100 * sum(r["hit"] for r in rows) / len(rows) if rows else None


def pool(cells: dict, seed: int = 0):
    """The paper's tables: one seed, pooled over whatever folds exist.

    Per-fold accuracy comes back too, because pooling can hide a bimodal run:
    two encoder-only rows scored ~74% on fold 0 and ~55% on folds 1-2, and only
    the per-fold view shows that their pooled figure is one excursion rather
    than a steady advantage.
    """
    per: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    folds: dict[str, set] = collections.defaultdict(set)
    byfold: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for (init, s, f), conds in cells.items():
        if s != seed:
            continue
        folds[init].add(f)
        a = acc(conds.get("local", []))
        if a is not None:
            byfold[init][f] = a
        for c, rows in conds.items():
            per[init][c].extend(rows)
    return per, folds, byfold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "numbers.tex")
    ap.add_argument("--check", action="store_true", help="list unresolved macros")
    a = ap.parse_args()

    cells = load_cells("eval.json", "eval_blank.json")
    per, folds, byfold = pool(cells, seed=0)
    short = json.loads((ROOT / "data" / "shortcuts.json").read_text())
    strata = json.loads((ROOT / "data" / "strata.json").read_text())["strata"]
    hard = {k for k, v in short["hits"]["lm_context"].items() if not v}

    out: list[str] = [
        "% Generated by numbers.py — do not edit by hand.",
        r"\newcommand{\TODO}[1]{\textbf{[TODO: #1]}}",
    ]

    seen_macros: dict[str, str] = {}

    def emit(macro: str, value) -> None:
        # \newcommand aborts the build on redefinition, so a macro emitted twice
        # is a compile error discovered at submission time rather than here. The
        # loops below overlap by design (SEED_ROWS and QUAD_ADAPT share rows), so
        # this is a live hazard and not a hypothetical.
        v = str(value)
        if macro in seen_macros:
            if seen_macros[macro] == v:
                return
            raise SystemExit(
                f"macro {macro!r} emitted twice with different values: "
                f"{seen_macros[macro]!r} then {v!r}")
        seen_macros[macro] = v
        out.append(f"\\newcommand{{\\{macro}}}{{{v}}}")

    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    emit("ChanceLevel", f"{CHANCE:.0f}")

    # ------------------------------------------------------------- main table
    for init, conds in per.items():
        m = NAME.get(init)
        if not m:
            continue
        rows_L = conds.get("local", [])
        L = [r["hit"] for r in rows_L]
        if not L:
            continue
        lo, hi = wilson(sum(L), len(L))
        emit(f"{m}Local", pct(sum(L) / len(L)))
        emit(f"{m}CIlo", f"{lo:.1f}")
        emit(f"{m}CIhi", f"{hi:.1f}")
        emit(f"{m}N", len(L))
        emit(f"{m}Folds", len(folds[init]))
        # `Blank` is blank_plain, not text_only. Both zero the target clip, but
        # text_only also *supplies* the preceding Chinese and switches the prompt
        # to PROMPT_CTX, so local -> text_only moves three factors at once and
        # cannot be captioned "what blanking the clip costs" (tsl/conditions.py).
        # blank_plain differs from local in exactly the one factor the column
        # claims to isolate. text_only is still reported, under its own name.
        for cond, tag in (("blank_plain", "Blank"), ("text_only", "TextOnly")):
            pr = paired(rows_L, conds.get(cond, []))
            dtag = "Video" if cond == "blank_plain" else "TextOnlyDelta"
            if not pr:
                emit(f"{m}{tag}", todo(f"{init} {cond}"))
                emit(f"{m}{dtag}", todo(f"{init} {cond} delta"))
                continue
            x, y = pr
            ax, ay = acc(x), acc(y)
            emit(f"{m}{tag}", f"{ay:.1f}")
            emit(f"{m}{dtag}", f"{ax - ay:+.1f}")
            # Unsigned twin, for prose that already carries the direction
            # ("loses N points"); the signed form is for table cells, where the
            # sign is the only thing distinguishing a gain from a loss.
            emit(f"{m}{dtag}Abs", f"{abs(ax - ay):.1f}")
        # diagnostic split, restricted to what a text-only model cannot solve
        for tag, sel in (("Ana", lambda r: r["item_type"] == "anaphoric"),
                         ("Dei", lambda r: r["item_type"] == "deictic")):
            h = [r["hit"] for r in conds["local"] if sel(r) and r["key"] in hard]
            allh = [r["hit"] for r in conds["local"] if sel(r)]
            if h:
                emit(f"{m}{tag}Hard", pct(sum(h) / len(h)))
                emit(f"{m}{tag}HardN", len(h))
            if allh:
                emit(f"{m}{tag}All", pct(sum(allh) / len(allh)))

    # ------------------------------------------------- encoder-only axis spread
    # The per-fold figures for the encoder-only rows, and the widest gap any of
    # them opens over random initialisation. Reported because the two rows with
    # a nominal advantage get all of it from a single fold, and a pooled number
    # alone would present that as a stable effect.
    rnd = per.get("random", {}).get("local", [])
    rnd_acc = 100 * sum(r["hit"] for r in rnd) / len(rnd) if rnd else float("nan")
    best_gap, best_name = -99.0, None
    # `random` gets per-fold macros too: it is the baseline row of the same
    # table, and a typed-in literal there would be the one number in the paper
    # that no longer tracks the runs.
    #
    # All five mixed-half cells are in here as of 8/17, when the sweep finished
    # folds 1 and 2 for every one of them. Until then this list stopped at the
    # two Chinese-on-Chinese pairings and the comment said the paper quoted fold
    # 0 throughout "because the cross-pairing cells are fold 0 only". They are
    # not any more, so Table 2 pools three folds like Table 1, and these macros
    # are what lets Limitations quote the per-fold spread instead of conceding
    # there is none to quote. The `Fz` macros further down are kept rather than
    # deleted: Appendix B still wants a fold-0 view, and the fold-0 figures are
    # what the earlier draft's claims were made from, so keeping them generated
    # means a sentence that was not rewritten yet still says something true.
    FOLD_ROWS = ENCODER_ONLY + [
        "random", "csl_daily", "csl_stage1", "csl_daily-mt5", "csl_stage1-mt5",
        "csl_daily-pose+csl_stage1-mt5", "csl_stage1-pose+csl_daily-mt5",
        "how2sign-pose+csl_daily-mt5", "wlasl-pose+csl_daily-mt5",
        "csl_daily-pose+how2sign-mt5",
    ]
    for init in FOLD_ROWS:
        m = NAME.get(init)
        accs = byfold.get(init, {})
        if not m or not accs:
            continue
        for f, v in accs.items():
            emit(f"{m}Fold{['Zero', 'One', 'Two'][f]}", f"{v:.1f}")
        if len(accs) > 1:
            emit(f"{m}Spread", f"{max(accs.values()) - min(accs.values()):.1f}")
    # EncBestGap ranges over the encoder-only axis alone — the whole rows above
    # are in FOLD_ROWS for their per-fold figures only and would otherwise win
    # this comparison trivially.
    for init in ENCODER_ONLY:
        accs = byfold.get(init, {})
        if not NAME.get(init) or not accs:
            continue
        pooled = per[init]["local"]
        gap = 100 * sum(r["hit"] for r in pooled) / len(pooled) - rnd_acc
        if len(accs) == 3 and gap > best_gap:
            best_gap, best_name = gap, NAME[init]
    if best_name:
        emit("EncBestGap", f"{best_gap:+.1f}")

    # ----------------------------------------------------- the factorial table
    # The mixed-half cells ran on folds 1 and 2 on 8/17, so Table 2 is pooled
    # over three folds from the ordinary `per` macros and no longer from the
    # `Fz` ones below. Those stay generated anyway: the fold-0 view is what
    # Appendix B compares its seeds against, and it is also the view the earlier
    # draft's claims were computed from, so keeping it live is what let the two
    # be compared instead of one quietly replacing the other. One of those
    # claims did not survive the comparison — see `Drop`/`LexDrop` below.
    FACTORIAL = [
        "csl_daily", "csl_stage1", "how2sign",
        "csl_daily-pose+csl_stage1-mt5", "csl_stage1-pose+csl_daily-mt5",
        "how2sign-pose+csl_daily-mt5", "wlasl-pose+csl_daily-mt5",
        "csl_daily-pose+how2sign-mt5",
        "csl_daily-pose", "csl_daily-mt5", "random",
    ]
    for init in FACTORIAL:
        m = NAME.get(init)
        conds = cells.get((init, 0, 0))
        if not m:
            continue
        if not conds:
            for suffix in ("FzAcc", "FzBlank", "FzVideo", "FzN"):
                emit(f"{m}{suffix}", todo(f"{init} {suffix}"))
            continue
        L = conds.get("local", [])
        emit(f"{m}FzAcc", f"{acc(L):.1f}")
        emit(f"{m}FzN", len(L))
        pr = paired(L, conds.get("blank_plain", []))
        if pr:
            emit(f"{m}FzBlank", f"{acc(pr[1]):.1f}")
            emit(f"{m}FzVideo", f"{acc(pr[0]) - acc(pr[1]):+.1f}")
        else:
            emit(f"{m}FzBlank", todo(f"{init} blank_plain f0"))
            emit(f"{m}FzVideo", todo(f"{init} blank_plain f0 delta"))

    # Distance below the intact CSL-Daily row, pooled over three folds. The
    # fold-0 twin for the lexical set is `LexDropFz`, further down, and the pair
    # is why the paper's second-strongest claim changed on 8/17. On fold 0 an ASL
    # pose branch cost 1 point on the referential set against 11 on the lexical
    # one, and the draft reported that as an order-of-magnitude dissociation
    # between the two diagnostics. Pooled it is `CrossHowCslDrop` against
    # `CrossHowCslLexDrop` — same direction, nothing like the same size, because
    # fold 0 was the one fold where the referential gap nearly vanished. The
    # dissociation that survives is in the video contribution, not the accuracy,
    # so the prose leads with `Video` and quotes these two as what does *not*
    # separate. Generated rather than subtracted in the prose for exactly the
    # reason this paragraph exists.
    ref_pool = per.get("csl_daily", {}).get("local", [])
    for init in FACTORIAL:
        m = NAME.get(init)
        L = per.get(init, {}).get("local", [])
        if m and ref_pool and L:
            emit(f"{m}Drop", f"{acc(ref_pool) - acc(L):.1f}")

    # ------------------------------------------- dev cross-entropy (fold 0)
    # `best_dev` from each run's log.json: teacher-forced cross-entropy on the
    # reference translations, i.e. a measure of written-language modelling that
    # owes nothing to the pronoun items.
    #
    # It is in the paper for one reason. A four-way choice among Chinese
    # pronouns could be suspected of simply rewarding whichever model writes
    # Chinese best, which would make the decoder result circular. These figures
    # are what shows it is not: `random` attains the *lowest* cross-entropy of
    # any system and still sits twenty points below CSL-Daily on the diagnostic.
    # Every named cell, not just the factorial: the body's one table carries a
    # dev-CE column for all of them, and a row whose accuracy is printed while
    # its cross-entropy is a placeholder invites exactly the "is this just
    # fluency?" reading the column exists to close.
    for init in dict.fromkeys(list(NAME)):
        m = NAME.get(init)
        p = ROOT / "runs" / f"f0_local_{init}_full_k4_s0" / "log.json"
        if not m:
            continue
        if not p.exists():
            emit(f"{m}Dev", todo(f"{init} Dev"))
            continue
        # 8/23: gate on epochs_done. `log.json` is rewritten EVERY epoch, so
        # unlike the accuracy macros -- which read eval files that only exist
        # once a cell is finished -- this one will happily print the best dev CE
        # of a run that is four epochs in. `watch_numbers.sh` regenerates every
        # two minutes while the queue is live, so without this gate a
        # half-trained number lands in numbers.tex and looks exactly like a
        # result. Same rule as run_wrongclip.sh's `cell_done`: 12 epochs unless
        # the run name says otherwise.
        lg = json.loads(p.read_text())
        want = 24 if init.endswith("_e24") else 20 if init.endswith("_e20") else 12
        if int(lg.get("epochs_done", 0)) < want:
            emit(f"{m}Dev", todo(f"{init} Dev: {lg.get('epochs_done', 0)}/{want} epochs"))
            continue
        emit(f"{m}Dev", f"{lg['best_dev']:.2f}")

    # ------------------------------------------------------------ seed variance
    # Fold 0 only: that is the fold the reruns were done on, and it is the fold
    # where two encoder-only rows made their excursion. `Seeds` is how many
    # seeds a row actually has, so a macro can never imply more evidence than
    # exists — a row with one seed emits a spread of \TODO rather than 0.0.
    for init in SEED_ROWS:
        m = NAME.get(init)
        if not m:
            continue
        got = sorted((s, conds) for (i, s, f), conds in cells.items()
                     if i == init and f == 0)
        emit(f"{m}Seeds", len(got))
        if len(got) < 2:
            for suffix in ("SeedLo", "SeedHi", "SeedSpread", "SeedVideoSpread"):
                emit(f"{m}{suffix}", todo(f"{init} {suffix}"))
            continue
        accs = [acc(c.get("local", [])) for _, c in got]
        accs = [x for x in accs if x is not None]
        emit(f"{m}SeedLo", f"{min(accs):.1f}")
        emit(f"{m}SeedHi", f"{max(accs):.1f}")
        emit(f"{m}SeedSpread", f"{max(accs) - min(accs):.1f}")
        # The video contribution is the column that actually decides whether a
        # run found a solution that reads the clip, so its spread across seeds
        # is the number that says whether that outcome is a coin flip.
        vids = [acc(c["local"]) - acc(c["blank_plain"]) for _, c in got
                if c.get("local") and c.get("blank_plain")]
        emit(f"{m}SeedVideoSpread",
             f"{max(vids) - min(vids):.1f}" if len(vids) > 1
             else todo(f"{init} SeedVideoSpread"))

    # -------------------------------------------------------------- JSL probe
    # From `data/signclip/similarity.json` (scripts/15). Kept behind a file
    # check so this script still runs before the probe has been executed.
    # `sp` is the primary concept set: glosses SignCLIP was actually trained on
    # in some language, so a miss cannot just mean the word was never seen.
    # `all` is the robustness check over every gloss, and the two do not agree
    # on everything — hence separate macros rather than one number.
    #
    # NOTE (2026-08-13): main.tex no longer references any of these. The probe
    # was moved out of the submission to `probe-reserve.tex` and is held for
    # rebuttal, so these macros exist to keep that file compilable and to keep
    # its numbers regenerating with the rest of the paper. They are deliberately
    # still emitted: a reserve whose numbers have gone stale is worse than no
    # reserve, because it would be quoted in a response without being re-derived.
    sim_path = ROOT / "data" / "signclip" / "similarity_sp.json"
    alt_path = ROOT / "data" / "signclip" / "similarity_all.json"
    if alt_path.exists():
        alt = json.loads(alt_path.read_text())
        emit("ProbeAllClips", f"{alt['n_clips']:,}".replace(",", "{,}"))
        emit("ProbeAllConcepts", f"{alt['n_candidates']:,}".replace(",", "{,}"))
        emit("ProbeAllJslRank", alt["focus"]["jsl"]["panel_rank"])
        emit("ProbeAllJslCentredRank", alt["centred_focus_rank"]["jsl"])
        emit("ProbeAllCslRank", alt["focus"]["csl"]["panel_rank"])
        emit("ProbeAllDJslCsl", f"{alt['pairs']['jsl-csl']['delta']:+.4f}")
        emit("ProbeAllDJslCsllo", f"{alt['pairs']['jsl-csl']['ci'][0]:+.4f}")
        emit("ProbeAllDJslCslhi", f"{alt['pairs']['jsl-csl']['ci'][1]:+.4f}")
    if sim_path.exists():
        sim = json.loads(sim_path.read_text())
        pretty = {"jsl": "Jsl", "csl": "Csl", "ase": "Ase", "gsg": "Gsg"}
        emit("ProbeClips", f"{sim['n_clips']:,}".replace(",", "{,}"))
        emit("ProbeConcepts", f"{sim['n_candidates']:,}".replace(",", "{,}"))
        emit("ProbeLangs", len(sim["languages"]))
        emit("ProbeChanceMRR", f"{sim['chance_mrr']:.4f}")
        emit("ProbeChanceRone", f"{100 / sim['n_candidates']:.2f}")
        for code, m in pretty.items():
            f = sim["focus"][code]
            emit(f"Probe{m}MRR", f"{f['mrr']:.4f}")
            emit(f"Probe{m}CIlo", f"{f['ci'][0]:.4f}")
            emit(f"Probe{m}CIhi", f"{f['ci'][1]:.4f}")
            emit(f"Probe{m}Rone", f"{f['r1']:.2f}")
            emit(f"Probe{m}Rank", f["panel_rank"])
            emit(f"Probe{m}Clips", f"{f['train_clips']:,.0f}".replace(",", "{,}"))
            emit(f"Probe{m}Wins", f"{sim['wins_pct'][code]:.1f}")
            emit(f"Probe{m}CentredRank", sim["centred_focus_rank"][code])
            emit(f"Probe{m}Centred", f"{sim['centred'][code]:.4f}")
        for k, v in sim["pairs"].items():
            m = "".join(pretty[x] for x in k.split("-"))
            emit(f"ProbeD{m}", f"{v['delta']:+.4f}")
            emit(f"ProbeD{m}lo", f"{v['ci'][0]:+.4f}")
            emit(f"ProbeD{m}hi", f"{v['ci'][1]:+.4f}")
        for k, v in sim["sign_tests"].items():
            m = "".join(pretty[x] for x in k.split("-"))
            emit(f"ProbeSign{m}Win", v["wins"])
            emit(f"ProbeSign{m}Loss", v["losses"])
            emit(f"ProbeSign{m}Z", f"{v['z']:+.2f}")
        emit("ProbePermLo", f"{sim['permuted_mrr_range'][0]:.4f}")
        emit("ProbePermHi", f"{sim['permuted_mrr_range'][1]:.4f}")
        emit("ProbeSplitHalf", f"{sim['split_half_spearman']:+.3f}")
        emit("ProbeVolRho", f"{sim['volume_spearman']:+.3f}")
        emit("ProbeVolRhoBig", f"{sim['volume_spearman_welltrained']:+.3f}")
        emit("ProbeSpreadRho", f"{sim['spread_spearman']:+.3f}")
        emit("ProbeMedianRank", f"{sim['focus']['jsl']['median_rank']:.0f}")
    else:
        # A comment, not `\TODO`: nothing in the submission cites these, so a
        # missing probe file is not an unresolved paper number and must not make
        # `--check` fail. It only means `probe-reserve.tex` cannot be compiled
        # until scripts/15 has been run.
        out.append("% probe macros absent: data/signclip/similarity_sp.json "
                   "not found (affects probe-reserve.tex only)")

    # -------------------------------------------------------- free generation
    # Beam-4 decoding of the entire fold-0 test split (`run_gen_sweep.sh`).
    # Two quantities, and the second is the one that carries an argument.
    #
    # BLEU is here because a reviewer will ask for a translation metric, but at
    # this corpus size every row is near zero and no reading of the *absolute*
    # numbers is supportable — only the ordering is.
    #
    # The share of distinct outputs is what free decoding genuinely adds.
    # Contrastive scoring can only rank four supplied candidates, so it cannot
    # see a model emitting the same string for every clip; decoding can. That
    # makes it a direct test of the collapse diagnosis rather than a restatement
    # of it.
    sys.path.insert(0, str(ROOT))
    from src.tsl.metrics import corpus_bleu_chrf  # noqa: E402
    # Added 8/23. The six baselines were the only rows with free-decoding macros,
    # which was fine while free decoding was a baseline-only column. It is not
    # any more: the degeneracy contrast (a stuck token against sentences) is
    # carried entirely by the head-swap cells, and the abolition cell is the one
    # place where a Delta of exactly zero and a BLEU of exactly zero agree. A
    # cell with no `eval_gen.json` yet emits \TODO, which is the point of \TODO.
    for init in ("csl_daily", "csl_stage1", "random",
                 "how2sign", "openasl", "wlasl",
                 "how2sign+mt5_base-lm_head", "csl_daily+mt5_base-lm_head",
                 "how2sign+csl_daily-lm_head", "csl_daily+how2sign-lm_head",
                 "how2sign+csl_daily-mt5_nohead",
                 # Camera-ready: the row-permutation control belongs here for the
                 # same reason the head-swap cells do. Section 3.5 reads the
                 # degeneracy ordering off free decoding, and a donor that scores
                 # at chance is only interesting if it also fails to *write* --
                 # contrastive ranking alone cannot show that.
                 "how2sign+mt5_base_perm-lm_head",
                 "how2sign+mt5_base_permws-lm_head"):
        m = NAME[init]
        p = ROOT / "runs" / f"f0_local_{init}_full_k4_s0" / "eval_gen.json"
        # RELEASE: eval_gen.json pairs every hypothesis with its corpus
        # reference, so it is not part of the public artifact. BLEU cannot be
        # recomputed without references under any redaction, so the statistics
        # it fed ship precomputed in data/gen_metrics.json instead.
        if not p.exists() and p.parent.name in GEN_METRICS:
            gm = GEN_METRICS[p.parent.name]
            emit(f"{m}Bleu", f"{gm['bleu']:.2f}")
            emit(f"{m}BleuNorm", f"{gm['bleu_norm']:.2f}")
            emit(f"{m}Uniq", f"{gm['uniq']:.1f}")
            emit(f"{m}Modal", f"{gm['modal']:.1f}")
            emit(f"{m}GenN", f"{gm['n']:,}".replace(",", "{,}"))
            emit(f"{m}Han", f"{gm['han']:.1f}")
            emit(f"{m}HanChar", f"{gm['han_char']:.1f}")
            continue
        if not p.exists():
            for suffix in ("Bleu", "BleuNorm", "Uniq", "Modal", "GenN",
                           "Han", "HanChar"):
                emit(f"{m}{suffix}", todo(f"{init} {suffix}"))
            continue
        g = [r for r in json.loads(p.read_text())["generation"]
             if r["condition"] == "local"]
        hyps = [r["hyp"] for r in g]
        sc = corpus_bleu_chrf(hyps, [r["ref"] for r in g])
        counts = collections.Counter(hyps)
        emit(f"{m}Bleu", f"{sc['bleu']:.2f}")
        emit(f"{m}BleuNorm", f"{sc['bleu_norm']:.2f}")
        emit(f"{m}Uniq", f"{100 * len(counts) / len(hyps):.1f}")
        emit(f"{m}Modal", f"{100 * counts.most_common(1)[0][1] / len(hyps):.1f}")
        emit(f"{m}GenN", f"{len(hyps):,}".replace(",", "{,}"))
        # 8/20, from review: a reviewer asked whether the mismatched systems are
        # failing to reach Chinese at all -- an English-output lm_head that
        # cannot score Chinese candidates would explain the near-chance rows
        # without any appeal to visual transfer. Two shares answer it: how many
        # hypotheses contain a Han character, and what fraction of all emitted
        # characters are Han. The first says whether the script is reachable, the
        # second whether anything but punctuation comes out.
        han = sum(1 for h in hyps if HAN.search(h))
        chars = [c for h in hyps for c in h]
        emit(f"{m}Han", f"{100 * han / len(hyps):.1f}")
        emit(f"{m}HanChar", f"{100 * sum(1 for c in chars if HAN.match(c)) / max(len(chars), 1):.1f}")

    # --------------------------------------------------------- shortcut gate
    for name, key in (("Majority", "majority"), ("LMprior", "lm_prior"),
                      ("LMcontext", "lm_context")):
        h = short["hits"][key]
        typ = {i["key"]: i["type"] for i in short["items"]}
        for tag, t in (("Ana", "anaphoric"), ("Dei", "deictic")):
            v = [x for k, x in h.items() if typ[k] == t]
            emit(f"{name}{tag}", pct(sum(v) / len(v)))
            emit(f"{name}{tag}N", len(v))
    emit("HardN", len(hard))
    emit("ItemsN", short["n_items"])
    # Share of the item set carried by its single commonest gold answer. The
    # paper puts this beside the lexical set's \LexTopPct, so the two must be
    # the same quantity: how skewed the answer distribution is, not how well
    # any baseline does on it.
    gl = collections.Counter(i["gloss"] for i in short["items"] if i["gloss"])
    emit("PronTopPct", f"{100 * gl.most_common(1)[0][1] / max(1, len(short['items'])):.1f}")

    # The same three baselines on the lexical set, which has one item type and so
    # takes no Ana/Dei split. `LexLMprior` is the one to read: the conditions this
    # set is evaluated under (local, blank_plain) are scored with PROMPT_PLAIN,
    # and the distractors are rank-balanced under that prompt, so a text-only LM
    # sits at chance there by construction. `LexLMcontext` is measured under
    # PROMPT_CTX, which no reported condition for this set uses; it is quoted only
    # to say what the preceding Chinese would have given away had we used it.
    lex_short = ROOT / "data" / "lex" / "shortcuts.json"
    if lex_short.exists():
        ls = json.loads(lex_short.read_text())
        # `*_sum` keys appear once 07_shortcuts.py has been re-run with both
        # normalisations; the paper cites the sum figure to say that the rank
        # balance is a property of (item set x scoring rule), not of the set.
        for name, key in (("LexMajority", "majority"), ("LexLMprior", "lm_prior"),
                          ("LexLMcontext", "lm_context"),
                          ("LexLMpriorSum", "lm_prior_sum"),
                          ("LexLMcontextSum", "lm_context_sum")):
            if key not in ls["hits"]:
                for suffix in ("", "CIlo", "CIhi"):
                    emit(f"{name}{suffix}", todo(f"shortcuts {key}"))
                continue
            v = list(ls["hits"][key].values())
            emit(name, pct(sum(v) / len(v)))
            # The interval matters for lm_prior specifically: the claim is that
            # the balance puts a text-only LM at chance, and "25.7%" alone does
            # not say whether 25.0 is inside the interval. Emitted for all three
            # so no one has to remember which of them carries a claim.
            lo, hi = wilson(sum(v), len(v))
            emit(f"{name}CIlo", f"{lo:.1f}")
            emit(f"{name}CIhi", f"{hi:.1f}")
        emit("LexHardN", sum(1 for x in ls["hits"]["lm_prior"].values() if not x))
        emit("LexCtxK", ls.get("ctx_k", 4))

    # Why a tolerance cannot do the job, as a number rather than an assertion.
    # Both figures re-select distractors from each item's own pool using the
    # cached plain-prompt scores, and ask how often the gold would still be the
    # likeliest of the four:
    #   Freq  the rule the superseded first pass used -- frequency-closest among
    #         the words inside the tolerance window. Reproduces the 49.4% that
    #         07_shortcuts.py measured on that set, which is what says this
    #         reconstruction is faithful.
    #   Near  the most favourable tolerance-based rule there is -- the three
    #         nearest in likelihood, tolerance as tight as the pool allows.
    # Both are far above chance, which is the point: the window fixes the size
    # of the gap and the argmax turns on its sign.
    lex_cache = ROOT / "data" / "lex" / "lm_scores.json"
    lex_recs = ROOT / "data" / "lex" / "records.jsonl"
    if lex_cache.exists() and lex_recs.exists():
        sc_cache = json.loads(lex_cache.read_text())
        rows = [r for r in (json.loads(l) for l in lex_recs.open())
                if r.get("item_type") == "lexical"]
        tau = max(abs(r["lm_gap"]) for r in rows)
        got = {"Freq": [0, 0], "Near": [0, 0]}
        for r in rows:
            gold_txt = r["candidates"][0]
            if gold_txt not in sc_cache:
                continue
            gs = sc_cache[gold_txt]
            lg = math.log(max(1, r.get("gold_freq") or 1))
            freq = dict(zip(r["distractor_pool"], r["pool_freq"]))
            cand = []
            for w in r["distractor_pool"]:
                c = gold_txt.replace(r["gold_word"], w, 1)
                if c in sc_cache:
                    cand.append((abs(sc_cache[c] - gs), sc_cache[c], w))
            fluent = [x for x in cand if x[0] <= tau]
            picks = {
                "Near": sorted(cand)[:3],
                "Freq": sorted(fluent, key=lambda x: abs(
                    math.log(max(1, freq.get(x[2], 1))) - lg))[:3],
            }
            for k, p in picks.items():
                if len(p) == 3:
                    got[k][0] += all(s < gs for _, s, _ in p)
                    got[k][1] += 1
        for k, (h, n) in got.items():
            if n:
                emit(f"LexTau{k}", f"{100 * h / n:.0f}")

    # -------------------------------------------- lexical set: system results
    # A second diagnostic over the same trained cells, so nothing here is a new
    # model — `05_eval.py` re-scores finished likelihood models against a
    # different candidate set. Only `local` and `blank_plain` were run: they are
    # the one-factor pair, and both are scored under PROMPT_PLAIN, which is the
    # prompt the distractors were rank-balanced under. `text_only` is the single
    # condition sitting in CTX_TEXT and would have brought an uncontrolled text
    # shortcut (66% for a text-only LM) into the same column as two conditions
    # without one, so it is deliberately absent from this set.
    # The post-hoc graft files ride along here rather than in their own loader:
    # they are the SAME 940-item lexical set (05_eval.py --data data/lex), and
    # `row_label` gives them an `@post-` suffix so they land in their own cells
    # instead of averaging into the row they are an intervention on. Their
    # conditions are `local` and the three wrong-clip draws, so the `LexBlank`
    # family comes out as \TODO for them and is never cited.
    POSTHOC_FILES = sorted({Path(q).name for q in glob.glob(
        str(ROOT / "runs" / "f[012]_local_*" / "eval_posthoc_*.json"))})
    lex_cells = load_cells("eval_lex.json", *POSTHOC_FILES)
    lex_per, lex_folds, lex_byfold = pool(lex_cells, seed=0)
    # The within-paragraph wrong clip is a separate readout of the same cells and
    # is kept in its own dict: merging it into `lex_cells` would put a
    # `swap_within@` condition in the bucket the `Lex` macros read `local` from,
    # and two `local` blocks from two files would double the item count.
    within_cells = load_cells("eval_within.json")
    within_per, _, _ = pool(within_cells, seed=0)
    for init, conds in lex_per.items():
        m = NAME.get(init)
        if not m:
            continue
        rows_L = conds.get("local", [])
        L = [r["hit"] for r in rows_L]
        if not L:
            continue
        lo, hi = wilson(sum(L), len(L))
        emit(f"{m}LexLocal", pct(sum(L) / len(L)))
        emit(f"{m}LexCIlo", f"{lo:.1f}")
        emit(f"{m}LexCIhi", f"{hi:.1f}")
        emit(f"{m}LexN", len(L))
        emit(f"{m}LexFolds", len(lex_folds[init]))
        # The same rows under summed rather than mean token log-probability.
        # `05_eval.py` records both per item, so this is a re-read of finished
        # runs, not a rerun; the appendix uses it to show the ordering does not
        # depend on the normalisation the item set was balanced under.
        emit(f"{m}LexLocalSum",
             f"{100 * sum(r['hit_sum'] for r in rows_L) / len(rows_L):.1f}")
        pr = paired(rows_L, conds.get("blank_plain", []))
        if pr:
            emit(f"{m}LexBlank", f"{acc(pr[1]):.1f}")
            emit(f"{m}LexVideo", f"{acc(pr[0]) - acc(pr[1]):+.1f}")
            # Unsigned twin, for the prose that carries the direction in words.
            # S3.2 needs it because zeroing the clip RAISES the ASL rows, and
            # "raises it by -4.3 points" is not a sentence.
            emit(f"{m}LexVideoAbs", f"{abs(acc(pr[0]) - acc(pr[1])):.1f}")
            emit(f"{m}LexVideoP", fmt_p(mcnemar(*pr)))
            # The frequency shortcut this set does not close is `majority`, so
            # the honest control is the subset where that prior answers wrongly.
            # If the video effect were really the prior in disguise it would
            # vanish here; the paper quotes these to show it does not.
            adv = [[r for r in side if r["prior_adv"]] for side in pr]
            if adv[0]:
                emit(f"{m}LexAdvLocal", f"{acc(adv[0]):.1f}")
                emit(f"{m}LexAdvBlank", f"{acc(adv[1]):.1f}")
                emit(f"{m}LexAdvVideo", f"{acc(adv[0]) - acc(adv[1]):+.1f}")
                emit(f"{m}LexAdvN", len(adv[0]))
        else:
            for suffix in ("LexBlank", "LexVideo", "LexVideoP", "LexAdvLocal",
                           "LexAdvBlank", "LexAdvVideo", "LexAdvN"):
                emit(f"{m}{suffix}", todo(f"{init} lexical blank_plain"))
    # Fold 0 for the factorial. Table 2 is pooled over three folds as of 8/17;
    # these are kept for Appendix B's fold-0 comparisons and as the record of
    # what the earlier draft's figures were.
    for init in FACTORIAL:
        m = NAME.get(init)
        conds = lex_cells.get((init, 0, 0))
        if not m:
            continue
        if not conds:
            for suffix in ("LexFzAcc", "LexFzBlank", "LexFzVideo", "LexFzN"):
                emit(f"{m}{suffix}", todo(f"{init} {suffix}"))
            continue
        L, B = conds.get("local", []), conds.get("blank_plain", [])
        emit(f"{m}LexFzAcc", f"{acc(L):.1f}")
        emit(f"{m}LexFzN", len(L))
        if B:
            emit(f"{m}LexFzBlank", f"{acc(B):.1f}")
            emit(f"{m}LexFzVideo", f"{acc(L) - acc(B):+.1f}")
        # Distance below the intact CSL-Daily cell on the same fold. Computed
        # here rather than left as a subtraction in the prose, where a later
        # rerun would silently make the sentence wrong — which is precisely what
        # the folds-1-and-2 sweep did to the sentence this used to support.
        ref = lex_cells.get(("csl_daily", 0, 0), {}).get("local", [])
        if ref and L:
            emit(f"{m}LexDropFz", f"{acc(ref) - acc(L):.1f}")

    # The pooled twin, and the one the body now quotes. See the note by `Drop`.
    lex_ref_pool = lex_per.get("csl_daily", {}).get("local", [])
    for init in FACTORIAL:
        m = NAME.get(init)
        L = lex_per.get(init, {}).get("local", [])
        if m and lex_ref_pool and L:
            emit(f"{m}LexDrop", f"{acc(lex_ref_pool) - acc(L):.1f}")

    # Per-fold on the lexical set, the mirror of the referential `Fold`/`Spread`
    # macros above. Worth having separately rather than assuming the two sets
    # move together, because they do not: the ASL-encoder cells swing
    # `CrossHowCslSpread` points across folds on pointing signs and
    # `CrossHowCslLexSpread` on content words. That is the same asymmetry the
    # encoder-only excursion shows — the unstable thing is a pointing cue, so it
    # is the referential set that sees it move — and Appendix B needs both halves
    # to say so.
    for init in FOLD_ROWS:
        m = NAME.get(init)
        accs = lex_byfold.get(init, {})
        if not m or not accs:
            continue
        for f, v in accs.items():
            # `FW` is not bound until the excursion section below; the referential
            # twin of this loop spells the list out for the same reason.
            emit(f"{m}LexFold{['Zero', 'One', 'Two'][f]}", f"{v:.1f}")
        if len(accs) > 1:
            emit(f"{m}LexSpread", f"{max(accs.values()) - min(accs.values()):.1f}")

    # ------------------------------------------ the two rungs of the encoder
    # Holding the decoder at CSL-Daily's, the pose branch's source moves the row
    # twice: untrained -> ASL-pretrained -> CSL-pretrained. Both steps tested on
    # the same items, pooled, seed 0, so the p is a paired McNemar between two
    # *cells* rather than between two conditions of one cell.
    #
    # These four numbers are the new form of the paper's claim about its two item
    # sets. They agree on the second rung (`EncSourceLexGap` against
    # `EncSourceGap`) and disagree by an order of magnitude on the first: an ASL
    # pose branch buys `EncPretrainLexGap` on content words and
    # `EncPretrainGap` on pointing signs. The draft had the dissociation on the
    # second rung, which is where fold 0 alone put it.
    LADDER = [("EncSource", "csl_stage1-pose+csl_daily-mt5",
               "how2sign-pose+csl_daily-mt5"),
              ("EncPretrain", "how2sign-pose+csl_daily-mt5", "csl_daily-mt5")]
    for tag, hi_init, lo_init in LADDER:
        # "" is the referential set, matching the unprefixed macros elsewhere.
        for suffix, C in (("", per), ("Lex", lex_per)):
            pr = paired(C.get(hi_init, {}).get("local", []),
                        C.get(lo_init, {}).get("local", []))
            if not pr:
                emit(f"{tag}{suffix}Gap", todo(f"{tag}{suffix} gap"))
                emit(f"{tag}{suffix}GapAbs", todo(f"{tag}{suffix} gap"))
                emit(f"{tag}{suffix}P", todo(f"{tag}{suffix} p"))
                continue
            d = acc(pr[0]) - acc(pr[1])
            emit(f"{tag}{suffix}Gap", f"{d:+.1f}")
            # Unsigned twin: the prose says "worth N points more", where a
            # leading + would read as a typo rather than as a direction.
            emit(f"{tag}{suffix}GapAbs", f"{abs(d):.1f}")
            emit(f"{tag}{suffix}P", fmt_p(mcnemar(*pr)))

    # ------------------------------------------- the encoder-only excursion
    # Two encoder-only rows — the How2Sign and WLASL pose branches on an
    # untouched mT5-base — reach ~70-74% on some runs and sit at `random`'s ~57%
    # on others, and that bimodality is the whole of their pooled advantage over
    # `random`.
    #
    # The comment here used to call it the *fold-0* excursion, because seeds 1
    # and 2 existed on fold 0 only and on fold 0 all three of them departed. The
    # 8/17 sweep filled in seeds 1 and 2 on folds 1 and 2 and the fold reading
    # did not survive: the high mode appears on every fold, on three of three
    # seeds on fold 0 and one of three on each of the others. What fold 0 changes
    # is how often the optimiser lands there, not whether it can — so this is a
    # seed effect with a fold-dependent rate, and `SeedAccs{Fold}` below is the
    # measurement that says so.
    #
    # This is also the case for generating rather than typing every figure in the
    # appendix: the previous comment here predicted this exact sweep and warned
    # that a hand-typed "three seeds on fold 0, one elsewhere" would survive it
    # as a false sentence. It would also have taken the section title with it.
    EXC = ["how2sign-pose", "wlasl-pose", "csl_daily-pose", "openasl-pose",
           "csl_stage1-pose", "random", "csl_daily"]
    FW = ["Zero", "One", "Two"]
    # The person split and the item-type split are the same split. Measured on
    # all three folds: every deictic item is 1st or 2nd person (我/我們/你/妳/
    # 你們) and every anaphoric item is 3rd (他/她/牠/他們), with no exceptions
    # in 975 items. That is not a coincidence but the construction — 1st and 2nd
    # person are body-anchored, so their referent is present by definition,
    # while 3rd person needs a locus assigned earlier — and it means the
    # deictic/anaphoric contrast anywhere in this paper is also a person
    # contrast. The excursion is reported on item type because that is the
    # paper's existing vocabulary; PERSON_12 is kept only to assert the
    # equivalence, so that a later change to the item builder that breaks it
    # cannot pass unnoticed.
    # RELEASE: the items ship with person labels in place of the Chinese
    # pronoun glosses, so the assertion below tests the same equivalence
    # against the same items using the labels the public records carry.
    PERSON_12 = {"1SG", "1PL", "2SG", "2PL"}
    for (i, s, f), conds in cells.items():
        for r in conds.get("local", []):
            want = "deictic" if r["pron_gloss"] in PERSON_12 else "anaphoric"
            if r["item_type"] != want:
                raise SystemExit(
                    f"person/item-type equivalence broken at {r['key']}: "
                    f"{r['pron_gloss']} is {r['item_type']}. The paper says "
                    "the deictic half is 1st/2nd person; fix the text or the "
                    "item builder before regenerating.")
    for init in EXC:
        m = NAME.get(init)
        if not m:
            continue
        for f in (0, 1, 2):
            conds = cells.get((init, 0, f))
            if not conds:
                continue
            L = conds.get("local", [])
            pr = paired(L, conds.get("blank_plain", []))
            if pr:
                emit(f"{m}Video{FW[f]}", f"{acc(pr[0]) - acc(pr[1]):+.1f}")
            # How many seeds this (row, fold) actually has. The excursion is
            # currently three seeds on fold 0 against one on each of the others,
            # and that asymmetry is the reason the fold reading is provisional.
            emit(f"{m}Seeds{FW[f]}",
                 sum(1 for (i, s, ff) in cells if i == init and ff == f))
            # Dev cross-entropy is measured on paragraphs disjoint from both
            # train and test, so it is the check that the excursion is a
            # property of the fitted model and not of the scoring.
            p = ROOT / "runs" / f"f{f}_local_{init}_full_k4_s0" / "log.json"
            if p.exists():
                lg = json.loads(p.read_text())
                emit(f"{m}Dev{FW[f]}", f"{lg['best_dev']:.2f}")
                # Dev after the first epoch, i.e. where a run starts from. One
                # of the three accounts of the excursion is that folds 1 and 2
                # simply had further to travel in the same 12 epochs, and this
                # is the figure that account rests on: it is in the paper, so
                # it is generated rather than read off a log by hand.
                emit(f"{m}DevFirst{FW[f]}", f"{lg['log'][0]['dev']:.2f}")
                emit(f"{m}Epochs{FW[f]}", lg["epochs_done"])
                # First epoch within half a nat of this run's *own* final dev,
                # i.e. when it stopped catching up and started refining. Scale-
                # free, so it compares folds that start from very different
                # places. This replaces an earlier "still descending at the last
                # epoch" reading of the same curves, which the curves do not
                # support: every run here is flat to within 0.006 nats over its
                # last three epochs. What differs is how much of the schedule
                # was spent getting there.
                settle = next((r["epoch"] for r in lg["log"]
                               if r["dev"] <= lg["best_dev"] + 0.5), None)
                emit(f"{m}Settle{FW[f]}",
                     settle if settle else todo(f"{init} settle f{f}"))
            else:
                emit(f"{m}Dev{FW[f]}", todo(f"{init} dev f{f}"))
                emit(f"{m}DevFirst{FW[f]}", todo(f"{init} dev1 f{f}"))
                emit(f"{m}Epochs{FW[f]}", todo(f"{init} epochs f{f}"))
            if f == 0 and L:
                # Fold 0 by item type. The pooled DeiAll/AnaAll macros cannot
                # be used for the excursion: pooling is what hides it.
                # Not `a`: `a` is main()'s argparse namespace, and rebinding it
                # here makes the name local to the whole function, so the write
                # at the end of main() raises AttributeError on a list. Same
                # trap as the `acc` and `todo` shadowing noted below.
                for tag, want in (("Dei", "deictic"), ("Ana", "anaphoric")):
                    sel = [r["hit"] for r in L if r["item_type"] == want]
                    if sel:
                        emit(f"{m}{tag}Fz", pct(sum(sel) / len(sel)))
                        emit(f"{m}{tag}FzN", len(sel))
        # Fold 0 on the lexical set, where the same cells are re-scored on
        # content words. The excursion does not survive there, which is what
        # makes it a pointing cue rather than a visual channel.
        lc = lex_cells.get((init, 0, 0), {}).get("local", [])
        emit(f"{m}LexFzOnly", f"{acc(lc):.1f}" if lc else todo(f"{init} lex f0"))
        # Seed range on fold 0 for the two excursion rows, so the prose can say
        # "three seeds spanning X to Y" without any figure being typed in.
        got = [c.get("local", []) for (i, s, f), c in cells.items()
               if i == init and f == 0]
        accs = [acc(x) for x in got if x]
        if len(accs) > 1:
            emit(f"{m}FzSeedLo", f"{min(accs):.1f}")
            emit(f"{m}FzSeedHi", f"{max(accs):.1f}")

    # ---- every seed of every fold, for the two rows that go bimodal
    # The measurement the earlier draft named as the evidence it was missing:
    # "the distinguishing evidence is seeds on the other folds". It arrived, and
    # it moved the excursion from a fold property to a seed one.
    #
    # Emitted as slash-joined strings rather than one macro per (row, fold, seed)
    # for two reasons. The seed counts are ragged — `random` has three seeds on
    # fold 0, two on fold 1, one on fold 2 — so a table cell citing three macros
    # would typeset a \TODO inside a row wherever a run does not exist; joining
    # here keeps the ragged case correct. And the *number* of seeds stays out of
    # main.tex, where it would be the one literal that a further sweep silently
    # falsifies, which is the failure this whole section exists to avoid.
    #
    # `SeedSpread{Fold}` is the number the argument turns on: fold 0's three
    # seeds agree to within a point and a half and folds 1 and 2 span thirteen
    # and seventeen, so the two modes are separated by the seed, not the fold.
    # Note the unsuffixed `SeedSpread` from SEED_ROWS above is the fold-0 one and
    # is what Limitations quotes; these carry the fold in the name.
    for init in ("how2sign-pose", "wlasl-pose", "random"):
        m = NAME.get(init)
        if not m:
            continue
        for f in (0, 1, 2):
            seen = sorted((s, c) for (i, s, ff), c in cells.items()
                          if i == init and ff == f)
            xs, vs = [], []
            for _, c in seen:
                L = c.get("local", [])
                if not L:
                    continue
                xs.append(acc(L))
                pr = paired(L, c.get("blank_plain", []))
                vs.append(acc(pr[0]) - acc(pr[1]) if pr else None)
            if not xs:
                continue
            emit(f"{m}SeedAccs{FW[f]}", "/".join(f"{x:.1f}" for x in xs))
            emit(f"{m}SeedVideos{FW[f]}",
                 "/".join("--" if v is None else f"{v:+.1f}" for v in vs))
            emit(f"{m}SeedHi{FW[f]}", f"{max(xs):.1f}")
            emit(f"{m}SeedLo{FW[f]}", f"{min(xs):.1f}")
            emit(f"{m}SeedSpread{FW[f]}", f"{max(xs) - min(xs):.1f}")

    # ---- the off-protocol long runs, read by name and pooled with nothing
    # The third account of the excursion is that a fixed 12-epoch schedule is not
    # enough on folds 1 and 2, which start from a worse dev cross-entropy and
    # spend most of the schedule catching up. Two cells test it directly: the
    # same (row, fold, seed 0) configuration at 20 epochs instead of 12.
    #
    # They are deliberately off-protocol — every figure the tables compare is at
    # a matched budget — so `CANONICAL_RUN` keeps them out of `load_cells` and
    # they are opened here by explicit path instead. That separation is the point:
    # the guard exists because these two files silently pooled into their
    # 12-epoch namesakes the first time they were generated, and reading them
    # under their own macro names is what makes the deliberate use possible
    # without reopening that door. Nothing here reaches a table.
    LONG_SUFFIX = "_e20"
    for init in ("how2sign-pose",):
        m = NAME.get(init)
        if not m:
            continue
        for f in (1, 2):
            d = ROOT / "runs" / f"f{f}_local_{init}_full_k4_s0{LONG_SUFFIX}"
            ev, bl = d / "eval.json", d / "eval_blank.json"
            L = [r for r in json.loads(ev.read_text())["contrastive"]
                 if r["condition"] == "local"] if ev.exists() else []
            B = [r for r in json.loads(bl.read_text())["contrastive"]
                 if r["condition"] == "blank_plain"] if bl.exists() else []
            pr = paired(L, B)
            if not pr:
                # A placeholder rather than a skip: the appendix cites these two
                # cells by name, so a missing one has to reach the page as a
                # marker and fail `--check`, not vanish into an undefined macro
                # that stops the build with a LaTeX error instead.
                for suffix in ("Long", "LongVideo", "LongEpochs"):
                    emit(f"{m}{suffix}{FW[f]}", todo(f"{init} 20ep f{f}"))
                continue
            emit(f"{m}Long{FW[f]}", f"{acc(pr[0]):.1f}")
            emit(f"{m}LongVideo{FW[f]}", f"{acc(pr[0]) - acc(pr[1]):+.1f}")
            lg = d / "log.json"
            emit(f"{m}LongEpochs{FW[f]}",
                 json.loads(lg.read_text())["epochs_done"] if lg.exists()
                 else todo(f"{init} 20ep epochs f{f}"))

    # ---- what fold 0 is not: composition, and difficulty, are flat across folds
    # The first account of the excursion is that fold 0's items differ. They do
    # not, and the direction of what difference there is runs against the
    # account: the gain is largest on 2nd-person items, and fold 0 holds the
    # *fewest* of them. Derived from the `random` cells rather than re-reading
    # items.tsv, because those records are exactly the fold's test slice.
    SECOND = {"2SG", "2PL"}          # RELEASE: see PERSON_12 above
    for f in (0, 1, 2):
        L = cells.get(("random", 0, f), {}).get("local", [])
        if not L:
            continue
        emit(f"ItemsFold{FW[f]}", len(L))
        emit(f"DeicticShareFold{FW[f]}",
             pct(sum(r["item_type"] == "deictic" for r in L) / len(L)))
        emit(f"AddresseeShareFold{FW[f]}",
             pct(sum(r["pron_gloss"] in SECOND for r in L) / len(L)))

    # ---- and what does order the rows within a fold: how well the run fitted.
    # Across the six encoder-only cells, dev cross-entropy and diagnostic
    # accuracy are rank-identical on folds 0 and 2. Fold 1 has no order to
    # recover — its six rows sit inside a point of each other — so its rho is
    # reported for completeness and carries nothing. This is the measurement
    # that moves the excursion from "fold 0 is different" to "these runs fitted
    # better, and fold 0 is where the runs spread furthest apart".
    ENC_ONLY = ["how2sign-pose", "wlasl-pose", "csl_daily-pose",
                "openasl-pose", "csl_stage1-pose", "random"]

    def _ranks(v: list[float]) -> list[int]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for j, i in enumerate(order):
            out[i] = j + 1
        return out

    for f in (0, 1, 2):
        dev, ac = [], []
        for init in ENC_ONLY:
            L = cells.get((init, 0, f), {}).get("local", [])
            p = ROOT / "runs" / f"f{f}_local_{init}_full_k4_s0" / "log.json"
            if not L or not p.exists():
                continue
            dev.append(json.loads(p.read_text())["best_dev"])
            ac.append(acc(L))
        if len(dev) < 3:
            emit(f"DevAccRho{FW[f]}", todo(f"rho f{f}"))
            emit(f"AccSpread{FW[f]}", todo(f"acc spread f{f}"))
            continue
        # Negated accuracy: lower dev should pair with *higher* accuracy, so
        # rho = +1 means "the better-fitting run is the better-scoring one".
        rx, ry = _ranks(dev), _ranks([-x for x in ac])
        n = len(dev)
        rho = 1 - 6 * sum((p - q) ** 2 for p, q in zip(rx, ry)) / (n * (n * n - 1))
        emit(f"DevAccRho{FW[f]}", f"{rho:+.2f}")
        emit(f"DevAccN{FW[f]}", n)
        emit(f"AccSpread{FW[f]}", f"{max(ac) - min(ac):.1f}")
        emit(f"DevSpread{FW[f]}", f"{max(dev) - min(dev):.2f}")

    # ------------------------------------------------------- axis 3: density
    cls = {k: v.get("density_class") for k, v in strata.items()}
    ref = "csl_daily"
    for init in per:
        m = NAME.get(init)
        if not m or init == ref or "local" not in per[init]:
            continue
        for c in ("low", "mid", "high"):
            # Not `acc`: the module-level `acc()` is used earlier in this
            # function, and a nested def of the same name makes the name local
            # to the whole of main(), so every earlier call raises
            # UnboundLocalError. Same trap as the `todo` shadowing above.
            def stratum_acc(i):
                h = [r["hit"] for r in per[i]["local"] if cls.get(r["key"]) == c]
                return (sum(h) / len(h)) if h else None
            a_ref, a_sys = stratum_acc(ref), stratum_acc(init)
            if a_ref is None or a_sys is None:
                continue
            emit(f"{m}Lor{c.capitalize()}", f"{logit(a_ref) - logit(a_sys):+.2f}")
    for c in ("low", "mid", "high"):
        n = sum(1 for r in per[ref]["local"] if cls.get(r["key"]) == c)
        emit(f"Density{c.capitalize()}N", n)

    # ------------------------------------- item-level agreement among the ASL rows
    hits = {i: {r["key"]: r["hit"] for r in c["local"]} for i, c in per.items()
            if "local" in c}
    asl = [i for i in ("how2sign", "openasl", "wlasl") if i in hits]
    if len(asl) > 1:
        keys = set.intersection(*[set(hits[i]) for i in asl])
        vals = [sum(hits[x][k] == hits[y][k] for k in keys) / len(keys)
                for n_, x in enumerate(asl) for y in asl[n_ + 1:]]
        emit("AslAgreeMin", pct(min(vals)))
        emit("AslAgreeMax", pct(max(vals)))
    # The largest video contribution any ASL row manages, for the abstract. It is
    # not zero — the two sentence-level ASL rows do read the clip a little — and
    # an abstract saying "costs them nothing" would contradict Table 1, which a
    # reviewer checks first. Quoted as a ceiling against the CSL rows instead.
    adv = []
    for init in ("how2sign", "openasl", "wlasl"):
        pr = paired(per.get(init, {}).get("local", []),
                    per.get(init, {}).get("blank_plain", []))
        if pr:
            adv.append(acc(pr[0]) - acc(pr[1]))
    emit("AslVideoMax", f"{max(adv):.1f}" if len(adv) == 3
         else todo("ASL blank_plain rows"))
    if "csl_daily" in hits and "csl_stage1" in hits:
        keys = set(hits["csl_daily"]) & set(hits["csl_stage1"])
        emit("CslAgree", pct(sum(hits["csl_daily"][k] == hits["csl_stage1"][k]
                                 for k in keys) / len(keys)))

    # The same agreement on the content-word set. The body leads with that set
    # now, so the collapse claim has to be made on it and not borrowed from the
    # pronoun items. It is weaker there and that is worth seeing: three models at
    # chance would agree on hit/miss about 63% of the time by construction, so
    # the figure to compare against is that, not zero.
    lex_hits = {i: {r["key"]: r["hit"] for r in c["local"]}
                for i, c in lex_per.items() if "local" in c}
    lex_asl = [i for i in ("how2sign", "openasl", "wlasl") if i in lex_hits]
    if len(lex_asl) > 1:
        keys = set.intersection(*[set(lex_hits[i]) for i in lex_asl])
        vals = [sum(lex_hits[x][k] == lex_hits[y][k] for k in keys) / len(keys)
                for n_, x in enumerate(lex_asl) for y in lex_asl[n_ + 1:]]
        emit("AslAgreeLexMin", pct(min(vals)))
        emit("AslAgreeLexMax", pct(max(vals)))
    if "csl_daily" in lex_hits and "csl_stage1" in lex_hits:
        keys = set(lex_hits["csl_daily"]) & set(lex_hits["csl_stage1"])
        emit("CslAgreeLex",
             pct(sum(lex_hits["csl_daily"][k] == lex_hits["csl_stage1"][k]
                     for k in keys) / len(keys)))

    # --------------------------------------------- split rows: what is missing
    for init in ("csl_daily-pose", "csl_daily-mt5",
                 "csl_stage1-pose", "csl_stage1-mt5",
                 "csl_daily-pose+csl_stage1-mt5", "csl_stage1-pose+csl_daily-mt5"):
        m = NAME[init]
        if init not in per:
            for suffix in ("Local", "Video", "Folds", "N"):
                emit(f"{m}{suffix}", todo(f"{init} {suffix}"))

    # ------------------------------------------- lexical set: iconicity check
    # Composition of the lexical item set, not a result: these describe which
    # concepts the items are built on and are fixed once the set is built, so
    # they resolve long before any run lands.
    lex = ROOT / "data" / "lex" / "records.jsonl"
    if lex.exists():
        sys.path.insert(0, str(ROOT / "src"))
        from tsl.asllex import ratings

        items = [r for r in (json.loads(l) for l in lex.open())
                 if r.get("item_type") == "lexical"]
        rated = [r["icon"] for r in items if r.get("icon") is not None]
        types = {r["gold_word"]: r["icon"] for r in items
                 if r.get("icon") is not None}
        # Anchored at ROOT, not cwd: this generator runs from writings/first,
        # where the module's own relative default resolves to nothing and the
        # population row would silently come out empty.
        # RELEASE: ASL-LEX is a third-party database and is not redistributed
        # here. See the README for where to place signdata.csv; without it the
        # population row below resolves to a placeholder.
        _asllex = ROOT / "data/external/asllex/signdata.csv"
        pop = ([v["icon"] for v in ratings(_asllex).values()]
               if _asllex.exists() else [])

        def p(v: list[float], q: float) -> str:
            s = sorted(v)
            return f"{s[int(q * (len(s) - 1))]:.2f}"

        def mu(v: list[float]) -> str:
            return f"{sum(v) / len(v):.2f}"

        for tag, v in (("Pop", pop), ("Item", rated), ("Type", list(types.values()))):
            if not v:                      # RELEASE: see the ASL-LEX note above
                for sfx in ("Lo", "Mean", "Hi"):
                    emit(f"Icon{tag}{sfx}", todo(f"Icon{tag}{sfx} needs ASL-LEX"))
                continue
            emit(f"Icon{tag}Lo", p(v, 0.33))
            emit(f"Icon{tag}Mean", mu(v))
            emit(f"Icon{tag}Hi", p(v, 0.67))
            emit(f"Icon{tag}N", len(v))
        emit("IconRated", len(rated))
        emit("IconPct", f"{100 * len(rated) / max(1, len(items)):.0f}")
        emit("LexItemsN", len(items))
        emit("LexTypesN", len({r["gold_word"] for r in items}))
        # The type itself is deliberately not emitted: the paper loads no CJK
        # package, and a macro expanding to Chinese would break the build the
        # first time someone cited it.
        top = collections.Counter(r["gold_word"] for r in items).most_common(1)
        emit("LexTopPct", f"{100 * top[0][1] / max(1, len(items)):.1f}")
        emit("LexPriorPct", f"{100 * sum(1 for r in items if r['prior_correct']) / max(1, len(items)):.1f}")
        # The rank balance, emitted rather than asserted in prose. `LexRankEach`
        # is the per-rank count; if the four ranks ever come out unequal the
        # macro turns into a \TODO instead of letting the paper keep claiming a
        # uniform distribution that the file no longer has.
        rk = collections.Counter(r.get("lm_rank") for r in items)
        emit("LexRankEach",
             rk[1] if len({rk[i] for i in (1, 2, 3, 4)}) == 1 and None not in rk
             else todo("lexical rank balance is not uniform"))
        emit("LexMaxGap", f"{max(abs(r['lm_gap']) for r in items):.2f}")
        emit("LexPoolMean", f"{sum(len(r['distractor_pool']) for r in items) / len(items):.0f}")
        # The two construction constants Appendix A now states, taken from the
        # module that built the items rather than typed, so a change to the build
        # cannot leave the appendix describing an older item set.
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from tsl import lexitems as _lx
            emit("LexMinFreq", _lx.MIN_FREQ)
            emit("LexPoolNear", _lx.POOL_NEAR)
        except Exception as exc:                            # pragma: no cover
            print(f"lexitems constants unavailable: {exc}", file=sys.stderr)
            emit("LexMinFreq", todo("lexitems MIN_FREQ"))
            emit("LexPoolNear", todo("lexitems POOL_NEAR"))

    # =================================================================== v3
    # Everything below was added on 8/18 in response to the three reviews. Each
    # block says which objection it answers, because a macro whose purpose is
    # forgotten is a macro that gets quoted in the wrong sentence later.

    # ------------------------------------------- the wrong-clip control (v3)
    # `local` - `blank_plain` was the paper's only measure of "does the model
    # read the clip", and zeroing 133 keypoints is an out-of-distribution input:
    # the ASL rows' NEGATIVE delta is exactly what an OOD reaction looks like, and
    # it carried the strongest claim in the paper. `eval_clip.json` adds two
    # in-distribution ablations over the same weights — a real clip of another
    # utterance (three draws, averaged per item) and this clip's own frames
    # permuted — so the claim rests on a control rather than on a caveat.
    # The post-hoc graft files carry `local` and the same three wrong-clip draws,
    # so they belong in this loader too: the point of grafting a head after
    # fine-tuning is partly to ask whether clip sensitivity comes with it.
    clip_cells = load_cells("eval_clip.json", *POSTHOC_FILES)
    clip_per, clip_folds, clip_byfold = pool(clip_cells, seed=0)

    def swap_rows(conds: dict) -> list | None:
        """The wrong-clip condition, averaged over draws inside each item.

        Averaging per item (rather than reporting three accuracies) keeps the
        contrast paired and stops one unlucky donor assignment carrying it.
        """
        draws = sorted(c for c in conds if c.startswith("swap_plain@"))
        if not draws:
            return None
        by_key: dict[str, list] = collections.defaultdict(list)
        for c in draws:
            for r in conds[c]:
                by_key[r["key"]].append(r["hit"])
        n = len(draws)
        return [{"key": k, "hit": sum(v) / len(v)}
                for k, v in by_key.items() if len(v) == n]

    def draw_spread(conds: dict) -> float | None:
        draws = sorted(c for c in conds if c.startswith("swap_plain@"))
        accs = [acc(conds[c]) for c in draws if conds[c]]
        return max(accs) - min(accs) if len(accs) > 1 else None

    spreads: list[float] = []
    for init, conds in clip_per.items():
        m = NAME.get(init)
        if not m:
            continue
        L = conds.get("local", [])
        if not L:
            continue
        # The clip conditions live in their own file, which lands one cell at a
        # time, so a row can hold three folds of `eval_lex.json` and two of
        # `eval_clip.json`. Accuracy would then be pooled over three folds and
        # the delta beside it over two, in the same table row, with nothing on
        # the page saying so. Refuse that: emit placeholders until the two files
        # cover the same items.
        lex_keys = {r["key"] for r in lex_per.get(init, {}).get("local", [])}
        if lex_keys and {r["key"] for r in L} != lex_keys:
            for tag in ("Swap", "Shuffle"):
                for suf in (f"Lex{tag}", f"Lex{tag}Delta", f"Lex{tag}DeltaAbs"):
                    emit(f"{m}{suf}", todo(f"{init} {tag.lower()}: clip file "
                                           "covers fewer folds than eval_lex"))
            continue
        sw = swap_rows(conds)
        # `blank_plain` is deliberately absent: the lexical section above already
        # emits it from eval_lex.json as `LexBlank`/`LexVideo`, and re-emitting it
        # here from a second file would either redefine the macro (a LaTeX error,
        # which is how this was caught) or, worse, silently disagree with it.
        for cond_rows, tag in ((sw, "Swap"),
                               (conds.get("shuffle_frames", []), "Shuffle")):
            if not cond_rows:
                for suf in (f"Lex{tag}", f"Lex{tag}Delta", f"Lex{tag}DeltaAbs"):
                    emit(f"{m}{suf}", todo(f"{init} {tag.lower()} clip condition"))
                continue
            pr = paired(L, cond_rows)
            if not pr:
                continue
            x, y = pr
            ax = 100 * sum(r["hit"] for r in x) / len(x)
            ay = 100 * sum(r["hit"] for r in y) / len(y)
            emit(f"{m}Lex{tag}", f"{ay:.1f}")
            emit(f"{m}Lex{tag}Delta", f"{ax - ay:+.1f}")
            emit(f"{m}Lex{tag}DeltaAbs", f"{abs(ax - ay):.1f}")
        d = draw_spread(conds)
        if d is not None:
            spreads.append(d)
    emit("SwapDrawSpread", f"{max(spreads):.1f}" if spreads
         else todo("wrong-clip draw spread"))
    # Paragraphs, not items, are the unit of resampling: the count belongs in the
    # paper because it is what makes the clustered interval necessary.
    lex_items_all = lex_per.get("csl_daily", {}).get("local", [])
    emit("LexClusters", len({r["key"].split(":")[0] for r in lex_items_all})
         if lex_items_all else todo("lexical paragraph count"))
    lex_rec_path = ROOT / "data" / "lex" / "records.jsonl"
    lex_golds = {r["gold_word"] for r in
                 (json.loads(l) for l in lex_rec_path.open())
                 if r.get("item_type") == "lexical"} if lex_rec_path.exists() else set()
    emit("LexTypeN", len(lex_golds) if lex_golds else todo("lexical gold types"))

    # ------------------------ the wrong-clip control, referential set (v3)
    # The same three ablations as above, asked of the referential items instead
    # of the lexical ones. This is what §3.5 rests on, and it is the reason the
    # two-instruments reading is a measurement claim rather than a story: if the
    # ASL rows lose a great deal here while losing nothing on the lexical set
    # (+0.5, CI crossing zero), then the two sets disagree about whether the clip
    # is read at all, and the disagreement is a property of the items.
    #
    # Scope is deliberate. `SCOPE=main ./run_wrongclip.sh` scores the six
    # released initialisations on three folds and nothing else, so these macros
    # exist for those rows only; any other init falls through to a placeholder
    # rather than to a number pooled over whichever folds happened to land.
    pron_cells = load_cells("eval_clip_pron.json")
    pron_per, pron_folds, _ = pool(pron_cells, seed=0)
    for init, conds in pron_per.items():
        m = NAME.get(init)
        if not m:
            continue
        L = conds.get("local", [])
        if not L:
            continue
        # Same guard as the lexical block, against the same failure: eval.json
        # covers three folds for every released row while this file lands one
        # cell at a time, and a table row carrying a three-fold accuracy beside a
        # two-fold delta says nothing on the page about the difference.
        ref_keys = {r["key"] for r in per.get(init, {}).get("local", [])}
        if ref_keys and {r["key"] for r in L} != ref_keys:
            for tag in ("Swap", "Shuffle"):
                for suf in (f"Pron{tag}", f"Pron{tag}Delta", f"Pron{tag}DeltaAbs"):
                    emit(f"{m}{suf}", todo(f"{init} referential {tag.lower()}: "
                                           "clip file covers fewer folds than eval"))
            continue
        # `blank_plain` is omitted here for the reason it is omitted above: the
        # main-table loop already emits it as `Blank`/`Video` from eval.json, and
        # a second definition is either a LaTeX error or a silent disagreement.
        sw = swap_rows(conds)
        for cond_rows, tag in ((sw, "Swap"),
                               (conds.get("shuffle_frames", []), "Shuffle")):
            if not cond_rows:
                for suf in (f"Pron{tag}", f"Pron{tag}Delta", f"Pron{tag}DeltaAbs"):
                    emit(f"{m}{suf}", todo(f"{init} referential {tag.lower()}"))
                continue
            pr = paired(L, cond_rows)
            if not pr:
                continue
            x, y = pr
            ax = 100 * sum(r["hit"] for r in x) / len(x)
            ay = 100 * sum(r["hit"] for r in y) / len(y)
            emit(f"{m}Pron{tag}", f"{ay:.1f}")
            emit(f"{m}Pron{tag}Delta", f"{ax - ay:+.1f}")
            emit(f"{m}Pron{tag}DeltaAbs", f"{abs(ax - ay):.1f}")
    emit("PronClipCells", len(pron_per) if pron_per
         else todo("referential wrong-clip sweep"))

    # ------------------------------------------------ interaction terms (v3)
    # The paper said "the decoder's written language sets the level, worth thirty
    # points", which is additive language for what is a gating interaction. These
    # four numbers are the interaction, and the prose now quotes them.
    def cell_acc(init: str) -> float | None:
        rows = lex_per.get(init, {}).get("local", [])
        return acc(rows) if rows else None

    def gap(a_init: str, b_init: str) -> str:
        x, y = cell_acc(a_init), cell_acc(b_init)
        return f"{abs(x - y):.1f}" if x is not None and y is not None else None

    for macro, (a_init, b_init) in {
        # The decoder factor at each level of the encoder factor, and both
        # contrasts hold the VISUAL half exactly constant. An earlier version
        # took the CSL-level contrast from the re-paired cell, which varies the
        # visual half (CSL-News against CSL-Daily) as well as the mT5 one and
        # therefore gave a second, slightly different number for the same
        # quantity the prose had already quoted -- which is the inconsistency the
        # reviews caught in v2.
        "DecoderOnCslEncAbs": ("csl_daily", "csl_daily-pose+how2sign-mt5"),
        "DecoderOnAslEncAbs": ("how2sign-pose+csl_daily-mt5", "how2sign"),
        # encoder factor, at each level of the decoder factor. The Chinese-decoder
        # leg is new on 8/19 and it is not `EncSourceLexGapAbs`: that macro is the
        # ladder's rung, whose CSL side is CSL-*News*, and it is the right number
        # where the paper prices its two item sets against each other
        # (`EncSourceGapAbs` is its referential twin, computed on the same pair).
        # Quoted as an interaction leg it was wrong, because the other three legs
        # are CSL-Daily's: 12.1 against 1.2 does not subtract to the 11.5 the
        # decoder axis gives. This one does. Same class of mismatch the reviews
        # caught in v2 on the decoder axis, surviving on the encoder axis.
        "EncSourceOnCslDecAbs": ("csl_daily", "how2sign-pose+csl_daily-mt5"),
        "EncSourceOnAslDecAbs": ("csl_daily-pose+how2sign-mt5", "how2sign"),
        # the single-tensor cross-load against the checkpoint it started from
        "HeadHowCslLexGapAbs": ("how2sign+csl_daily-lm_head", "how2sign"),
        "BodyHowCslLexGapAbs": ("how2sign+csl_daily-mt5_nohead", "how2sign"),
        "HeadCslHowLexGapAbs": ("csl_daily", "csl_daily+how2sign-lm_head"),
        # Oriented as "what the swap does to the recipient", which is the sign
        # Appendix M's table column needs. `HeadCslHowLexGapAbs` above is the
        # same magnitude read the other way round, because the prose says "a loss
        # of X points" and carries the direction in the word.
        "HeadCslHowLexDropAbs": ("csl_daily+how2sign-lm_head", "csl_daily"),
        "HeadGainOpenAbs": ("openasl+csl_daily-lm_head", "openasl"),
        "HeadGainNewsAbs": ("how2sign+csl_stage1-lm_head", "how2sign"),
    }.items():
        g = gap(a_init, b_init)
        emit(macro, g if g else todo(f"{a_init} vs {b_init}"))
        # Signed twin, for table cells. `gap` is unsigned because the prose
        # carries the direction in words; a table column cannot, and Appendix M
        # puts the initialization-time swaps in the same column as the post-hoc
        # ones, where a gain and a loss have to be distinguishable.
        if macro.endswith("Abs"):
            x, y = cell_acc(a_init), cell_acc(b_init)
            emit(macro[:-3],
                 f"{x - y:+.1f}" if x is not None and y is not None
                 else todo(f"{a_init} vs {b_init} signed"))

    # --------------------------------------------- clustered statistics (v3)
    # Wilson intervals over pooled items assume independence the items do not
    # have (940 items inside ~400 paragraphs). `19_bootstrap.py` resamples
    # paragraphs and permutes at the paragraph level; these macros carry its
    # output so the paper can stop quoting an interval it does not believe.
    boot_path = ROOT / "data" / "bootstrap.json"
    boot = json.loads(boot_path.read_text())["contrasts"] if boot_path.exists() else {}

    _CI_KEYS: dict[str, str] = {}

    def ci(key: str, macro: str) -> None:
        _CI_KEYS[macro] = key
        r = boot.get(key)
        if not r:
            emit(macro, todo(f"bootstrap {key}"))
            emit(f"{macro}Pclust", todo(f"bootstrap {key} p"))
            return
        # \mbox: a confidence interval must not break across lines. Without
        # it TeX will leave "([-25.5, -" at a column end and carry "17.5])"
        # to the next, which a reader sees as a dropped sign.
        emit(macro, f"\\mbox{{[{r['lo']:+.1f}, {r['hi']:+.1f}]}}")
        emit(f"{macro}Pclust", fmt_p(r["p"]))
        # The point estimate beside its interval. Added 8/19 after the third
        # fold turned `HeadResidual` from -2.4 [-7.3, +2.3] into -4.7
        # [-7.7, -1.7]: an interval quoted without the estimate it brackets let
        # a sentence saying "adds nothing detectable" survive the sign change.
        base = macro[:-2] if macro.endswith("CI") else macro
        emit(f"{base}Delta", f"{r['delta']:+.1f}")
        emit(f"{base}DeltaAbs", f"{abs(r['delta']):.1f}")

    ci("csl_stage1-pose+csl_daily-mt5::vs::how2sign-pose+csl_daily-mt5",
       "EncSourceLexCI")
    # The 2x2's encoder leg, visual halves both CSL-Daily's/How2Sign's own. See
    # the note on `EncSourceOnCslDecAbs`: this is the interval the factorial
    # claim needs, `EncSourceLexCI` above the one the two-instrument claim needs,
    # and they are different contrasts rather than two numbers for one.
    ci("csl_daily::vs::how2sign-pose+csl_daily-mt5", "EncSourceOnCslDecLexCI")
    # Same pair as `DecoderOnCslEncAbs` above, visual half held constant. Quoting
    # an interval for a neighbouring contrast is how a paper ends up with two
    # numbers for one quantity.
    ci("csl_daily::vs::csl_daily-pose+how2sign-mt5", "DecoderLexCI")
    ci("how2sign-pose+csl_daily-mt5::vs::how2sign", "DecoderAslLexCI")
    # The two independently fine-tuned English sentence-level decoders against
    # each other. This is the sharpest form of the replication a review asked
    # for: not "both land near chance" but "indistinguishable from each other".
    ci("csl_daily-pose+openasl-mt5::vs::csl_daily-pose+how2sign-mt5",
       "EnglishPairLexCI")
    ci("csl_daily-pose+wlasl-mt5::vs::csl_daily-pose+how2sign-mt5",
       "EnglishTrioLexCI")
    # The same replication on the encoder axis. Added 8/20 because S3.3 asserted
    # the three ASL-visual rows were indistinguishable "as do[es]" the English
    # trio while citing intervals for the English trio only -- and the WLASL pair
    # was not even computed. Two intervals here, two there, same sentence.
    ci("openasl-pose+csl_daily-mt5::vs::how2sign-pose+csl_daily-mt5",
       "AslPairLexCI")
    ci("wlasl-pose+csl_daily-mt5::vs::how2sign-pose+csl_daily-mt5",
       "AslTrioLexCI")
    # The single-tensor decomposition. `HeadGain` is what loading only lm_head
    # buys, `BodyGain` what loading everything except it buys, and `HeadResidual`
    # is what the whole-half swap adds on top of the tensor alone -- the three
    # numbers the localisation claim rests on.
    ci("how2sign+csl_daily-lm_head::vs::how2sign", "HeadGainLexCI")
    ci("how2sign+csl_daily-mt5_nohead::vs::how2sign", "BodyGainLexCI")
    ci("how2sign-pose+csl_daily-mt5::vs::how2sign+csl_daily-lm_head",
       "HeadResidualLexCI")
    ci("csl_daily::vs::csl_daily+how2sign-lm_head", "HeadLossLexCI")
    ci("openasl+csl_daily-lm_head::vs::openasl", "HeadGainOpenLexCI")
    ci("how2sign+csl_stage1-lm_head::vs::how2sign", "HeadGainNewsLexCI")
    # ------------------------------------------------ the interaction (v3, 8/19)
    # Four simple effects were being reported as if their pattern were the
    # finding. It is not: "output language gates visual transfer" says two of
    # them DIFFER, and one interval excluding zero next to another containing it
    # is not that test. This is the difference of differences itself, paired
    # across all four cells at once and clustered by paragraph like everything
    # else here.
    ci("interaction::decoder-x-encoder", "InteractionLexCI")

    # The same interaction as a ratio of odds ratios. In points it is not
    # scale-free: two of the four cells sit on the \ChanceLevel-point floor the
    # item set was rank-balanced to produce and the fourth is past 56, so some of
    # an 11.5-point double difference is arithmetic on a bounded scale. The odds
    # ratio is not bounded that way, so the two scales together say whether the
    # gate survives the rescaling. What neither scale can do is turn an
    # at-chance readout into a measured null: with both English-output cells at
    # chance the encoder effect there is limited by the instrument's resolution,
    # which the prose has to say rather than test.
    lor = boot.get("interaction::decoder-x-encoder::logit")
    if lor:
        emit("InteractionLexLor", f"{lor['delta']:+.2f}")
        emit("InteractionLexLorCI", f"[{lor['lo']:+.2f}, {lor['hi']:+.2f}]")
        emit("InteractionLexOR", f"{lor['or_ratio']:.2f}")
        emit("InteractionLexORCI",
             f"[{lor['or_ratio_lo']:.2f}, {lor['or_ratio_hi']:.2f}]")
        # Named `Pboot` and not `Pclust` on purpose: the percentile bootstrap's
        # two-sided tail, not the sign-flip permutation the `Pclust` macros
        # carry. A log odds ratio is a function of four aggregates, so there is
        # no per-item difference whose sign a permutation could flip.
        emit("InteractionLexLorPboot", fmt_p(lor["p"]))
    else:
        for suf in ("Lor", "LorCI", "OR", "ORCI", "LorPboot"):
            emit(f"InteractionLex{suf}", todo("interaction logit"))

    # Cross-source closure guard. The four leg macros come from this file's
    # loader and the interaction from `19_bootstrap.py`'s, so requiring that the
    # legs subtract to the interaction checks both that no leg has drifted to a
    # fifth cell and that the two loaders still agree about what a cell is.
    # Cheap, and it is the check whose absence let 12.1 stand next to 31.4 and
    # 19.9 for three drafts.
    inter = boot.get("interaction::decoder-x-encoder")
    if inter:
        q = [cell_acc(i) for i in ("csl_daily", "csl_daily-pose+how2sign-mt5",
                                   "how2sign-pose+csl_daily-mt5", "how2sign")]
        if all(x is not None for x in q):
            A, B, C, D = q
            for label, mine in (("decoder axis", (A - B) - (C - D)),
                                ("encoder axis", (A - C) - (B - D))):
                if abs(mine - inter["delta"]) > 0.05:
                    raise SystemExit(
                        f"interaction disagrees on the {label}: "
                        f"{mine:+.2f} here vs {inter['delta']:+.2f} in "
                        f"bootstrap.json — a leg is on the wrong cell")

    # --------------------------- the floor-free factorial (v4, 8/20 from review)
    # Two reviewers made the same objection to v3's headline: the factorial's
    # two English-output cells sit on the \ChanceLevel-point level the item set
    # was rank-balanced to produce, so its visual-source leg there is bounded by
    # the instrument rather than measured by it, and an interaction whose small
    # arm is a bound is not an effect size. The answer is a second 2x2 that was
    # already in the runs and quoted in v3 as two bare accuracies with no
    # interval: switch the text side off by leaving mT5-base UNADAPTED instead of
    # by making it English-output. Every cell then stands at least fourteen
    # points clear of chance, and the interaction is larger, not smaller.
    # The comparison S3.1 needs and v3 did not make. "Worse than no
    # sign-language pretraining FOR THE VISUAL BRANCH" requires holding the mT5
    # half fixed and changing only the visual one, and v3 read it off 23.9
    # against 40.1, which changes both halves at once. Held fixed, the
    # mismatched visual half is worth slightly MORE than no visual pretraining,
    # which is the opposite sign to the sentence it was supporting.
    ci("how2sign-pose+csl_daily-mt5::vs::csl_daily-mt5",
       "AslVisualOverUntrainedLexCI")
    ci("random::vs::how2sign", "IntactMismatchLexCI")
    ci("interaction::adapt-x-encoder", "InteractionAdaptLexCI")
    ci("csl_daily::vs::csl_daily-mt5", "VisualOnCslMtfiveLexCI")
    ci("csl_daily-pose::vs::random", "VisualOnBaseMtfiveLexCI")
    ci("csl_daily::vs::csl_daily-pose", "MtfiveOnCslVisualLexCI")
    ci("csl_daily-mt5::vs::random", "MtfiveOnUntrainedLexCI")

    # ------------------------------------------- the v5 contrasts (8/22 reviews)
    # The head factor's neutral third level. Two of these decide which of the two
    # readings of Table 1c the paper is allowed to put in its title: `BaseGain` is
    # what an ASL model gains from having its English-specialized projection
    # merely REMOVED, and `CslOverBase` is what the Chinese one ADDS on top of
    # that. If the first carries most of the effect, the finding is about
    # de-specialization and not about written language.
    ci("how2sign+mt5_base-lm_head::vs::how2sign", "HeadBaseGainLexCI")
    ci("how2sign+csl_daily-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadCslOverBaseLexCI")
    ci("csl_daily+mt5_base-lm_head::vs::csl_daily", "HeadBaseLossLexCI")
    ci("csl_daily+mt5_base-lm_head::vs::csl_daily+how2sign-lm_head",
       "HeadAslHarmLexCI")
    # Added 8/23. Reviewer O's P3 in the video regime. `RandGain` is what a
    # RESET of the projection buys on its own; `RandVsBase` is the gap to
    # mT5-base's actual tensor, which is the number that decides between
    # "de-specialization" and "restore the pretrained geometry"; `RandVsRand`
    # is against RAND-VIS, because a random head at init landing BELOW
    # no-sign-language-pretraining is the sharper form of the same point.
    ci("how2sign+rand_head_nm-lm_head::vs::how2sign", "HeadRandGainLexCI")
    ci("how2sign+rand_head_nm-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadRandVsBaseLexCI")
    ci("how2sign+rand_head_nm-lm_head::vs::random", "HeadRandVsRandvisLexCI")
    ci("csl_daily+rand_head_nm-lm_head::vs::csl_daily", "HeadRandLossLexCI")
    ci("csl_daily+rand_head_nm-lm_head::vs::csl_daily+mt5_base-lm_head",
       "HeadRandVsBaseCslLexCI")
    # The equivalence that fixes finding 0f's mechanism: on the WORKING model a
    # random norm-matched head and How2Sign's fine-tuned one are the same thing.
    # A null here is the result, so it is quoted as an interval and never as "no
    # significant difference".
    ci("csl_daily+rand_head_nm-lm_head::vs::csl_daily+how2sign-lm_head",
       "HeadRandVsAslheadCslLexCI")
    ci("csl_daily+rand_head_nm-lm_head::vs::random", "HeadRandVsRandvisCslLexCI")
    # Scale, with no timing confound: two random heads at initialization whose
    # row norms differ by 2.2x (27.7 against 12.746). Another null quoted as an
    # interval -- it is the result, not a failure to find one.
    ci("how2sign+rand_head-lm_head::vs::how2sign+rand_head_nm-lm_head",
       "HeadRandScaleLexCI")
    ci("how2sign+rand_head-lm_head::vs::how2sign", "HeadRandLibGainLexCI")
    ci("how2sign+rand_head-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadRandLibVsBaseLexCI")
    # A-B3. The per-script rescale as an initialization, against its own head,
    # against the neutral one, and against a random projection. The first is the
    # one A-W2 predicts should be large.
    ci("how2sign+how2sign_cjk-lm_head::vs::how2sign", "HeadCjkInitGainLexCI")
    ci("how2sign+how2sign_cjk-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadCjkInitVsBaseLexCI")
    ci("how2sign+how2sign_cjk-lm_head::vs::how2sign+rand_head_nm-lm_head",
       "HeadCjkInitVsRandLexCI")
    ci("how2sign+how2sign_cjkgl-lm_head::vs::how2sign", "HeadCjkglInitGainLexCI")
    ci("how2sign+how2sign_cjkgl-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadCjkglInitVsBaseLexCI")
    ci("how2sign+how2sign_gl-lm_head::vs::how2sign", "HeadGlInitGainLexCI")
    ci("how2sign+how2sign_gl-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadGlInitVsBaseLexCI")
    # Camera-ready reviewer 2's mechanism control. A row permutation of
    # mT5-base's projection keeps the matrix's contents exactly -- same rows,
    # same Frobenius norm, same row-norm distribution, same singular values --
    # and destroys only which token each row scores. `PermVsBase` is the one the
    # argument turns on; `PermVsRand` asks whether a permuted pretrained head is
    # simply a random one; `PermWsVsPerm` asks whether keeping each row inside
    # its own script group recovers anything.
    ci("how2sign+mt5_base_perm-lm_head::vs::how2sign", "HeadPermGainLexCI")
    ci("how2sign+mt5_base_perm-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadPermVsBaseLexCI")
    ci("how2sign+mt5_base_perm-lm_head::vs::how2sign+rand_head_nm-lm_head",
       "HeadPermVsRandLexCI")
    ci("how2sign+mt5_base_perm-lm_head::vs::random", "HeadPermVsRandvisLexCI")
    ci("how2sign+mt5_base_permws-lm_head::vs::how2sign", "HeadPermWsGainLexCI")
    ci("how2sign+mt5_base_permws-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadPermWsVsBaseLexCI")
    ci("how2sign+mt5_base_permws-lm_head::vs::how2sign+mt5_base_perm-lm_head",
       "HeadPermWsVsPermLexCI")
    # Camera-ready reviewer 2's provenance control on the visual factor. Every
    # released visual state inherits Uni-Sign's CSL-News stage-1 pose
    # pretraining, so the factorial's visual leg bundles that shared stage with
    # CSL-Daily's adaptation on top of it. `StageOneAdapt` is what the
    # adaptation adds once the shared stage is held fixed -- the contrast the
    # submitted version could not make -- and the next two are what the shared
    # stage alone is worth under each mT5 state.
    # The clustering the intervals are actually computed over. The camera-ready
    # reviewer notes that the paper reports paragraph-level folds but leaves the
    # "clustered statistical structure harder to reconstruct"; the resampling
    # unit is the paragraph, so the number of paragraphs a pooled contrast spans
    # is the missing figure, and it is read off the contrast rather than
    # recomputed so it cannot drift from what the bootstrap actually did.
    _mainc = boot.get("how2sign+mt5_base-lm_head::vs::how2sign")
    if _mainc and _mainc.get("clusters"):
        emit("LexClustersN", _mainc["clusters"])
    else:
        emit("LexClustersN", todo("bootstrap cluster count"))

    # Exact split sizes and signer allocation (scripts/34_split_stats.py). The
    # camera-ready reviewer reads the evaluation as held-out paragraph rather
    # than cross-signer; that reading is right, and these macros let the paper
    # say so with the counts behind it instead of leaving it to be inferred.
    ss_path = ROOT / "data" / "split_stats.json"
    if ss_path.exists():
        ss = json.loads(ss_path.read_text())
        emit("SplitParagraphs", f"{ss['paragraphs']:,}".replace(",", "{,}"))
        emit("SplitUtterances", f"{ss['utterances']:,}".replace(",", "{,}"))
        emit("SplitSigners", ss["signers"])
        emit("SplitMonoParas", ss["monologue_paragraphs"])
        emit("SplitDialogueParas", ss["dialogue_paragraphs"])
        # Ranges, not per-fold lists: three folds differing by two paragraphs do
        # not need three numbers in a short paper's appendix.
        for split in ("train", "dev", "test"):
            key = split.capitalize()
            for field, macro in (("paragraphs", "Paras"), ("utterances", "Utts")):
                vals = [f[split][field] for f in ss["folds"]]
                lo, hi = min(vals), max(vals)
                emit(f"Split{key}{macro}",
                     f"{lo:,}".replace(",", "{,}") if lo == hi
                     else f"{lo:,}--{hi:,}".replace(",", "{,}"))
        emit("SplitTestSignersUnseen", ss["test_signers_unseen_max"])
    else:
        for m in ("SplitParagraphs", "SplitUtterances", "SplitSigners",
                  "SplitMonoParas", "SplitDialogueParas",
                  "SplitTrainParas", "SplitTrainUtts", "SplitDevParas",
                  "SplitDevUtts", "SplitTestParas", "SplitTestUtts",
                  "SplitTestSignersUnseen"):
            emit(m, todo("scripts/34_split_stats.py"))

    ci("csl_daily::vs::csl_stage1-pose+csl_daily-mt5", "StageOneAdaptLexCI")
    ci("csl_stage1-pose+csl_daily-mt5::vs::csl_daily-mt5",
       "StageOneOnCslMtfiveLexCI")
    ci("csl_stage1-pose::vs::random", "StageOneOnBaseMtfiveLexCI")
    ci("csl_daily-pose::vs::csl_stage1-pose", "StageOneAdaptOnBaseLexCI")
    ci("interaction::stageone-x-encoder", "InteractionStageOneLexCI")
    # O-P2/O-W3. The replication on two other ASL checkpoints, and each against
    # How2Sign's own rescue so the paper can say whether the SIZE replicates too
    # and not only the sign.
    ci("openasl+mt5_base-lm_head::vs::openasl", "HeadBaseGainOpenLexCI")
    ci("wlasl+mt5_base-lm_head::vs::wlasl", "HeadBaseGainWlaslLexCI")
    ci("openasl+mt5_base-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadBaseOpenVsHowLexCI")
    ci("wlasl+mt5_base-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadBaseWlaslVsHowLexCI")
    # The Chinese donor on the same recipients, completing the 3x2 grid, and the
    # neutral-vs-Chinese ordering read per recipient. None of the three ordering
    # contrasts clears p<0.05 on its own -- quote them as three positive point
    # estimates, never as a significant difference.
    ci("openasl+csl_daily-lm_head::vs::openasl", "HeadCslGainOpenLexCI")
    ci("wlasl+csl_daily-lm_head::vs::wlasl", "HeadCslGainWlaslLexCI")
    ci("openasl+mt5_base-lm_head::vs::openasl+csl_daily-lm_head",
       "HeadBaseOverCslOpenLexCI")
    ci("wlasl+mt5_base-lm_head::vs::wlasl+csl_daily-lm_head",
       "HeadBaseOverCslWlaslLexCI")
    # The constructed Chinese-output head: written-language match set rather than
    # inherited.
    # A-B2's downstream cell. The three nulls are the result -- a Chinese-output
    # head with no video in its history matches every video-trained donor -- so
    # they are quoted as intervals, never as "no significant difference".
    ci("how2sign+gloss_lm-lm_head::vs::how2sign", "HeadGlossGainLexCI")
    ci("how2sign+gloss_lm-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadGlossVsBaseLexCI")
    ci("how2sign+gloss_lm-lm_head::vs::how2sign+csl_daily-lm_head",
       "HeadGlossVsCslLexCI")
    ci("how2sign+gloss_lm-lm_head::vs::how2sign+tsl_text-lm_head",
       "HeadGlossVsTextLexCI")
    ci("how2sign+gloss_lm-lm_head::vs::how2sign+rand_head_nm-lm_head",
       "HeadGlossVsRandLexCI")
    ci("how2sign+tsl_text-lm_head::vs::how2sign", "HeadTextGainLexCI")
    ci("how2sign+tsl_text-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadTextOverBaseLexCI")
    ci("how2sign+csl_daily-lm_head::vs::how2sign+tsl_text-lm_head",
       "HeadCslOverTextLexCI")
    # Reviewer G's drift-matched rung. The two that carry the answer are
    # `...OverBase...` and `...OverText...`: if pushing the donor's projection out
    # to a released fine-tune's displacement bought nothing over the neutral level,
    # the original null was not an artefact of an undertrained donor.
    ci("how2sign+tsl_text_strong-lm_head::vs::how2sign",
       "HeadTextStrongGainLexCI")
    ci("how2sign+tsl_text_strong-lm_head::vs::how2sign+mt5_base-lm_head",
       "HeadTextStrongOverBaseLexCI")
    ci("how2sign+tsl_text_strong-lm_head::vs::how2sign+tsl_text-lm_head",
       "HeadTextStrongOverTextLexCI")
    ci("how2sign+csl_daily-lm_head::vs::how2sign+tsl_text_strong-lm_head",
       "HeadCslOverTextStrongLexCI")
    ci("how2sign+gloss_lm-lm_head::vs::how2sign+tsl_text_strong-lm_head",
       "HeadGlossOverTextStrongLexCI")
    ci("how2sign+tsl_text_strong-lm_head::local-swap_mean",
       "HeadTextStrongGroundLexCI")
    ci("how2sign+tsl_text_strong-lm_head::dswap::how2sign+mt5_base-lm_head",
       "HeadTextStrongGroundVsBaseCI")
    # Sufficiency against the existing necessity result.
    ci("csl_daily-pose+csl_daily-lm_head::vs::csl_daily-pose",
       "HeadSufficientLexCI")
    ci("csl_daily::vs::csl_daily-pose+csl_daily-lm_head", "HeadShortfallLexCI")
    # The post-hoc graft, and the comparison that adjudicates: an
    # initialization-time swap works, so does a post-hoc one close the gap?
    ci("how2sign@post-csl_daily::vs::how2sign", "PostHowCslLexCI")
    ci("csl_daily@post-how2sign::vs::csl_daily", "PostCslHowLexCI")
    ci("how2sign@post-how2sign::vs::how2sign", "PostHowSelfLexCI")
    ci("how2sign@post-how2sign+csl_daily-lm_head::vs::how2sign",
       "PostHowRescuedLexCI")
    ci("how2sign+csl_daily-lm_head::vs::how2sign@post-csl_daily",
       "PostVersusInitLexCI")
    # The wrong clip drawn from inside the paragraph, and the two wrong clips
    # against each other -- the second is what says whether signer and session
    # change carried any of the between-paragraph number.
    # `@within` because the within-paragraph readout is its own cell in
    # bootstrap.json: it re-scores existing weights in one forward pass, and
    # merging its `local` block into the canonical cell would both overwrite the
    # rows the tables are built from and unpair the contrast (19_bootstrap.py).
    for init, macro in (("csl_daily", "Csldaily"), ("how2sign", "Howtwosign"),
                        ("random", "Randominit"),
                        ("how2sign+csl_daily-lm_head", "HeadHowCsl")):
        ci(f"{init}@within::local-within_mean", f"{macro}WithinCI")
        ci(f"{init}::within_vs_swap", f"{macro}WithinVsSwapCI")
    # Label smoothing: the optimization-collapse reading's own prediction.
    ci("how2sign_ls00::vs::how2sign", "LsZeroLexCI")
    ci("how2sign_ls00::vs::random", "LsZeroVsRandLexCI")
    # The learning rate as an intervention in its own right, with the same
    # intervals every other intervention in the paper gets.
    ci("how2sign_lr1e4::vs::how2sign", "LrOneEFourGainLexCI")
    ci("how2sign_lr3e4::vs::how2sign", "LrThreeEFourGainLexCI")
    ci("how2sign_lr3e4::vs::how2sign+mt5_base-lm_head", "LrThreeEFourVsBaseLexCI")
    ci("how2sign_lr3e4::vs::random", "LrThreeEFourVsRandvisLexCI")
    # The head effect AT each learning rate -- the quantity that says whether the
    # readout result survives a tuned baseline. `HeadGainAtLr*`, so a reader
    # cannot mistake it for the learning-rate effect itself.
    ci("how2sign+mt5_base-lm_head_lr1e4::vs::how2sign_lr1e4",
       "HeadGainAtLrOneEFourLexCI")
    ci("how2sign+mt5_base-lm_head_lr3e4::vs::how2sign_lr3e4",
       "HeadGainAtLrThreeEFourLexCI")
    ci("how2sign+mt5_base-lm_head_lr3e4::vs::how2sign+mt5_base-lm_head",
       "LrGainAtBaseHeadLexCI")
    ci("how2sign+mt5_base-lm_head_lr3e4::vs::csl_daily+mt5_base-lm_head",
       "BestAslVsBestCslLexCI")
    # --------------------------------------------------------------- A-B6, 8/23
    # Model selection. The review's alternative reading is that the head effect
    # is an artefact of WHICH epoch each arm is read at: the dev-CE early stop
    # could be stopping the two arms at systematically different points, and a
    # difference between two selection rules would then be reported as a
    # difference between two initializations.
    #
    # `SelHeadAtBest` and `SelHeadAtLast` are the answer -- the same contrast
    # under the two selection rules. They are the pair to quote; everything
    # below them says how far each arm individually moves, which is what makes
    # the pair interpretable rather than a coincidence.
    ci("how2sign+mt5_base-lm_head_sl::vs::how2sign_sl", "SelHeadAtBestLexCI")
    ci("how2sign+mt5_base-lm_head_sl@last::vs::how2sign_sl@last",
       "SelHeadAtLastLexCI")
    # Within-cell: last epoch minus best epoch, same run, same weights
    # trajectory. No seed noise at all, so these are the tightest intervals in
    # the whole set and a null here means the early stop was not doing anything.
    ci("how2sign_sl@last::vs::how2sign_sl", "SelLastHowLexCI")
    ci("csl_daily_sl@last::vs::csl_daily_sl", "SelLastCslLexCI")
    ci("how2sign+mt5_base-lm_head_sl@last::vs::how2sign+mt5_base-lm_head_sl",
       "SelLastHeadLexCI")
    # The retrains against the cells they reproduce: same config, same seed, a
    # fresh run. This is the floor on how much of any small effect above is
    # run-to-run noise, and it belongs beside the effects rather than in a
    # footnote -- a +1.8 last-minus-best is only readable next to it.
    ci("how2sign_sl::vs::how2sign", "SelRerunHowLexCI")
    ci("csl_daily_sl::vs::csl_daily", "SelRerunCslLexCI")
    ci("how2sign+mt5_base-lm_head_sl::vs::how2sign+mt5_base-lm_head",
       "SelRerunHeadLexCI")
    # And the grounding diagnostic at the last epoch, for all three arms. The
    # accuracy contrast could in principle survive while the video dependence
    # did not, and that would be a different paper.
    ci("how2sign_sl@last::local-blank_plain", "SelBlankHowLastLexCI")
    ci("csl_daily_sl@last::local-blank_plain", "SelBlankCslLastLexCI")
    ci("how2sign+mt5_base-lm_head_sl@last::local-blank_plain",
       "SelBlankHeadLastLexCI")
    # The floor-free interaction inside each seed, as intervals rather than as the
    # bare accuracies `InteractionAdaptSeedAccs` carries.
    # Spelled-out digits: a LaTeX control sequence is letters only, so
    # `\InteractionAdaptSeed0CI` would not be a macro name at all.
    for sd, word in enumerate(("Zero", "One", "Two")):
        ci(f"interaction::adapt-x-encoder::seed{sd}",
           f"InteractionAdaptSeed{word}CI")
    lora = boot.get("interaction::adapt-x-encoder::logit")
    if lora:
        emit("InteractionAdaptLexOR", f"{lora['or_ratio']:.2f}")
        emit("InteractionAdaptLexORCI",
             f"[{lora['or_ratio_lo']:.2f}, {lora['or_ratio_hi']:.2f}]")
    else:
        for suf in ("OR", "ORCI"):
            emit(f"InteractionAdaptLex{suf}", todo("adapt interaction logit"))

    # The same two interactions on the score MARGIN. Accuracy is a threshold on
    # this quantity -- gold's mean token log probability minus the best
    # distractor's, in nats -- so it is the readout that has no floor to sit on,
    # and it is the one measurement that can say whether v3's points-scale
    # interaction was arithmetic on a bounded scale. Nats, so three decimals.
    for key, macro in (
            ("interaction::decoder-x-encoder::margin", "InteractionLexMargin"),
            ("interaction::adapt-x-encoder::margin", "InteractionAdaptLexMargin"),
            ("how2sign+csl_daily-lm_head::vs::how2sign::margin",
             "HeadGainLexMargin"),
            ("csl_daily::vs::csl_daily+how2sign-lm_head::margin",
             "HeadLossLexMargin"),
            ("csl_daily::vs::how2sign-pose+csl_daily-mt5::margin",
             "EncSourceOnCslDecLexMargin"),
            # 8/22. One review closes by asking whether a post-hoc swap would
            # recover performance ON THE MARGIN, which is the right place to look
            # because accuracy could hide a partial rescue. It does not: the
            # margin moves the wrong way.
            ("how2sign@post-csl_daily::vs::how2sign::margin",
             "PostHowCslLexMargin"),
            ("csl_daily@post-how2sign::vs::csl_daily::margin",
             "PostCslHowLexMargin"),
            ("how2sign@post-how2sign::vs::how2sign::margin",
             "PostHowSelfLexMargin"),
            ("how2sign+csl_daily-lm_head::vs::how2sign@post-csl_daily::margin",
             "PostVersusInitLexMargin")):
        r = boot.get(key)
        if not r:
            emit(f"{macro}Delta", todo(f"margin {key}"))
            emit(f"{macro}CI", todo(f"margin {key} CI"))
            continue
        emit(f"{macro}Delta", f"{r['delta']:+.3f}")
        emit(f"{macro}DeltaAbs", f"{abs(r['delta']):.3f}")
        emit(f"{macro}CI", f"[{r['lo']:+.3f}, {r['hi']:+.3f}]")
        emit(f"{macro}CIPclust", fmt_p(r["p"]))

    # Does the wrong clip move the score when it does not move the accuracy? If
    # the mismatched rows were reading the clip and only the output projection
    # hid it, this is where it would show. Same three draws, same items.
    for init, macro in (("csl_daily", "Csldaily"), ("how2sign", "Howtwosign"),
                        ("random", "Randominit"),
                        ("how2sign+csl_daily-lm_head", "HeadHowCsl"),
                        ("how2sign-pose+csl_daily-mt5", "CrossHowCsl"),
                        ("csl_daily+how2sign-lm_head", "HeadCslHow")):
        r = boot.get(f"{init}::local-swap_mean::margin")
        if not r:
            emit(f"{macro}SwapMargin", todo(f"margin swap {init}"))
            emit(f"{macro}SwapMarginCI", todo(f"margin swap {init} CI"))
            continue
        emit(f"{macro}SwapMargin", f"{r['delta']:+.3f}")
        emit(f"{macro}SwapMarginAbs", f"{abs(r['delta']):.3f}")
        emit(f"{macro}SwapMarginCI", f"[{r['lo']:+.3f}, {r['hi']:+.3f}]")

    # The three headline effects under the other contrastive scoring rule.
    # Appendix A establishes that the 25% reference level belongs to mean token
    # log probability; these say whether the effects do too.
    for key, macro in (
            ("interaction::decoder-x-encoder::sum", "InteractionLexSum"),
            ("interaction::adapt-x-encoder::sum", "InteractionAdaptLexSum"),
            ("how2sign+csl_daily-lm_head::vs::how2sign::sum", "HeadGainLexSum"),
            ("csl_daily::local-swap_mean::sum", "CsldailySwapSum"),
            ("how2sign::local-swap_mean::sum", "HowtwosignSwapSum")):
        r = boot.get(key)
        if not r:
            emit(f"{macro}Delta", todo(f"sum-scored {key}"))
            emit(f"{macro}CI", todo(f"sum-scored {key} CI"))
            continue
        emit(f"{macro}Delta", f"{r['delta']:+.1f}")
        emit(f"{macro}DeltaAbs", f"{abs(r['delta']):.1f}")
        emit(f"{macro}CI", f"\\mbox{{[{r['lo']:+.1f}, {r['hi']:+.1f}]}}")

    # Clustering sensitivity: the same two interactions with the gold word type
    # as the resampling unit instead of the paragraph. Neither nesting contains
    # the other, so this is a genuine second reading of the same estimate.
    for key, macro in (
            ("interaction::decoder-x-encoder::bytype", "InteractionLexByTypeCI"),
            ("interaction::adapt-x-encoder::bytype",
             "InteractionAdaptLexByTypeCI")):
        r = boot.get(key)
        if not r:
            emit(macro, todo(f"type-clustered {key}"))
            continue
        emit(macro, f"\\mbox{{[{r['lo']:+.1f}, {r['hi']:+.1f}]}}")
        emit(f"{macro}TypesN", r["clusters"])

    # A cell's own accuracy with a paragraph-clustered interval, for the
    # one-sample sentences. S3.1 says the ASL rows' intervals include chance; the
    # only intervals that existed for it were the Wilson ones emitted above,
    # which assume independence these items do not have -- so the appendix's
    # claim that every main-text interval resamples paragraphs was false for
    # precisely that sentence. Clustered intervals are wider, so the claim
    # survives; it just needs the interval it actually describes.
    for init in sorted(set(NAME) | {"random"}):
        m = NAME.get(init)
        if not m:
            continue
        r = boot.get(f"{init}::acc")
        if r:
            emit(f"{m}LexAccCI", f"[{r['lo']:.1f}, {r['hi']:.1f}]")
            emit(f"{m}LexAccCIlo", f"{r['lo']:.1f}")
            emit(f"{m}LexAccCIhi", f"{r['hi']:.1f}")

    # Per-row intervals on the two target-clip contrasts. The blank one is quoted
    # because the earlier draft called two of the ASL rows' negative deltas
    # significant on an unclustered McNemar test, and clustered by paragraph they
    # are not: that correction is the reason this block exists.
    for init in sorted(set(NAME) | {"random"}):
        m = NAME.get(init)
        if not m:
            continue
        for key, macro in ((f"{init}::local-swap_mean", f"{m}SwapCI"),
                           (f"{init}::local-blank_plain", f"{m}BlankCI")):
            _CI_KEYS[macro] = key
            r = boot.get(key)
            if r:
                emit(macro, f"$[{r['lo']:+.1f}, {r['hi']:+.1f}]$, "
                            f"$p{fmt_p(r['p'])}$")

    # ------------------------- clustered statistics, referential set (v3)
    # A separate file because it is a separate instrument: different items, in
    # different paragraphs, so nothing here may be paired against the lexical
    # numbers above. Where the paper compares the two it compares two independent
    # intervals and says so.
    pboot_path = ROOT / "data" / "bootstrap_pron.json"
    pboot = (json.loads(pboot_path.read_text())["contrasts"]
             if pboot_path.exists() else {})

    # Added 8/23. Until now the referential set emitted only `*PronSwapCI` and
    # the `dswap` family, so a cell-vs-cell contrast could be quoted on content
    # words and not on pointing. That is exactly the wrong asymmetry for the
    # results where the two instruments DISAGREE -- Reviewer G's drift-matched
    # donor loses to mT5-base on content words and beats CSL-Daily's head on
    # referential pointing, and only one of those had a macro.
    def cip(key: str, macro: str) -> None:
        r = pboot.get(key)
        if not r:
            emit(macro, todo(f"bootstrap pron {key}"))
            emit(f"{macro}Pclust", todo(f"bootstrap pron {key} p"))
            return
        emit(macro, f"\\mbox{{[{r['lo']:+.1f}, {r['hi']:+.1f}]}}")
        emit(f"{macro}Pclust", fmt_p(r["p"]))
        base = macro[:-2] if macro.endswith("CI") else macro
        emit(f"{base}Delta", f"{r['delta']:+.1f}")
        emit(f"{base}DeltaAbs", f"{abs(r['delta']):.1f}")

    cip("how2sign+tsl_text_strong-lm_head::vs::how2sign",
        "HeadTextStrongGainPronCI")
    cip("how2sign+tsl_text_strong-lm_head::vs::how2sign+mt5_base-lm_head",
        "HeadTextStrongOverBasePronCI")
    cip("how2sign+tsl_text_strong-lm_head::vs::how2sign+tsl_text-lm_head",
        "HeadTextStrongOverTextPronCI")
    cip("how2sign+csl_daily-lm_head::vs::how2sign+tsl_text_strong-lm_head",
        "HeadCslOverTextStrongPronCI")
    cip("how2sign+gloss_lm-lm_head::vs::how2sign+tsl_text_strong-lm_head",
        "HeadGlossOverTextStrongPronCI")
    cip("how2sign+tsl_text-lm_head::vs::how2sign", "HeadTextGainPronCI")
    cip("how2sign+tsl_text-lm_head::vs::how2sign+mt5_base-lm_head",
        "HeadTextOverBasePronCI")
    cip("how2sign+csl_daily-lm_head::vs::how2sign+tsl_text-lm_head",
        "HeadCslOverTextPronCI")

    for init in sorted(set(NAME) | {"random"}):
        m = NAME.get(init)
        if not m:
            continue
        r = pboot.get(f"{init}::local-swap_mean")
        if r:
            emit(f"{m}PronSwapCI", f"$[{r['lo']:+.1f}, {r['hi']:+.1f}]$, "
                                   f"$p{fmt_p(r['p'])}$")
            emit(f"{m}PronSwapDeltaCI", f"{r['delta']:+.1f}")
            emit(f"{m}PronSwapDeltaAbsCI", f"{abs(r['delta']):.1f}")

    # Added 8/23. The same difference-of-differences applied to the output
    # projection. `HeadClipKill` is the abolition result -- how much clip
    # dependence a collapsed donor head removes from a working model -- and the
    # two `Vs` rows are what turn "its interval covers zero" into "it is
    # indistinguishable from a cell that has none": one that has none by
    # construction (RAND-VIS) and one that has none by collapse (How2Sign).
    for a_cell, b_cell, tag in (
            ("csl_daily+how2sign-lm_head", "csl_daily", "Kill"),
            ("csl_daily+how2sign-lm_head", "random", "KillVsRand"),
            ("csl_daily+how2sign-lm_head", "how2sign", "KillVsHow"),
            ("how2sign+mt5_base-lm_head", "how2sign", "BaseGain"),
            ("csl_daily+mt5_base-lm_head", "csl_daily", "BaseGainCsl"),
            ("csl_daily+rand_head_nm-lm_head", "csl_daily", "RandKillCsl"),
            ("csl_daily+rand_head_nm-lm_head", "csl_daily+how2sign-lm_head",
             "RandVsAslheadCsl"),
            ("how2sign+rand_head_nm-lm_head", "how2sign", "RandGainHow"),
            ("how2sign+rand_head_nm-lm_head", "how2sign+mt5_base-lm_head",
             "RandVsBaseHow"),
            ("how2sign+rand_head-lm_head", "how2sign+rand_head_nm-lm_head",
             "RandScale"),
            ("how2sign+how2sign_cjk-lm_head", "how2sign", "CjkInitGain"),
            ("how2sign+how2sign_cjk-lm_head", "how2sign+mt5_base-lm_head",
             "CjkInitVsBase"),
            ("how2sign+how2sign_cjkgl-lm_head", "how2sign", "CjkglInitGain"),
            ("how2sign+how2sign_cjkgl-lm_head", "how2sign+mt5_base-lm_head",
             "CjkglInitVsBase"),
            ("how2sign+how2sign_gl-lm_head", "how2sign", "GlInitGain"),
            ("openasl+mt5_base-lm_head", "openasl", "BaseGainOpen"),
            ("wlasl+mt5_base-lm_head", "wlasl", "BaseGainWlasl"),
            ("openasl+csl_daily-lm_head", "openasl", "CslGainOpen"),
            ("wlasl+csl_daily-lm_head", "wlasl", "CslGainWlasl"),
            ("how2sign_lr3e4", "how2sign", "LrThreeEFourGain"),
            ("how2sign_lr3e4", "how2sign+mt5_base-lm_head", "LrThreeEFourVsBase"),
            ("how2sign_lr1e4", "how2sign", "LrOneEFourGain"),
            # Added 8/23 once run_rc5m.sh landed eval_clip.json for the two
            # head x LR cells. S3.1's claim is about clip dependence, not
            # accuracy, so until these existed the interaction could only be
            # stated in the quantity the claim is NOT about.
            ("how2sign+mt5_base-lm_head_lr1e4", "how2sign_lr1e4",
             "BaseGainAtLrOneEFour"),
            ("how2sign+mt5_base-lm_head_lr3e4", "how2sign_lr3e4",
             "BaseGainAtLrThreeEFour"),
            ("how2sign+mt5_base-lm_head_lr3e4", "how2sign+mt5_base-lm_head",
             "BaseLrThreeEFourVsBase"),
            # Reviewer G's drift-matched donor. The `VsBase` row is the one that
            # answers the objection in the quantity the grounding claim is about.
            ("how2sign+tsl_text_strong-lm_head", "how2sign", "TextStrongGain"),
            ("how2sign+tsl_text_strong-lm_head", "how2sign+mt5_base-lm_head",
             "TextStrongVsBase"),
            ("how2sign+tsl_text_strong-lm_head", "how2sign+tsl_text-lm_head",
             "TextStrongVsText")):
        key = f"{a_cell}::dswap::{b_cell}"
        for src_boot, suffix in ((boot, "Lex"), (pboot, "Pron")):
            r = src_boot.get(key)
            macro = f"HeadClip{tag}{suffix}"
            if src_boot is boot:
                _CI_KEYS[f"{macro}CI"] = key
            if not r:
                emit(macro, todo(f"head dswap {suffix} {tag}"))
                emit(f"{macro}CI", todo(f"head dswap {suffix} {tag} CI"))
                continue
            emit(macro, f"{r['delta']:+.1f}")
            emit(f"{macro}CI", f"$[{r['lo']:+.1f}, {r['hi']:+.1f}]$, "
                               f"$p{fmt_p(r['p'])}$")
            emit(f"{macro}A", f"{r['delta_a']:+.1f}")
            emit(f"{macro}B", f"{r['delta_b']:+.1f}")

    # What the pose branch's source is worth in clip-reading, mT5 half held at
    # CSL-Daily's. The paper's claim is that the two item sets price this
    # differently, so both instruments emit the same three macros and the body
    # quotes them side by side.
    for src, tag in (("how2sign-pose+csl_daily-mt5", "How"),
                     ("wlasl-pose+csl_daily-mt5", "Wlasl"),
                     ("openasl-pose+csl_daily-mt5", "Open")):
        key = f"csl_stage1-pose+csl_daily-mt5::dswap::{src}"
        for src_boot, suffix in ((boot, "Lex"), (pboot, "Pron")):
            r = src_boot.get(key)
            macro = f"EncClipCost{suffix}{tag}"
            if src_boot is boot:
                _CI_KEYS[f"{macro}CI"] = key
            if not r:
                emit(macro, todo(f"dswap {suffix} {tag}"))
                emit(f"{macro}CI", todo(f"dswap {suffix} {tag} CI"))
                continue
            emit(macro, f"{r['delta']:+.1f}")
            emit(f"{macro}CI", f"$[{r['lo']:+.1f}, {r['hi']:+.1f}]$, "
                               f"$p{fmt_p(r['p'])}$")

    # ------------------------------------- third-party source pin (8/20)
    # The paper has to name the Uni-Sign revision it was built against, and a
    # hand-typed SHA in the .tex is the same hazard as a hand-typed number: it
    # cannot follow a re-fetch. Read it from the provenance file the fetcher
    # writes instead, so the manuscript and the working tree cannot disagree.
    notice = ROOT / "third_party" / "unisign" / "README.NOTICE"
    ref = None
    if notice.exists():
        m = re.search(r"Fetched ref:\s*([0-9a-f]{7,40})", notice.read_text())
        ref = m.group(1) if m else None
    emit("UniSignRef", f"\\texttt{{{ref[:12]}}}" if ref
         else todo("Uni-Sign source not pinned: re-run scripts/09\\_fetch\\_unisign.py"))

    # --------------------------------------- the interaction, by seed (8/20)
    # The interaction is the abstract's claim, so "would another seed have given
    # this?" is the question a reviewer asks first. Fold 0 has all three seeds
    # for all four cells, so the 2x2 is rebuilt inside each seed rather than
    # inferred from the spread of any single row.
    seed_ints = [(sd, boot.get(f"interaction::decoder-x-encoder::seed{sd}"))
                 for sd in (0, 1, 2)]
    got = [(sd, r) for sd, r in seed_ints if r]
    if len(got) >= 2:
        xs = [r["delta"] for _, r in got]
        emit("InteractionSeedN", len(got))
        emit("InteractionSeedAccs", "/".join(f"{x:+.1f}" for x in xs))
        emit("InteractionSeedLo", f"{min(xs):+.1f}")
        emit("InteractionSeedHi", f"{max(xs):+.1f}")
        emit("InteractionSeedSpread", f"{max(xs) - min(xs):.1f}")
        # Fold 0 alone, so these are ~a third of the items: quoted so the reader
        # can see why the intervals are wider than the pooled one.
        emit("InteractionSeedItemsN", got[0][1]["n"])
    else:
        for suf in ("N", "Accs", "Lo", "Hi", "Spread", "ItemsN"):
            emit(f"InteractionSeed{suf}", todo("per-seed interaction"))

    # ------------------------------------------- multiplicity (v3, 8/20)
    # Limitations used to say the paper reports its contrasts "without a
    # multiplicity correction". Correcting them is cheap, so the sentence is now
    # a measurement rather than a caveat.
    #
    # The family is the contrasts the MAIN TEXT quotes with an interval, derived
    # from main.tex rather than listed here. A hand-kept list is what let the
    # dagger legend outlive its daggers: it cannot track a sentence being added
    # or cut, and a multiplicity family that silently drifts is worse than none.
    # Contrasts computed but never quoted are excluded, since correcting for
    # tests the paper does not report would be self-punishment; the reverse
    # choice (all 147) is reported in the reply, not here.
    #
    # Holm rather than Benjamini-Hochberg: the family is small and confirmatory,
    # so controlling the family-wise error rate is the stricter and simpler
    # claim. Both were computed and neither changes any verdict.
    main_tex_p = ROOT / "writings" / "fourth" / "main.tex"
    mult_worst = None
    if main_tex_p.exists():
        cited_macros = set(re.findall(r"\\([A-Z][A-Za-z]+)", main_tex_p.read_text()))
        fam = sorted({key for macro, key in _CI_KEYS.items()
                      if macro in cited_macros and key in boot})
        ps = [(boot[k]["p"], k) for k in fam]
        n = len(ps)
        adj, run = {}, 0.0
        for rank, (pv, k) in enumerate(sorted(ps)):
            run = max(run, (n - rank) * pv)
            adj[k] = min(run, 1.0)
        sig = [k for k in fam if boot[k]["p"] < 0.05]
        survive = [k for k in sig if adj[k] < 0.05]
        emit("MultTestsN", n)
        emit("MultSigN", len(sig))
        emit("MultSurviveN", len(survive))
        # The largest Holm-adjusted p among those significant before correction.
        # Quoted rather than the individual adjusted values because most raw p
        # here sit on the permutation's 1/B floor, so their adjusted values are
        # a property of B; this one is not.
        if sig:
            mult_worst = max(adj[k] for k in sig)
            emit("MultWorstAdjP", fmt_p(mult_worst))
        if len(survive) != len(sig):
            emit("MultLostN", len(sig) - len(survive))
        # 8/20, from review: a reviewer asked which contrasts the family is. It
        # is derived, so it can be printed rather than described. Cell names are
        # the paper's own notation for a cross-loaded cell, visual half first.
        def pretty(k: str) -> str:
            k = k.replace("interaction::decoder-x-encoder",
                          "visual source $\\times$ output language")
            k = k.replace("interaction::adapt-x-encoder",
                          "visual source $\\times$ mT5 adaptation")
            k = k.replace("::vs::", " vs.\\ ")
            k = k.replace("::local-swap_mean", ": local vs.\\ wrong clip")
            k = k.replace("::local-blank_plain", ": local vs.\\ blanked")
            k = k.replace("::dswap::", ": wrong-clip cost vs.\\ ")
            k = k.replace("_", "\\_")
            return k
        emit("MultFamilyList", "; ".join(pretty(k) for k in fam))
    if mult_worst is None:
        for m in ("MultTestsN", "MultSigN", "MultSurviveN", "MultWorstAdjP"):
            emit(m, todo("multiplicity"))

    # -------------------------------------------- inter-model agreement (v3)
    # "The three ASL rows converged on one and the same solution" rested on
    # hit/miss agreement, which three systems sharing a difficulty profile would
    # also produce. `18_agreement.py` computes agreement on the four-way pick,
    # kappa against the marginals, and the share of both-wrong items where the
    # two chose the SAME wrong distractor — which is the statistic that cannot be
    # produced by two systems at chance.
    agree_path = ROOT / "data" / "agreement.json"
    if agree_path.exists():
        ag = json.loads(agree_path.read_text())["summary"]
        for grp, prefix in (("asl_intact", "Asl"), ("csl_intact", "Csl"),
                            ("asl_vs_csl", "Cross")):
            g = ag.get(grp, {})
            for field, suffix in (("pick_agree", "PickAgree"),
                                  ("pick_agree_chance", "PickChance"),
                                  ("kappa", "Kappa"),
                                  ("wrong_pick_match", "WrongMatch")):
                v = g.get(field)
                if not v:
                    emit(f"{prefix}{suffix}Min", todo(f"agreement {grp} {field}"))
                    emit(f"{prefix}{suffix}Max", todo(f"agreement {grp} {field}"))
                    continue
                lo, hi = v
                fmt = "{:.2f}" if field == "kappa" else "{:.1f}"
                emit(f"{prefix}{suffix}Min", fmt.format(lo))
                emit(f"{prefix}{suffix}Max", fmt.format(hi))
                # A range that collapses when it has to. With one pair in a
                # group the min and max are the same number, and "87.0--87.0%"
                # on the page reads as a bug rather than as a range of one.
                emit(f"{prefix}{suffix}Range",
                     fmt.format(lo) if abs(hi - lo) < 5e-3
                     else f"{fmt.format(lo)}--{fmt.format(hi)}")
                if suffix == "PickChance":
                    # One number in the prose: the range is uninteresting, the
                    # level is what makes 87% agreement mean something.
                    emit(f"{prefix}PickChance", fmt.format((lo + hi) / 2))
    else:
        for prefix in ("Asl", "Csl", "Cross"):
            for suffix in ("PickAgree", "Kappa", "WrongMatch"):
                emit(f"{prefix}{suffix}Min", todo("agreement not computed"))
                emit(f"{prefix}{suffix}Max", todo("agreement not computed"))

    # ----------------------------------------------- checkpoint lineage (v3)
    # Uni-Sign fine-tunes every downstream model from its CSL-News stage 1, which
    # means the five checkpoints are not independent and "never trained together"
    # was not ours to claim. Measured from the weights rather than read off the
    # recipe, and the same measurement is what located the decoder factor in
    # `lm_head`.
    lin_path = ROOT / "data" / "lineage.json"
    if lin_path.exists():
        lin = json.loads(lin_path.read_text())
        pw = lin.get("pairwise", {})
        CSL, ASL = {"csl_stage1", "csl_daily"}, {"how2sign", "openasl", "wlasl"}

        def head_pairs(pred):
            out_ = []
            for k, v in pw.items():
                x, y, h = k.split("|")
                if h == "lm_head" and pred(x, y):
                    out_.append(v["cos"])
            return out_
        within_csl = head_pairs(lambda x, y: {x, y} <= CSL)
        within_asl = head_pairs(lambda x, y: {x, y} <= ASL)
        across = head_pairs(lambda x, y: (x in CSL) != (y in CSL)
                            and {x, y} <= (CSL | ASL))
        emit("LinHeadCosCsl", f"{min(within_csl):.2f}" if within_csl
             else todo("lm_head cosine within CSL"))
        emit("LinHeadCosAsl", f"{min(within_asl):.2f}" if within_asl
             else todo("lm_head cosine within ASL"))
        emit("LinHeadCosCrossLo", f"{min(across):.2f}" if across
             else todo("lm_head cosine across"))
        emit("LinHeadCosCrossHi", f"{max(across):.2f}" if across
             else todo("lm_head cosine across"))
        # The bodies, for the sentence that says they barely differ. Quoted as a
        # percentage because "0.006 relative L2" reads as noise to most readers.
        body = [v["rel_l2"] for k, v in pw.items()
                if k.endswith("|mt5_encoder") or k.endswith("|mt5_decoder")]
        emit("LinBodyRelPct", f"{100 * max(body):.1f}" if body
             else todo("mt5 body distance"))
        pose = [v["cos"] for name, row in lin.items()
                if isinstance(row, dict) and name != "pairwise"
                for key, v in row.items() if key == "vs_stage1_pose" and v]
        emit("LinPoseCosLo", f"{min(pose):.2f}" if pose else todo("pose cosine"))
        emit("LinPoseCosHi", f"{max(pose):.2f}" if pose else todo("pose cosine"))
    else:
        for macro in ("LinHeadCosCsl", "LinHeadCosAsl", "LinHeadCosCrossLo",
                      "LinHeadCosCrossHi", "LinBodyRelPct", "LinPoseCosLo",
                      "LinPoseCosHi"):
            emit(macro, todo("lineage not measured"))

    # ------------------------------------------------- second scoring LM (v3)
    # The rank balance is a property of (item set x rule x MODEL), and the model
    # it was balanced against is the one that chose the distractors. A second LM
    # from an unrelated family, run on the finished set with nothing re-chosen,
    # is what turns "a text-only LLM is at chance" into a claim about more than
    # our own construction model.
    lm2_path = ROOT / "data" / "lex" / "shortcuts.glm.json"
    if lm2_path.exists():
        # `07_shortcuts.py` writes per-item hits, not aggregates: {"hits":
        # {baseline: {item_key: bool}}}. The accuracy and its interval are
        # computed here so the two LMs are summarised by the same code, and so a
        # change to that script's aggregation cannot silently move this number.
        lm2 = json.loads(lm2_path.read_text())
        hits = lm2["hits"]["lm_prior"]
        k, n = sum(bool(v) for v in hits.values()), len(hits)
        lo, hi = wilson(k, n)
        emit("LexLMtwoPrior", f"{100 * k / n:.1f}")
        emit("LexLMtwoCIlo", f"{lo:.1f}")
        emit("LexLMtwoCIhi", f"{hi:.1f}")
        emit("LexLMtwoN", n)
        # Named in the appendix, and read from the file rather than typed: the
        # sentence says "a different family", and the file is what knows which.
        emit("LexLMtwoName", str(lm2.get("llm", "")).split("/")[-1]
             .replace("_", "\\_"))
    else:
        for macro in ("LexLMtwoPrior", "LexLMtwoCIlo", "LexLMtwoCIhi",
                      "LexLMtwoN", "LexLMtwoName"):
            emit(macro, todo("second scoring LM not run"))

    # ------------------------------------------------------ dev curves (v3)
    # "Worse than no pretraining" is measured at a fixed budget, so the paper has
    # to say whether the rows had finished fitting. The last-three-epoch movement
    # in dev cross-entropy is the cheapest honest answer and it comes from logs
    # that already exist.
    def dev_last_three(init: str) -> str | None:
        vals = []
        for f in (0, 1, 2):
            lp = ROOT / "runs" / f"f{f}_local_{init}_full_k4_s0" / "log.json"
            if not lp.exists():
                continue
            dev = [e["dev"] for e in json.loads(lp.read_text())["log"]]
            if len(dev) >= 4:
                vals.append(dev[-4] - dev[-1])
        return f"{max(vals):.3f}" if vals else None

    for init, macro in (("how2sign", "HowtwosignDevLastThree"),
                        ("csl_daily", "CsldailyDevLastThree")):
        v = dev_last_three(init)
        emit(macro, v if v else todo(f"{init} dev curve"))

    # --------------------------------------------- the long-budget cells (v3)
    # Off-protocol by design: a different budget from everything the tables
    # compare, so they get their own macros and are never pooled. LongEpochs is
    # read from the runs rather than typed, so a change of schedule cannot leave
    # a stale number in the prose.
    long_eps: set[int] = set()
    for init, macro in (("how2sign", "HowtwosignLongLexLocal"),
                        ("random", "RandominitLongLexLocal"),
                        ("csl_daily", "CsldailyLongLexLocal")):
        accs = []
        for f in (0, 1, 2):
            d = ROOT / "runs" / f"f{f}_local_{init}_full_k4_s0_e24"
            ev, lg = d / "eval_lex.json", d / "log.json"
            if not (ev.exists() and lg.exists()):
                continue
            rows = [r for r in json.loads(ev.read_text())["contrastive"]
                    if r["condition"] == "local"]
            if rows:
                accs.append(100 * sum(r["hit"] for r in rows) / len(rows))
                long_eps.add(json.loads(lg.read_text())["epochs_done"])
        emit(macro, f"{sum(accs) / len(accs):.1f}" if accs
             else todo(f"{init} long-budget cell"))
    emit("LongEpochs", str(min(long_eps)) if len(long_eps) == 1
         else (todo("long-budget cells disagree on epochs") if long_eps
               else todo("long-budget cells not run")))

    # --------------------------- neighbourhood density on the lexical set (v3)
    # The density result was the strongest evidence that the models read the sign
    # rather than the sentence, and it was computed on the referential set only —
    # i.e. on the set the paper itself argues is contaminated. Same computation,
    # content-word items, CSL rows against ASL rows.
    # `density_class`, keyed by utterance, exactly as the referential version
    # above reads it — the strata file is per utterance and the lexical items are
    # utterances too, so this is the same computation on a different item set.
    dens = {k: v.get("density_class") for k, v in strata.items()}
    csl_rows = lex_per.get("csl_daily", {}).get("local", [])
    asl_rows = lex_per.get("how2sign", {}).get("local", [])
    if csl_rows and asl_rows:
        by_key_asl = {r["key"]: r["hit"] for r in asl_rows}
        buckets: dict[str, list[tuple[bool, bool]]] = collections.defaultdict(list)
        for r in csl_rows:
            t = dens.get(r["key"])
            if t in ("low", "high") and r["key"] in by_key_asl:
                buckets[t].append((r["hit"], by_key_asl[r["key"]]))
        for t, macro in (("low", "LexLorLow"), ("high", "LexLorHigh")):
            b = buckets.get(t, [])
            if len(b) < 20:
                emit(macro, todo(f"lexical density {t} (n={len(b)})"))
                continue
            p_csl = sum(x for x, _ in b) / len(b)
            p_asl = sum(y for _, y in b) / len(b)
            emit(macro, f"{logit(p_csl, len(b)) - logit(p_asl, len(b)):+.2f}")
    else:
        emit("LexLorLow", todo("lexical density strata"))
        emit("LexLorHigh", todo("lexical density strata"))

    # ---------------------------------------------- accuracy by gold type (v3)
    # A review asked how many distinct gold types the 940 items rest on, and what
    # the accuracy looks like averaged over types rather than over items. The
    # concern is real: if a handful of frequent words carried many items each, an
    # item-average would be a weighted vote over far fewer independent things
    # than n suggests. Macro-averaging over types is the answer to that, and it
    # is reported beside the item figure rather than instead of it.
    gold_by_key: dict[str, str] = {}
    if lex_rec_path.exists():
        for line in lex_rec_path.open():
            r = json.loads(line)
            if r.get("item_type") == "lexical":
                gold_by_key[r["key"]] = r["gold_word"]
    for init, macro in (("csl_daily", "Csldaily"), ("how2sign", "Howtwosign"),
                        ("random", "Randominit")):
        rows = lex_per.get(init, {}).get("local", [])
        if not rows or not gold_by_key:
            emit(f"{macro}LexByType", todo(f"{init} by-type accuracy"))
            continue
        per_type: dict[str, list] = collections.defaultdict(list)
        for r in rows:
            g = gold_by_key.get(r["key"])
            if g:
                per_type[g].append(r["hit"])
        if not per_type:
            emit(f"{macro}LexByType", todo(f"{init} by-type accuracy"))
            continue
        macro_acc = sum(sum(v) / len(v) for v in per_type.values()) / len(per_type)
        emit(f"{macro}LexByType", f"{100 * macro_acc:.1f}")
    emit("LexTypesScored", len({g for k, g in gold_by_key.items()
                                if any(r["key"] == k for r in
                                       lex_per.get("csl_daily", {}).get("local", []))})
         if gold_by_key else todo("scored gold types"))

    # ---------------------------------- seed spread on the mixed cells (v3)
    # The two cells carrying the encoder-source effect, at seeds 0/1/2. Every
    # review asked for this, and Appendix B is why: elsewhere in this model the
    # seed decides 13-17 points.
    for init, macro in (("how2sign-pose+csl_daily-mt5", "CrossHowCslSeedSpread"),
                        ("csl_stage1-pose+csl_daily-mt5", "CrossNewsDailySeedSpread")):
        accs = []
        for sd in (0, 1, 2):
            rows = lex_cells.get((init, sd, 0), {}).get("local", [])
            if rows:
                accs.append(acc(rows))
        emit(macro, f"{max(accs) - min(accs):.1f}" if len(accs) > 1
             else todo(f"{init} seeds"))

    # =================================================================== v5
    # Added 8/22 for the three reviews of v4. Each block below names the
    # objection it answers; the ordering is the order the reviews put them in.

    # -------------------------------------- 1. the post-hoc lm_head graft (v5)
    # All three reviews name this as the highest-return experiment outstanding,
    # and S4 of v4 named it too and did not run it. The swaps in S3.4 happen at
    # INITIALIZATION, so they cannot separate the two readings S4 leaves open:
    # a mismatched output projection may stop the network learning to use the
    # video, or the video may be used and the projection unable to express it.
    # Grafting a FINE-TUNED head onto a FINE-TUNED model separates them, because
    # only the second reading predicts a rescue.
    #
    # `PostHowSelf` is what makes the other rows readable. Two independently
    # fine-tuned models do not share a coordinate system, so replacing 192.1M
    # parameters with another run's version of them costs something whatever the
    # written language is; the seed-1 graft is that cost with the language held
    # fixed. Reported beside every other graft rather than in a footnote.
    for init in ("how2sign@post-csl_daily", "csl_daily@post-how2sign",
                 "how2sign@post-how2sign",
                 "how2sign@post-how2sign+csl_daily-lm_head",
                 "csl_daily@post-csl_daily"):
        m = NAME.get(init)
        if not m:
            continue
        rows = lex_per.get(init, {}).get("local", [])
        parent = init.split("@")[0]
        base = lex_per.get(parent, {}).get("local", [])
        # Intersected rather than required equal, which is what `paired` does and
        # why it is not used here. A graft is a re-scoring of ONE trained model, so
        # recipient and recipient-plus-graft are the same weights on whatever
        # items the graft file covers; a graft run on fold 0 alone against a
        # three-fold parent is a valid 330-item contrast, not the two-different-
        # item-sets error `paired` exists to catch. `PostFolds` records the
        # coverage and the row carries a dagger.
        keys = {r["key"] for r in rows} & {r["key"] for r in base}
        if not keys:
            for suf in ("PostDelta", "PostDeltaAbs", "PostFolds"):
                emit(f"{m}{suf}", todo(f"{init} post-hoc graft"))
            continue
        x = [r for r in rows if r["key"] in keys]
        y = [r for r in base if r["key"] in keys]
        emit(f"{m}PostDelta", f"{acc(x) - acc(y):+.1f}")
        emit(f"{m}PostDeltaAbs", f"{abs(acc(x) - acc(y)):.1f}")
        emit(f"{m}PostFolds", len(lex_folds.get(init, ())))
        emit(f"{m}PostN", len(x))
        # The graft's own geometry, read off the eval file rather than recomputed:
        # how far apart the two fine-tuned heads were before one replaced the
        # other. A rescue that came with a cosine of 0.99 would be a different
        # claim from one that came with 0.31.
        for q in sorted(glob.glob(str(ROOT / "runs" / "f0_local_*"
                                      / "eval_posthoc_*.json"))):
            d = json.loads(Path(q).read_text())
            if row_label(d) != init:
                continue
            g = d.get("posthoc_head") or {}
            if "cos_recipient_donor" in g:
                emit(f"{m}PostCos", f"{g['cos_recipient_donor']:+.2f}")
                emit(f"{m}PostParams", f"{g.get('params_M', 192.1):.1f}")
            break

    # ------------------------------------------------------- A-B6, 8/23: @last
    # The last-epoch readout's own accuracy. Read straight off the file rather
    # than through `load_cells`, deliberately: `eval_lex_last.json` carries the
    # SAME run name as `eval_lex.json`, so passing it to the pooled loader would
    # give it the best-epoch cell's label and its `local` block would overwrite
    # the number every other macro for that cell is built from. That is the exact
    # collision `@within` was suffixed to avoid, and here the safer fix is to
    # keep the file out of the pool entirely -- three cells do not need pooling
    # machinery, and the contrast that matters is already in `bootstrap.json`
    # under the `@last` labels.
    for init, m in (("how2sign_sl", "HowtwosignSl"),
                    ("csl_daily_sl", "CsldailySl"),
                    ("how2sign+mt5_base-lm_head_sl", "HeadHowBaseSl")):
        q = ROOT / "runs" / f"f0_local_{init}_full_k4_s0" / "eval_lex_last.json"
        if not q.exists():
            for suf in ("LastLexLocal", "LastLexCIlo", "LastLexCIhi",
                        "LastLexN", "LastEpoch"):
                emit(f"{m}{suf}", todo(f"{init} last-epoch readout"))
            continue
        d = json.loads(q.read_text())
        L = [r["hit"] for r in d["contrastive"] if r["condition"] == "local"]
        if not L:
            emit(f"{m}LastLexLocal", todo(f"{init} last-epoch local"))
            continue
        lo, hi = wilson(sum(L), len(L))
        emit(f"{m}LastLexLocal", pct(sum(L) / len(L)))
        emit(f"{m}LastLexCIlo", f"{lo:.1f}")
        emit(f"{m}LastLexCIhi", f"{hi:.1f}")
        emit(f"{m}LastLexN", len(L))
        # WHICH epoch the dev-CE rule actually picked, so a reader can tell a
        # null last-minus-best that means "the rule changed nothing" from one
        # that means "the rule picked the last epoch anyway". CSL-Daily's is the
        # second case and the interval alone would not say so.
        lg = ROOT / "runs" / f"f0_local_{init}_full_k4_s0" / "log.json"
        if lg.exists():
            g = json.loads(lg.read_text())
            rows = [r for r in g.get("log", []) if "dev" in r and r.get("epoch")]
            if rows:
                best = min(rows, key=lambda r: r["dev"])
                emit(f"{m}BestEpoch", best["epoch"])
                emit(f"{m}LastEpoch", g.get("epochs_done", len(rows)))

    # How far apart two scorings of the SAME weights on the SAME items land. The
    # pose branch's convolutions are not run deterministically, so `local` is not
    # bit-reproducible across eval files, and Appendix E says so rather than
    # leaving a reader to notice that two tables disagree by a third of a point.
    # Every delta in the paper is computed inside one file, so this cannot enter
    # a contrast -- but it bounds how much of a cross-table difference is real.
    _jit = []
    for key in sorted(set(lex_cells) | set(clip_cells) | set(within_cells)):
        got = [acc(d.get(key, {}).get("local", []))
               for d in (lex_cells, clip_cells, within_cells)]
        got = [x for x in got if x is not None]
        if len(got) > 1:
            _jit.append(max(got) - min(got))
    emit("LocalJitter", f"{max(_jit):.1f}" if _jit else todo("local jitter"))

    # ------------------------------- 2. the within-paragraph wrong clip (v5)
    # `swap_plain` draws its donor from a DIFFERENT paragraph, which was chosen so
    # that a donor could not share topic and lexis with the target. A review
    # points out the cost: it changes signer, recording session and discourse
    # topic along with the clip-sentence correspondence, so part of the drop could
    # be any of those. `swap_within` draws from the target's own paragraph, which
    # holds all three fixed and matches duration less tightly. Neither supersedes
    # the other; together they bracket the effect.
    for init, conds in within_per.items():
        m = NAME.get(init)
        if not m:
            continue
        L = conds.get("local", [])
        draws = sorted(c for c in conds if c.startswith("swap_within@"))
        if not L or not draws:
            continue
        by_key: dict[str, list] = collections.defaultdict(list)
        for c in draws:
            for r in conds[c]:
                by_key[r["key"]].append(r["hit"])
        w = [{"key": k, "hit": sum(v) / len(v)}
             for k, v in by_key.items() if len(v) == len(draws)]
        pr = paired(L, w)
        if not pr:
            continue
        x, y = pr
        emit(f"{m}LexWithin", f"{acc(y):.1f}")
        emit(f"{m}LexWithinDelta", f"{acc(x) - acc(y):+.1f}")
        emit(f"{m}LexWithinDeltaAbs", f"{abs(acc(x) - acc(y)):.1f}")
        emit(f"{m}LexWithinN", len(x))
        emit(f"{m}LexWithinFolds",
             len({f for (i, _s, f) in within_cells if i == init}))
        emit(f"{m}LexWithinClusters", len({r["key"].split(":")[0] for r in x}))

    # The duration match each wrong clip achieves, which is the price the
    # within-paragraph version pays for holding the signer fixed. Computed here
    # from the same functions the evaluation uses, so the two cannot drift.
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from tsl.conditions import (assign_target_swaps, assign_within_swaps,
                                    load_records as _lr, split_records as _sr)
        from tsl.splits import load as _lf
        _recs = _lr(ROOT / "data" / "lex" / "records.jsonl")
        _folds = _lf(ROOT / "data" / "folds.json")
        for tag, fn in (("Swap", assign_target_swaps),
                        ("Within", assign_within_swaps)):
            gaps = []
            for f in (0, 1, 2):
                te = _sr(_recs, _folds, f)["test"]
                its = [r for r in te if r["item_type"] and r["candidates"]]
                bk = {r["key"]: r for r in te}
                for d in (0, 1, 2):
                    mp = fn(te, seed=0, perm=d, pool=te)
                    gaps += [abs((bk[mp[i["key"]]]["t2"] - bk[mp[i["key"]]]["t1"])
                                 - (i["t2"] - i["t1"]))
                             for i in its if i["key"] in mp]
            gaps.sort()
            emit(f"Lex{tag}DurMedian", f"{gaps[len(gaps)//2]:.0f}" if gaps
                 else todo(f"{tag} duration match"))
    except Exception as exc:                                # pragma: no cover
        print(f"duration match not computed: {exc}", file=sys.stderr)
        for tag in ("Swap", "Within"):
            emit(f"Lex{tag}DurMedian", todo(f"{tag} duration match"))

    # ------------------------------------------- 3. the head's geometry (v5)
    # Appendix D reported cosines BETWEEN checkpoints' output projections and a
    # review points out that this does not distinguish "two written languages"
    # from "one group barely moved". Distance to the common ancestor does, and it
    # is what every one of these heads started as. The second half answers the
    # sharper version of the same worry: if a CSL head simply rescales the rows of
    # Chinese tokens, "the output projection carries the written language" is
    # close to a tautology and the paper's claim has to move to how that
    # conditions visual transfer.
    hg_path = ROOT / "data" / "head_geometry.json"
    CSLK, ASLK = ("csl_daily", "csl_stage1"), ("how2sign", "openasl", "wlasl")
    if hg_path.exists():
        hg = json.loads(hg_path.read_text())
        emit("HeadVocabRows", hg["vocab"]["vocab_rows"])
        # The output projection's size, computed rather than typed. A review
        # objected that "one tensor" makes the result sound more localized than
        # it is, and it is right: this is a fifth of the mT5 half.
        emit("HeadParamsM",
             f"{hg['vocab']['vocab_rows'] * 768 / 1e6:.1f}")
        emit("HeadCjkRows", f"{hg['vocab']['cjk_rows']:,}".replace(",", "{,}"))
        emit("HeadCjkPct",
             f"{100 * hg['vocab']['cjk_rows'] / hg['vocab']['vocab_rows']:.1f}")
        tb, rn = hg["to_mt5base"], hg["row_norms"]

        def span(macro, vals, fmt="{:.2f}"):
            if not vals:
                emit(f"{macro}Lo", todo(macro))
                emit(f"{macro}Hi", todo(macro))
                return
            emit(f"{macro}Lo", fmt.format(min(vals)))
            emit(f"{macro}Hi", fmt.format(max(vals)))

        # The unsuffixed span is over the five RELEASED checkpoints only. It used
        # to be `tuple(tb)`, which was harmless while that was the same set --- and
        # stopped being harmless on 8/22, when scripts/23_text_lm.py added a
        # `tsl_text` entry whose lm_head sits 0.026 from mt5-base. That turned
        # "all five have moved almost equally far, relative L2 0.92-0.98" into
        # "0.03-0.98", which is a different and false sentence.
        RELEASED = CSLK + ASLK
        for tag, group in (("", RELEASED), ("Csl", CSLK), ("Asl", ASLK)):
            g = [k for k in group if k in tb]
            span(f"HeadBaseCos{tag}", [tb[k]["cos"] for k in g])
            span(f"HeadBaseRel{tag}", [tb[k]["rel_l2"] for k in g])
            span(f"HeadBaseRowCos{tag}", [tb[k]["row_cos_mean"] for k in g])
        for tag, group in (("Csl", CSLK), ("Asl", ASLK)):
            g = [k for k in group if k in rn]
            span(f"HeadNormRatio{tag}", [rn[k]["ratio_cjk_other"] for k in g])
            span(f"HeadGrowthCjk{tag}", [rn[k]["cjk_growth"] for k in g])
            span(f"HeadGrowthOther{tag}", [rn[k]["other_growth"] for k in g])
            # The symmetric group, added 8/22. Splitting the non-CJK side into
            # Latin-script rows and the remainder turns "an ASL fine-tune
            # attenuates Chinese harder" into something a global rescaling cannot
            # produce: the two families spare OPPOSITE groups, and each spares the
            # script of its own output language.
            span(f"HeadGrowthLatin{tag}",
                 [rn[k]["latin_growth"] for k in g if "latin_growth" in rn[k]])
            span(f"HeadGrowthRest{tag}",
                 [rn[k]["rest_growth"] for k in g if "rest_growth" in rn[k]])
            span(f"HeadRatioCjkLatin{tag}",
                 [rn[k]["ratio_cjk_latin"] for k in g if "ratio_cjk_latin" in rn[k]])
        if "mt5_base" in rn:
            emit("HeadNormRatioBase", f"{rn['mt5_base']['ratio_cjk_other']:.2f}")
            emit("HeadNormCjkBase", f"{rn['mt5_base']['cjk_mean_norm']:.1f}")
            if "ratio_cjk_latin" in rn["mt5_base"]:
                emit("HeadRatioCjkLatinBase",
                     f"{rn['mt5_base']['ratio_cjk_latin']:.2f}")
        if "latin_rows" in hg["vocab"]:
            emit("HeadLatinRows",
                 f"{hg['vocab']['latin_rows']:,}".replace(",", "{,}"))
            emit("HeadLatinPct",
                 f"{100 * hg['vocab']['latin_rows'] / hg['vocab']['vocab_rows']:.0f}")
        for k, macro in (("csl_daily", "Csldaily"), ("csl_stage1", "Cslstageone"),
                         ("how2sign", "Howtwosign"), ("openasl", "Openasl"),
                         ("wlasl", "Wlasl"), ("tsl_text", "Textlm")):
            if k in tb:
                emit(f"{macro}HeadBaseRel", f"{tb[k]['rel_l2']:.2f}")
                emit(f"{macro}HeadBaseCos", f"{tb[k]['cos']:+.2f}")
            if k in rn:
                emit(f"{macro}HeadNormRatio", f"{rn[k]['ratio_cjk_other']:.2f}")
    else:
        for macro in ("HeadVocabRows", "HeadCjkRows", "HeadCjkPct",
                      "HeadNormRatioBase", "HeadNormCjkBase", "HeadParamsM",
                      "HeadRatioCjkLatinBase", "HeadLatinRows", "HeadLatinPct"):
            emit(macro, todo("head geometry"))
        for stem in ("HeadBaseCos", "HeadBaseRel", "HeadBaseRowCos"):
            for tag in ("", "Csl", "Asl"):
                for end in ("Lo", "Hi"):
                    emit(f"{stem}{tag}{end}", todo("head geometry"))
        for stem in ("HeadNormRatio", "HeadGrowthCjk", "HeadGrowthOther",
                     "HeadGrowthLatin", "HeadGrowthRest", "HeadRatioCjkLatin"):
            for tag in ("Csl", "Asl"):
                for end in ("Lo", "Hi"):
                    emit(f"{stem}{tag}{end}", todo("head geometry"))

    # RAND-VIS against the reference level a video-free system actually has. v4
    # said "close to a unigram prior" and left the reader to subtract; a review
    # asked for the right reference line to be explicit, and once it is explicit
    # the distance to it belongs in a macro rather than in the adjective.
    _rv = acc(lex_per.get("random", {}).get("local", []))
    _prior = next((float(m_.group(1)) for m_ in
                   (re.match(r"\\newcommand\{\\LexMajority\}\{([0-9.]+)\}", ln)
                    for ln in out) if m_), None)
    emit("RandMinusPriorAbs",
         f"{abs(_rv - _prior):.1f}" if _rv is not None and _prior is not None
         else todo("rand-vis minus unigram prior"))
    # How far the floor-free 2x2's three low cells are from that prior. A review
    # observed that three of the four sit almost on it, so the interaction
    # contrasts one configuration against three at the video-free level. The
    # observation is correct and the paper states it; stating it needs the number.
    _q3 = [acc(lex_per.get(i, {}).get("local", []))
           for i in ("csl_daily-mt5", "csl_daily-pose", "random")]
    emit("QuadPriorSpread",
         f"{max(abs(x - _prior) for x in _q3):.1f}"
         if all(x is not None for x in _q3) and _prior is not None
         else todo("floor-free quad against the prior"))

    # ------------------- 3b. how far a run's projection moves at all (v5)
    # The measurement that closes the post-hoc argument. If the two interventions
    # install nearly the same tensor, then two models that differ by the whole
    # effect have essentially the same output projection, and the difference
    # cannot be in the projection. scripts/26_head_drift.py.
    hd_path = ROOT / "data" / "head_drift.json"
    if hd_path.exists():
        hd = json.loads(hd_path.read_text())
        own = [v["cos"] for v in hd.get("vs_released", {}).values()]
        if own:
            emit("HeadDriftOwnLo", f"{min(own):.3f}")
            emit("HeadDriftOwnHi", f"{max(own):.3f}")
        pr = hd.get("pairs", {})

        # Not `a`, `b`: `a` is the argparse namespace in this scope and shadowing
        # it here made `a.out` fail two hundred lines later, which is a long way
        # from the cause.
        def pcos(x, y):
            return (pr.get(f"{x}|{y}") or pr.get(f"{y}|{x}") or {}).get("cos")

        for macro, (lhs, rhs) in {
                # the init-swapped cell's FINAL projection against the donor the
                # post-hoc graft installs
                "HeadDriftInitSwapCos": ("csl_daily", "init_swapped"),
                # two runs of the same cell: near-identity, which is why the
                # seed-to-seed graft in Appendix M costs nothing
                "HeadDriftSeedCos": ("how2sign", "how2sign_seed1"),
                "HeadDriftSeedCslCos": ("csl_daily", "csl_daily_seed1"),
                # across lineages, for scale
                "HeadDriftCrossCos": ("csl_daily", "how2sign"),
        }.items():
            c = pcos(lhs, rhs)
            emit(macro,
                 f"{c:.3f}" if c is not None else todo(f"head drift {lhs}|{rhs}"))
    else:
        for macro in ("HeadDriftOwnLo", "HeadDriftOwnHi", "HeadDriftInitSwapCos",
                      "HeadDriftSeedCos", "HeadDriftSeedCslCos",
                      "HeadDriftCrossCos"):
            emit(macro, todo("head drift"))

    # ---------------------------- 4. the constructed Chinese-output head (v5)
    # v4's Limitations said no Chinese-output text component adapted without a
    # sign-language checkpoint is AVAILABLE. A review's answer is that this is a
    # statement about what we did: mT5-base can be adapted to the TSL Chinese
    # translations with no video anywhere in it. What the numbers below say is
    # that the cell is an honest attempt and a weak instrument -- text-only
    # adaptation at this data scale moves the output projection two orders of
    # magnitude less than the released fine-tunes did, so it cannot stand in for
    # a Chinese-output head matched in displacement.
    tl_path = ROOT / "data" / "text_lm.json"
    if tl_path.exists():
        tl = json.loads(tl_path.read_text())
        emit("TextLmEpochs", tl["epochs"])
        emit("TextLmNTrain", f"{tl['n_train']:,}".replace(",", "{,}"))
        emit("TextLmDevStart", f"{tl['log'][0]['dev']:.2f}")
        emit("TextLmDevBest", f"{tl['best_dev']:.2f}")
        emit("TextLmHeadRel", f"{tl['lm_head_rel_l2_from_mt5base']:.3f}")
        emit("TextLmHeadCos", f"{tl['lm_head_cos_to_mt5base']:.4f}")
    else:
        for macro in ("TextLmEpochs", "TextLmNTrain", "TextLmDevStart",
                      "TextLmDevBest", "TextLmHeadRel", "TextLmHeadCos"):
            emit(macro, todo("text-only adaptation"))

    # ----------------- 4b. the drift-matched rung of that ladder (8/23, G)
    # The objection the block above invites: the null belongs to a donor whose
    # projection barely moved, so it may be a null about mT5-base rather than
    # about Chinese-without-sign-language. The answer is to keep the objective
    # and push the optimiser until the displacement matches -- three rungs, stop
    # at the first that clears the target. What the ladder cannot match is
    # QUALITY: `TextLmStrongDevBest` is deliberately emitted next to the drift so
    # the write-up cannot quote a matched displacement without also quoting the
    # language model it was bought with.
    rungs = sorted(ROOT.glob("data/text_lm_strong_lr*.json"),
                   key=lambda q: json.loads(q.read_text())
                                     ["lm_head_rel_l2_from_mt5base"])
    if rungs:
        emit("TextLmStrongRungs", len(rungs))
        emit("TextLmStrongTarget", "0.85")
        # `3e-4` is not a control-sequence name, and the file already has a
        # spelling for exactly this rate: A-B1's `\LrThreeEFour...`. Reuse it
        # rather than invent a second one -- two spellings of one learning rate
        # is how a reader ends up believing they are two rates.
        RUNG = {"3e-4": ("ThreeEFour", "3\\times10^{-4}"),
                "1e-3": ("OneEThree", "1\\times10^{-3}"),
                "3e-3": ("ThreeEThree", "3\\times10^{-3}"),
                "3e-5": ("ThreeEFive", "3\\times10^{-5}")}
        for q in rungs:
            r = json.loads(q.read_text())
            key = q.stem.split("_lr")[1]
            tag = RUNG.get(key, (key.replace("-", "M"), key))[0]
            emit(f"TextLmStrong{tag}Drift",
                 f"{r['lm_head_rel_l2_from_mt5base']:.3f}")
            emit(f"TextLmStrong{tag}Cos", f"{r['lm_head_cos_to_mt5base']:.3f}")
            emit(f"TextLmStrong{tag}DevBest", f"{r['best_dev']:.2f}")
        chosen_path = rungs[-1]
        chosen = json.loads(chosen_path.read_text())
        ckey = chosen_path.stem.split("_lr")[1]
        emit("TextLmStrongLr", f"${RUNG.get(ckey, (ckey, ckey))[1]}$")
        emit("TextLmStrongEpochs", chosen["epochs"])
        emit("TextLmStrongDrift",
             f"{chosen['lm_head_rel_l2_from_mt5base']:.3f}")
        emit("TextLmStrongCos", f"{chosen['lm_head_cos_to_mt5base']:.3f}")
        emit("TextLmStrongDevBest", f"{chosen['best_dev']:.2f}")
    else:
        for macro in ("TextLmStrongRungs", "TextLmStrongTarget",
                      "TextLmStrongLr", "TextLmStrongEpochs",
                      "TextLmStrongDrift", "TextLmStrongCos",
                      "TextLmStrongDevBest"):
            emit(macro, todo("drift-matched ladder"))

    # ------------------------- 4c. the gradient at step 0 (8/23, Reviewer O P4)
    # The paper's trajectory claim -- a mismatched projection stops the visual
    # representation FORMING -- has been carried entirely by endpoints. This is
    # the only measurement of the trajectory itself: one backward pass on real
    # batches with the model built exactly as training would build it, before any
    # weight moves. `VisShare` is the visual side's fraction of the total
    # gradient norm; `Coh` is the across-batch cosine of the pose gradient, and
    # the two together separate starvation from misdirection.
    gp_path = ROOT / "data" / "grad_probe.json"
    GP = {"how2sign+own": "GradHowOwn", "how2sign+mt5_base": "GradHowBase",
          "how2sign+csl_daily": "GradHowCsl",
          "how2sign+rand_head_nm": "GradHowRand",
          "csl_daily+own": "GradCslOwn", "csl_daily+mt5_base": "GradCslBase",
          "csl_daily+how2sign": "GradCslHow",
          "csl_daily+rand_head_nm": "GradCslRand"}
    if gp_path.exists():
        gp = json.loads(gp_path.read_text())
        emit("GradProbeBatches", gp[next(iter(gp))]["n_batches"])
        shares = {}
        for key, m in GP.items():
            r = gp.get(key)
            if not r:
                for suf in ("VisShare", "Coh", "Loss", "Pose", "Body"):
                    emit(f"{m}{suf}", todo(f"grad probe {key}"))
                continue
            shares[key] = r["visual_share"]
            emit(f"{m}VisShare", f"{r['visual_share']:.3f}")
            emit(f"{m}Coh", f"{r['pose_grad_cos_mean']:+.3f}")
            emit(f"{m}Loss", f"{r['loss']:.2f}")
            emit(f"{m}Pose", f"{r['g_pose']:.2f}")
            emit(f"{m}Body", f"{r['g_mt5_body']:.2f}")
        # The separation itself. Four of the eight configurations train to a
        # working model and four do not; if the two groups do not overlap in
        # visual share, that is the whole result and it should be one macro
        # rather than eight the reader has to sort by hand.
        WORKS = {"how2sign+mt5_base", "how2sign+csl_daily",
                 "csl_daily+own", "csl_daily+mt5_base"}
        wv = [v for k, v in shares.items() if k in WORKS]
        dv = [v for k, v in shares.items() if k not in WORKS]
        if wv and dv:
            emit("GradWorksMinShare", f"{min(wv):.3f}")
            emit("GradDeadMaxShare", f"{max(dv):.3f}")
            emit("GradSeparates", "yes" if min(wv) > max(dv) else "no")
            emit("GradCohMaxAbs",
                 f"{max(abs(v['pose_grad_cos_mean']) for v in gp.values()):.3f}")
    else:
        emit("GradProbeBatches", todo("grad probe"))
        for m in GP.values():
            for suf in ("VisShare", "Coh", "Loss", "Pose", "Body"):
                emit(f"{m}{suf}", todo("grad probe"))
        for m in ("GradWorksMinShare", "GradDeadMaxShare", "GradSeparates",
                  "GradCohMaxAbs"):
            emit(m, todo("grad probe"))

    # ------------------------- 5. where the wrong clip moves the network (v5)
    # A review asks the question S4 leaves open in a form that needs no training:
    # if the mismatched model's encoder state moves as much as a matched model's
    # under a wrong clip while its output distribution does not, the video is read
    # and not expressible. If neither moves, it is not read. The two live at
    # different depths of one forward pass (scripts/25_encoder_shift.py).
    es_path = ROOT / "data" / "encoder_shift.json"
    ES = {"f0_local_csl_daily_full_k4_s0": "Csldaily",
          "f0_local_how2sign_full_k4_s0": "Howtwosign",
          "f0_local_random_full_k4_s0": "Randominit",
          "f0_local_how2sign+csl_daily-lm_head_full_k4_s0": "HeadHowCsl"}
    if es_path.exists():
        es = json.loads(es_path.read_text())
        for run, macro in ES.items():
            row = (es.get(run) or {}).get("swap_plain")
            if not row:
                for suf in ("EncCos", "EncRel", "OutJS"):
                    emit(f"{macro}{suf}", todo(f"encoder shift {run}"))
                continue
            emit(f"{macro}EncCos", f"{row['enc_cos']:.3f}")
            emit(f"{macro}EncRel", f"{row['enc_rel_l2']:.3f}")
            # Four decimals, not three: the first-step divergences are a few
            # ten-thousandths of a nat for the mismatched rows, and at three
            # decimals \RandominitOutJS rounds to 0.000, which reads as an
            # exact zero the measurement does not support.
            emit(f"{macro}OutJS", f"{row['out_js']:.4f}")
    else:
        for macro in ES.values():
            for suf in ("EncCos", "EncRel", "OutJS"):
                emit(f"{macro}{suf}", todo("encoder shift"))

    # ---------------------- 6. the floor-free interaction, seed by seed (v5)
    # Two reviews rank this above every other GPU-hour outstanding, and for the
    # same reason: a pooled 15.5 with a [+12.0, +17.9] interval reads as "the
    # training procedure is stable" when the interval holds the trained weights
    # fixed by construction. What a reader wants to know is whether re-training
    # reproduces the sign. Fold 0 only, so each figure rests on a third of the
    # items.
    QUAD_ADAPT = ("csl_daily", "csl_daily-mt5", "csl_daily-pose", "random")
    seed_inter: list[float] = []
    for sd in (0, 1, 2):
        q = [lex_cells.get((i, sd, 0), {}).get("local", []) for i in QUAD_ADAPT]
        if not all(q):
            continue
        a_, b_, c_, d_ = [acc(x) for x in q]
        seed_inter.append((a_ - b_) - (c_ - d_))
    emit("InteractionAdaptSeedN", len(seed_inter))
    emit("InteractionAdaptSeedAccs",
         ", ".join(f"{x:+.1f}" for x in seed_inter) if seed_inter
         else todo("floor-free interaction by seed"))
    emit("InteractionAdaptSeedMin",
         f"{min(seed_inter):+.1f}" if seed_inter else todo("seed min"))
    emit("InteractionAdaptSeedMax",
         f"{max(seed_inter):+.1f}" if seed_inter else todo("seed max"))
    # Every row that was re-run under more than one seed, not just the 2x2:
    # the neutral-head cells' headline is on THIS set, so a seed spread
    # reported only on the referential set does not cover it (O-P1, A-A2).
    for init in dict.fromkeys(QUAD_ADAPT + tuple(SEED_ROWS)):
        m = NAME.get(init)
        accs = [acc(lex_cells.get((init, sd, 0), {}).get("local", []))
                for sd in (0, 1, 2)]
        accs = [x for x in accs if x is not None]
        if m and len(accs) > 1:
            emit(f"{m}LexSeedRange",
                 f"{min(accs):.1f}--{max(accs):.1f}")
            emit(f"{m}LexSeedSpread", f"{max(accs) - min(accs):.1f}")
            emit(f"{m}LexSeedN", len(accs))
        elif m:
            for suf in ("LexSeedRange", "LexSeedSpread", "LexSeedN"):
                emit(f"{m}{suf}", todo(f"{init} fold-0 seeds"))

    # ---------------------------------------------- fold-coverage marks (v3)
    # Cells land fold by fold, so a table can hold a three-fold row beside a
    # one-fold row with nothing on the page to say so. Every row carries a mark
    # that is empty at three folds and a dagger below it; the caption explains
    # the dagger once. When the missing folds land the mark disappears by itself.
    for init in sorted(NAME):
        m = NAME.get(init)
        if not m:
            continue
        nf = len(lex_folds.get(init, ()))
        emit(f"{m}LexMark", "" if nf >= 3 else r"$^{\dagger}$")
        emit(f"{m}LexNfolds", nf)

    # ... and the caption's legend for the mark, which has to disappear with it.
    # On 8/20 the last incomplete row in Table 1 (`mt5_nohead`) got its two
    # missing folds, every mark in that table went empty, and the caption was
    # left explaining a symbol that appears nowhere -- the mirror image of the
    # failure the marks were added to prevent. So the legend is generated from
    # the same fold counts as the marks.
    TAB_MAIN = ["csl_daily", "csl_stage1", "how2sign", "openasl", "wlasl",
                "random", "csl_stage1-pose+csl_daily-mt5",
                "csl_daily-pose+csl_stage1-mt5", "how2sign-pose+csl_daily-mt5",
                "openasl-pose+csl_daily-mt5", "wlasl-pose+csl_daily-mt5",
                "csl_daily-pose+how2sign-mt5", "csl_daily-pose+openasl-mt5",
                "how2sign+csl_daily-lm_head", "how2sign+csl_daily-mt5_nohead",
                "csl_daily+how2sign-lm_head"]
    short_rows = [i for i in TAB_MAIN if len(lex_folds.get(i, ())) < 3]
    emit("TabMainMarkNote",
         r" $^{\dagger}$ marks an incomplete three-fold row." if short_rows else "")

    # ------------------------------------------- the placeholder safety net
    # Macros for cells that have not landed are emitted as visible \TODO markers
    # rather than left undefined. The difference matters: an undefined control
    # sequence is a BUILD FAILURE, so a paper citing a cell that is still in the
    # sweep cannot be compiled at all, and the pressure at that point is to
    # hand-type the number. A \TODO compiles, shows up in the PDF in bold, and is
    # listed by --check. Every macro below is one the paper cites and no run has
    # produced yet.
    #
    # Restricted to this project's naming convention, and \Delta and friends are
    # excluded by name: defining a LaTeX primitive here would break maths that is
    # currently fine.
    LATEX_RESERVED = {"TODO", "Delta", "LaTeX", "S", "P", "Roman", "Large",
                      "Huge", "Big", "Bigg", "Alpha", "Beta", "Gamma", "Sigma",
                      "Omega", "Lambda", "Theta", "Phi", "Psi", "Pi", "Xi"}
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", "\n".join(out)))
    main_path = a.out.parent / "main.tex"
    if main_path.exists():
        cited = set(re.findall(r"\\([A-Z][A-Za-z]{3,})", main_path.read_text()))
        for name in sorted(cited - defined - LATEX_RESERVED):
            emit(name, todo(f"no run yet: {name}"))

    a.out.write_text("\n".join(out) + "\n")
    unresolved = [l for l in out
                  if "\\TODO" in l and "newcommand{\\TODO}" not in l]
    print(f"-> {a.out}  ({len(out) - 2} macros, {len(unresolved)} unresolved)")
    for t in unresolved:
        print("   " + re.sub(r"\\newcommand\{\\(\w+)\}.*", r"\1", t))
    n_folds = {NAME.get(i, i): len(f) for i, f in sorted(folds.items())}
    print("folds per system:", n_folds)

    # Table 2's caption says "over three folds", with one daggered exception. A
    # cell whose folds are still landing pools whatever exists, so mid-sweep the
    # table silently shows a two-fold mean under a three-fold caption -- which is
    # exactly the state this file is read in while a sweep runs. Name the rows
    # the caption covers and say so out loud.
    THREE_FOLD_ROWS = [
        "Howtwosign", "HeadHowBase", "HeadHowCsl", "HeadHowPerm",
        "Openasl", "HeadOpenBase", "HeadOpenCsl",
        "Wlasl", "HeadWlaslBase", "HeadWlaslCsl",
        "Csldaily", "HeadCslHow", "HeadCslBase",
        # Table 1's six factorial cells.
        "Csldailypose", "CrossNewsDaily", "Cslstageonepose",
        "Csldailymtfive", "Randominit",
    ]
    short = {m: n_folds.get(m, 0) for m in THREE_FOLD_ROWS if n_folds.get(m, 0) < 3}
    if short:
        print("!! rows the captions call three-fold that are not yet:", short)
    else:
        print("every caption-declared three-fold row has 3 folds")

    # An unresolved macro only reaches the page if the paper cites it, and the
    # generator deliberately emits placeholders for cells that are still queued.
    # So the submittability gate is the intersection: macros main.tex actually
    # expands *and* that have no number behind them. Those print as
    # "[TODO: ...]" in the PDF, which is what must never survive a commit; the
    # rest are inventory for rows not yet written up.
    cited = _cited_todos(a.out.parent / "main.tex", unresolved)
    if cited:
        print(f"\n!! main.tex cites {len(cited)} unresolved macro(s) — the PDF "
              f"will show [TODO: ...]:")
        for m in cited:
            print(f"   \\{m}")
    return 1 if (a.check and cited) else 0


def _cited_todos(main_tex: Path, unresolved: list[str]) -> list[str]:
    """Names of \\TODO macros that main.tex expands. Empty if it is submittable."""
    if not main_tex.exists():
        return []
    names = {re.sub(r"\\newcommand\{\\(\w+)\}.*", r"\1", l) for l in unresolved}
    # Strip comment tails first: a `% queued, see TODO.md` note mentioning a
    # macro is not a citation of it, and counting it as one would make the gate
    # unfixable except by deleting the comment.
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", l)
                     for l in main_tex.read_text().splitlines())
    return sorted(names & set(re.findall(r"\\([A-Za-z]+)", body)))


if __name__ == "__main__":
    raise SystemExit(main())
