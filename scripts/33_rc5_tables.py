#!/usr/bin/env python3
"""Markdown tables for the review-response write-up, straight out of the JSON.

Every number in `writings/fifth/rc5-review-experiments.md` comes through here
rather than being retyped, for the same reason the paper's numbers come through
`make_numbers.py`: a hand-copied figure is a figure that can silently stop
matching the run it came from. Missing cells print as `—` so a table shows its
own gaps instead of omitting rows.

    ../.venv/bin/python scripts/33_rc5_tables.py            # every table
    ../.venv/bin/python scripts/33_rc5_tables.py --only seeds,gen
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = json.loads((ROOT / "data" / "rc5_analysis.json").read_text())


def g(*path, default=None):
    cur = A
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def f1(x, plus=False):
    if x is None:
        return "—"
    return f"{x:+.1f}" if plus else f"{x:.1f}"


def f3(x, plus=False):
    if x is None:
        return "—"
    return f"{x:+.3f}" if plus else f"{x:.3f}"


def ci(r, scale=1):
    if not r:
        return "—"
    lo, hi = r.get("lo"), r.get("hi")
    d = r.get("delta")
    fmt = f1 if scale == 1 else f3
    return f"{fmt(d, True)} [{fmt(lo, True)}, {fmt(hi, True)}]"


def pval(r):
    if not r or r.get("p") is None:
        return "—"
    p = r["p"]
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table(head, rows):
    w = [len(h) for h in head]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(str(c)))
    out = ["| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(head)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(head))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w[i])
                                     for i, c in enumerate(r)) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------- tables

CELL_ORDER = [
    ("how2sign", "How2Sign (ASL→En)", "own"),
    ("how2sign+mt5_base-lm_head", "How2Sign", "mT5-base (neutral)"),
    ("how2sign+csl_daily-lm_head", "How2Sign", "CSL-Daily (Chinese)"),
    ("how2sign+csl_stage1-lm_head", "How2Sign", "CSL-News (Chinese)"),
    ("how2sign+tsl_text-lm_head", "How2Sign", "TSL text (span-denoise)"),
    ("how2sign+tsl_text_strong-lm_head", "How2Sign", "TSL text (drift-matched)"),
    ("how2sign+gloss_lm-lm_head", "How2Sign", "gloss→Chinese"),
    ("how2sign+rand_head_nm-lm_head", "How2Sign", "random, norm-matched"),
    ("how2sign+rand_head-lm_head", "How2Sign", "random, library init"),
    ("how2sign+how2sign_cjk-lm_head", "How2Sign", "own, CJK rescaled"),
    ("how2sign+how2sign_gl-lm_head", "How2Sign", "own, radius rescaled"),
    ("how2sign+how2sign_cjkgl-lm_head", "How2Sign", "own, both rescaled"),
    ("openasl", "OpenASL (ASL→En)", "own"),
    ("openasl+mt5_base-lm_head", "OpenASL", "mT5-base (neutral)"),
    ("openasl+csl_daily-lm_head", "OpenASL", "CSL-Daily (Chinese)"),
    ("wlasl", "WLASL (ASL ISLR)", "own"),
    ("wlasl+mt5_base-lm_head", "WLASL", "mT5-base (neutral)"),
    ("wlasl+csl_daily-lm_head", "WLASL", "CSL-Daily (Chinese)"),
    ("csl_daily", "CSL-Daily (CSL→Zh)", "own"),
    ("csl_daily+mt5_base-lm_head", "CSL-Daily", "mT5-base (neutral)"),
    ("csl_daily+how2sign-lm_head", "CSL-Daily", "How2Sign (English)"),
    ("csl_daily+rand_head_nm-lm_head", "CSL-Daily", "random, norm-matched"),
    ("random", "RAND-VIS", "mT5-base"),
]


def t_cells():
    rows = []
    for key, vis, head in CELL_ORDER:
        c = g("cells", key) or {}
        v = g("video", key) or {}
        h = (v or {}).get("hit") or {}
        m = (v or {}).get("margin") or {}
        acc = c.get("acc")
        rows.append([
            vis, head,
            "—" if acc is None else f"{acc:.1f}",
            "—" if acc is None else f"[{c['lo']:.1f}, {c['hi']:.1f}]",
            "—" if c.get("n") is None else str(c["n"]),
            ci(h) if h else "—",
            ci(m, scale=3) if m else "—",
        ])
    return table(["visual + body", "lm_head", "acc", "95% CI", "n",
                  "Δ local−swap", "Δ margin (nats)"], rows)


def t_contrasts():
    rows = []
    for label, v in (g("contrasts") or {}).items():
        acc, mar = v.get("accuracy"), v.get("margin")
        rows.append([label,
                     f"{acc['acc_a']:.1f}" if acc else "—",
                     f"{acc['acc_b']:.1f}" if acc else "—",
                     ci(acc), pval(acc), ci(mar, scale=3),
                     str(acc["n"]) if acc else "—"])
    return table(["contrast", "a", "b", "Δ acc [95% CI]", "p",
                  "Δ margin [95% CI]", "n"], rows)


def t_seeds():
    rows = []
    for k, v in (g("seeds", "accuracy") or {}).items():
        if not v or v.get("n_seeds", 0) < 2:
            continue
        p = v["per_seed"]
        rows.append([k, f1(p.get("0")), f1(p.get("1")), f1(p.get("2")),
                     f1(v.get("mean")), f1(v.get("sd")), f1(v.get("range"))])
    a = table(["cell", "s0", "s1", "s2", "mean", "SD", "range"], rows)
    rows = []
    for k, v in (g("seeds", "deltas") or {}).items():
        if not v:
            continue
        p = v["per_seed"]
        rows.append([k, f1(p.get("0"), True), f1(p.get("1"), True),
                     f1(p.get("2"), True), f1(v.get("mean"), True),
                     f1(v.get("sd")), str(v.get("n_seeds"))])
    b = table(["contrast (computed inside each seed)", "s0", "s1", "s2",
               "mean", "SD", "n"], rows)
    return a + "\n\n" + b


def t_posthoc():
    rows = []
    for seed in ("0", "1", "2"):
        for k, v in sorted((g("posthoc", "per_seed", seed) or {}).items()):
            a, m = v.get("accuracy_delta"), v.get("margin_delta")
            rows.append([k, seed, f1(v.get("acc_base")), f1(v.get("acc")),
                         ci(a), ci(m, scale=3)])
    t = table(["graft", "seed", "recipient", "after graft", "Δ acc [95% CI]",
               "Δ margin [95% CI]"], rows)
    h = g("posthoc", "headline") or {}
    p = h.get("per_seed") or {}
    s = (f"\nAcross-seed summary for `{h.get('cell')}`: "
         f"{', '.join(f1(p.get(k), True) for k in ('0','1','2'))} — "
         f"mean {f1(h.get('mean'), True)}, SD {f1(h.get('sd'))}, "
         f"n={h.get('n_seeds')} seeds.\n")
    return t + "\n" + s


def t_synthead():
    rows = []
    for k, v in sorted((g("synthead") or {}).items()):
        a, m = v.get("accuracy_delta"), v.get("margin_delta")
        st = v.get("donor_stats") or {}
        vid = v.get("video")
        rows.append([k, f1(v.get("acc_base")), f1(v.get("acc")), ci(a),
                     ci(vid) if vid else "—",
                     f3(st.get("ratio_cjk_latin")) if st else "—",
                     f3(st.get("all_mean_norm")) if st else "—",
                     f3(v.get("cos_recipient_donor"))])
    return table(["recipient @ synthetic head", "before", "after",
                  "Δ acc [95% CI]", "Δ local−swap [95% CI]", "CJK:Latin",
                  "mean row norm", "cos(recip, donor)"], rows)


def t_gen():
    rows = []
    for k, v in sorted((g("gen") or {}).items()):
        rows.append([v.get("run", k),
                     v.get("posthoc_donor") or "—",
                     f3(v.get("bleu")), f3(v.get("bleu_norm")),
                     f3(v.get("chrf")),
                     f1((v.get("distinct") or 0) * 100),
                     f1((v.get("modal") or 0) * 100),
                     f1(v.get("mean_len")), str(v.get("n"))])
    return table(["run", "post-hoc head", "BLEU", "BLEU (norm)", "chrF",
                  "distinct %", "modal %", "mean len", "n"], rows)


def t_lr():
    rows = []
    for k, v in sorted((g("lr") or {}).items()):
        rows.append([k, str(v.get("lr_mt5")), f3(v.get("best_dev")),
                     f1(v.get("acc")), str(v.get("epochs"))])
    return table(["run", "lr (mT5)", "best dev CE", "diagnostic acc",
                  "epochs"], rows)


def t_lastepoch():
    rows = []
    for k, v in sorted((g("lastepoch") or {}).items()):
        rows.append([k, str(v.get("best_epoch")), f3(v.get("best_dev")),
                     f3(v.get("last_dev")), f1(v.get("acc_best_dev")),
                     f1(v.get("acc_last_epoch")), f1(v.get("delta"), True)])
    return table(["run", "best epoch", "best dev CE", "last dev CE",
                  "acc (best.pt)", "acc (last.pt)", "Δ"], rows)


def t_params():
    rows = []
    for k, v in (g("params") or {}).items():
        rows.append([k, f1(v.get("mt5_state_dict_M")),
                     f1(v.get("duplicate_embedding_M")),
                     f1(v.get("mt5_distinct_M")), f1(v.get("lm_head_M")),
                     f1(100 * v["lm_head_M"] / v["mt5_distinct_M"])
                     if v.get("lm_head_M") else "—",
                     str(v.get("lm_head_tied_to_embedding")),
                     f3(v.get("lm_head_cos_to_embedding")),
                     f3(v.get("embedding_rel_l2_from_mt5base")),
                     f3(v.get("lm_head_rel_l2_from_mt5base"))])
    return table(["checkpoint", "mT5 state dict (M)", "dup. embedding (M)",
                  "distinct (M)", "lm_head (M)", "lm_head % of distinct",
                  "tied?", "cos(head, emb)", "emb drift", "head drift"], rows)


def t_gloss():
    rows = []
    for k, v in (g("gloss") or {}).items():
        t = v.get("test") or {}
        rows.append([k, str(v.get("n_train")), f3(v.get("best_dev_ce")),
                     f3(t.get("bleu")), f3(t.get("chrf")),
                     f1((t.get("distinct") or 0) * 100),
                     f1((t.get("modal") or 0) * 100),
                     f3(v.get("lm_head_rel_l2_from_init")),
                     f3(v.get("lm_head_rel_l2_from_mt5base"))])
    return table(["init head", "n train", "dev CE", "BLEU", "chrF",
                  "distinct %", "modal %", "head drift from init",
                  "drift from mT5-base"], rows)


def t_interface():
    rows = []
    for pair, v in (g("interface") or {}).items():
        for method, m in (v.get("methods") or {}).items():
            rows.append([pair, method,
                         f3(m.get("holdout_r2")),
                         f3(m.get("holdout_ce_own_head")),
                         f3(m.get("holdout_ce_donor_head_raw")),
                         f3(m.get("holdout_ce_donor_head_aligned")),
                         f3(m.get("m_dist_from_identity"))])
    t = table(["recipient __ donor", "map", "held-out R²", "CE own head",
               "CE donor raw", "CE donor + map", "‖M−I‖/√d"], rows)
    rows = []
    for k, v in sorted((g("synthead") or {}).items()):
        if "align" not in k:
            continue
        a = v.get("accuracy_delta")
        rows.append([k, f1(v.get("acc_base")), f1(v.get("acc")), ci(a),
                     ci(v.get("margin_delta"), scale=3)])
    return t + "\n\n" + table(["aligned graft, scored", "before", "after",
                               "Δ acc [95% CI]", "Δ margin"], rows)


# The probe keys are `init+head`; the trained cell that pairs with each one is
# named differently, and the whole point of the table is to read a step-0
# quantity against a twelve-epoch one, so the join has to be explicit.
GRAD_CELL = {
    "how2sign+own": "how2sign",
    "how2sign+mt5_base": "how2sign+mt5_base-lm_head",
    "how2sign+csl_daily": "how2sign+csl_daily-lm_head",
    "how2sign+rand_head_nm": "how2sign+rand_head_nm-lm_head",
    "csl_daily+own": "csl_daily",
    "csl_daily+mt5_base": "csl_daily+mt5_base-lm_head",
    "csl_daily+how2sign": "csl_daily+how2sign-lm_head",
    "csl_daily+rand_head_nm": "csl_daily+rand_head_nm-lm_head",
}


def t_grad():
    rows = []
    # Sorted by the quantity the result is about, so the separation is visible
    # in the table rather than something the reader has to reconstruct.
    items = sorted((g("grad") or {}).items(),
                   key=lambda kv: kv[1].get("visual_share") or 0.0)
    for k, v in items:
        acc = g("cells", GRAD_CELL.get(k, ""), "acc")
        # `|` inside a cell ends the cell: the norm bars have to be U+2016.
        rows.append([k, f3(v.get("loss")), f3(v.get("g_pose")),
                     f3(v.get("g_pose_proj")), f3(v.get("g_embeds_pose")),
                     f3(v.get("g_lm_head")), f3(v.get("g_mt5_body")),
                     f3(v.get("visual_share")),
                     f3(v.get("pose_grad_cos_mean")),
                     f1(acc) if acc is not None else "—"])
    return table(["init + head", "loss", "‖g‖ pose", "‖g‖ pose_proj",
                  "‖g‖ at pose embeds", "‖g‖ lm_head", "‖g‖ mT5 body",
                  "visual share", "pose-grad coherence",
                  "trained cell, lex local %"], rows)


def t_ladder():
    rows = []
    for k, v in (g("ladder") or {}).items():
        rows.append([k, str(v.get("epochs")), f3(v.get("best_dev")),
                     f3(v.get("drift_from_mt5base")),
                     f3(v.get("cos_to_mt5base"))])
    return table(["text-LM rung", "epochs", "dev CE", "drift from mT5-base",
                  "cos to mT5-base"], rows)


def t_items():
    p = ROOT / "data" / "item_audit.json"
    if not p.exists():
        return "_(item audit not run)_"
    s = json.loads(p.read_text())["summary"]
    rows = []
    for group, d in s.items():
        if not isinstance(d, dict):
            rows.append([group, str(d)])
            continue
        for k, v in d.items():
            if k in ("note", "where"):
                continue
            rows.append([f"{group}.{k}",
                         f"{v:.4f}" if isinstance(v, float) else str(v)])
    return table(["measure", "value"], rows)


def t_videodelta():
    """Between-cell clip dependence. `a` and `b` are each cell's own
    Delta local-swap; `difference` is the paired test that one is smaller."""
    rows = []
    for k, v in (g("videodelta") or {}).items():
        if not v:
            continue
        a, b = k.split(" - ", 1)
        rows.append([a, b, f1(v.get("delta_a"), True), f1(v.get("delta_b"), True),
                     ci(v), pval(v), str(v.get("n"))])
    return table(["cell a", "cell b", "Δ a", "Δ b",
                  "difference [95% CI]", "p", "n"], rows)


TABLES = {
    "cells": t_cells, "contrasts": t_contrasts, "seeds": t_seeds,
    "videodelta": t_videodelta,
    "posthoc": t_posthoc, "synthead": t_synthead, "gen": t_gen,
    "lr": t_lr, "lastepoch": t_lastepoch, "params": t_params,
    "gloss": t_gloss, "interface": t_interface, "grad": t_grad,
    "ladder": t_ladder,
    "items": t_items,
}


def render(name: str) -> str:
    try:
        return TABLES[name]()
    except Exception as e:
        return f"_(table `{name}` failed: {type(e).__name__}: {e})_"


def inject(path: Path) -> None:
    """Replace each `<!--TABLE:name-->` marker with the rendered table.

    Idempotent, and idempotent for ANY block content. The first implementation
    recognised a previous injection by its shape -- table rows, `_(` notes,
    interleaved blanks -- which silently failed for the one block that ends in a
    prose sentence (`t_posthoc`'s across-seed summary). Every refresh appended
    another copy of that sentence; by the time it was noticed the write-up
    carried eight. So the block is now delimited explicitly and replaced between
    its markers, with the old shape-based sweep kept only to absorb blocks
    written before the end marker existed.
    """
    text = path.read_text()
    out, n = [], 0
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        m = re.match(r"^<!--TABLE:([a-z]+)-->$", line.strip())
        if not m:
            continue
        name = m.group(1)
        end = f"<!--/TABLE:{name}-->"
        # Preferred path: a delimited block from a previous run of this script.
        j = next((k for k in range(i, len(lines)) if lines[k].strip() == end),
                 None)
        if j is not None:
            i = j + 1
        else:
            # Legacy path: consume the un-delimited block a previous version
            # wrote. Table rows, notes, blanks between them -- and the trailing
            # prose paragraph, which is what the shape test used to miss.
            while i < len(lines) and (
                    lines[i].startswith("|")
                    or lines[i].startswith("_(")
                    or lines[i].startswith("Across-seed summary")
                    or (lines[i] == "" and i + 1 < len(lines)
                        and (lines[i + 1].startswith("|")
                             or lines[i + 1].startswith("_(")
                             or lines[i + 1].startswith("Across-seed summary")))):
                i += 1
        out.append("")
        out.append(render(name))
        out.append(end)
        n += 1
    path.write_text("\n".join(out))
    print(f"injected {n} tables -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--inject", type=Path, default=None,
                    help="write the tables into a markdown file's "
                         "<!--TABLE:name--> markers instead of stdout")
    a = ap.parse_args()
    if a.inject:
        inject(a.inject)
        return
    want = [x.strip() for x in a.only.split(",") if x.strip()] or list(TABLES)
    for name in want:
        print(f"\n\n### {name}\n")
        print(render(name))


if __name__ == "__main__":
    main()
