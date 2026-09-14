#!/usr/bin/env python3
"""Pool folds and emit the paper's tables (plain text + LaTeX).

    python3 scripts/06_report.py runs/f*_context_pose_gated_s0/eval.json
    python3 scripts/06_report.py runs/*/eval.json --latex tables.tex

Folds are pooled by item, not averaged over folds: every diagnostic item is
held out in exactly one fold, so concatenating the per-fold rows gives one
evaluation over the whole item set with no item counted twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.metrics import (  # noqa: E402
    cell, corpus_bleu_chrf, delta_var, interaction, paired_delta,
)

MAIN = ["local", "video", "text", "both"]
CONTROL = ["mismatch_video", "mismatch_text", "mismatch_both", "shuffle_video",
           "text_only", "blank_plain"]
LABEL = {
    "local": "Local (target clip only)",
    "video": "+ correct context video",
    "text": "+ correct context text",
    "both": "+ correct context video and text",
    "mismatch_video": "+ mismatched context video",
    "mismatch_text": "+ mismatched context text",
    "mismatch_both": "+ mismatched context, both",
    "shuffle_video": "+ shuffled same-paragraph video",
    "text_only": "blank target video, context text only",
    # The paper's Blank column and every Delta in it. Listed here because a
    # condition absent from MAIN/CONTROL is silently dropped from Table 2, so
    # adding a condition to tsl.conditions is not enough to make it visible.
    "blank_plain": "blank target video, no context (one-factor)",
}


def load(paths: list[Path]) -> dict[str, list[dict]]:
    """system -> pooled contrastive rows (folds concatenated)."""
    per: dict[str, list[dict]] = defaultdict(list)
    gen: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    for p in paths:
        d = json.loads(Path(p).read_text())
        # A Uni-Sign run is identified by what it was initialised from and how it
        # was tuned; a from-scratch run by its feature backend and fusion. Folds
        # of the same system pool, so everything else must be in the name.
        #
        # `init` alone is not enough: a half-checkpoint row records the same
        # `init` as the whole checkpoint it was cut from, so `csl_daily`,
        # `csl_daily-pose` and `csl_daily-mt5` would land in one bucket — and
        # because they cover the same items on the same folds, the de-duplication
        # below would then silently discard the two split rows entirely rather
        # than merely mixing them.
        if d.get("arch") == "unisign":
            init = d["init"]
            if d["cfg"].get("init_parts", "all") != "all":
                init = f"{init}-{d['cfg']['init_parts']}"
            if d["cfg"].get("init_mt5"):
                init = f"{init}+{d['cfg']['init_mt5']}-mt5"
            sysname = f"{d['system']}/{init}/{d['tune']}/s{d['seed']}"
        else:
            sysname = f"{d['system']}/{d['backend']}/{d['fusion']}/s{d['seed']}"
        for r in d["contrastive"]:
            k = (r["key"], r["condition"])
            if k in seen[sysname]:
                continue  # same item in two folds would double-count
            seen[sysname].add(k)
            per[sysname].append(r)
        gen[sysname].extend(d.get("generation", []))
    return per, gen


def sel(rows, condition=None, item_type=None, prior_adv=False):
    out = rows
    if condition:
        out = [r for r in out if r["condition"] == condition]
    if item_type:
        out = [r for r in out if r["item_type"] == item_type]
    if prior_adv:
        out = [r for r in out if r.get("prior_adv")]
    return out


def table_main(rows: list[dict], f) -> None:
    p = lambda *a: print(*a, file=f)
    p("\nTable 2: contrastive referent accuracy (gold vs 3 person/number distractors)")
    p(f"{'':34s} {'--- all items ---':>53s} {'-- prior-adversarial --':>53s}")
    p(f"{'condition':34s} {'anaphoric':>26s} {'deictic':>26s} "
      f"{'anaphoric':>26s} {'deictic':>26s}")
    p("-" * 142)
    for c in MAIN + CONTROL:
        r_a, r_d = sel(rows, c, "anaphoric"), sel(rows, c, "deictic")
        if not r_a and not r_d:
            continue
        ca, cd = cell([r["hit"] for r in r_a]), cell([r["hit"] for r in r_d])
        pa = cell([r["hit"] for r in sel(rows, c, "anaphoric", True)])
        pd = cell([r["hit"] for r in sel(rows, c, "deictic", True)])
        p(f"{LABEL.get(c, c):34s} {ca.fmt():>26s} {cd.fmt():>26s} "
          f"{pa.fmt():>26s} {pd.fmt():>26s}")
    p("\nChance is 25% (4 candidates) throughout. Brackets are Wilson 95% intervals.")


SPECS = [
    ("Context benefit   (both - local)", "both", "local"),
    ("  video only      (video - local)", "video", "local"),
    ("  text only       (text - local)", "text", "local"),
    ("Selectivity       (both - mismatch)", "both", "mismatch_both"),
    ("  video           (video - mm video)", "video", "mismatch_video"),
    ("  text            (text - mm text)", "text", "mismatch_text"),
    ("Robustness        (local - shuffled)", "local", "shuffle_video"),
]


def table_flat(rows: list[dict], f) -> None:
    """One row per condition, for item sets with no anaphoric/deictic split.

    The lexical set (`data/lex`, `eval_lex.json`) carries a single item_type, so
    every pronoun-shaped table above it comes out with a header and no rows —
    which prints as a clean run rather than as nothing happening. This is what
    gets printed instead, chosen by what is in the file rather than by a flag.
    """
    p = lambda *a: print(*a, file=f)
    types = sorted({r["item_type"] for r in rows if r["item_type"]})
    p(f"\nTable 2: contrastive accuracy ({'/'.join(types)} items, gold vs 3 distractors)")
    p(f"{'condition':34s} {'all items':>26s} {'prior-adversarial':>26s}")
    p("-" * 88)
    base = None
    for c in MAIN + CONTROL:   # blank_plain lives in CONTROL; do not re-append
        r_all = sel(rows, c)
        if not r_all:
            continue
        ca = cell([r["hit"] for r in r_all])
        cp = cell([r["hit"] for r in sel(rows, c, prior_adv=True)])
        if c == "local":
            base = r_all
        p(f"{LABEL.get(c, c):34s} {ca.fmt():>26s} {cp.fmt():>26s}")
    # The one contrast this set is built to support: local vs the same model with
    # the target clip zeroed and nothing else changed.
    if base:
        bp = sel(rows, "blank_plain")
        if bp:
            d = paired_delta(base, bp)
            p(f"\nlocal - blank_plain: {100 * d['delta']:+.1f} points "
              f"(n={d['n']}, +{d['n_gain']}/-{d['n_loss']}, p={d['p_exact']:.3g})")
    p("\nChance is 25% (4 candidates). Brackets are Wilson 95% intervals.")


def table_deltas(rows: list[dict], f, prior_adv: bool = False) -> None:
    p = lambda *a: print(*a, file=f)
    which = "prior-adversarial items only" if prior_adv else "all items"
    p(f"\nTable 3{'b' if prior_adv else 'a'}: context effects, paired per item "
      f"({which})")
    p(f"{'':34s} {'anaphoric':>22s} {'deictic':>22s} {'interaction':>14s}")
    p("-" * 96)
    for label, ca, cb in SPECS:
        ra1 = sel(rows, ca, "anaphoric", prior_adv)
        ra2 = sel(rows, cb, "anaphoric", prior_adv)
        rd1 = sel(rows, ca, "deictic", prior_adv)
        rd2 = sel(rows, cb, "deictic", prior_adv)
        da, dd = paired_delta(ra1, ra2), paired_delta(rd1, rd2)
        if not da["n"] or not dd["n"]:
            continue
        da["var"], dd["var"] = delta_var(ra1, ra2), delta_var(rd1, rd2)
        it = interaction(da, dd)
        p(f"{label:34s} "
          f"{da['delta']:+7.1%} (p={da['p_exact']:.3f}) "
          f"{dd['delta']:+7.1%} (p={dd['p_exact']:.3f}) "
          f"{it['interaction']:+9.1%}")
    p("\np is an exact McNemar sign test on discordant items.")
    p("Interaction = benefit(anaphoric) - benefit(deictic): the paper's key number.")
    p("A gain that is target-side fluency shows interaction ~ 0.")
    if prior_adv:
        p("This is the comparison to quote. 74% of the deictic items are answerable")
        p("by 'always say 我' (the commonest pointing gloss); restricting both subsets")
        p("to items where that frequency prior FAILS matches their prior difficulty,")
        p("so the interaction cannot be an artefact of class imbalance.")


def table_breakdown(rows: list[dict], f) -> None:
    p = lambda *a: print(*a, file=f)
    p("\nTable 4: anaphoric items broken down (condition = both)")
    r = sel(rows, "both", "anaphoric")
    loc = {x["key"]: x["hit"] for x in sel(rows, "local", "anaphoric")}
    for group, name in (("pron_gloss", "pointing gloss"), ("n_ctx", "context utterances")):
        p(f"\n  by {name}:")
        b: dict[str, list[dict]] = defaultdict(list)
        for x in r:
            b[str(x[group])].append(x)
        for g, v in sorted(b.items(), key=lambda kv: -len(kv[1])):
            c = cell([x["hit"] for x in v])
            d = sum(x["hit"] for x in v) / len(v) - sum(
                loc.get(x["key"], False) for x in v) / len(v)
            p(f"    {g:12s} {c.fmt():>28s}   benefit {d:+.1%}")
    av = [x for x in r if x["agreeing_verb"]]
    if av:
        p(f"\n  utterances containing an agreeing/spatial verb: {cell([x['hit'] for x in av]).fmt()}")


def table_generation(gen: list[dict], f) -> None:
    if not gen:
        return
    p = lambda *a: print(*a, file=f)
    p("\nTable 5: free translation (secondary) and pronoun-slot accuracy")
    p(f"{'condition':34s} {'BLEU':>7s} {'chrF':>7s} {'slot acc (ana)':>18s} "
      f"{'slot acc (dei)':>18s}")
    p("-" * 88)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in gen:
        by[r["condition"]].append(r)
    for c in MAIN + CONTROL:
        rs = by.get(c)
        if not rs:
            continue
        sc = corpus_bleu_chrf([r["hyp"] for r in rs], [r["ref"] for r in rs])
        cells = []
        for t in ("anaphoric", "deictic"):
            hits = [r["slot_hit"] for r in rs
                    if r["item_type"] == t and r["slot_hit"] is not None]
            n_abst = sum(1 for r in rs if r["item_type"] == t and r["slot_hit"] is None)
            cells.append(f"{(sum(hits)/len(hits)):.1%} n={len(hits)} (+{n_abst} n/a)"
                         if hits else "-")
        p(f"{LABEL.get(c, c):34s} {sc['bleu']:7.2f} {sc['chrf']:7.2f} "
          f"{cells[0]:>18s} {cells[1]:>18s}")
    p("\n'n/a' = the hypothesis committed to no pronoun at all; excluded rather than")
    p("scored wrong, so slot accuracy is not diluted by omission.")


STRATA_NOTE = {
    "density": ("phonological neighbourhood density",
                "signs sharing a slot-1 (location, handshape); high = many visual "
                "competitors"),
    "reduction": ("duration reduction",
                  "token duration over that sign type's median in the corpus; "
                  "low = rushed, coarticulated"),
}


def table_strata(per: dict, gen: dict, strata: dict, f, field: str,
                 condition: str = "local", ref_system: str | None = None) -> None:
    """Axis 3: does the initialisation matter more where the form is harder to read?

    The transfer table alone cannot say whether a CSL advantage is its decoder
    already speaking Chinese or its encoder already reading sign. The two make
    opposite predictions here: decoder credit is a constant offset across strata,
    encoder credit concentrates where the signs are hard to see. So the row that
    matters is not any system's accuracy but the *gap between systems within a
    stratum*, printed at the bottom.
    """
    p = lambda *a: print(*a, file=f)
    rows = strata.get("strata", {})
    cuts = next((c for c in strata.get("cuts", []) if c["field"] == field), None)
    label, gloss = STRATA_NOTE[field]
    classes = ["low", "mid", "high"]

    p(f"\n{'=' * 96}\nTable 6-{field}: initialisation × {label}\n{'=' * 96}")
    p(f"{gloss}.")
    if cuts and cuts.get("bins"):
        p(f"Terciles over {cuts['n']} utterances, taken inside "
          f"{cuts['within']} bins so the classes cannot differ in utterance "
          f"length; condition = {condition}. Cut points per bin: " + "  ".join(
              f"{b['cuts'][0]:.2f}/{b['cuts'][1]:.2f}" for b in cuts["bins"]))

    cls = {k: v.get(f"{field}_class") for k, v in rows.items()}
    dur = {c: [rows[k]["dur_ms"] for k, v in cls.items() if v == c] for c in classes}
    p(f"\n{'system':34s} " + " ".join(f"{c:>18s}" for c in classes))
    p("-" * 92)

    acc: dict[str, dict[str, float]] = {}
    for sysname in sorted(per):
        cells, a = [], {}
        for c in classes:
            hits = [r["hit"] for r in per[sysname]
                    if r["condition"] == condition and cls.get(r["key"]) == c]
            if hits:
                a[c] = sum(hits) / len(hits)
                cells.append(f"{a[c]:.1%} n={len(hits)}")
            else:
                cells.append("-")
        acc[sysname] = a
        p(f"{sysname:34s} " + " ".join(f"{x:>18s}" for x in cells))

    g = {s: v for s, v in gen.items() if v}
    if g:
        p(f"\n{'system (chrF, free translation)':34s} "
          + " ".join(f"{c:>18s}" for c in classes))
        p("-" * 92)
        for sysname in sorted(g):
            cells = []
            for c in classes:
                rs = [r for r in g[sysname]
                      if r["condition"] == condition and cls.get(r["key"]) == c]
                if rs:
                    sc = corpus_bleu_chrf([r["hyp"] for r in rs], [r["ref"] for r in rs])
                    cells.append(f"{sc['chrf']:.1f} n={len(rs)}")
                else:
                    cells.append("-")
            p(f"{sysname:34s} " + " ".join(f"{x:>18s}" for x in cells))

    ref = ref_system or next((s for s in sorted(acc) if "csl_daily" in s), None)
    if ref and len(acc) > 1:
        import math

        def logit(x, n=313):  # Haldane correction keeps 0% and 100% finite
            x = min(max(x, 0.5 / n), 1 - 0.5 / n)
            return math.log(x / (1 - x))

        p(f"\ngap against {ref} (contrastive), the number the hypothesis is about:")
        for sysname in sorted(acc):
            if sysname == ref:
                continue
            cells = [f"{acc[ref][c] - acc[sysname][c]:+.1%}"
                     if c in acc[ref] and c in acc[sysname] else "-" for c in classes]
            p(f"  vs {sysname:30s} " + " ".join(f"{x:>18s}" for x in cells))
        # Percentage-point gaps are not comparable across strata of different
        # difficulty: a system near ceiling in one column and mid-range in
        # another will show a "widening" gap from the scale alone. Log odds
        # ratios are the scale-free version, so a widening that survives here is
        # an interaction and not an artefact of where the accuracies happen to
        # sit.
        p(f"\nsame gaps as log odds ratios (scale-free — this is the honest test):")
        for sysname in sorted(acc):
            if sysname == ref:
                continue
            cells = [f"{logit(acc[ref][c]) - logit(acc[sysname][c]):+.2f}"
                     if c in acc[ref] and c in acc[sysname] else "-" for c in classes]
            p(f"  vs {sysname:30s} " + " ".join(f"{x:>18s}" for x in cells))
        p("\nA gap that is flat across the three columns is decoder credit: fluent")
        p("Chinese helps everywhere equally. A gap that widens towards 'high' is")
        p("encoder credit — it is worth most exactly where the sign form has to be")
        p("read, which is what the genealogy account predicts and the written-")
        p("language account does not.")

    p(f"\nmean utterance duration by class: "
      + "  ".join(f"{c} {sum(dur[c])/len(dur[c]):.0f}ms" for c in classes if dur[c]))
    if field == "density":
        p("Flat by construction — the terciles are taken inside utterance-length")
        p("bins, so this stratum is not a length stratum in disguise. It is")
        p("therefore the cleaner of the two tests.")
    else:
        p("NOT flat, and it cannot be made flat: reduction is duration per token,")
        p("so a rushed utterance is a short one. 'Reduced is harder' and 'short is")
        p("harder' stay entangled here, and any effect has to be reported as the")
        p("pair. The density stratum is the length-controlled one.")


def latex_main(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"& \multicolumn{2}{c}{Anaphoric} & \multicolumn{2}{c}{Deictic} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Condition & Acc. & $n$ & Acc. & $n$ \\", r"\midrule",
    ]
    for c in MAIN + CONTROL:
        ra, rd = sel(rows, c, "anaphoric"), sel(rows, c, "deictic")
        if not ra and not rd:
            continue
        ca, cd = cell([r["hit"] for r in ra]), cell([r["hit"] for r in rd])
        label = LABEL.get(c, c).replace("&", r"\&").replace("_", r"\_")
        lines.append(
            f"{label} & {ca.acc*100:.1f} & {ca.n} & {cd.acc*100:.1f} & {cd.n} \\\\"
        )
        if c == "both":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+", type=Path)
    ap.add_argument("--latex", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strata", type=Path, default=Path("data/strata.json"),
                    help="difficulty strata from 12_strata.py; absent = skip axis 3")
    ap.add_argument("--strata-condition", default="local",
                    help="which condition the stratum tables read")
    a = ap.parse_args()

    per, gen = load(a.evals)
    strata = json.loads(a.strata.read_text()) if a.strata.exists() else None
    f = a.out.open("w") if a.out else sys.stdout
    for sysname in sorted(per):
        rows = per[sysname]
        print(f"\n{'=' * 96}\nsystem: {sysname}   ({len(rows)} scored rows)\n{'=' * 96}",
              file=f)
        # Dispatch on what the file actually holds. Tables 2-4 are built around
        # the anaphoric/deictic contrast and print empty for any set without it.
        if {r["item_type"] for r in rows} & {"anaphoric", "deictic"}:
            table_main(rows, f)
            table_deltas(rows, f, prior_adv=False)
            table_deltas(rows, f, prior_adv=True)
            table_breakdown(rows, f)
        else:
            table_flat(rows, f)
        table_generation(gen.get(sysname, []), f)
    if strata:
        for field in ("density", "reduction"):
            table_strata(per, gen, strata, f, field, a.strata_condition)
    if a.latex:
        main_sys = max(per, key=lambda s: len(per[s]))
        a.latex.write_text(latex_main(per[main_sys]))
        print(f"\nLaTeX (system {main_sys}) -> {a.latex}", file=f)
    if a.out:
        f.close()
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
