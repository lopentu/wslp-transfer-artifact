#!/usr/bin/env python3
"""How close is the TSL form to each other sign language's form of the same concept?

Stage 3 of the JSL probe (`experiment-suggestion-first.md` §3), on the
embeddings `14_signclip_embed.py` writes.

The measurement
---------------
SignCLIP's text prompt ``<en> <jsl> house`` was trained against *Japanese*
videos of "house", so it encodes how JSL signs that concept. Scoring a TSL clip
of "house" against that prompt therefore asks: **does the Taiwanese form look
like the Japanese one?** TSL is not in the model's training inventory, so this
is a measurement rather than a lookup.

Why the headline metric is a rank and not a cosine
--------------------------------------------------
Raw cosine is not comparable across language tags. Each ``<xx>`` tag sits
somewhere of its own in the text space, and a language whose tag happens to lie
nearer the video manifold gets a higher cosine for every concept alike, which
says nothing about form. Retrieval rank is immune to it: within one language we
rank that language's prompts for all concepts against one clip, so any constant
per-language offset cancels exactly. Mean cosine is reported too, purely so the
size of that artefact is visible.

The controls, all of which are needed
-------------------------------------
* **Unrelated languages.** German (`gsg`) is the pre-registered "not a
  relative" control, and the full 41-language panel puts the four languages of
  interest on a scale rather than leaving them free-floating. Sign languages
  share iconicity everywhere, so a bare TSL--JSL number is uninterpretable.
* **Training volume.** The panel's languages differ 20-fold in training clips,
  and a better-trained language could retrieve better for that reason alone.
  Reported as a rank correlation across the panel, and JSL is the strong case
  here: it has the *fewest* clips of the four (9,238 against DGS's 15,657), so
  volume works against the hypothesis rather than for it.
* **Label permutation.** Shuffling which clip belongs to which concept must
  drop every language to chance. If it does not, the metric is reading
  something other than concept identity.
* **Concept attestation.** `--vocab sp` restricts to glosses that appear in
  SignCLIP's own training concept list, so a miss cannot just mean the model
  never saw the word in any language.

    .venv_signclip/bin/python scripts/15_signclip_similarity.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "signclip" / "embeddings.npz"
# One file per concept set. They are not interchangeable -- JSL's rank after
# centring is 1 under `sp` and 2 under `all` -- so writing both to one path
# would let whichever ran last silently become "the" result.
OUT = ROOT / "data" / "signclip" / "similarity_{vocab}.json"
FOCUS = ["jsl", "csl", "ase", "gsg"]
PRETTY = {"jsl": "Japanese", "csl": "Chinese", "ase": "American",
          "gsg": "German", "tsm": "Turkish", "bfi": "British"}


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def rank_of_correct(video: np.ndarray, text: np.ndarray,
                    correct: np.ndarray) -> np.ndarray:
    """(L, N) 1-based rank of each clip's own concept, per language.

    Ties are counted as "how many candidates strictly beat it, plus one", which
    is the optimistic convention; with float32 cosines exact ties essentially do
    not occur, and any that do are identical prompts.
    """
    out = np.zeros((text.shape[0], video.shape[0]), np.int32)
    n = np.arange(video.shape[0])
    for i in range(text.shape[0]):
        s = video @ text[i].T                      # (N, C)
        out[i] = 1 + (s > s[n, correct][:, None]).sum(1)
    return out


def mrr(ranks: np.ndarray) -> np.ndarray:
    return (1.0 / ranks).mean(axis=1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", choices=["sp", "all"], default="sp",
                    help="'sp' keeps only glosses attested in SignCLIP's own "
                         "training concept list")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = np.load(EMB, allow_pickle=True)
    langs = [str(x) for x in z["langs"]]
    video = unit(z["video"].astype(np.float32))
    text = unit(z["text"].astype(np.float32))
    gidx = z["gloss_index"]
    uniq = [str(x) for x in z["uniq_gloss"]]
    vol = z["lang_train_clips"].astype(float)
    in_sp = z["in_sp_vocab"]

    # Candidate concepts and the clips scored against them. Restricting the
    # clips without restricting the candidates would leave the model able to
    # answer with a concept no longer in play, so both move together.
    if a.vocab == "sp":
        keep_clip = in_sp.astype(bool)
        cand = sorted({int(g) for g in gidx[keep_clip]})
    else:
        keep_clip = np.ones(len(gidx), bool)
        cand = sorted(set(int(g) for g in gidx))
    pos = {g: i for i, g in enumerate(cand)}
    correct = np.array([pos[int(g)] for g in gidx[keep_clip]])
    V = video[keep_clip]
    T = text[:, cand, :]
    n_clip, n_cand = V.shape[0], len(cand)
    print(f"{n_clip} clips over {n_cand} candidate concepts "
          f"(chance MRR ~= {np.log(n_cand) / n_cand:.4f}, chance R@1 = "
          f"{100 / n_cand:.2f}%), {len(langs)} languages")

    ranks = rank_of_correct(V, T, correct)
    rr = 1.0 / ranks
    score = mrr(ranks)
    r1 = (ranks == 1).mean(axis=1) * 100
    r5 = (ranks <= 5).mean(axis=1) * 100
    med = np.median(ranks, axis=1)
    # Raw mean cosine to the correct concept: reported only to show how large
    # the per-language offset is, i.e. why the ranking metric is the headline.
    cos_correct = np.array([(V * T[i][correct]).sum(1).mean()
                            for i in range(len(langs))])

    order = np.argsort(-score)
    print(f"\n{'=' * 78}\nForm similarity of TSL to each sign language "
          f"(SignCLIP retrieval, TSL unseen)\n{'=' * 78}")
    print(f"{'':4s} {'lang':8s} {'MRR':>7s} {'R@1':>7s} {'R@5':>7s} "
          f"{'medRk':>6s} {'meanCos':>8s} {'trainClips':>11s}")
    for rank, i in enumerate(order, 1):
        star = "*" if langs[i] in FOCUS else " "
        name = PRETTY.get(langs[i], "")
        print(f"{rank:>3d}{star} {langs[i]:8s} {score[i]:7.4f} {r1[i]:7.2f} "
              f"{r5[i]:7.2f} {med[i]:6.0f} {cos_correct[i]:8.4f} "
              f"{vol[i]:11,.0f}  {name}")

    # ------------------------------------------------------------- the contrast
    idx = {l: i for i, l in enumerate(langs)}
    rng = np.random.default_rng(a.seed)
    boot = rng.integers(0, n_clip, size=(a.boot, n_clip))

    def ci(v: np.ndarray) -> tuple[float, float]:
        s = v[boot].mean(axis=1)
        return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

    print(f"\n{'=' * 78}\nThe four languages the probe is about\n{'=' * 78}")
    focus_rows = {}
    for l in FOCUS:
        i = idx[l]
        lo, hi = ci(rr[i])
        panel_rank = int(np.where(order == i)[0][0]) + 1
        focus_rows[l] = {"mrr": float(score[i]), "ci": [lo, hi],
                         "r1": float(r1[i]), "r5": float(r5[i]),
                         "median_rank": float(med[i]),
                         "panel_rank": panel_rank,
                         "train_clips": float(vol[i]),
                         "mean_cos": float(cos_correct[i])}
        print(f"  {PRETTY[l]:10s} ({l})  MRR {score[i]:.4f} "
              f"[{lo:.4f}, {hi:.4f}]   R@1 {r1[i]:5.2f}%   "
              f"rank {panel_rank}/{len(langs)} of the panel")

    print("\n  paired differences (bootstrap over clips, 95% CI):")
    pairs = [("jsl", "csl"), ("jsl", "ase"), ("jsl", "gsg"),
             ("csl", "ase"), ("csl", "gsg"), ("ase", "gsg")]
    diffs = {}
    for x, y in pairs:
        d = rr[idx[x]] - rr[idx[y]]
        lo, hi = ci(d)
        sig = "" if lo <= 0 <= hi else "  <-- excludes 0"
        diffs[f"{x}-{y}"] = {"delta": float(d.mean()), "ci": [lo, hi]}
        print(f"    {x} - {y}: {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")

    # per-clip winner among the four, by rank (offset-free)
    fi = [idx[l] for l in FOCUS]
    fr = ranks[fi]
    best = fr.min(axis=0)
    wins = {}
    for k, l in enumerate(FOCUS):
        tied = (fr == best).sum(axis=0)
        wins[l] = float((((fr[k] == best) / tied)).sum() / n_clip * 100)
    print("\n  share of clips whose form each language predicts best "
          "(ties split):")
    for l in sorted(wins, key=lambda x: -wins[x]):
        print(f"    {PRETTY[l]:10s} {wins[l]:5.1f}%")

    # ------------------------------------------------------------- the controls
    print(f"\n{'=' * 78}\nControls\n{'=' * 78}")
    rho = spearman(score, np.log(vol))
    print(f"  training volume: Spearman(MRR, log train clips) over "
          f"{len(langs)} languages = {rho:+.3f}")
    print(f"    JSL has {vol[idx['jsl']]:,.0f} training clips against DGS's "
          f"{vol[idx['gsg']]:,.0f} — the fewest of the four, so volume cannot "
          f"explain a JSL advantage.")

    perm = rng.permutation(n_clip)
    pr = rank_of_correct(V, T, correct[perm])
    ps = mrr(pr)
    print(f"  label permutation: MRR falls to "
          f"{ps.min():.4f}-{ps.max():.4f} across languages "
          f"(chance ~= {np.log(n_cand) / n_cand:.4f}); "
          f"real range {score.min():.4f}-{score.max():.4f}")

    # Split-half reliability. The mid-field of the panel orders languages that
    # differ by less than the bootstrap width, and a one-clip language (`ysl`)
    # lands in the top ten, so the ordering as a whole has to be shown to be
    # reproducible before any of it is read. Scoring two disjoint halves of the
    # clips and correlating the two league tables is the direct test.
    half = rng.permutation(n_clip)
    a_half, b_half = half[: n_clip // 2], half[n_clip // 2:]
    sa, sb = mrr(ranks[:, a_half]), mrr(ranks[:, b_half])
    rel = spearman(sa, sb)
    print(f"  split-half reliability of the 41-language ordering: "
          f"Spearman = {rel:+.3f}")
    print(f"    JSL is rank {int(np.argsort(-sa).argsort()[idx['jsl']]) + 1} "
          f"and {int(np.argsort(-sb).argsort()[idx['jsl']]) + 1} "
          f"in the two halves independently")

    # Text-space spread. Retrieval improves mechanically if a language's concept
    # embeddings are simply further apart, whatever they encode — an anisotropy
    # artefact rather than a fact about form. If that drove the result, MRR
    # would track spread across the panel.
    spread = np.array([1.0 - float((T[i] @ T[i].T).mean()) for i in range(len(langs))])
    print(f"  text-space spread: Spearman(MRR, 1 - mean pairwise cos) = "
          f"{spearman(score, spread):+.3f}  "
          f"(jsl {spread[idx['jsl']]:.4f} vs panel median {np.median(spread):.4f})")

    # Volume again, restricted to languages whose tag is actually trained. A tag
    # the model barely saw cannot encode a language, so those rows measure a
    # near-language-neutral prompt and do not belong in a volume regression.
    big = vol >= 5000
    print(f"  volume among the {big.sum()} well-trained languages "
          f"(>=5,000 clips): Spearman = {spearman(score[big], np.log(vol[big])):+.3f}; "
          f"JSL rank {int(np.argsort(-score[big]).argsort()[big[:idx['jsl']].sum()]) + 1}"
          f"/{big.sum()} among them")

    # Anisotropy, removed rather than argued about. Centring each language's
    # concept embeddings on their own mean deletes the direction they all share
    # — the tag's own position and whatever global spread goes with it — and
    # leaves only how each concept differs from that language's average
    # concept. If the JSL result is an artefact of its text space being more
    # compact than the panel's, it does not survive this; if it is about form,
    # it does.
    Tc = unit(T - T.mean(axis=1, keepdims=True))
    ranks_c = rank_of_correct(V, Tc, correct)
    score_c = mrr(ranks_c)
    order_c = np.argsort(-score_c)
    print("  centred text space (per-language mean removed):")
    print("    top 5: " + ", ".join(
        f"{langs[i]} {score_c[i]:.4f}" for i in order_c[:5]))
    for l in FOCUS:
        i = idx[l]
        print(f"    {PRETTY[l]:10s} MRR {score_c[i]:.4f}  "
              f"rank {int(np.where(order_c == i)[0][0]) + 1}/{len(langs)}")

    # Paired sign test on the focus pairs: how many individual clips prefer one
    # language's form to the other's, which needs no distributional assumption.
    print("  per-clip paired sign test (clips strictly preferring the first):")
    signs = {}
    for x, y in [("jsl", "csl"), ("jsl", "ase"), ("jsl", "gsg")]:
        d = ranks[idx[y]] - ranks[idx[x]]        # positive = x ranks correct higher
        w, l = int((d > 0).sum()), int((d < 0).sum())
        # Normal approximation to the binomial, ties excluded.
        z = (w - (w + l) / 2) / np.sqrt((w + l) / 4) if w + l else 0.0
        signs[f"{x}-{y}"] = {"wins": w, "losses": l, "z": float(z)}
        print(f"    {x} > {y} on {w} clips, < on {l}  (z = {z:+.2f})")

    out = {
        "n_clips": int(n_clip), "n_candidates": int(n_cand),
        "vocab": a.vocab, "languages": langs,
        "panel": [{"lang": langs[i], "mrr": float(score[i]),
                   "r1": float(r1[i]), "r5": float(r5[i]),
                   "median_rank": float(med[i]),
                   "mean_cos": float(cos_correct[i]),
                   "train_clips": float(vol[i])} for i in order],
        "focus": focus_rows, "pairs": diffs, "wins_pct": wins,
        "volume_spearman": rho,
        "volume_spearman_welltrained": spearman(score[big], np.log(vol[big])),
        "split_half_spearman": rel,
        "spread_spearman": spearman(score, spread),
        "centred": {langs[i]: float(score_c[i]) for i in order_c},
        "centred_focus_rank": {l: int(np.where(order_c == idx[l])[0][0]) + 1
                               for l in FOCUS},
        "sign_tests": signs,
        "permuted_mrr_range": [float(ps.min()), float(ps.max())],
        "chance_mrr": float(np.log(n_cand) / n_cand),
    }
    dst = Path(str(OUT).format(vocab=a.vocab))
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
