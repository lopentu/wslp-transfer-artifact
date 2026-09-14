#!/usr/bin/env python3
"""Synthesise `lm_head` donors that no released checkpoint can provide.

Three of the 8/22 reviews converge on the same objection to the neutral-head
result (`how2sign + mt5_base lm_head`, 47.9 against 24.5 intact): the paper shows
that *removing* an English-output-specialised projection restores transfer, but
not *what about its removal* matters. Two cheap alternatives survive:

  W2 (ReviewerA)  a per-script logit SCALE artefact. `how2sign`'s CJK rows end at
                  norm 2.67 against its Latin rows' 4.21 -- ratio 0.64, against
                  1.02 at the shared mT5-base origin. If the collapse on a
                  Chinese contrastive set is just those rows being quiet, then
                  rescaling them is enough and no account of "adaptation" is
                  needed.
  P3 (ReviewerO)  a RESET artefact. If a freshly randomised projection rescues as
                  well as mT5-base's, the mechanism is "remove the maladapted
                  matrix", not "install a good multilingual geometry".

Both need a donor that does not exist on disk, so build them:

    --rand              transformers' own init for this tensor, untrained
    --rand-normmatched  random directions, per-row norms taken from mT5-base --
                        the scale control for --rand, since the library init and
                        a trained head need not live at the same radius
    --rescale NAME      NAME's own head with its CJK rows scaled by
                        (mT5-base cjk/latin) / (NAME cjk/latin), i.e. the ratio
                        restored and nothing else moved
    --rescale-global    ... and/or every row scaled so the mean row norm matches
                        mT5-base's, which is the other scale confound: these
                        heads sit at a third of mT5-base's radius overall, not
                        only in the wrong proportion
    --scale-cjk F       an explicit CJK factor, so the calibrated point can be
                        read against a sweep rather than on its own

Each donor is written in BOTH shapes, because the two interventions the reviews
ask for consume different files:

  external shape  {"model": {"mt5_model.lm_head.weight": W}}  -> `--init-mt5 PATH
                  --init-mt5-parts lm_head`, i.e. the intervention at
                  initialisation, which needs training.
  run shape       {"arch": "unisign", "state": {"core.mt5_model.lm_head.weight":
                  W}} -> `05_eval.py --posthoc-head PATH`, i.e. the same tensor
                  installed after fine-tuning, which needs no training at all.

The run shape is deliberately NOT a full run checkpoint: `graft_head` moves one
tensor by name and raises on anything else, so a two-key file is all it reads and
a partial file cannot silently displace something else.

    ../.venv/bin/python scripts/27_head_donors.py --all   # -> data/external/, ~1 min/donor

CPU only. Each file is 0.72 GiB (250112x768 in fp32), so `--all` writes ~7 GiB;
they go to data/external/ beside the released checkpoints, which is on /mnt/md0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
DONORS = EXT / "head_donors"
OUT_JSON = ROOT / "data" / "head_donors.json"

HEAD_KEY = "mt5_model.lm_head.weight"
RUN_KEY = "core.mt5_model.lm_head.weight"

# Same partition as scripts/24_head_geometry.py, and it must stay the same one:
# the factors below are calibrated against that script's published ratios, so a
# different tokenizer split would silently recalibrate them.
HAN = re.compile(r"[一-鿿㐀-䶿]")
LATIN = re.compile(r"[A-Za-z]")
SPM_SPACE = "▁"

FILES = {
    "csl_stage1": "csl_stage1_weight.pth",
    "csl_daily": "csl_daily_pose_only_slt.pth",
    "how2sign": "how2sign_pose_only_slt.pth",
    "openasl": "openasl_pose_only_slt.pth",
    "wlasl": "wlasl_pose_only_islr.pth",
    "mt5_base": "mt5_base.pth",
}


# --------------------------------------------------------------------- loading


def load_head(path: Path) -> torch.Tensor:
    """The `lm_head` out of either checkpoint shape, as fp32."""
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = obj
    # "state" is 04_train.py's own key, so a donor can be derived from a TRAINED
    # run and not only from a released file. That matters for the post-hoc arm:
    # ReviewerA's A3 rescales the head the fine-tuned model is actually wearing,
    # which is not quite the released one (cos 0.988-1.000 from its init).
    for key in ("model", "state", "state_dict", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    for k, v in sd.items():
        kk = k.replace("module.", "", 1)
        if kk in (HEAD_KEY, RUN_KEY):
            return v.detach().float().clone()
    raise SystemExit(f"{path} has no lm_head tensor")


def row_groups(n_rows: int) -> dict[str, torch.Tensor]:
    """Boolean masks over vocabulary rows: CJK / Latin-with-no-Han / the rest."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/mt5-base")
    cjk = torch.zeros(n_rows, dtype=torch.bool)
    latin = torch.zeros(n_rows, dtype=torch.bool)
    for i in range(min(len(tok), n_rows)):
        piece = tok.convert_ids_to_tokens(i) or ""
        piece = piece.replace(SPM_SPACE, "")
        if HAN.search(piece):
            cjk[i] = True
        elif LATIN.search(piece):
            latin[i] = True
    rest = ~(cjk | latin)
    return {"cjk": cjk, "latin": latin, "rest": rest}


def norm_stats(w: torch.Tensor, masks: dict[str, torch.Tensor]) -> dict:
    norms = w.norm(dim=1)
    out = {f"{g}_mean_norm": float(norms[m].mean()) for g, m in masks.items()}
    out["all_mean_norm"] = float(norms.mean())
    out["ratio_cjk_latin"] = out["cjk_mean_norm"] / out["latin_mean_norm"]
    return out


# ---------------------------------------------------------------- constructors


def rand_head(n_rows: int, d: int) -> torch.Tensor:
    """transformers' own initialisation for this tensor, never trained.

    Built by instantiating the config rather than by drawing from a normal here,
    so that "randomly initialised" means what the library means by it (T5 scales
    `lm_head` by `initializer_factor`, which a hand-rolled draw would miss) and
    ReviewerO's P3 answers the question actually asked.
    """
    from transformers import MT5Config, MT5ForConditionalGeneration

    cfg = MT5Config.from_pretrained("google/mt5-base")
    # transformers 5.15 reports `tie_word_embeddings=True` for this config, but
    # every released checkpoint -- and `from_pretrained`'s own mt5-base -- stores
    # an `lm_head.weight` that is orthogonal to the embedding (cos 0.003-0.022,
    # norms 26x apart), so the published model is in fact untied and the tensor
    # the paper swaps is a real separate matrix. Untie explicitly, or a fresh
    # init would hand back the embedding and this control would silently be a
    # different experiment.
    cfg.tie_word_embeddings = False
    torch.manual_seed(0)
    m = MT5ForConditionalGeneration(cfg)
    w = m.lm_head.weight.detach().float().clone()
    if tuple(w.shape) != (n_rows, d):
        raise SystemExit(f"fresh lm_head is {tuple(w.shape)}, expected {(n_rows, d)}")
    return w


def permute_rows(w: torch.Tensor, seed: int,
                 masks: dict[str, torch.Tensor] | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder the vocabulary rows of `w`, changing nothing else.

    The camera-ready reviewer's control on what makes a projection a good
    initialisation. Every donor so far varies the matrix's CONTENT -- a fresh
    draw, a rescaling, a different fine-tune -- so each one changes the row
    statistics and the token correspondence together, and the rescue could be
    credited to either. A row permutation separates them exactly: the multiset
    of rows is untouched, so the Frobenius norm, the row-norm distribution, the
    per-script norm structure in aggregate, the singular values and hence the
    conditioning are all bit-for-bit those of mT5-base (a row permutation is a
    left multiplication by a permutation matrix). The ONLY thing destroyed is
    which token each pretrained row scores.

    So if the permuted head rescues as well as mT5-base's, what the recipient
    needs is generic pretrained matrix structure; if it does not, the rescue
    depends on rows sitting against the tokens they were trained for. The input
    embedding is left alone in both cases -- `--init-mt5-parts lm_head` moves
    this tensor and no other -- so the model still READS the vocabulary
    correctly and only its output side is scrambled.

    With `masks`, the permutation is drawn within each script group instead of
    over the whole vocabulary. That keeps every row in a slot of its own script,
    which is the sharper version of the same question: the paper's CJK/Latin
    rescaling donors already show that per-script SCALE does not explain the
    rescue, and a within-script permutation asks whether per-script membership
    is all the surviving structure amounts to.
    """
    g = torch.Generator().manual_seed(seed)
    n = w.shape[0]
    if masks is None:
        perm = torch.randperm(n, generator=g)
    else:
        perm = torch.arange(n)
        for _, m in sorted(masks.items()):
            idx = m.nonzero(as_tuple=True)[0]
            perm[idx] = idx[torch.randperm(len(idx), generator=g)]
    return w[perm].contiguous(), perm


def norm_match(w: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rescale each row of `w` to the norm of the corresponding row of `target`.

    Directions untouched. This is the scale control for `--rand`: if a fresh head
    behaves differently from mT5-base's only because it sits at a different
    radius, this donor removes that difference and leaves the geometry.
    """
    wn = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    tn = target.norm(dim=1, keepdim=True)
    return w * (tn / wn)


def rescale(w: torch.Tensor, masks: dict, cjk_factor: float | None,
            global_match: torch.Tensor | None) -> torch.Tensor:
    out = w.clone()
    if cjk_factor is not None:
        out[masks["cjk"]] *= cjk_factor
    if global_match is not None:
        # One scalar over every row, chosen so the MEAN row norm matches. Applied
        # after the CJK factor so the two are composable and the per-script ratio
        # a `--rescale` donor was built for survives the global step.
        cur = float(out.norm(dim=1).mean())
        tgt = float(global_match.norm(dim=1).mean())
        out *= tgt / cur
    return out


# ------------------------------------------------------------------- emission


def emit(name: str, w: torch.Tensor, stats: dict) -> dict:
    DONORS.mkdir(parents=True, exist_ok=True)
    ext = DONORS / f"{name}.pth"
    run = DONORS / f"{name}.run.pth"
    torch.save({"model": {HEAD_KEY: w}}, ext)
    torch.save({"arch": "unisign", "state": {RUN_KEY: w},
                "cfg": {"synthetic_head": name}}, run)
    rec = dict(stats)
    rec["external"] = str(ext.relative_to(ROOT))
    rec["run"] = str(run.relative_to(ROOT))
    rec["gib"] = round(ext.stat().st_size / 2**30, 3)
    print(f"  -> {ext.name} / {run.name}  ({rec['gib']} GiB)  "
          f"cjk/latin={stats.get('ratio_cjk_latin', float('nan')):.4f}  "
          f"mean={stats.get('all_mean_norm', float('nan')):.3f}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rand", action="store_true")
    ap.add_argument("--rand-normmatched", action="store_true")
    ap.add_argument("--rescale", action="append", default=[],
                    help="checkpoint name or runs/<run> whose head to rescale")
    ap.add_argument("--scale-cjk", type=float, action="append", default=[],
                    help="explicit CJK factors; default is the calibrated one")
    ap.add_argument("--rescale-global", action="store_true",
                    help="also match the overall mean row norm to mT5-base's")
    ap.add_argument("--passthrough", action="append", default=[],
                    help="re-emit a head unchanged, in RUN shape. The point is "
                         "the post-hoc arm: `--posthoc-head` reads a run "
                         "checkpoint, so an external file (mt5_base.pth) cannot "
                         "be grafted onto a fine-tuned model without this.")
    ap.add_argument("--permute", action="append", default=[],
                    help="checkpoint name whose head to row-permute: the same "
                         "pretrained rows against the wrong tokens. Preserves "
                         "every global statistic of the matrix exactly, so a "
                         "null here separates lexical alignment from generic "
                         "pretrained structure (camera-ready reviewer 2).")
    ap.add_argument("--permute-within-script", action="store_true",
                    help="draw the permutation inside each CJK/Latin/other "
                         "group rather than over the whole vocabulary")
    ap.add_argument("--permute-seed", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="every donor the three reviews ask for")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    if a.all:
        a.rand = a.rand_normmatched = a.rescale_global = True
        a.rescale = a.rescale or ["how2sign"]

    base = load_head(EXT / FILES["mt5_base"])
    n_rows, d = base.shape
    print(f"mt5-base lm_head {n_rows}x{d}")
    masks = row_groups(n_rows)
    print("rows: " + "  ".join(f"{g}={int(m.sum())}" for g, m in masks.items()))
    base_stats = norm_stats(base, masks)
    print(f"mt5-base cjk/latin = {base_stats['ratio_cjk_latin']:.4f}, "
          f"mean row norm {base_stats['all_mean_norm']:.3f}")

    out: dict = {"vocab": {g: int(m.sum()) for g, m in masks.items()},
                 "mt5_base": base_stats, "donors": {}}

    if a.rand:
        print("rand: transformers init, untrained")
        w = rand_head(n_rows, d)
        st = norm_stats(w, masks)
        st["kind"] = "library-random"
        out["donors"]["rand_head"] = emit("rand_head", w, st)

        if a.rand_normmatched:
            print("rand, norm-matched to mt5-base per row")
            wn = norm_match(w, base)
            stn = norm_stats(wn, masks)
            stn["kind"] = "library-random, per-row norms from mt5-base"
            out["donors"]["rand_head_nm"] = emit("rand_head_nm", wn, stn)

    for src in a.permute:
        path = (ROOT / src / "best.pt") if src.startswith("runs/") else EXT / FILES[src]
        tag = src.replace("runs/", "").replace("/", "_")
        w = load_head(path)
        for within in ([False, True] if a.permute_within_script else [False]):
            lab = f"{tag}_perm" + ("ws" if within else "")
            wp, perm = permute_rows(w, a.permute_seed,
                                    masks if within else None)
            moved = int((perm != torch.arange(len(perm))).sum())
            # A row permutation cannot change these, so a mismatch means the
            # permutation was applied to the wrong axis -- the one failure mode
            # that would leave the control looking fine while testing nothing.
            #
            # In float64, not the tensor's own float32: summing 192 M squares in
            # fp32 is accumulation-order dependent, and reordering the rows moves
            # the reported Frobenius norm by 3e-4 relative -- 30x `allclose`'s
            # tolerance -- while the true value is identical to 13 figures. The
            # fp32 sum is also 3% below the fp64 one for both matrices, so this
            # is the summation and not the permutation.
            assert torch.allclose(wp.double().norm(), w.double().norm()), \
                "Frobenius norm moved"
            assert torch.allclose(wp.double().norm(dim=1).sort().values,
                                  w.double().norm(dim=1).sort().values), \
                "row norms moved"
            st = norm_stats(wp, masks)
            st.update(kind="row-permuted" + ("-within-script" if within else ""),
                      source=src, permute_seed=a.permute_seed,
                      rows_moved=moved, rows=len(perm))
            print(f"{src}: row-permuted{' within script' if within else ''}, "
                  f"{moved}/{len(perm)} rows moved")
            out["donors"][lab] = emit(lab, wp, st)

    for src in a.passthrough:
        path = (ROOT / src / "best.pt") if src.startswith("runs/") else EXT / FILES[src]
        tag = src.replace("runs/", "").replace("/", "_")
        w = load_head(path)
        st = norm_stats(w, masks)
        st.update(kind="passthrough", source=src)
        print(f"passthrough {src}")
        out["donors"][f"{tag}_pass"] = emit(f"{tag}_pass", w, st)

    for src in a.rescale:
        path = (ROOT / src / "best.pt") if src.startswith("runs/") else EXT / FILES[src]
        tag = src.replace("runs/", "").replace("/", "_")
        w = load_head(path)
        st0 = norm_stats(w, masks)
        print(f"{src}: cjk/latin={st0['ratio_cjk_latin']:.4f}, "
              f"mean row norm {st0['all_mean_norm']:.3f}")
        calibrated = base_stats["ratio_cjk_latin"] / st0["ratio_cjk_latin"]
        factors = a.scale_cjk or [round(calibrated, 4)]
        for f in factors:
            lab = f"{tag}_cjk{f:g}".replace(".", "p")
            wr = rescale(w, masks, f, None)
            st = norm_stats(wr, masks)
            st.update(kind="cjk-rescaled", source=src, cjk_factor=f,
                      calibrated_factor=round(calibrated, 4))
            out["donors"][lab] = emit(lab, wr, st)
        if a.rescale_global:
            # The other half of the scale objection: these heads are not only in
            # the wrong PROPORTION, they are at a third of mT5-base's radius.
            for f, lab in ((None, f"{tag}_gl"),
                           (round(calibrated, 4), f"{tag}_cjkgl")):
                wr = rescale(w, masks, f, base)
                st = norm_stats(wr, masks)
                st.update(kind="global-rescaled" if f is None else "cjk+global",
                          source=src, cjk_factor=f)
                out["donors"][lab] = emit(lab, wr, st)

    # Merged, not overwritten: the donors are built in several passes (the
    # released heads first, then ones derived from a trained run once it exists),
    # and a fresh write would drop the earlier passes' records.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        prev = json.loads(a.out.read_text())
        merged = dict(prev.get("donors", {}))
        merged.update(out["donors"])
        out["donors"] = merged
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n{len(out['donors'])} donors on record -> {a.out}")


if __name__ == "__main__":
    main()
